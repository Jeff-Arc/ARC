#!/usr/bin/env python3
"""
arc_discovery_v5.py — Systematic feature discovery for arc_v5.

Ports arc_discovery.py to the arc_v5 schema.  Same analytical approach:
  Supervised   → LassoCV / L1 LogisticRegressionCV on full feature matrix
  Unsupervised → KMeans on trajectory fingerprint (T0 + T-1/T-2/T-3 lags)

PRIMARY SOURCE TABLE: cluster_snapshot
PERIOD METRICS:       cloud_f_period
DEATH SCORES:         ml_death_scores
LABELS:               cluster_labels

TARGET SETS (--targets flag):
  default    → is_dead (binary), drift_magnitude_next (continuous)
  hidden     → spectral_gap_next, algebraic_connectivity_next, leiden_modularity_next
  strategic  → void_count_adjacent_next, growth_rate_next, phase_transition_score_next
  all        → union of all above

RESULTS:
  discovery_results    — supervised scores + top feature coefficients (JSONB)
  discovery_archetypes — unsupervised KMeans archetype profiles

Usage:
  python3 arc_discovery_v5.py --corpus G06N_quarterly --mode supervised
  python3 arc_discovery_v5.py --corpus G06N_quarterly --mode supervised --targets strategic
  python3 arc_discovery_v5.py --corpus H01L_quarterly --mode unsupervised
  python3 arc_discovery_v5.py --corpus all --mode both --targets all
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import psycopg2
import psycopg2.extras

try:
    import pandas as pd
    from sklearn.linear_model import LassoCV, LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\nInstall: pip install scikit-learn pandas")

try:
    from mlxtend.frequent_patterns import apriori, association_rules as mlxtend_assoc_rules
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False

try:
    from pysr import PySRRegressor
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ── Column exclusions ─────────────────────────────────────────────────────────

# Non-feature columns to exclude from the cluster_snapshot feature matrix
SNAPSHOT_EXCLUDE = frozenset([
    # Identity / keys
    'corpus_id', 'period_start', 'cluster_id', 'persistent_cluster_id',
    'match_type',
    # Non-scalar (1024-dim vectors)
    'centroid', 'drift_vector',
    # LLM text
    'cluster_label', 'cluster_summary',
    # Primary targets — excluded globally (circular regardless of active target)
    'is_dead', 'is_new',
    # Date column in disguise
    'reborn_from_period',
])

# Exclude from field-level (cloud_f_period) features
FIELD_EXCLUDE = frozenset([
    'corpus_id', 'period_start',
])

# Per-target circularity guards: drop these features only for specific targets
TARGET_CONDITIONAL_EXCLUDE: dict = {
    'is_dead': frozenset(['death_probability']),
}

# The 12 features used in the arc_v4 cross-corpus death model.
# Features NOT in this set are flagged as "new v5" in results.
V4_12_FEATURES = frozenset([
    'cohesion', 'size', 'elongation_ratio', 'drift_magnitude',
    'boundary_pressure_rate', 'persistence_score', 'convergence_score',
    'mean_betweenness', 'cohesion_percentile', 'drift_percentile',
    'size_percentile', 'jerk',
])

_NUMERIC_TYPES = (
    'real', 'integer', 'bigint', 'smallint', 'numeric',
    'double precision', 'boolean',
)

MIN_LAG_ROWS    = 30
MIN_TRAJ_PIDS   = 8
MIN_SIL_SCORE   = 0.25

# Death model feature order (must match pgml arc_death_model_v5 training)
DEATH_FEATURES = [
    'cohesion', 'size', 'elongation_ratio', 'drift_magnitude',
    'boundary_pressure_rate', 'persistence_score', 'convergence_score',
    'mean_betweenness', 'cohesion_percentile', 'drift_percentile',
    'size_percentile', 'jerk', 'age_periods', 'n_attractors',
    'mean_triangle_count', 'volume_estimate', 'velocity_alignment',
    'max_adjacent_void_persistence',
]

# Outcome columns used as rule consequents in association mining
RULE_OUTCOME_COLS = [
    'is_dead', 'die_within_4',
    'elongation_ratio_high', 'drift_magnitude_high',
    'size_high', 'n_attractors_high',
]

# ── Target sets ───────────────────────────────────────────────────────────────
# (col_name, is_binary, metric_key)
_TARGET_SETS = {
    'default': [
        ('is_dead',              True,  'roc_auc'),   # Will this cluster die?
        ('drift_magnitude_next', False, 'r2'),         # How far will it move next period?
    ],
    'hidden': [
        ('spectral_gap_next',            False, 'r2'), # Field gap next period
        ('algebraic_connectivity_next',  False, 'r2'), # Graph cohesion next period
        ('leiden_modularity_next',       False, 'r2'), # Community structure next period
    ],
    'strategic': [
        ('void_count_adjacent_next',     False, 'r2'), # Will adjacent voids change?
        ('growth_rate_next',             False, 'r2'), # Will cluster grow or shrink?
        ('phase_transition_score_next',  False, 'r2'), # Is a structural shift coming?
    ],
}
_TARGET_SETS['all'] = (
    _TARGET_SETS['default'] +
    _TARGET_SETS['hidden'] +
    _TARGET_SETS['strategic']
)


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(
        host=os.environ.get('PGHOST', '/var/run/postgresql'),
        dbname=os.environ.get('PGDATABASE', 'arc_v5'),
        user=os.environ.get('PGUSER', 'jeff'),
    )


def get_active_corpora(conn):
    """Return all corpus_ids that have data in cluster_snapshot."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT corpus_id FROM cluster_snapshot ORDER BY corpus_id"
        )
        return [r[0] for r in cur.fetchall()]


def ensure_tables(conn):
    """Create discovery_results and discovery_archetypes if they don't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS discovery_results (
                id                serial PRIMARY KEY,
                corpus_id         text NOT NULL,
                mode              text NOT NULL,
                target            text NOT NULL,
                model_type        text NOT NULL,
                auc_roc           real,
                r2_score          real,
                n_features_used   integer,
                n_nonzero_coefs   integer,
                top_features      jsonb,
                n_samples         integer,
                corpus_ids        text[],
                run_at            timestamptz DEFAULT NOW(),
                notes             text
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS discovery_archetypes (
                id                serial PRIMARY KEY,
                corpus_id         text NOT NULL,
                archetype_id      integer NOT NULL,
                n_clusters        integer,
                death_rate        real,
                reborn_rate       real,
                mean_age_at_death real,
                mean_drift        real,
                mean_cohesion     real,
                label             text,
                run_at            timestamptz DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS discovery_patterns (
                id                serial PRIMARY KEY,
                corpus_id         text,
                mode              text,
                antecedent        text,
                consequent        text,
                support           real,
                confidence        real,
                lift              real,
                combined_score    real,
                n_rows_matching   integer,
                includes_neighbor boolean DEFAULT false,
                lag_periods       integer DEFAULT 0,
                run_at            timestamptz DEFAULT NOW(),
                notes             text
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS dp_corpus_mode ON discovery_patterns (corpus_id, mode)")
        cur.execute("CREATE INDEX IF NOT EXISTS dp_lift ON discovery_patterns (lift DESC NULLS LAST)")
    conn.commit()


def write_supervised_result(conn, corpus_id, mode, target, model_type,
                            score, metric_key, n_features, n_nonzero,
                            top_features, n_samples, corpus_ids, notes):
    """Insert one row into discovery_results."""
    auc  = score if metric_key == 'roc_auc' else None
    r2   = score if metric_key == 'r2'      else None
    with conn.cursor() as cur:
        # Delete previous run for same corpus+target
        cur.execute(
            "DELETE FROM discovery_results WHERE corpus_id=%s AND target=%s AND mode=%s",
            (corpus_id, target, mode),
        )
        cur.execute("""
            INSERT INTO discovery_results
              (corpus_id, mode, target, model_type, auc_roc, r2_score,
               n_features_used, n_nonzero_coefs, top_features, n_samples,
               corpus_ids, notes)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """, (
            corpus_id, mode, target, model_type,
            auc, r2,
            n_features, n_nonzero,
            json.dumps(top_features),
            n_samples,
            corpus_ids or [corpus_id],
            notes,
        ))
        return cur.fetchone()[0]


def write_archetype_result(conn, corpus_id, archetype_id, n_clusters,
                           death_rate, reborn_rate, mean_age_at_death,
                           mean_drift, mean_cohesion, label):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO discovery_archetypes
              (corpus_id, archetype_id, n_clusters, death_rate, reborn_rate,
               mean_age_at_death, mean_drift, mean_cohesion, label)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """, (
            corpus_id, archetype_id, n_clusters,
            float(death_rate) if death_rate is not None else None,
            float(reborn_rate) if reborn_rate is not None else None,
            float(mean_age_at_death) if mean_age_at_death is not None else None,
            float(mean_drift) if mean_drift is not None else None,
            float(mean_cohesion) if mean_cohesion is not None else None,
            label,
        ))
        return cur.fetchone()[0]


# ── Feature column discovery ──────────────────────────────────────────────────

_col_cache: dict = {}


def _get_numeric_cols(conn, table_name, exclude_set):
    """Return list of (col_name, data_type) for numeric/bool columns in relation.

    Uses pg_attribute so it works for tables, views, AND materialized views
    (information_schema.columns only covers tables and regular views).
    """
    if table_name not in _col_cache:
        ph = ','.join(f"'{t}'" for t in _NUMERIC_TYPES)
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT a.attname,
                       a.atttypid::regtype::text AS data_type
                FROM pg_attribute a
                WHERE a.attrelid = %s::regclass
                  AND a.attnum   > 0
                  AND NOT a.attisdropped
                  AND a.atttypid::regtype::text IN ({ph})
                ORDER BY a.attnum
            """, (table_name,))
            _col_cache[table_name] = cur.fetchall()
    return [(c, dt) for c, dt in _col_cache[table_name] if c not in exclude_set]


def _cast(alias, col, dtype):
    """SQL cast expression: boolean → int, everything else → real, with COALESCE."""
    ref = f"{alias}.{col}" if alias else col
    if dtype == 'boolean':
        return f"COALESCE({ref}::int, 0) AS {col}"
    return f"COALESCE({ref}::real, 0) AS {col}"


# ── Supervised: data loading ──────────────────────────────────────────────────

def load_supervised_data(conn, corpus_id):
    """
    Build the full feature matrix for supervised discovery.

    Sources:
      cluster_snapshot     → ~70 numeric/bool features (minus SNAPSHOT_EXCLUDE)
      cluster_snapshot     → T-1/T-2/T-3 lag features via LAG() window
      ml_death_scores      → death_probability
      cloud_f_period       → field-level metrics (joined by corpus+period)

    Returns (df, feat_cols, field_feat_cols) where:
      feat_cols       → cluster-level feature names (T0 + lags + death_prob)
      field_feat_cols → field-level feature names from cloud_f_period
    """
    snap_cols  = _get_numeric_cols(conn, 'cluster_snapshot', SNAPSHOT_EXCLUDE)
    field_cols = _get_numeric_cols(conn, 'cloud_f_period',   FIELD_EXCLUDE)

    # Explicit SELECT for cluster features (no cs.* to avoid vector columns)
    c_select = ',\n            '.join(_cast('cs', c, dt) for c, dt in snap_cols)

    # Prefixed field features to avoid collision with cluster cols
    f_select = ',\n            '.join(
        f"COALESCE(fp.{c}::real, 0) AS fp_{c}" for c, dt in field_cols
    )

    # Lag aliases we expose (T-1, T-2, T-3)
    lag_spec = [
        # (alias, source_col, lag)
        ('size_t1',        'size::real',        1),
        ('cohesion_t1',    'cohesion',          1),
        ('drift_t1',       'drift_magnitude',   1),
        ('jerk_t1',        'jerk',              1),
        ('betweenness_t1', 'mean_betweenness',  1),
        ('persistence_t1', 'persistence_score', 1),
        ('elongation_t1',  'elongation_ratio',  1),
        ('attractors_t1',  'n_attractors::real',1),
        ('size_t2',        'size::real',        2),
        ('drift_t2',       'drift_magnitude',   2),
        ('cohesion_t2',    'cohesion',          2),
        ('size_t3',        'size::real',        3),
        ('drift_t3',       'drift_magnitude',   3),
    ]
    lag_select = ',\n            '.join(
        f"LAG(COALESCE(cs.{col}, 0), {n}) OVER w AS {alias}"
        for alias, col, n in lag_spec
    )

    # Corpus filter: 'all' = no filter
    corpus_filter = (
        "AND cs.corpus_id = %(corpus_id)s" if corpus_id != 'all' else ""
    )
    corpus_param = {'corpus_id': corpus_id} if corpus_id != 'all' else {}

    sql = f"""
        WITH lagged AS (
            SELECT
                cs.corpus_id,
                cs.period_start,
                cs.cluster_id,
                cs.persistent_cluster_id,
                cs.is_dead::int    AS is_dead,
                cs.is_new::int     AS is_new,
                {c_select},
                {lag_select},
                COALESCE(mds.death_probability::real, 0) AS death_probability
            FROM cluster_snapshot cs
            LEFT JOIN ml_death_scores mds
                ON  mds.corpus_id             = cs.corpus_id
                AND mds.persistent_cluster_id = cs.persistent_cluster_id
                AND mds.period_start          = cs.period_start
                AND mds.model_version         = 'cross_corpus_v5_12feat'
            WHERE cs.size >= 3
              AND cs.is_new = false
              {corpus_filter}
            WINDOW w AS (
                PARTITION BY cs.corpus_id, cs.persistent_cluster_id
                ORDER BY cs.period_start
            )
        ),
        field_next AS (
            SELECT
                fp.corpus_id,
                fp.period_start,
                LEAD(fp.spectral_gap) OVER (
                    PARTITION BY fp.corpus_id ORDER BY fp.period_start
                ) AS spectral_gap_next,
                LEAD(fp.algebraic_connectivity) OVER (
                    PARTITION BY fp.corpus_id ORDER BY fp.period_start
                ) AS algebraic_connectivity_next,
                LEAD(fp.leiden_modularity) OVER (
                    PARTITION BY fp.corpus_id ORDER BY fp.period_start
                ) AS leiden_modularity_next,
                LEAD(fp.phase_transition_score) OVER (
                    PARTITION BY fp.corpus_id ORDER BY fp.period_start
                ) AS phase_transition_score_next
            FROM cloud_f_period fp
            {('WHERE fp.corpus_id = %(corpus_id)s' if corpus_id != 'all' else '')}
        )
        SELECT
            l.*,
            {f_select},
            LEAD(l.drift_magnitude) OVER (
                PARTITION BY l.corpus_id, l.persistent_cluster_id
                ORDER BY l.period_start
            ) AS drift_magnitude_next,
            LEAD(l.void_count_adjacent) OVER (
                PARTITION BY l.corpus_id, l.persistent_cluster_id
                ORDER BY l.period_start
            ) AS void_count_adjacent_next,
            LEAD(l.growth_rate) OVER (
                PARTITION BY l.corpus_id, l.persistent_cluster_id
                ORDER BY l.period_start
            ) AS growth_rate_next,
            fn.spectral_gap_next,
            fn.algebraic_connectivity_next,
            fn.leiden_modularity_next,
            fn.phase_transition_score_next
        FROM lagged l
        LEFT JOIN field_next fn
            ON  fn.corpus_id    = l.corpus_id
            AND fn.period_start = l.period_start
        LEFT JOIN cloud_f_period fp
            ON  fp.corpus_id    = l.corpus_id
            AND fp.period_start = l.period_start
        ORDER BY l.period_start
    """

    with conn.cursor() as cur:
        cur.execute(sql, corpus_param)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()

    if not rows:
        raise ValueError(f"No supervised data for corpus '{corpus_id}'")

    df = pd.DataFrame(rows, columns=cols)

    # Interaction features
    df['size_x_drift']       = df.get('size', pd.Series(0.0)).fillna(0) * \
                               df.get('drift_magnitude', pd.Series(0.0)).fillna(0)
    df['drift_t1_x_drift']   = df.get('drift_t1', pd.Series(0.0)).fillna(0) * \
                               df.get('drift_magnitude', pd.Series(0.0)).fillna(0)
    df['bpr_x_cohesion']     = df.get('boundary_pressure_rate', pd.Series(0.0)).fillna(0) * \
                               df.get('cohesion', pd.Series(0.0)).fillna(0)

    snap_feat_cols  = [c for c, _ in snap_cols]
    lag_feat_cols   = [alias for alias, _, _ in lag_spec]
    field_feat_cols = [f'fp_{c}' for c, _ in field_cols]
    interaction_cols = ['size_x_drift', 'drift_t1_x_drift', 'bpr_x_cohesion']

    feat_cols = (
        snap_feat_cols +
        lag_feat_cols +
        ['death_probability'] +
        field_feat_cols +
        interaction_cols
    )
    feat_cols = [f for f in feat_cols if f in df.columns]

    return df, feat_cols, field_feat_cols


# ── Supervised: chronological split ──────────────────────────────────────────

def _chronological_split(df, target_col, frac_train=0.8):
    df = df.dropna(subset=[target_col]).copy()
    periods = sorted(df['period_start'].unique())
    cut = max(1, int(len(periods) * frac_train))
    train_p = set(periods[:cut])
    mask = df['period_start'].isin(train_p)
    return df[mask], df[~mask]


# ── Supervised: run ───────────────────────────────────────────────────────────

def run_supervised(conn, corpus_id, target_set='default', all_corpus_ids=None):
    label = corpus_id if corpus_id != 'all' else 'all_corpora'
    print(f"\n── Supervised: {label} (targets={target_set}) ──────────────────")

    df, feat_cols, field_feat_cols = load_supervised_data(conn, corpus_id)
    n_total = len(df)
    n_lag   = int(df['size_t3'].notna().sum()) if 'size_t3' in df.columns else 0
    print(f"  {n_total:,} rows, {n_lag:,} with T-3 lag, {len(feat_cols)} candidate features")

    if n_lag < MIN_LAG_ROWS:
        print(f"  WARNING: only {n_lag} T-3 lag rows — lag features may be sparse")

    targets = _TARGET_SETS[target_set]
    results_summary = []

    for target_col, is_binary, metric_key in targets:
        if target_col not in df.columns:
            print(f"  SKIP {target_col}: column not in dataframe")
            continue

        df_t = df.dropna(subset=[target_col]).copy()
        if is_binary:
            df_t[target_col] = df_t[target_col].astype(int)
            n_pos = int(df_t[target_col].sum())
            n_neg = len(df_t) - n_pos
            if n_pos < 10 or n_neg < 10:
                print(f"  SKIP {target_col}: pos={n_pos}, neg={n_neg} (need ≥10 each)")
                continue
        else:
            n_valid = df_t[target_col].notna().sum()
            if n_valid < 20:
                print(f"  SKIP {target_col}: {n_valid} valid rows (< 20)")
                continue

        df_train, df_test = _chronological_split(df_t, target_col)
        if len(df_train) < 20 or len(df_test) < 5:
            print(f"  SKIP {target_col}: train={len(df_train)}, test={len(df_test)}")
            continue

        # Conditionally exclude circular features
        cond_excl = TARGET_CONDITIONAL_EXCLUDE.get(target_col, frozenset())
        present = [f for f in feat_cols if f in df_t.columns and f not in cond_excl]

        X_tr = df_train[present].fillna(0).values.astype(np.float32)
        X_te = df_test[present].fillna(0).values.astype(np.float32)
        y_tr = df_train[target_col].values
        y_te = df_test[target_col].values

        # Drop constant features
        feat_std = X_tr.std(axis=0)
        nonconstant = feat_std > 0
        X_tr  = X_tr[:, nonconstant]
        X_te  = X_te[:, nonconstant]
        feat_std_nc = feat_std[nonconstant]
        present_nc  = [present[i] for i in range(len(present)) if nonconstant[i]]

        if is_binary and len(set(y_tr)) < 2:
            print(f"  SKIP {target_col}: only one class in training split")
            continue

        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_te_s   = scaler.transform(X_te)

        try:
            if is_binary:
                model = LogisticRegressionCV(
                    Cs=10, penalty='l1', solver='saga',
                    class_weight='balanced', cv=5,
                    max_iter=500, random_state=42, n_jobs=-1,
                )
                model.fit(X_tr_s, y_tr)
                score = float(roc_auc_score(y_te, model.predict_proba(X_te_s)[:, 1]))
                coefs = model.coef_[0]
                model_type = 'logistic_l1'
            else:
                model = LassoCV(cv=5, max_iter=5000, random_state=42, n_jobs=-1)
                model.fit(X_tr_s, y_tr)
                score = float(r2_score(y_te, model.predict(X_te_s)))
                coefs = model.coef_
                model_type = 'lasso'
        except Exception as exc:
            print(f"  SKIP {target_col}: {exc}")
            continue

        # Degenerate regression guard
        if metric_key == 'r2' and score < -10:
            print(f"  SKIP {target_col}: degenerate R²={score:.2f}")
            continue

        std_imp = coefs * feat_std_nc
        nonzero = [
            (present_nc[i], float(std_imp[i]), float(coefs[i]))
            for i in range(len(present_nc)) if coefs[i] != 0.0
        ]
        nonzero.sort(key=lambda x: abs(x[1]), reverse=True)

        # Tag each feature: is it new in v5 vs v4's 12-feature model?
        top_features_payload = [
            {
                'feature':    feat,
                'coefficient': coef,
                'std_impact':  simp,
                'is_new_v5':  feat not in V4_12_FEATURES,
            }
            for feat, simp, coef in nonzero[:20]
        ]

        metric_disp = metric_key.upper()
        print(f"  {target_col}: {metric_disp}={score:.4f}, "
              f"{len(nonzero)}/{len(present_nc)} non-zero features")
        for feat, simp, _ in nonzero[:5]:
            new_tag = ' [NEW-v5]' if feat not in V4_12_FEATURES else ''
            print(f"    {feat:<46} {simp:+.4f}{new_tag}")

        corpus_ids_stored = all_corpus_ids or [corpus_id]
        row_id = write_supervised_result(
            conn, label, 'supervised', target_col, model_type,
            score, metric_key,
            n_features=len(present_nc),
            n_nonzero=len(nonzero),
            top_features=top_features_payload,
            n_samples=len(df_t),
            corpus_ids=corpus_ids_stored,
            notes=f"n_train={len(df_train)}, n_test={len(df_test)}",
        )
        conn.commit()
        results_summary.append((target_col, metric_key, score, nonzero))
        print(f"    → saved to discovery_results id={row_id}")

    return results_summary


# ── Unsupervised: trajectory matrix ──────────────────────────────────────────

def _load_trajectory_matrix(conn, corpus_id):
    """
    Build trajectory fingerprint per persistent_cluster_id.
    Uses direct LAG() window on cluster_snapshot.

    Fingerprint: T0 (14 features) + T-1 (8) + T-2 (6) + T-3 (4) = 32 dims.
    Returns (pid_list, corpus_id_list, X, feat_names).
    """
    t0_feats = [
        ('size',                   'cs.size::real'),
        ('cohesion',               'COALESCE(cs.cohesion, 0)'),
        ('drift_magnitude',        'COALESCE(cs.drift_magnitude, 0)'),
        ('jerk',                   'COALESCE(cs.jerk, 0)'),
        ('acceleration',           'COALESCE(cs.acceleration, 0)'),
        ('velocity',               'COALESCE(cs.velocity, 0)'),
        ('boundary_pressure_rate', 'COALESCE(cs.boundary_pressure_rate, 0)'),
        ('convergence_score',      'COALESCE(cs.convergence_score, 0)'),
        ('mean_betweenness',       'COALESCE(cs.mean_betweenness, 0)'),
        ('persistence_score',      'COALESCE(cs.persistence_score, 0)'),
        ('age_periods',            'cs.age_periods::real'),
        ('elongation_ratio',       'COALESCE(cs.elongation_ratio, 0)'),
        ('death_probability',      'COALESCE(mds.death_probability, 0)'),
        ('n_attractors',           'COALESCE(cs.n_attractors, 0)::real'),
        # New v5 features
        ('mean_pagerank',          'COALESCE(cs.mean_pagerank, 0)'),
        ('mean_belief_persistence','COALESCE(cs.mean_belief_persistence, 0)'),
        ('void_count_adjacent',    'COALESCE(cs.void_count_adjacent, 0)::real'),
        ('growth_rate',            'COALESCE(cs.growth_rate, 0)'),
    ]
    lag_specs = [
        # (alias, source_expr, lag_n)
        ('size_t1',              'cs.size::real',                    1),
        ('cohesion_t1',          'COALESCE(cs.cohesion, 0)',         1),
        ('drift_t1',             'COALESCE(cs.drift_magnitude, 0)',  1),
        ('jerk_t1',              'COALESCE(cs.jerk, 0)',             1),
        ('bpr_t1',               'COALESCE(cs.boundary_pressure_rate, 0)', 1),
        ('is_dead_t1',           'COALESCE(cs.is_dead::int, 0)',     1),
        ('pagerank_t1',          'COALESCE(cs.mean_pagerank, 0)',    1),
        ('belief_t1',            'COALESCE(cs.mean_belief_persistence, 0)', 1),
        ('size_t2',              'cs.size::real',                    2),
        ('cohesion_t2',          'COALESCE(cs.cohesion, 0)',         2),
        ('drift_t2',             'COALESCE(cs.drift_magnitude, 0)',  2),
        ('jerk_t2',              'COALESCE(cs.jerk, 0)',             2),
        ('bpr_t2',               'COALESCE(cs.boundary_pressure_rate, 0)', 2),
        ('is_dead_t2',           'COALESCE(cs.is_dead::int, 0)',     2),
        ('size_t3',              'cs.size::real',                    3),
        ('drift_t3',             'COALESCE(cs.drift_magnitude, 0)',  3),
        ('jerk_t3',              'COALESCE(cs.jerk, 0)',             3),
        ('is_dead_t3',           'COALESCE(cs.is_dead::int, 0)',     3),
    ]

    t0_select  = ',\n                '.join(
        f"{expr} AS {name}" for name, expr in t0_feats
    )
    lag_select = ',\n                '.join(
        f"LAG({expr}, {n}) OVER w AS {alias}" for alias, expr, n in lag_specs
    )

    corpus_filter = (
        "AND cs.corpus_id = %(corpus_id)s" if corpus_id != 'all' else ""
    )
    corpus_param = {'corpus_id': corpus_id} if corpus_id != 'all' else {}

    sql = f"""
        WITH lagged AS (
            SELECT
                cs.corpus_id,
                cs.persistent_cluster_id,
                cs.period_start,
                {t0_select},
                {lag_select}
            FROM cluster_snapshot cs
            LEFT JOIN ml_death_scores mds
                ON  mds.corpus_id             = cs.corpus_id
                AND mds.persistent_cluster_id = cs.persistent_cluster_id
                AND mds.period_start          = cs.period_start
                AND mds.model_version         = 'cross_corpus_v5_12feat'
            WHERE cs.size >= 3
              AND cs.is_new = false
              AND cs.persistent_cluster_id IS NOT NULL
              {corpus_filter}
            WINDOW w AS (
                PARTITION BY cs.corpus_id, cs.persistent_cluster_id
                ORDER BY cs.period_start
            )
        )
        SELECT DISTINCT ON (corpus_id, persistent_cluster_id)
            corpus_id,
            persistent_cluster_id,
            {', '.join(name for name, _ in t0_feats)},
            {', '.join(alias for alias, _, _ in lag_specs)}
        FROM lagged
        WHERE size_t3 IS NOT NULL
        ORDER BY corpus_id, persistent_cluster_id, period_start DESC
    """

    with conn.cursor() as cur:
        cur.execute(sql, corpus_param)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()

    feat_names = (
        [name for name, _ in t0_feats] +
        [alias for alias, _, _ in lag_specs]
    )

    if not rows:
        return [], [], np.empty((0, len(feat_names)), dtype=np.float32), feat_names

    df    = pd.DataFrame(rows, columns=cols)
    pids  = list(df['persistent_cluster_id'])
    cids  = list(df['corpus_id'])
    X     = df[feat_names].fillna(0).values.astype(np.float32)
    return pids, cids, X, feat_names


def _compute_terminal_states(conn, corpus_id, pids, cids, assignments, k):
    """Compute per-archetype terminal event rates."""
    if not pids:
        return {}

    # Build lookup: (corpus_id, pid) → assignment
    pid_to_asgn = {}
    for i, (pid, cid) in enumerate(zip(pids, cids)):
        pid_to_asgn[(cid, pid)] = assignments[i]

    corpus_filter = (
        "AND cs.corpus_id = %s" if corpus_id != 'all' else ""
    )
    cid_param = (corpus_id,) if corpus_id != 'all' else ()

    with conn.cursor() as cur:
        cur.execute(f"""
            WITH last_per_pid AS (
                SELECT corpus_id, persistent_cluster_id, MAX(period_start) AS last_period
                FROM cluster_snapshot
                WHERE persistent_cluster_id = ANY(%s)
                  AND size >= 3
                  {corpus_filter}
                GROUP BY corpus_id, persistent_cluster_id
            )
            SELECT cs.corpus_id, cs.persistent_cluster_id,
                   cs.is_dead, cs.drift_magnitude, cs.cohesion,
                   cs.age_periods, cs.reborn_after_periods
            FROM cluster_snapshot cs
            JOIN last_per_pid lp
                ON  lp.corpus_id             = cs.corpus_id
                AND lp.persistent_cluster_id = cs.persistent_cluster_id
                AND lp.last_period           = cs.period_start
        """, (list(pids),) + cid_param)
        state_rows = cur.fetchall()

    state_map = {}
    for row in state_rows:
        cid_r, pid_r, is_dead, drift, cohesion, age, reborn = row
        state_map[(cid_r, pid_r)] = {
            'is_dead':   bool(is_dead) if is_dead is not None else False,
            'drift':     float(drift)  if drift   is not None else 0.0,
            'cohesion':  float(cohesion) if cohesion is not None else 0.0,
            'age':       int(age)      if age      is not None else 0,
            'reborn':    bool(reborn is not None and reborn > 0),
        }

    stats = {}
    for a in range(k):
        members = [(cids[i], pids[i]) for i, asgn in enumerate(assignments) if asgn == a]
        if not members:
            stats[a] = None
            continue
        n = len(members)
        deaths  = sum(1 for key in members if state_map.get(key, {}).get('is_dead', False))
        reborns = sum(1 for key in members if state_map.get(key, {}).get('reborn', False))
        mean_drift    = float(np.mean([state_map.get(k2, {}).get('drift', 0) for k2 in members]))
        mean_cohesion = float(np.mean([state_map.get(k2, {}).get('cohesion', 0) for k2 in members]))
        ages_dead     = [state_map.get(k2, {}).get('age', 0) for k2 in members
                         if state_map.get(k2, {}).get('is_dead', False)]
        mean_age_dead = float(np.mean(ages_dead)) if ages_dead else None
        stats[a] = {
            'n':              n,
            'death_rate':     deaths  / n,
            'reborn_rate':    reborns / n,
            'survival_rate':  (n - deaths) / n,
            'mean_drift':     mean_drift,
            'mean_cohesion':  mean_cohesion,
            'mean_age_dead':  mean_age_dead,
            'example_pids':   [p for _, p in members[:3]],
        }
    return stats


def _name_archetype(stats):
    """Name an archetype by its dominant terminal behavior."""
    if stats['death_rate'] > 0.6:
        return 'death_precursor'
    if stats['reborn_rate'] > 0.3:
        return 'cyclical_reborn'
    if stats['survival_rate'] > 0.85 and stats['mean_drift'] < 0.1:
        return 'stable_persistent'
    if stats['survival_rate'] > 0.75 and stats['mean_drift'] > 0.2:
        return 'active_drifter'
    if stats['death_rate'] > 0.3:
        return 'at_risk'
    return 'mixed'


# ── Unsupervised: run ─────────────────────────────────────────────────────────

def run_unsupervised(conn, corpus_id):
    """
    KMeans on trajectory fingerprints.
    Returns list of archetype dicts for cross-corpus comparison.
    """
    label = corpus_id if corpus_id != 'all' else 'all_corpora'
    print(f"\n── Unsupervised: {label} ──────────────────────────────────────────")

    pids, cids, X, feat_names = _load_trajectory_matrix(conn, corpus_id)
    if len(pids) < MIN_TRAJ_PIDS:
        print(f"  Only {len(pids)} trajectory-eligible PIDs (< {MIN_TRAJ_PIDS}) — skip")
        return []

    print(f"  {len(pids)} PIDs, trajectory matrix {X.shape} ({len(feat_names)} dims)")

    # Min-max normalize each feature
    X_norm = X.copy()
    for fi in range(X.shape[1]):
        vmin, vmax = X[:, fi].min(), X[:, fi].max()
        if vmax > vmin:
            X_norm[:, fi] = (X[:, fi] - vmin) / (vmax - vmin)

    # Select k by silhouette score (k=3..min(12, n//2))
    k_max    = min(12, len(pids) // 2)
    best_k   = 3
    best_sil = -1.0
    best_labels = None
    sil_scores  = {}
    for k in range(3, k_max + 1):
        if len(pids) <= k:
            continue
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_norm)
        sil = float(silhouette_score(X_norm, lbl)) if len(set(lbl)) > 1 else 0.0
        sil_scores[k] = sil
        if sil > best_sil:
            best_sil, best_k, best_labels = sil, k, lbl

    print("  Silhouette: " +
          ', '.join(f"k={k}:{s:.3f}" for k, s in sorted(sil_scores.items())))
    print(f"  Best k={best_k} (sil={best_sil:.3f})")

    if best_sil < MIN_SIL_SCORE:
        print(f"  No stable archetypes — sil={best_sil:.3f} < {MIN_SIL_SCORE}")
        return []

    km_final    = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    assignments = km_final.fit_predict(X_norm)
    centers     = km_final.cluster_centers_

    terminal    = _compute_terminal_states(conn, corpus_id, pids, cids, assignments, best_k)
    archetypes  = []

    # Clear previous archetypes for this corpus
    with conn.cursor() as cur:
        cur.execute("DELETE FROM discovery_archetypes WHERE corpus_id=%s", (label,))

    for a in range(best_k):
        stats = terminal.get(a)
        if stats is None:
            continue
        arch_label = _name_archetype(stats)

        print(
            f"  Archetype {a}: {arch_label:<22} "
            f"n={stats['n']}, death={stats['death_rate']:.1%}, "
            f"reborn={stats['reborn_rate']:.1%}, survive={stats['survival_rate']:.1%}, "
            f"drift={stats['mean_drift']:.3f}"
        )

        # Print top distinguishing features
        top_feat_idx = np.argsort(np.abs(centers[a]))[::-1][:8]
        for fi in top_feat_idx:
            new_tag = ' [NEW-v5]' if feat_names[fi] not in V4_12_FEATURES else ''
            print(f"    {feat_names[fi]:<40} center={centers[a][fi]:+.3f}{new_tag}")

        row_id = write_archetype_result(
            conn, label, a, best_k,
            stats['death_rate'],
            stats['reborn_rate'],
            stats['mean_age_dead'],
            stats['mean_drift'],
            stats['mean_cohesion'],
            arch_label,
        )
        archetypes.append({
            'label':      arch_label,
            'idx':        a,
            'center':     centers[a],
            'feat_names': feat_names,
            'stats':      stats,
            'row_id':     row_id,
        })

    conn.commit()
    print(f"  Saved {len(archetypes)} archetypes to discovery_archetypes.")
    return archetypes


# ── Cross-corpus comparison ───────────────────────────────────────────────────

def cross_corpus_supervised(conn, corpus_ids):
    """Find features that are significant predictors across ALL corpora."""
    print(f"\n── Cross-corpus supervised: universal predictors ─────────────────")

    per_corpus = {}
    for cid in corpus_ids:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT target, top_features
                FROM discovery_results
                WHERE corpus_id = %s AND mode = 'supervised'
            """, (cid,))
            rows = cur.fetchall()
        feat_map = {}
        for target, top_feats in rows:
            if top_feats:
                tf = top_feats if isinstance(top_feats, list) else json.loads(top_feats)
                for item in tf:
                    feat_map[(target, item['feature'])] = item['std_impact']
        per_corpus[cid] = feat_map

    all_keys = set()
    for d in per_corpus.values():
        all_keys |= set(d.keys())

    universal = sorted(
        (t, f) for (t, f) in all_keys
        if all((t, f) in per_corpus[cid] for cid in corpus_ids)
    )

    print(f"\n  Universal predictors (significant in all {len(corpus_ids)} corpora):")
    last_t = None
    for t, f in universal[:40]:
        if t != last_t:
            print(f"  [{t}]")
            last_t = t
        new_tag = ' [NEW-v5]' if f not in V4_12_FEATURES else ''
        vals = '  '.join(
            f"{cid.split('_')[0]}:{per_corpus[cid].get((t, f), 0):+.3f}"
            for cid in corpus_ids
        )
        print(f"    {f:<46} {vals}{new_tag}")

    return universal


def cross_corpus_unsupervised(corpus_ids, all_archetypes):
    """Cosine similarity between archetype centers across corpora."""
    print(f"\n── Cross-corpus archetype similarity ─────────────────────────────")

    active = {cid: archs for cid, archs in all_archetypes.items() if archs}
    if len(active) < 2:
        print("  Fewer than 2 corpora with archetypes — no comparison.")
        return

    cids = list(active.keys())
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            ca, cb = cids[i], cids[j]
            print(f"\n  {ca} × {cb}:")
            for arch_a in active[ca]:
                best_sim, best_lbl = 0.0, None
                va = arch_a['center']
                for arch_b in active[cb]:
                    vb  = arch_b['center']
                    n   = min(len(va), len(vb))
                    sim = float(
                        np.dot(va[:n], vb[:n]) /
                        (np.linalg.norm(va[:n]) * np.linalg.norm(vb[:n]) + 1e-9)
                    )
                    if sim > best_sim:
                        best_sim, best_lbl = sim, arch_b['label']
                flag = 'UNIVERSAL PATTERN' if best_sim >= 0.85 else ''
                print(
                    f"    {arch_a['label']:<24} → {best_lbl or '?':<24} "
                    f"sim={best_sim:.3f}  {flag}"
                )


# ── Final report ──────────────────────────────────────────────────────────────

def print_report(conn, corpus_ids, all_results, all_archetypes):
    print("\n" + "═" * 70)
    print("  ARC DISCOVERY v5 — SUMMARY REPORT")
    print("═" * 70)

    # 1. is_dead AUC by corpus vs v4 baseline (0.853)
    print("\n1. DEATH PREDICTION (is_dead AUC vs v4 baseline 0.853):")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT corpus_id, auc_roc, n_features_used, n_nonzero_coefs
            FROM discovery_results
            WHERE target = 'is_dead' AND mode = 'supervised'
            ORDER BY corpus_id
        """)
        for row in cur.fetchall():
            cid, auc, n_feat, n_nz = row
            delta = (auc - 0.853) if auc else None
            delta_str = f" (Δ{delta:+.3f} vs v4)" if delta is not None else ""
            print(f"  {cid:<25} AUC={auc:.4f}{delta_str}  "
                  f"features={n_feat} nonzero={n_nz}")

    # 2. New v5 features with significant is_dead coefficients
    print("\n2. NEW v5 FEATURES — significant for is_dead:")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT corpus_id, top_features
            FROM discovery_results
            WHERE target = 'is_dead' AND mode = 'supervised'
        """)
        for cid, tf in cur.fetchall():
            if not tf:
                continue
            items = tf if isinstance(tf, list) else json.loads(tf)
            new_feats = [x for x in items if x.get('is_new_v5')]
            if new_feats:
                print(f"  [{cid}]")
                for x in new_feats[:8]:
                    print(f"    {x['feature']:<40} coef={x['coefficient']:+.4f}")

    # 3. Best drift predictors
    print("\n3. DRIFT PREDICTORS (drift_magnitude_next R²):")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT corpus_id, r2_score, top_features
            FROM discovery_results
            WHERE target = 'drift_magnitude_next' AND mode = 'supervised'
            ORDER BY corpus_id
        """)
        for cid, r2, tf in cur.fetchall():
            if r2 is None:
                continue
            print(f"  {cid:<25} R²={r2:.4f}")
            if tf:
                items = tf if isinstance(tf, list) else json.loads(tf)
                for x in items[:5]:
                    new_tag = ' [NEW-v5]' if x.get('is_new_v5') else ''
                    print(f"    {x['feature']:<40} {x['std_impact']:+.4f}{new_tag}")

    # 4. Void precursor signals
    print("\n4. VOID PRECURSOR SIGNALS (void_count_adjacent_next R²):")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT corpus_id, r2_score, top_features
            FROM discovery_results
            WHERE target = 'void_count_adjacent_next' AND mode = 'supervised'
            ORDER BY corpus_id
        """)
        for cid, r2, tf in cur.fetchall():
            if r2 is None:
                continue
            print(f"  {cid:<25} R²={r2:.4f}")
            if tf:
                items = tf if isinstance(tf, list) else json.loads(tf)
                for x in items[:5]:
                    new_tag = ' [NEW-v5]' if x.get('is_new_v5') else ''
                    print(f"    {x['feature']:<40} {x['std_impact']:+.4f}{new_tag}")

    # 5. Lifecycle archetypes
    print("\n5. LIFECYCLE ARCHETYPES (unsupervised KMeans):")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT corpus_id, archetype_id, label, n_clusters,
                   death_rate, reborn_rate, mean_drift, mean_cohesion
            FROM discovery_archetypes
            ORDER BY corpus_id, archetype_id
        """)
        rows = cur.fetchall()
    if rows:
        for row in rows:
            cid, aid, lbl, k, dr, rr, md, mc = row
            print(f"  [{cid}] k={k} arch={aid}: {lbl:<24} "
                  f"death={dr:.1%} reborn={rr:.1%} drift={md:.3f} cohesion={mc:.3f}")
    else:
        print("  No archetypes found (silhouette below threshold).")

    # 6. cuGraph feature significance
    print("\n6. cuGRAPH FEATURE SIGNIFICANCE:")
    cugraph_feats = {
        'mean_pagerank', 'max_pagerank', 'mean_eigenvector_centrality',
        'mean_katz_centrality', 'mean_harmonic_centrality',
        'mean_in_degree_centrality', 'mean_triangle_count',
    }
    with conn.cursor() as cur:
        cur.execute("""
            SELECT corpus_id, target, top_features
            FROM discovery_results
            WHERE mode = 'supervised'
            ORDER BY corpus_id, target
        """)
        hits = []
        for cid, target, tf in cur.fetchall():
            if not tf:
                continue
            items = tf if isinstance(tf, list) else json.loads(tf)
            for x in items:
                if x['feature'] in cugraph_feats:
                    hits.append((cid, target, x['feature'], x['std_impact']))
    if hits:
        for cid, target, feat, impact in sorted(hits, key=lambda x: abs(x[3]), reverse=True)[:15]:
            print(f"  {cid:<25} {target:<35} {feat:<35} {impact:+.4f}")
    else:
        print("  No cuGraph features selected by Lasso/LogReg.")

    print("\n" + "═" * 70)


# ── Mode: neighbor ────────────────────────────────────────────────────────────

def build_neighborhood_features(conn, corpus_id):
    """
    For each cluster-period, aggregate features of adjacent clusters via cloud_f_edge.
    Returns a DataFrame indexed by (corpus_id, persistent_cluster_id, period_start).
    Returns None if cloud_f_edge is empty or missing required columns.
    """
    corpus_filter = "AND cs.corpus_id = %(corpus_id)s" if corpus_id != 'all' else ""
    corpus_param  = {'corpus_id': corpus_id} if corpus_id != 'all' else {}

    sql = f"""
        SELECT
            cs.corpus_id,
            cs.persistent_cluster_id,
            cs.period_start,
            AVG(mds2.death_probability)              AS neighbor_avg_dp,
            MAX(mds2.death_probability)              AS neighbor_max_dp,
            MIN(mds2.death_probability)              AS neighbor_min_dp,
            AVG(cs2.size)                            AS neighbor_avg_size,
            AVG(cs2.cohesion)                        AS neighbor_avg_cohesion,
            AVG(cs2.mean_betweenness)                AS neighbor_avg_betweenness,
            MAX(cs2.mean_betweenness)                AS neighbor_max_betweenness,
            AVG(cs2.mean_triangle_count)             AS neighbor_avg_triangles,
            AVG(cs2.drift_magnitude)                 AS neighbor_avg_drift,
            MAX(cs2.drift_magnitude)                 AS neighbor_max_drift,
            AVG(e.connection_weight)                 AS avg_edge_weight,
            MAX(e.connection_weight)                 AS max_edge_weight,
            COUNT(*)                                 AS n_neighbors,
            SUM(CASE WHEN e.is_new_edge  THEN 1 ELSE 0 END) AS n_new_edges,
            SUM(CASE WHEN e.is_lost_edge THEN 1 ELSE 0 END) AS n_lost_edges,
            SUM(e.connection_weight * COALESCE(mds2.death_probability, 0))
                                                     AS weighted_dying_pressure,
            MAX(CASE WHEN cst.stage = 4 THEN 1 ELSE 0 END)
                                                     AS has_crystallizing_neighbor,
            COUNT(*) FILTER (WHERE mds2.death_probability > 0.7)
                                                     AS n_dying_neighbors
        FROM cluster_snapshot cs
        JOIN cloud_f_edge e
            ON  e.corpus_id    = cs.corpus_id
            AND e.period_start = cs.period_start
            AND (e.cluster_a = cs.cluster_id OR e.cluster_b = cs.cluster_id)
        JOIN cloud_matching m_neighbor
            ON  m_neighbor.corpus_id    = cs.corpus_id
            AND m_neighbor.period_start = cs.period_start
            AND m_neighbor.cluster_id   = CASE
                WHEN e.cluster_a = cs.cluster_id THEN e.cluster_b
                ELSE e.cluster_a END
        JOIN cluster_snapshot cs2
            ON  cs2.corpus_id             = m_neighbor.corpus_id
            AND cs2.period_start          = m_neighbor.period_start
            AND cs2.persistent_cluster_id = m_neighbor.persistent_cluster_id
        LEFT JOIN ml_death_scores mds2
            ON  mds2.corpus_id             = m_neighbor.corpus_id
            AND mds2.persistent_cluster_id = m_neighbor.persistent_cluster_id
            AND mds2.period_start          = m_neighbor.period_start
            AND mds2.model_version         = 'arc_death_model_v5'
        LEFT JOIN crystallization_stages cst
            ON  cst.corpus_id    = cs.corpus_id
            AND cst.period_start = cs.period_start
        WHERE cs.persistent_cluster_id IS NOT NULL
          AND cs2.persistent_cluster_id IS NOT NULL
          {corpus_filter}
        GROUP BY cs.corpus_id, cs.persistent_cluster_id, cs.period_start
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, corpus_param)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    except Exception as exc:
        print(f"  WARNING: neighborhood query failed: {exc}")
        conn.rollback()
        return None

    if not rows:
        print("  WARNING: no neighborhood edges found — cloud_f_edge may be empty")
        return None

    df = pd.DataFrame(rows, columns=cols)
    df = df.set_index(['corpus_id', 'persistent_cluster_id', 'period_start'])
    print(f"  Neighborhood features: {len(df):,} rows, "
          f"{df.columns.tolist().__len__()} neighbor cols")
    return df


# ── Mode: rules ───────────────────────────────────────────────────────────────

def run_association_rules(conn, corpus_id, df_base, min_support=0.03,
                          min_confidence=0.6, min_lift=2.0,
                          includes_neighbor=False):
    """
    Mine IF-THEN patterns across all discretized features.
    Adds T+1 lead columns for outcome consequents.
    Writes results to discovery_patterns; returns summary dict.
    """
    if not HAS_MLXTEND:
        print("  SKIP rules: mlxtend not installed")
        return {}

    label = corpus_id if corpus_id != 'all' else 'all_corpora'
    tag   = '+neighbor' if includes_neighbor else ''
    print(f"\n── Association rules: {label}{tag} ──────────────────────────────────")

    df = df_base.copy()

    # ── Build T+1 outcome columns via lead ────────────────────────────────────
    df = df.sort_values(['corpus_id', 'persistent_cluster_id', 'period_start'])
    grp = df.groupby(['corpus_id', 'persistent_cluster_id'])

    for col in ['elongation_ratio', 'drift_magnitude', 'size', 'n_attractors']:
        if col in df.columns:
            df[f'{col}_lead1'] = grp[col].shift(-1)

    # die_within_4: is_dead true in next 4 periods
    if 'is_dead' in df.columns:
        def _any_dead_next4(s):
            return s.shift(-1).rolling(4, min_periods=1).max()
        df['die_within_4'] = grp['is_dead'].transform(_any_dead_next4).fillna(0).astype(int)

    # ── Select numeric feature columns ───────────────────────────────────────
    # Restrict feature set to keep Apriori tractable.
    # C(n_bins, max_len) must stay well under ~100K combinations.
    # DEATH_FEATURES (18) × 3 bins = 54 binary cols → C(54,3) ≈ 24K  ✓
    # Adding neighbor cols when available: keep total ≤ 70 binary cols.
    RULES_FEATURE_WHITELIST = set(DEATH_FEATURES) | {
        # neighborhood (if present) — a few high-signal ones only
        'neighbor_avg_dp', 'neighbor_max_dp',
        'weighted_dying_pressure', 'n_dying_neighbors',
        'has_crystallizing_neighbor',
    }
    id_cols  = {'corpus_id', 'persistent_cluster_id', 'period_start',
                'cluster_id', 'match_type', 'reborn_from_period',
                'cluster_label', 'cluster_summary', 'centroid', 'drift_vector',
                'is_new', 'is_dead'}
    num_cols = [c for c in df.select_dtypes(include='number').columns
                if c not in id_cols
                and not c.endswith('_lead1')
                and c in RULES_FEATURE_WHITELIST]

    # ── Discretize into terciles ──────────────────────────────────────────────
    # Build all columns at once via pd.concat to avoid fragmentation warnings
    disc_parts = []
    for col in num_cols:
        if df[col].nunique() < 3:
            continue
        q33 = df[col].quantile(0.33)
        q67 = df[col].quantile(0.67)
        part = pd.DataFrame({
            f'{col}_low':  (df[col] <= q33).astype(int),
            f'{col}_mid':  ((df[col] > q33) & (df[col] <= q67)).astype(int),
            f'{col}_high': (df[col] > q67).astype(int),
        }, index=df.index)
        disc_parts.append(part)

    # ── Add outcome columns (T+1 leads as binary high/dead) ───────────────────
    outcome_parts = []
    for col in ['elongation_ratio', 'drift_magnitude', 'size', 'n_attractors']:
        lead_col = f'{col}_lead1'
        if lead_col in df.columns and df[lead_col].nunique() >= 2:
            q67 = df[lead_col].quantile(0.67)
            outcome_parts.append(
                pd.Series((df[lead_col] > q67).astype(int),
                          index=df.index, name=f'{col}_high')
            )

    extra_parts = []
    if 'is_dead' in df.columns:
        extra_parts.append(
            pd.Series(df['is_dead'].fillna(0).astype(int), index=df.index, name='is_dead')
        )
    if 'die_within_4' in df.columns:
        extra_parts.append(
            pd.Series(df['die_within_4'].fillna(0).astype(int),
                      index=df.index, name='die_within_4')
        )

    all_parts = disc_parts + [pd.DataFrame(dict(zip(
        [s.name for s in outcome_parts + extra_parts],
        outcome_parts + extra_parts
    )), index=df.index)] if (outcome_parts or extra_parts) else disc_parts
    discretized = pd.concat(all_parts, axis=1) if all_parts else pd.DataFrame(index=df.index)

    n_rows = len(discretized)
    n_cols = len(discretized.columns)
    print(f"  Discretized matrix: {n_rows:,} rows × {n_cols} binary columns")

    if n_rows < 50:
        print("  SKIP: too few rows for reliable rules")
        return {}

    # ── Run Apriori ───────────────────────────────────────────────────────────
    print(f"  Running Apriori (min_support={min_support}, max_len=3)…")
    try:
        freq_items = apriori(
            discretized.astype(bool),
            min_support=min_support,
            use_colnames=True,
            max_len=3,
            verbose=0,
        )
    except Exception as exc:
        print(f"  Apriori failed: {exc}")
        return {}

    if freq_items.empty:
        print("  No frequent itemsets found at this support threshold")
        return {}

    print(f"  {len(freq_items):,} frequent itemsets found")

    try:
        rules = mlxtend_assoc_rules(
            freq_items,
            metric='lift',
            min_threshold=min_lift,
        )
    except Exception as exc:
        print(f"  association_rules failed: {exc}")
        return {}

    if rules.empty:
        print(f"  No rules found at lift≥{min_lift}")
        return {}

    # ── Filter for confidence and outcome consequents ─────────────────────────
    rules = rules[rules['confidence'] >= min_confidence].copy()

    outcome_rules = rules[
        rules['consequents'].apply(
            lambda x: any(o in str(item) for item in x for o in RULE_OUTCOME_COLS)
        )
    ].copy()

    rules['combined_score']         = rules['lift'] * rules['confidence']
    outcome_rules['combined_score'] = outcome_rules['lift'] * outcome_rules['confidence']
    rules         = rules.sort_values('combined_score', ascending=False)
    outcome_rules = outcome_rules.sort_values('combined_score', ascending=False)

    # ── Counts ────────────────────────────────────────────────────────────────
    n_total    = len(rules)
    n_outcome  = len(outcome_rules)
    n_lift3    = int((rules['lift'] > 3.0).sum())
    n_lift5    = int((rules['lift'] > 5.0).sum())
    n_death    = int(outcome_rules['consequents'].apply(
        lambda x: any('dead' in str(i) or 'die' in str(i) for i in x)
    ).sum())
    n_elong    = int(outcome_rules['consequents'].apply(
        lambda x: any('elongation' in str(i) for i in x)
    ).sum())

    print(f"  Rules found: {n_total:,} total | {n_outcome:,} with outcome consequent")
    print(f"  Lift >3.0: {n_lift3:,} | Lift >5.0: {n_lift5:,}")
    print(f"  Death/survival outcomes: {n_death} | Elongation outcomes: {n_elong}")

    # ── Top 20 by combined_score ──────────────────────────────────────────────
    print(f"\n  TOP 20 RULES by combined_score (lift × confidence):")
    print(f"  {'Antecedent':<60} {'Consequent':<30} {'sup':>5} {'conf':>5} {'lift':>5}")
    print(f"  {'-'*60} {'-'*30} {'-'*5} {'-'*5} {'-'*5}")
    for _, row in outcome_rules.head(20).iterrows():
        ant = ', '.join(sorted(row['antecedents']))
        con = ', '.join(sorted(row['consequents']))
        print(f"  {ant[:58]:<60} {con[:28]:<30} "
              f"{row['support']:5.3f} {row['confidence']:5.3f} {row['lift']:5.2f}")

    # ── Write to discovery_patterns ───────────────────────────────────────────
    rows_written = 0
    with conn.cursor() as cur:
        # Clear previous run for this corpus+mode combination
        cur.execute(
            "DELETE FROM discovery_patterns WHERE corpus_id=%s AND mode=%s "
            "AND includes_neighbor=%s",
            (label, 'rules', includes_neighbor),
        )
        for _, row in outcome_rules.iterrows():
            ant_str = ', '.join(sorted(row['antecedents']))
            con_str = ', '.join(sorted(row['consequents']))
            n_match = int(round(row['support'] * n_rows))
            cur.execute("""
                INSERT INTO discovery_patterns
                  (corpus_id, mode, antecedent, consequent, support, confidence,
                   lift, combined_score, n_rows_matching, includes_neighbor, lag_periods)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                label, 'rules',
                ant_str, con_str,
                float(row['support']), float(row['confidence']),
                float(row['lift']), float(row['combined_score']),
                n_match, includes_neighbor, 1,
            ))
            rows_written += 1
    conn.commit()
    print(f"\n  Wrote {rows_written:,} outcome rules to discovery_patterns")

    return {
        'n_total': n_total,
        'n_outcome': n_outcome,
        'n_lift3': n_lift3,
        'n_lift5': n_lift5,
        'n_death': n_death,
        'n_elong': n_elong,
        'includes_neighbor': includes_neighbor,
        'rules_df': outcome_rules,
    }


# ── Mode: symbolic ────────────────────────────────────────────────────────────

def run_symbolic_regression(conn, corpus_id, df_base):
    """
    Use PySR to find a mathematical formula for death probability.
    Target: death_probability from v_cluster_snapshot_with_dp (pgml live scores).
    """
    if not HAS_PYSR:
        print("  SKIP symbolic: pysr not installed")
        return {}

    label = corpus_id if corpus_id != 'all' else 'all_corpora'
    print(f"\n── Symbolic regression: {label} ────────────────────────────────────")

    # Use death_probability from cluster_snapshot (pgml inline) as target
    corpus_filter = "WHERE corpus_id = %(corpus_id)s" if corpus_id != 'all' else ""
    corpus_param  = {'corpus_id': corpus_id} if corpus_id != 'all' else {}

    feat_select = ',\n    '.join(
        f"COALESCE({f}::real, 0) AS {f}" for f in DEATH_FEATURES
    )
    sql = f"""
        SELECT
            {feat_select},
            death_probability
        FROM v_cluster_snapshot_with_dp
        {corpus_filter}
          {'AND' if corpus_filter else 'WHERE'} death_probability IS NOT NULL
          AND size >= 3
          AND is_new = false
        LIMIT 5000
    """
    # Fix: if no corpus_filter, the WHERE clause logic above will break; use cleaner form
    if corpus_id == 'all':
        sql = f"""
            SELECT
                {feat_select},
                death_probability
            FROM v_cluster_snapshot_with_dp
            WHERE death_probability IS NOT NULL
              AND size >= 3
              AND is_new = false
            LIMIT 5000
        """

    try:
        with conn.cursor() as cur:
            cur.execute(sql, corpus_param if corpus_id != 'all' else {})
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    except Exception as exc:
        print(f"  Query failed: {exc}")
        conn.rollback()
        return {}

    if not rows:
        print("  No scored rows found")
        return {}

    df = pd.DataFrame(rows, columns=cols)
    feat_cols = [c for c in DEATH_FEATURES if c in df.columns]
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df['death_probability'].values.astype(np.float32)

    # Drop constant features
    nonconstant = X.std(axis=0) > 0
    X = X[:, nonconstant]
    active_feats = [feat_cols[i] for i in range(len(feat_cols)) if nonconstant[i]]

    print(f"  {len(df):,} samples, {len(active_feats)} non-constant features")
    print(f"  Target: death_probability — mean={y.mean():.4f}, std={y.std():.4f}")

    model = PySRRegressor(
        niterations=50,
        binary_operators=['+', '-', '*', '/'],
        unary_operators=['log', 'sqrt', 'square'],
        populations=20,
        maxsize=15,
        verbosity=0,
        progress=False,
        random_state=42,
    )

    print("  Running PySR (50 iterations, populations=20)…")
    try:
        model.fit(X, y, variable_names=active_feats)
    except Exception as exc:
        print(f"  PySR failed: {exc}")
        return {}

    eqs = model.equations_
    if eqs is None or len(eqs) == 0:
        print("  No equations found")
        return {}

    # Best simple formula: lowest complexity with R² > 0.1
    eqs = eqs.copy()
    eqs['r2'] = eqs.get('r2', eqs.get('score', 0.0))
    simple = eqs[eqs['complexity'] < 8].sort_values('r2', ascending=False)

    print(f"\n  PySR found {len(eqs)} equations across complexity/accuracy frontier")
    print(f"\n  PARETO FRONT (complexity vs R²):")
    print(f"  {'Complexity':>10} {'R²':>8}  Equation")
    print(f"  {'-'*10} {'-'*8}  {'-'*50}")
    for _, eq_row in eqs.iterrows():
        try:
            r2_val   = float(eq_row.get('r2', eq_row.get('score', 0.0)))
            comp_val = int(eq_row.get('complexity', 0))
            eq_str   = str(eq_row.get('equation', eq_row.get('sympy_format', '?')))
            print(f"  {comp_val:>10} {r2_val:>8.4f}  {eq_str[:70]}")
        except Exception:
            continue

    best_simple = {}
    if len(simple) > 0:
        best_row   = simple.iloc[0]
        best_r2    = float(best_row.get('r2', best_row.get('score', 0.0)))
        best_eq    = str(best_row.get('equation', best_row.get('sympy_format', '?')))
        best_comp  = int(best_row.get('complexity', 0))
        print(f"\n  BEST SIMPLE FORMULA (complexity < 8):")
        print(f"  death ≈ {best_eq}")
        print(f"  R² = {best_r2:.4f}, complexity = {best_comp}")
        best_simple = {'equation': best_eq, 'r2': best_r2, 'complexity': best_comp}

        # Write best formula to discovery_patterns as a note
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM discovery_patterns WHERE corpus_id=%s AND mode='symbolic'",
                (label,),
            )
            cur.execute("""
                INSERT INTO discovery_patterns
                  (corpus_id, mode, antecedent, consequent, confidence, lift, notes)
                VALUES (%s, 'symbolic', %s, 'death_probability', %s, %s, %s)
            """, (
                label,
                ', '.join(active_feats[:5]) + ('…' if len(active_feats) > 5 else ''),
                float(best_r2),
                float(best_r2),  # lift = R² for symbolic
                f"complexity={best_comp}; eq={best_eq}; n_samples={len(df)}",
            ))
        conn.commit()

    return {'equations': eqs, 'best_simple': best_simple, 'n_samples': len(df)}


# ── Cross-corpus rule overlap analysis ────────────────────────────────────────

def cross_corpus_rule_overlap(conn, corpus_ids):
    """Find rules that appear in 3+ corpora — candidates for genuine laws."""
    print(f"\n── Cross-corpus rule overlap ─────────────────────────────────────")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT antecedent, consequent, COUNT(DISTINCT corpus_id) as n_corpora,
                   ROUND(AVG(lift)::numeric, 3) as avg_lift,
                   ROUND(AVG(confidence)::numeric, 3) as avg_conf,
                   ARRAY_AGG(DISTINCT corpus_id ORDER BY corpus_id) as corpora
            FROM discovery_patterns
            WHERE mode = 'rules'
            GROUP BY antecedent, consequent
            HAVING COUNT(DISTINCT corpus_id) >= 2
            ORDER BY n_corpora DESC, avg_lift DESC
            LIMIT 30
        """)
        rows = cur.fetchall()

    if not rows:
        print("  No overlapping rules found across corpora yet.")
        return []

    n_3plus = sum(1 for r in rows if r[2] >= 3)
    print(f"  Rules in 2+ corpora: {len(rows)} | Rules in 3+ corpora: {n_3plus}")
    print(f"\n  {'Antecedent':<50} {'Consequent':<28} {'N':>3} {'lift':>5}")
    print(f"  {'-'*50} {'-'*28} {'-'*3} {'-'*5}")
    for ant, con, n_corp, avg_lift, avg_conf, corpora in rows:
        marker = ' ★' if n_corp >= 3 else ''
        print(f"  {ant[:48]:<50} {con[:26]:<28} {n_corp:>3} {avg_lift:>5.2f}{marker}")

    return rows


# ── Threshold recommendation ──────────────────────────────────────────────────

def recommend_thresholds(conn):
    """Based on pattern counts, recommend thresholds for manageable investigation set."""
    print(f"\n── Threshold recommendation ──────────────────────────────────────")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*)                                       as total,
                COUNT(*) FILTER (WHERE lift > 2.0)            as lift_2,
                COUNT(*) FILTER (WHERE lift > 3.0)            as lift_3,
                COUNT(*) FILTER (WHERE lift > 5.0)            as lift_5,
                COUNT(*) FILTER (WHERE combined_score > 2.0)  as cs_2,
                COUNT(*) FILTER (WHERE combined_score > 3.0)  as cs_3,
                COUNT(*) FILTER (WHERE combined_score > 5.0)  as cs_5,
                COUNT(*) FILTER (WHERE includes_neighbor)     as neighbor_rules
            FROM discovery_patterns
            WHERE mode = 'rules'
        """)
        row = cur.fetchone()

    if not row or row[0] == 0:
        print("  No rules in discovery_patterns yet.")
        return

    total, l2, l3, l5, cs2, cs3, cs5, nb = row
    print(f"  Total outcome rules: {total:,}")
    print(f"  Lift  >2.0: {l2:,} | >3.0: {l3:,} | >5.0: {l5:,}")
    print(f"  Score >2.0: {cs2:,} | >3.0: {cs3:,} | >5.0: {cs5:,}")
    print(f"  With neighbor features: {nb:,}")

    # Recommend threshold to land in 50–200 range
    for thresh, count, label in [
        (5.0, l5, 'lift>5.0'),
        (3.0, l3, 'lift>3.0'),
        (5.0, cs5, 'combined_score>5.0'),
        (3.0, cs3, 'combined_score>3.0'),
        (2.0, l2, 'lift>2.0'),
    ]:
        if 50 <= count <= 200:
            print(f"\n  ✓ RECOMMENDED: {label} → {count} rules (target 50–200)")
            return
    # Fallback
    if total <= 200:
        print(f"\n  ✓ RECOMMENDED: all rules → {total} total (already within target)")
    elif l3 > 0:
        print(f"\n  ✓ RECOMMENDED: lift>3.0 → {l3} rules (closest to 50–200 range)")
    else:
        print(f"\n  ⚠ No threshold gives 50–200 rules — total={total}, lift>5: {l5}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _ALL_MODES = ['supervised', 'unsupervised', 'both', 'rules', 'neighbor', 'symbolic']

    parser = argparse.ArgumentParser(description='ARC Discovery v5 — feature signal analysis')
    parser.add_argument('--corpus',  default='all',
                        help='corpus_id to analyse, or "all" for all corpora')
    parser.add_argument('--mode',    choices=_ALL_MODES, action='append', dest='modes',
                        help='One or more modes: supervised unsupervised both rules neighbor symbolic')
    parser.add_argument('--targets', choices=['default', 'hidden', 'strategic', 'all'],
                        default='default')
    parser.add_argument('--min-support',    type=float, default=0.03,
                        help='Apriori min support (default 0.03)')
    parser.add_argument('--min-confidence', type=float, default=0.6,
                        help='Apriori min confidence (default 0.6)')
    parser.add_argument('--min-lift',       type=float, default=2.0,
                        help='Apriori min lift (default 2.0)')
    args = parser.parse_args()

    # Normalise modes: None → ['both'], expand 'both' → supervised+unsupervised
    modes = set(args.modes or ['both'])
    if 'both' in modes:
        modes.discard('both')
        modes.add('supervised')
        modes.add('unsupervised')

    conn = get_conn()
    ensure_tables(conn)

    # Resolve corpora
    if args.corpus == 'all':
        corpus_ids = get_active_corpora(conn)
        print(f"Running on all {len(corpus_ids)} corpora: {corpus_ids}")
        run_corpus_id = 'all'
    else:
        corpus_ids    = [args.corpus]
        run_corpus_id = args.corpus

    all_archetypes = {}
    all_results    = {}

    # ── Supervised ────────────────────────────────────────────────────────────
    if 'supervised' in modes:
        results = run_supervised(
            conn, run_corpus_id,
            target_set=args.targets,
            all_corpus_ids=corpus_ids,
        )
        all_results[run_corpus_id] = results

        if args.corpus == 'all' and len(corpus_ids) > 1:
            for cid in corpus_ids:
                try:
                    res = run_supervised(
                        conn, cid,
                        target_set=args.targets,
                        all_corpus_ids=[cid],
                    )
                    all_results[cid] = res
                except ValueError as e:
                    print(f"  SKIP {cid}: {e}")
            cross_corpus_supervised(conn, corpus_ids)

    # ── Unsupervised ──────────────────────────────────────────────────────────
    if 'unsupervised' in modes:
        archetypes = run_unsupervised(conn, run_corpus_id)
        all_archetypes[run_corpus_id] = archetypes

        if args.corpus == 'all' and len(corpus_ids) > 1:
            per_corpus_archs = {}
            for cid in corpus_ids:
                try:
                    a = run_unsupervised(conn, cid)
                    per_corpus_archs[cid] = a
                except Exception as e:
                    print(f"  SKIP {cid} unsupervised: {e}")
                    per_corpus_archs[cid] = []
            cross_corpus_unsupervised(corpus_ids, per_corpus_archs)

    # ── Neighbor + Rules ─────────────────────────────────────────────────────
    # Load base feature matrix if rules or neighbor mode requested
    if 'rules' in modes or 'neighbor' in modes:
        print(f"\n── Loading base feature matrix for rules/neighbor… ──────────────")
        try:
            df_base, feat_cols, _ = load_supervised_data(conn, run_corpus_id)
        except ValueError as e:
            print(f"  SKIP rules/neighbor: {e}")
            df_base = None

        if df_base is not None:
            # Optional: augment with neighborhood features
            neighbor_df = None
            if 'neighbor' in modes:
                neighbor_df = build_neighborhood_features(conn, run_corpus_id)
                if neighbor_df is not None:
                    # Join neighborhood features onto df_base
                    df_base = df_base.set_index(
                        ['corpus_id', 'persistent_cluster_id', 'period_start']
                    ).join(neighbor_df, how='left').reset_index()
                    print(f"  Augmented matrix: {df_base.shape[1]} total columns")

            if 'rules' in modes:
                # Run without neighbor features first (baseline)
                run_association_rules(
                    conn, run_corpus_id, df_base,
                    min_support=args.min_support,
                    min_confidence=args.min_confidence,
                    min_lift=args.min_lift,
                    includes_neighbor=False,
                )
                # Run with neighbor features if available
                if neighbor_df is not None:
                    run_association_rules(
                        conn, run_corpus_id, df_base,
                        min_support=args.min_support,
                        min_confidence=args.min_confidence,
                        min_lift=args.min_lift,
                        includes_neighbor=True,
                    )

        # Cross-corpus overlap
        cross_corpus_rule_overlap(conn, corpus_ids)
        recommend_thresholds(conn)

    # ── Symbolic ──────────────────────────────────────────────────────────────
    if 'symbolic' in modes:
        try:
            df_base, _, _ = load_supervised_data(conn, run_corpus_id)
        except ValueError as e:
            print(f"  SKIP symbolic: {e}")
            df_base = None
        if df_base is not None:
            run_symbolic_regression(conn, run_corpus_id, df_base)

    # ── Summary report (only for supervised/unsupervised runs) ────────────────
    if 'supervised' in modes or 'unsupervised' in modes:
        print_report(conn, corpus_ids, all_results, all_archetypes)

    conn.close()


if __name__ == '__main__':
    main()

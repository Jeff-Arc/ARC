#!/usr/bin/env python3
"""
arc_discovery.py — Systematic cross-primitive pattern discovery.

ARC Discovery Layer — Systematic Pattern Discovery
===================================================

PURPOSE: Systematically test all geometric features as predictors
of strategic outcomes. Subsumes all existing ML — every model is
a specific point in the feature-lag-target space.

TARGET SETS (--targets flag):
  default    → Core cluster lifecycle: death, splits, future drift
               Strategic Q: "Will this cluster survive?"
  hidden     → Hidden structure: dark matter, graph connectivity,
               spectral gap
               Strategic Q: "Is the field fragmenting structurally?"
  strategic  → Commercial signals: size growth, prosecution heat,
               mergers, void closure, persistence
               Strategic Q: "Where should we file? What is about
               to be contested? Which voids are about to close?"
  all        → All targets combined

FEATURE GROUPS:
  Cluster geometric  → size, drift, cohesion, convergence etc
  Period structural  → spectral_gap, algebraic_connectivity,
                       leiden_modularity etc
  Chunk aggregates   → betweenness, novelty, bridge/boundary counts
  Assignee metadata  → HHI concentration, top share, inventor type
                       (requires assignee_normalized table populated
                       by arc_assignee_ingest.py)
  Lag features       → T-1, T-2, T-3 versions of above

RESULTS LOCATION:
  ml_results table — query by model_name LIKE 'lasso_discovery%'
  findings table   — laws and hypotheses filed from significant results

KEY RESULTS TO DATE (update as new runs complete):
  is_dead G06N_quarterly          AUC=0.9926  (full feature set)
  is_dead H01L_quarterly          AUC=0.5000  (topology-driven, needs own model)
  drift_magnitude_next G06N       R²=0.383
  drift_magnitude_next H01L_23    R²=0.527    (strongest drift predictor)
  spectral_gap_next G06N/H01L     R²=0.894/0.940  (near-perfect)
  dark_matter_chunks_next G06N    R²=0.928
  alg_conn_next H01L              R²=0.840    (G06N=0.050, corpus-specific)

SUPERVISED MODE: LassoCV / L1 LogReg discovers which feature-lag combinations
  predict labeled events (is_dead, is_split_child, phase_transition_score,
  future drift_magnitude). Full ~130-feature matrix from clusters, period_stats,
  chunk_periods aggregates, v_cluster_trajectory lag columns, and assignee metadata.

UNSUPERVISED MODE: KMeans on 33-dim v_cluster_trajectory fingerprints discovers
  recurring cluster lifecycle archetypes; names them by terminal event rates.

Usage:
  python3 arc_discovery.py --corpus G06N_quarterly --mode supervised
  python3 arc_discovery.py --corpus G06N_quarterly --mode supervised --targets strategic
  python3 arc_discovery.py --corpus G06N_quarterly --mode unsupervised
  python3 arc_discovery.py --corpus G06N_quarterly --mode both
  python3 arc_discovery.py --corpus all --mode both
"""

import argparse
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
    from sklearn.metrics import r2_score, roc_auc_score, f1_score
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\nInstall: pip install scikit-learn pandas")

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ── Column exclusions ─────────────────────────────────────────────────────────

# Admin / non-feature columns to exclude from clusters feature matrix
CLUSTER_EXCLUDE = frozenset([
    # Identity / key cols
    'cluster_id', 'corpus_id', 'corpus_type', 'period', 'period_start',
    'period_end', 'persistent_cluster_id', 'match_type', 'is_junk',
    # Non-scalar (1024-dim vectors)
    'centroid', 'velocity_vector',
    # LLM text / timestamps
    'cluster_label', 'cluster_summary', 'labelled_at',
    # Administrative metadata
    'event_model_version',
    # Primary targets — excluded globally (circularity regardless of target)
    'is_dead', 'is_split_child',
    # Dead column: 99–100% NULL across all corpora (bug #99)
    'inheritance_score',
    # Corrupted values pending backfill: point_density formula used 1/distance
    # without distance floor — near-duplicate patent clusters produce values up
    # to 8.3M.  Exclude until compute_point_density backfill is complete.
    'mean_density',
])

# Features excluded only when a specific target is active (circularity guard).
# death_probability / birth_probability are valid inputs for non-death targets
# (e.g. drift_magnitude_next, spectral_gap_next) but circular for is_dead.
TARGET_CONDITIONAL_EXCLUDE: dict = {
    'is_dead':        frozenset(['death_probability', 'birth_probability']),
    'is_split_child': frozenset(['death_probability', 'birth_probability']),
}

# Admin / non-feature columns to exclude from period_stats feature matrix
PSTAT_EXCLUDE = frozenset([
    'corpus_id', 'corpus_type', 'period', 'period_start', 'period_end',
    'anomaly_model_version', 'is_refresh_period',
])

# Admin columns in v_cluster_trajectory (not used in trajectory fingerprint)
TRAJ_ADMIN_COLS = frozenset([
    'corpus_id', 'period_start', 'persistent_cluster_id', 'cluster_id',
    'is_dead', 'is_new', 'is_split_child',   # T0 event labels — exclude from fingerprint
])

# Lag columns provided by v_cluster_trajectory (T-1, T-2, T-3)
TRAJ_LAG_COLS = [
    'size_t1', 'cohesion_t1', 'drift_t1', 'jerk_t1', 'boundary_t1',
    'boundary_pressure_t1', 'split_t1', 'dead_t1',
    'size_t2', 'cohesion_t2', 'drift_t2', 'jerk_t2', 'boundary_t2', 'split_t2',
    'size_t3', 'drift_t3', 'split_t3', 'jerk_t3',
]

# T0 columns in v_cluster_trajectory not present in clusters table
VCT_T0_EXTRA = ['boundary_fraction', 'perturbation_score']

# Assignee metadata features joined from v_cluster_assignee_features
# (requires assignee_normalized table — populated by arc_assignee_ingest.py)
CAF_FEAT_COLS = [
    'assignee_hhi',             # HHI concentration: 0=distributed, 1=monopoly
    'top_assignee_share',       # largest single assignee's share in cluster
    'n_assignees',              # distinct assignees count
    'individual_inventor_ratio', # fraction of chunks from individual inventors
    'university_ratio',         # fraction from universities/research institutes
    'ibm_share',                # IBM-specific share (largest filer in G06N)
]

# chunk_periods aggregates joined to cluster level: (alias, SQL expression)
CPRD_AGGS = [
    ('avg_betweenness',           'AVG(cp.betweenness)'),
    ('avg_uncertainty_score',     'AVG(cp.uncertainty_score)'),
    ('avg_perturbation_score',    'AVG(cp.perturbation_score)'),
    ('avg_belief_persistence',    'AVG(cp.belief_persistence_score)'),
    ('avg_membership_volatility', 'AVG(cp.membership_volatility)'),
    ('avg_novelty_score',         'AVG(cp.novelty_score)'),
    ('avg_point_density',         'AVG(cp.point_density)'),
    ('avg_idea_lineage_depth',    'AVG(cp.idea_lineage_depth)'),
    ('n_boundary_chunks',         'SUM(cp.is_boundary::int)'),
    ('n_bridge_chunks',           'SUM(cp.is_bridge::int)'),
    ('n_dark_matter_chunks',      'SUM(cp.is_dark_matter::int)'),
    ('n_outlier_chunks',          'SUM(cp.is_outlier::int)'),
    # Geometry aggregates added 2026-03-14
    ('avg_boundary_proximity',    'AVG(cp.boundary_proximity)'),
    ('avg_distance_to_centroid',  'AVG(cp.distance_to_centroid)'),
    ('avg_intrinsic_dim',         'AVG(cp.intrinsic_dim)'),
    ('avg_boundary_score',        'AVG(cp.boundary_score)'),
    ('avg_energy',                'AVG(cp.energy)'),
    ('avg_novelty_percentile',    'AVG(cp.novelty_percentile)'),
]

# chunk_periods aggregates excluded until point_density backfill is complete
# (1/distance formula without floor produces extreme outliers for near-duplicate clusters)
CPRD_EXCLUDE = frozenset(['avg_point_density'])

# cluster_edges aggregates joined to cluster level via UNION ALL (both sides)
EDGE_AGGS = [
    ('avg_connection_weight',  'AVG(e.connection_weight::real)'),
    ('avg_weight_change',      'AVG(e.weight_change::real)'),
    ('n_new_edges',            'SUM(CASE WHEN e.is_new  THEN 1 ELSE 0 END)'),
    ('n_lost_edges',           'SUM(CASE WHEN e.is_lost THEN 1 ELSE 0 END)'),
    ('avg_semantic_overlap',   'AVG(e.semantic_overlap_max::real)'),
]

_NUMERIC_TYPES = (
    'real', 'integer', 'bigint', 'smallint', 'numeric',
    'double precision', 'boolean',
)

MIN_LAG_ROWS          = 50
MIN_TRAJ_PIDS         = 10
MIN_SIL_SCORE         = 0.3
ARCHETYPE_SIM_THRESHOLD = 0.85
N_SHAP_FEATURES       = 5

# ── Target sets ───────────────────────────────────────────────────────────────
# Tuples: (column_name, is_binary, metric_key)
_TARGET_SETS = {
    "default": [
        # "Will this cluster die?" — AUC 0.9926 G06N, 0.500 H01L
        ("is_dead",              True,  "roc_auc"),
        # "Will a new cluster form nearby?" — split tracking incomplete
        ("is_split_child",       True,  "roc_auc"),
        # "How far will this cluster move next period?" — R²=0.38 cross-corpus
        ("drift_magnitude_next", False, "r2"),
    ],
    "hidden": [
        # "Will unclusterable content increase?" — R²=0.928 G06N
        ("dark_matter_chunks_next", False, "r2"),
        # "Will graph connectivity change?" — R²=0.840 H01L, 0.050 G06N
        ("alg_conn_next",           False, "r2"),
        # "Will community separation change?" — R²=0.894/0.940 cross-corpus law
        ("spectral_gap_next",       False, "r2"),
    ],
    "strategic": [
        # "Is this space about to get crowded?" — untested
        ("size_percentile_next",   False, "r2"),
        # "Is filing activity about to intensify?" — untested
        ("prosecution_heat_next",  False, "r2"),
        # persistence_score_next REMOVED (bug #68): R²=1.0 label leakage.
        # persistence_score_next = persistence_score + 1/N by construction;
        # any linear model trivially fits this with R²=1.0 — not a real signal.
        # "Will this cluster be absorbed by another?" — untested, low base rate
        ("is_merger_next",         True,  "roc_auc"),
        # "Is the uninvented space about to be filled?" — untested,
        # commercially highest value if signal exists
        ("void_fill_next",         True,         "roc_auc"),
        # "How will this cluster die and where do patents land?"
        # 4-class: absorption/fragmentation/maturation/obsolescence
        # Note: G06N dominated by obsolescence (87%); absorption/fragmentation rare
        ("death_type",             "multiclass", "f1_macro"),
    ],
}
_TARGET_SETS["all"] = (
    _TARGET_SETS["default"] + _TARGET_SETS["hidden"] + _TARGET_SETS["strategic"]
)


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "/var/run/postgresql"),
        dbname=os.environ.get("PGDATABASE", "arc_v4"),
        user=os.environ.get("PGUSER", "jeff"),
    )


def _write_ml_results(conn, rows):
    if not rows:
        return
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO ml_results
              (model_name, corpus_id, target, metric_name,
               metric_value, feature_name, n_samples, notes, recorded_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, rows)
    conn.commit()


def _get_top_shap_features(conn, corpus_id, n=5):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT feature_name
            FROM pipe_cluster_shap_values
            WHERE corpus_id = %s
            GROUP BY feature_name
            ORDER BY AVG(ABS(shap_value)) DESC
            LIMIT %s
        """, (corpus_id, n))
        return [r[0] for r in cur.fetchall()]


def _get_active_corpora(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT legacy_name AS name FROM sys_run_config WHERE status='active' ORDER BY legacy_name")
        return [r[0] for r in cur.fetchall()]


def _discover_feature_cols(conn, corpus_id: str):
    """
    Return (cluster_col_defs, pstat_col_defs) for the given corpus.

    Strategy:
    1. If ml_feature_registry has rows for this corpus, use only features
       marked included=TRUE from source_table in ('clusters', 'period_stats').
       data_type is looked up from information_schema for each included feature.
    2. Fallback: scan information_schema directly (original behaviour).

    cluster_col_defs : list of (name, data_type) tuples (no 'ps_' prefix)
    pstat_col_defs   : list of (name, data_type) tuples (no 'ps_' prefix)
    """
    ph = ','.join(["'%s'" % t for t in _NUMERIC_TYPES])

    # ── Try registry first ────────────────────────────────────────────────────
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM ml_feature_registry
            WHERE corpus_id = %s
        """, (corpus_id,))
        has_registry = cur.fetchone()[0] > 0

    if has_registry:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT feature_name, source_table
                FROM ml_feature_registry
                WHERE corpus_id = %s AND included = TRUE
                  AND source_table IN ('cluster', 'period_stats')
                ORDER BY feature_name
            """, (corpus_id,))
            registry_rows = cur.fetchall()  # [(feature_name, source_table), ...]

        # Build dtype lookup from information_schema
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT column_name, table_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name IN ('clusters', 'pipe_period_stats')
                  AND data_type IN ({ph})
            """)
            # information_schema uses 'clusters' (plural); registry uses 'cluster' (singular)
            dtype_map = {(r[1], r[0]): r[2] for r in cur.fetchall()}

        cluster_col_defs = []
        pstat_col_defs   = []
        for feat, src in registry_rows:
            if src == 'cluster':
                col = feat
                dtype = dtype_map.get(('clusters', col), 'real')
                if col not in CLUSTER_EXCLUDE:
                    cluster_col_defs.append((col, dtype))
            elif src == 'period_stats':
                # period_stats features stored with 'ps_' prefix in registry
                col = feat[3:] if feat.startswith('ps_') else feat
                dtype = dtype_map.get(('period_stats', col), 'real')
                if col not in PSTAT_EXCLUDE:
                    pstat_col_defs.append((col, dtype))

        if cluster_col_defs or pstat_col_defs:
            return cluster_col_defs, pstat_col_defs
        # Fall through if registry was empty/all-excluded

    # ── Fallback: information_schema scan ────────────────────────────────────
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'clusters'
              AND data_type IN ({ph})
            ORDER BY ordinal_position
        """)
        cluster_col_defs = [
            (r[0], r[1]) for r in cur.fetchall() if r[0] not in CLUSTER_EXCLUDE
        ]

        cur.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'pipe_period_stats'
              AND data_type IN ({ph})
            ORDER BY ordinal_position
        """)
        pstat_col_defs = [
            (r[0], r[1]) for r in cur.fetchall() if r[0] not in PSTAT_EXCLUDE
        ]

    return cluster_col_defs, pstat_col_defs


# ── Supervised: data loading ──────────────────────────────────────────────────

_table_types_cache: dict = {}


def _get_table_col_types(conn, table_name):
    """Return {col: data_type} dict for all numeric/bool cols in a table/view."""
    if table_name not in _table_types_cache:
        ph = ','.join(["'%s'" % t for t in _NUMERIC_TYPES])
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                  AND data_type IN ({ph})
            """, (table_name,))
            _table_types_cache[table_name] = {r[0]: r[1] for r in cur.fetchall()}
    return _table_types_cache[table_name]


def _cast_expr(table_alias, col, dtype):
    """Return SQL cast expression for a column, handling boolean → int (no ::real on bool).
    table_alias may be empty string for unqualified column references."""
    ref = f"{table_alias}.{col}" if table_alias else col
    if dtype == 'boolean':
        return f"COALESCE({ref}::int, 0)"
    return f"COALESCE({ref}::real, 0)"


def load_supervised_data(conn, corpus_id):
    """
    Build the full feature matrix for supervised discovery.

    Sources:
    - clusters table: numeric/bool non-admin columns (~59 cols); registry-filtered if available
    - period_stats table: numeric/bool non-admin columns (~42 cols), prefixed ps_; registry-filtered
    - v_cluster_trajectory: T-1/T-2/T-3 lag columns (18 cols) + T0 extras (2 cols)
    - chunk_periods: 18 aggregated cols (avg_*/n_*_chunks) — CPRD_AGGS
    - cluster_edges: 5 aggregated cols (avg_connection_weight etc.) via UNION ALL CTE — EDGE_AGGS
    - Interaction features: 3 (computed in Python)
    - SHAP features: top-N from cluster_shap_values

    Returns (df, feat_groups) where feat_groups is a dict
    mapping group_name -> [col, ...] for tagging.
    """
    cluster_col_defs, pstat_col_defs = _discover_feature_cols(conn, corpus_id)
    cluster_cols = [name for name, _ in cluster_col_defs]
    pstat_cols   = [name for name, _ in pstat_col_defs]

    c_select = ",\n            ".join(
        f"{_cast_expr('c', col, dtype)} AS {col}"
        for col, dtype in cluster_col_defs
    )
    ps_select = ",\n            ".join(
        f"{_cast_expr('ps', col, dtype)} AS ps_{col}"
        for col, dtype in pstat_col_defs
    )
    vct_types = _get_table_col_types(conn, 'v_cluster_trajectory')
    vct_lag_select = ",\n            ".join(
        f"{_cast_expr('vt', col, vct_types.get(col, 'real'))} AS {col}"
        for col in TRAJ_LAG_COLS
    )
    vct_extra_select = ",\n            ".join(
        f"{_cast_expr('vt', col, vct_types.get(col, 'real'))} AS vct_{col}"
        for col in VCT_T0_EXTRA
    )
    cprd_agg_select = ",\n            ".join(
        f"COALESCE({expr}::real, 0) AS {alias}"
        for alias, expr in CPRD_AGGS if alias not in CPRD_EXCLUDE
    )
    cprd_ref_select = ",\n            ".join(
        f"COALESCE(cprd.{alias}, 0) AS {alias}"
        for alias, _ in CPRD_AGGS if alias not in CPRD_EXCLUDE
    )
    edge_agg_select = ",\n                ".join(
        f"COALESCE({expr}, 0) AS {alias}" for alias, expr in EDGE_AGGS
    )
    edge_ref_select = ",\n            ".join(
        f"COALESCE(eagg.{alias}, 0) AS {alias}" for alias, _ in EDGE_AGGS
    )
    # Assignee concentration features — requires assignee_normalized table
    # (populated from g_assignee_disambiguated.tsv by arc_assignee_ingest.py)
    # Strategic Q: "Does corporate ownership concentration predict
    # cluster fate differently than fragmented ownership?"
    # HHI=1.0 means one company owns all patents in cluster
    # HHI=0.0 means perfectly distributed ownership
    caf_select = ",\n            ".join(
        f"COALESCE(caf.{col}::real, 0) AS {col}" for col in CAF_FEAT_COLS
    )
    ps_cols_aliased = ",\n            ".join(
        f"{_cast_expr('ps', col, dtype)} AS ps_{col}"
        for col, dtype in pstat_col_defs
    )

    sql = f"""
        WITH cprd AS (
            SELECT
                cp.corpus_id, cp.cluster_id, cp.period_start,
                {cprd_agg_select}
            FROM chunk_periods cp
            WHERE cp.corpus_id = %(cid)s
            GROUP BY cp.corpus_id, cp.cluster_id, cp.period_start
        ),
        edge_agg AS (
            -- Unnest both sides so each cluster gets stats for all its edges
            SELECT
                corpus_id, period_start, cluster_id,
                {edge_agg_select}
            FROM (
                SELECT corpus_id, period_start, cluster_a AS cluster_id,
                       connection_weight, weight_change, is_new, is_lost,
                       semantic_overlap_max
                  FROM pipe_cluster_edges WHERE corpus_id = %(cid)s
                UNION ALL
                SELECT corpus_id, period_start, cluster_b AS cluster_id,
                       connection_weight, weight_change, is_new, is_lost,
                       semantic_overlap_max
                  FROM pipe_cluster_edges WHERE corpus_id = %(cid)s
            ) e
            GROUP BY corpus_id, period_start, cluster_id
        ),
        ps_next AS (
            SELECT
                corpus_id,
                period_start,
                LEAD(n_dark_matter_chunks::real) OVER (
                    PARTITION BY corpus_id ORDER BY period_start
                ) AS dark_matter_chunks_next,
                LEAD(algebraic_connectivity) OVER (
                    PARTITION BY corpus_id ORDER BY period_start
                ) AS alg_conn_next,
                LEAD(spectral_gap) OVER (
                    PARTITION BY corpus_id ORDER BY period_start
                ) AS spectral_gap_next
            FROM pipe_period_stats
            WHERE corpus_id = %(cid)s
        ),
        void_fill AS (
            -- Pre-compute which (corpus_id, period_start, persistent_cluster_id)
            -- had a void close during that period (last_seen = that period AND
            -- status = 'inactive').  Using LEAD() on this flags void closure at
            -- T+1 from T's perspective, eliminating the label leakage where the
            -- old code used last_seen = c.period_start (= closure AT T, not T+1).
            SELECT v.corpus_id,
                   v.last_seen  AS period_start,
                   v.persistent_cluster_a AS pcid
            FROM   pipe_voids v
            WHERE  v.corpus_id = %(cid)s
              AND  v.status    = 'inactive'
              AND  v.last_seen IS NOT NULL
            UNION
            SELECT v.corpus_id,
                   v.last_seen,
                   v.persistent_cluster_b
            FROM   pipe_voids v
            WHERE  v.corpus_id = %(cid)s
              AND  v.status    = 'inactive'
              AND  v.last_seen IS NOT NULL
        )
        SELECT
            c.corpus_id,
            c.period_start,
            c.cluster_id,
            c.persistent_cluster_id,
            c.is_dead::int        AS is_dead,
            c.is_split_child::int AS is_split_child,
            {c_select},
            {vct_lag_select},
            {vct_extra_select},
            {cprd_ref_select},
            {edge_ref_select},
            {ps_cols_aliased},
            LEAD(c.drift_magnitude) OVER (
                PARTITION BY c.persistent_cluster_id
                ORDER BY c.period_start
            ) AS drift_magnitude_next,
            psn.dark_matter_chunks_next,
            psn.alg_conn_next,
            psn.spectral_gap_next,
            LEAD(c.size_percentile) OVER (
                PARTITION BY c.persistent_cluster_id ORDER BY c.period_start
            ) AS size_percentile_next,
            LEAD(c.prosecution_heat_index) OVER (
                PARTITION BY c.persistent_cluster_id ORDER BY c.period_start
            ) AS prosecution_heat_next,
            LEAD(c.persistence_score) OVER (
                PARTITION BY c.persistent_cluster_id ORDER BY c.period_start
            ) AS persistence_score_next,
            LEAD(c.is_merger_target::int) OVER (
                PARTITION BY c.persistent_cluster_id ORDER BY c.period_start
            ) AS is_merger_next,
            LEAD(
                CASE WHEN vf.pcid IS NOT NULL THEN 1 ELSE 0 END
            ) OVER (
                PARTITION BY c.persistent_cluster_id ORDER BY c.period_start
            ) AS void_fill_next,
            {caf_select},
            vdt.death_type
        FROM pipe_clusters c
        LEFT JOIN v_cluster_trajectory vt
               ON vt.corpus_id    = c.corpus_id
              AND vt.cluster_id   = c.cluster_id
              AND vt.period_start = c.period_start
        LEFT JOIN cprd
               ON cprd.corpus_id    = c.corpus_id
              AND cprd.cluster_id   = c.cluster_id
              AND cprd.period_start = c.period_start
        LEFT JOIN edge_agg eagg
               ON eagg.corpus_id    = c.corpus_id
              AND eagg.cluster_id   = c.cluster_id
              AND eagg.period_start = c.period_start
        LEFT JOIN pipe_period_stats ps
               ON ps.corpus_id    = c.corpus_id
              AND ps.period_start = c.period_start
        LEFT JOIN ps_next psn
               ON psn.corpus_id    = c.corpus_id
              AND psn.period_start = c.period_start
        LEFT JOIN void_fill vf
               ON vf.corpus_id    = c.corpus_id
              AND vf.period_start = c.period_start
              AND vf.pcid         = c.persistent_cluster_id
        LEFT JOIN v_cluster_assignee_features caf
               ON caf.corpus_id             = c.corpus_id
              AND caf.persistent_cluster_id  = c.persistent_cluster_id
              AND caf.period_start           = c.period_start
        LEFT JOIN v_cluster_death_type vdt
               ON vdt.corpus_id             = c.corpus_id
              AND vdt.persistent_cluster_id  = c.persistent_cluster_id
              AND vdt.period_start           = c.period_start
        WHERE c.corpus_id = %(cid)s
          AND c.is_junk   = FALSE
          AND c.cohesion  IS NOT NULL
        ORDER BY c.period_start, c.cluster_id
    """

    with conn.cursor() as cur:
        cur.execute(sql, {"cid": corpus_id})
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()

    if not rows:
        raise ValueError(f"No supervised data for {corpus_id}")

    df = pd.DataFrame(rows, columns=cols)

    # SHAP features (top-N pivoted from cluster_shap_values)
    top_shap = _get_top_shap_features(conn, corpus_id, N_SHAP_FEATURES)
    shap_feat_names = []
    if top_shap:
        ph = ", ".join(["%s"] * len(top_shap))
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT period_start, cluster_id, feature_name, shap_value "
                f"FROM pipe_cluster_shap_values "
                f"WHERE corpus_id = %s AND feature_name IN ({ph})",
                [corpus_id] + top_shap,
            )
            shap_rows = cur.fetchall()
        if shap_rows:
            shap_df = pd.DataFrame(
                shap_rows,
                columns=["period_start", "cluster_id", "feature_name", "shap_value"],
            )
            pivot = shap_df.pivot_table(
                index=["period_start", "cluster_id"],
                columns="feature_name",
                values="shap_value",
                aggfunc="first",
            ).reset_index()
            pivot.columns = [
                f"shap_{c}" if c not in ("period_start", "cluster_id") else c
                for c in pivot.columns
            ]
            df = df.merge(pivot, on=["period_start", "cluster_id"], how="left")
            shap_feat_names = [f"shap_{f}" for f in top_shap]

    # Interaction features
    df["size_x_drift"]              = df.get("size", pd.Series(0.0)).fillna(0) * \
                                      df.get("drift_magnitude", pd.Series(0.0)).fillna(0)
    df["betweenness_x_persistence"] = df.get("mean_betweenness", pd.Series(0.0)).fillna(0) * \
                                      df.get("persistence_score", pd.Series(0.0)).fillna(0)
    df["drift_t1_x_drift"]          = df.get("drift_t1", pd.Series(0.0)).fillna(0) * \
                                      df.get("drift_magnitude", pd.Series(0.0)).fillna(0)
    interaction_feats = ["size_x_drift", "betweenness_x_persistence", "drift_t1_x_drift"]

    # Feature groups for tagging
    vct_extra_cols  = [f"vct_{c}" for c in VCT_T0_EXTRA]
    cprd_agg_cols   = [alias for alias, _ in CPRD_AGGS if alias not in CPRD_EXCLUDE]
    edge_agg_cols   = [alias for alias, _ in EDGE_AGGS]
    pstat_feat_cols = [f"ps_{c}" for c in pstat_cols]
    feat_groups = {
        "cluster":     cluster_cols,
        "lag":         TRAJ_LAG_COLS,
        "vct_extra":   vct_extra_cols,
        "chunk_agg":   cprd_agg_cols,
        "edge_agg":    edge_agg_cols,
        "period":      pstat_feat_cols,
        "shap":        shap_feat_names,
        "interaction": interaction_feats,
        "assignee":    CAF_FEAT_COLS,
    }
    return df, feat_groups


def _file_supervised_findings(conn, corpus_id, target_results):
    """
    Phase 6: File findings in the findings table for notable supervised results.

    Rules:
    - Any target with R² > 0.15 or AUC > 0.65   → type='finding'
    - Any assignee feature with nonzero Lasso coef → type='finding' (ownership signal)
    - void_fill_next AUC > 0.60                   → type='law' (commercial core signal)
    """
    ASSIGNEE_COLS = set(CAF_FEAT_COLS)
    to_insert = []

    for target_col, is_binary, metric_key, score, nonzero in target_results:
        thresh = (metric_key == "r2"       and score > 0.15) or \
                 (metric_key == "roc_auc"  and score > 0.65) or \
                 (metric_key == "f1_macro" and score > 0.35)

        if thresh:
            top_lines = "\n".join(
                f"  {feat:<44} std_impact={simp:+.4f}"
                for feat, simp, _ in nonzero[:10]
            )
            body = (
                f"Corpus: {corpus_id}\n"
                f"Target: {target_col}\n"
                f"Metric: {metric_key} = {score:.4f}\n"
                f"Non-zero features: {len(nonzero)}\n\n"
                f"Top predictors:\n{top_lines}"
            )
            to_insert.append((
                "finding",
                f"Supervised Discovery: {target_col} — {corpus_id}",
                body, [corpus_id],
            ))

        # Assignee concentration predicts death?
        if target_col == "is_dead":
            assignee_hits = [(f, s) for f, s, _ in nonzero if f in ASSIGNEE_COLS]
            if assignee_hits:
                coef_lines = "\n".join(
                    f"  {feat:<44} std_impact={simp:+.4f}"
                    for feat, simp in assignee_hits
                )
                to_insert.append((
                    "finding",
                    "Assignee Concentration as Cluster Predictor",
                    f"Corpus: {corpus_id}\n"
                    f"Target: {target_col} ({metric_key}={score:.4f})\n\n"
                    f"Assignee features with nonzero Lasso coefficients:\n{coef_lines}\n\n"
                    f"Interpretation: Concentrated cluster ownership (high HHI) may indicate "
                    f"a different death mechanism — institutional abandonment vs organic decline. "
                    f"Positive HHI coef → concentrated clusters die more; "
                    f"negative → concentrated clusters are protected by their owner.",
                    [corpus_id],
                ))

        # void_fill_next — law candidate (commercially highest value)
        if target_col == "void_fill_next" and metric_key == "roc_auc" and score > 0.60:
            top_lines = "\n".join(
                f"  {feat:<44} std_impact={simp:+.4f}"
                for feat, simp, _ in nonzero[:10]
            )
            to_insert.append((
                "law",
                f"Void Fill Prediction — Law Candidate — {corpus_id}",
                f"Corpus: {corpus_id}\n"
                f"AUC = {score:.4f} for predicting void closure in next period.\n\n"
                f"Predicting which voids close is the core commercial output of the ARC system.\n"
                f"Cluster-level geometric features at period T predict void closure at T+1.\n\n"
                f"Top predictors:\n{top_lines}",
                [corpus_id],
            ))

    if not to_insert:
        return

    with conn.cursor() as cur:
        for ftype, title, body, corpus_ids in to_insert:
            cur.execute("""
                INSERT INTO sci_findings
                  (type, title, body, corpus_ids, created_by, source, outcome)
                VALUES (%s, %s, %s, %s, 'arc_discovery', 'arc_discovery.py', 'pending')
                RETURNING id
            """, (ftype, title, body, corpus_ids))
            fid = cur.fetchone()[0]
            print(f"  Filed {ftype} id={fid}: {title}")
    conn.commit()


def _chronological_split(df, target_col, frac_train=0.8):
    df = df.dropna(subset=[target_col]).copy()
    periods = sorted(df["period_start"].unique())
    cut = int(len(periods) * frac_train)
    train_p = set(periods[:cut])
    mask = df["period_start"].isin(train_p)
    return df[mask], df[~mask]


# ── Supervised: run ────────────────────────────────────────────────────────────

def run_supervised(conn, corpus_id, target_set="default"):
    print(f"\n── Supervised: {corpus_id} (targets={target_set}) ────────────────")

    df, feat_groups = load_supervised_data(conn, corpus_id)
    n_total = len(df)

    # Check lag coverage
    n_lag_rows = int(df["size_t3"].notna().sum()) if "size_t3" in df.columns else 0
    print(f"  {n_total:,} clusters, {n_lag_rows:,} with T-3 lag history")
    if n_lag_rows < MIN_LAG_ROWS:
        print(f"  WARNING: {n_lag_rows} lag rows < {MIN_LAG_ROWS} — lag features omitted")
        feat_groups.pop("lag", None)

    # Build flat feature list with tags
    feat_tag = {}
    feat_names = []
    for group, cols in feat_groups.items():
        for col in cols:
            if col in df.columns or col in feat_names:
                feat_tag[col] = group
                if col not in feat_names:
                    feat_names.append(col)

    print(f"  Feature matrix: {len(feat_names)} features "
          f"({', '.join(f'{g}:{len(c)}' for g, c in feat_groups.items() if c)})")

    targets = _TARGET_SETS[target_set]

    now = datetime.now(timezone.utc)
    all_ml_rows = []
    target_results = []   # (target_col, is_binary, metric_key, score, nonzero)

    for target_col, is_binary, metric_key in targets:
        if target_col not in df.columns:
            print(f"  SKIP {target_col}: column not found")
            continue

        df_t = df.dropna(subset=[target_col]).copy()
        if is_binary == "multiclass":
            df_t[target_col] = df_t[target_col].astype(str)
            class_counts = df_t[target_col].value_counts()
            if len(class_counts) < 2:
                print(f"  SKIP {target_col}: only {len(class_counts)} class in data")
                continue
            print(f"  {target_col}: {dict(class_counts)} class distribution")
        elif is_binary:
            df_t[target_col] = df_t[target_col].astype(int)
            n_pos = int(df_t[target_col].sum())
            if n_pos < 10:
                print(f"  SKIP {target_col}: {n_pos} positives (< 10)")
                continue

        df_train, df_test = _chronological_split(df_t, target_col)
        if len(df_train) < 20 or len(df_test) < 5:
            print(f"  SKIP {target_col}: train={len(df_train)}, test={len(df_test)}")
            continue

        cond_excl = TARGET_CONDITIONAL_EXCLUDE.get(target_col, frozenset())
        present = [f for f in feat_names if f in df_t.columns and f not in cond_excl]
        if cond_excl:
            dropped = [f for f in feat_names if f in cond_excl and f in df_t.columns]
            if dropped:
                print(f"    [leakage guard] dropped for {target_col}: {dropped}")
        X_tr = df_train[present].fillna(0).values.astype(np.float32)
        X_te = df_test[present].fillna(0).values.astype(np.float32)
        y_tr = df_train[target_col].values
        y_te = df_test[target_col].values

        # Drop constant features
        feat_std = X_tr.std(axis=0)
        nonconstant = feat_std > 0
        X_tr = X_tr[:, nonconstant]
        X_te = X_te[:, nonconstant]
        feat_std = feat_std[nonconstant]
        present_nc = [present[i] for i in range(len(present)) if nonconstant[i]]

        # Guard: non-regression targets need multiple classes in training set
        if is_binary and len(set(y_tr)) < 2:
            print(f"  SKIP {target_col}: training set has only one class "
                  f"(chronological split imbalance)")
            continue

        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_te_s  = scaler.transform(X_te)

        try:
            if is_binary == "multiclass":
                # L1 logistic regression (sklearn 1.8+: no multi_class kwarg needed)
                # Use fixed C=0.1 (sparse solution, equivalent to Lasso for multiclass)
                from sklearn.linear_model import LogisticRegression as LR
                model = LR(
                    C=0.1, penalty="l1", solver="saga",
                    class_weight="balanced", max_iter=1000, random_state=42,
                    n_jobs=-1,
                )
                model.fit(X_tr_s, y_tr)
                score = float(f1_score(
                    y_te, model.predict(X_te_s), average="macro", zero_division=0
                ))
                # coef_ shape: (n_classes, n_features) — use max abs across classes
                coefs = np.abs(model.coef_).max(axis=0)
            elif is_binary:
                model = LogisticRegressionCV(
                    Cs=10, penalty="l1", solver="saga",
                    class_weight="balanced", cv=5, max_iter=500, random_state=42,
                    n_jobs=-1,
                )
                model.fit(X_tr_s, y_tr)
                score = float(roc_auc_score(y_te, model.predict_proba(X_te_s)[:, 1]))
                coefs = model.coef_[0]
            else:
                model = LassoCV(cv=5, max_iter=5000, random_state=42, n_jobs=-1)
                model.fit(X_tr_s, y_tr)
                score = float(r2_score(y_te, model.predict(X_te_s)))
                coefs = model.coef_
        except Exception as exc:
            print(f"  SKIP {target_col}: {exc}")
            continue

        # Guard: degenerate regression results indicate train/test distribution
        # shift or a SQL bug in the target (e.g. missing PARTITION BY on LEAD).
        # Writing R²=-283 to ml_results and findings is misleading — skip instead.
        if metric_key == "r2" and score < -10:
            print(f"  SKIP {target_col}: degenerate R²={score:.1f} "
                  f"(train/test distribution shift or target construction error)")
            continue

        std_imp = coefs * feat_std
        nonzero = [
            (present_nc[i], float(std_imp[i]), float(coefs[i]))
            for i in range(len(present_nc)) if coefs[i] != 0.0
        ]
        nonzero.sort(key=lambda x: abs(x[1]), reverse=True)
        n_samples = len(df_t)
        model_name = f"lasso_discovery_{target_col}"

        if is_binary == "multiclass":
            model_desc = "L1 Multinomial LogReg"
        elif is_binary:
            model_desc = "L1 LogReg"
        else:
            model_desc = "LassoCV"

        print(f"  {target_col}: {metric_key}={score:.4f}, {len(nonzero)} non-zero features")
        for feat, simp, _ in nonzero[:5]:
            print(f"    {feat:<44} {simp:+.4f}  ({feat_tag.get(feat, '?')})")

        all_ml_rows.append((
            model_name, corpus_id, target_col, metric_key,
            score, None, n_samples,
            f"{model_desc}; "
            f"{'lag+' if 'lag' in feat_groups else 'no lag; '}"
            f"chronological split; {len(nonzero)} non-zero",
            now,
        ))
        for feat, simp, _ in nonzero:
            all_ml_rows.append((
                model_name, corpus_id, target_col, "lasso_coefficient",
                simp, feat, n_samples, feat_tag.get(feat, "cluster"), now,
            ))
        target_results.append((target_col, is_binary, metric_key, score, nonzero))

    with conn.cursor() as cur:
        target_cols = [t[0] for t in targets]
        for target_col in target_cols:
            cur.execute(
                "DELETE FROM ml_results "
                "WHERE model_name=%s AND corpus_id=%s "
                "AND target = ANY(%s)",
                (f"lasso_discovery_{target_col}", corpus_id, target_cols),
            )
    _write_ml_results(conn, all_ml_rows)
    print(f"  Wrote {len(all_ml_rows)} rows to ml_results.")
    _file_supervised_findings(conn, corpus_id, target_results)


# ── Unsupervised: trajectory matrix ──────────────────────────────────────────

def _load_trajectory_matrix(conn, corpus_id):
    """
    Build a trajectory fingerprint per persistent_cluster_id using direct
    window functions on the clusters table (faster than querying v_cluster_trajectory
    which scans the full view).

    Fingerprint dimensions: T0 (15 feats) + T-1 (8) + T-2 (6) + T-3 (4) = 33 cols.
    Returns (pid_list, X, traj_feat_names).
    """
    # T0 features (15): match v_cluster_trajectory T0 structure
    t0_feats = [
        ("size",                   "c.size::real",                    "real"),
        ("cohesion",               "c.cohesion",                      "real"),
        ("drift_magnitude",        "c.drift_magnitude",               "real"),
        ("jerk",                   "c.jerk",                          "real"),
        ("acceleration",           "c.acceleration",                  "real"),
        ("velocity",               "c.velocity",                      "real"),
        ("boundary_pressure_rate", "c.boundary_pressure_rate",        "real"),
        ("convergence_score",      "c.convergence_score",             "real"),
        ("mean_betweenness",       "c.mean_betweenness",              "real"),
        ("persistence_score",      "c.persistence_score",             "real"),
        ("age_periods",            "c.age_periods::real",             "real"),
        ("elongation_ratio",       "c.elongation_ratio",              "real"),
        ("death_probability",      "c.death_probability",             "real"),
        ("birth_probability",      "c.birth_probability",             "real"),
        ("n_saddle_points",        "c.n_saddle_points::real",         "real"),
        ("distance_entropy",       "c.distance_entropy",              "real"),
        ("cluster_reachability",   "c.cluster_reachability::real",    "real"),
        ("n_attractors",           "c.n_attractors::real",            "real"),
    ]
    # Lag columns (T-1: 8, T-2: 6, T-3: 4)
    lag_feats = [
        # T-1
        ("size_t1",              "c.size::real",              1),
        ("cohesion_t1",          "c.cohesion",                1),
        ("drift_t1",             "c.drift_magnitude",         1),
        ("jerk_t1",              "c.jerk",                    1),
        ("boundary_t1",          "c.mean_boundary_score",     1),
        ("boundary_pressure_t1", "c.boundary_pressure_rate",  1),
        ("split_t1",             "c.is_split_child::int",     1),
        ("dead_t1",              "c.is_dead::int",            1),
        # T-2
        ("size_t2",    "c.size::real",          2),
        ("cohesion_t2","c.cohesion",            2),
        ("drift_t2",   "c.drift_magnitude",     2),
        ("jerk_t2",    "c.jerk",                2),
        ("boundary_t2","c.mean_boundary_score", 2),
        ("split_t2",   "c.is_split_child::int", 2),
        # T-3
        ("size_t3",  "c.size::real",          3),
        ("drift_t3", "c.drift_magnitude",     3),
        ("jerk_t3",  "c.jerk",                3),
        ("split_t3", "c.is_split_child::int", 3),
    ]

    t0_select = ",\n            ".join(
        f"COALESCE({expr}, 0) AS {name}" for name, expr, _ in t0_feats
    )
    lag_select = ",\n            ".join(
        f"LAG(COALESCE({expr}, 0), {lag}) OVER w AS {name}"
        for name, expr, lag in lag_feats
    )

    sql = f"""
        WITH lagged AS (
            SELECT
                c.persistent_cluster_id,
                c.period_start,
                {t0_select},
                {lag_select}
            FROM pipe_clusters c
            WHERE c.corpus_id           = %(cid)s
              AND c.is_junk             = FALSE
              AND c.persistent_cluster_id IS NOT NULL
            WINDOW w AS (
                PARTITION BY c.persistent_cluster_id
                ORDER BY c.period_start
            )
        )
        SELECT DISTINCT ON (persistent_cluster_id)
            persistent_cluster_id,
            {", ".join(name for name, _, *_ in t0_feats)},
            {", ".join(name for name, *_ in lag_feats)}
        FROM lagged
        WHERE size_t3 IS NOT NULL
        ORDER BY persistent_cluster_id, period_start DESC
    """

    with conn.cursor() as cur:
        cur.execute(sql, {"cid": corpus_id})
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()

    traj_feat_names = [name for name, _, *_ in t0_feats] + \
                      [name for name, *_ in lag_feats]

    if not rows:
        return [], np.empty((0, len(traj_feat_names)), dtype=np.float32), traj_feat_names

    df = pd.DataFrame(rows, columns=cols)
    pids = list(df["persistent_cluster_id"])
    X    = df[traj_feat_names].fillna(0).values.astype(np.float32)

    return pids, X, traj_feat_names


def _compute_terminal_states(conn, corpus_id, pids, assignments, k):
    """Return per-archetype stats: death_rate, split_rate, survival_rate, drift_declining."""
    if not pids:
        return {}

    with conn.cursor() as cur:
        cur.execute("""
            WITH last_per_pid AS (
                SELECT persistent_cluster_id, MAX(period_start) AS last_period
                FROM pipe_clusters
                WHERE corpus_id = %s AND persistent_cluster_id = ANY(%s)
                  AND is_junk = FALSE
                GROUP BY persistent_cluster_id
            ),
            last_state AS (
                SELECT c.persistent_cluster_id,
                       c.is_dead, c.is_split_child,
                       c.drift_magnitude AS drift_last
                FROM pipe_clusters c
                JOIN last_per_pid lp
                     ON lp.persistent_cluster_id = c.persistent_cluster_id
                    AND lp.last_period = c.period_start
                WHERE c.corpus_id = %s AND c.is_junk = FALSE
            ),
            penultimate AS (
                SELECT c.persistent_cluster_id,
                       c.drift_magnitude AS drift_prev
                FROM pipe_clusters c
                JOIN last_per_pid lp
                     ON lp.persistent_cluster_id = c.persistent_cluster_id
                WHERE c.corpus_id = %s AND c.is_junk = FALSE
                  AND c.period_start = (
                      SELECT MAX(c2.period_start)
                      FROM pipe_clusters c2
                      WHERE c2.corpus_id = %s
                        AND c2.persistent_cluster_id = c.persistent_cluster_id
                        AND c2.period_start < lp.last_period
                  )
            )
            SELECT ls.persistent_cluster_id,
                   ls.is_dead, ls.is_split_child, ls.drift_last,
                   p.drift_prev
            FROM last_state ls
            LEFT JOIN penultimate p USING (persistent_cluster_id)
        """, (corpus_id, list(pids), corpus_id, corpus_id, corpus_id))
        state_rows = cur.fetchall()

    state_map = {
        r[0]: {
            "is_dead":    bool(r[1]),
            "is_split":   bool(r[2]) if r[2] is not None else False,
            "drift_last": r[3] or 0.0,
            "drift_prev": r[4] or 0.0,
        }
        for r in state_rows
    }

    stats = {}
    for a in range(k):
        members = [pids[i] for i, asgn in enumerate(assignments) if asgn == a]
        if not members:
            stats[a] = None
            continue
        n         = len(members)
        deaths    = sum(1 for p in members if state_map.get(p, {}).get("is_dead", False))
        splits    = sum(1 for p in members if state_map.get(p, {}).get("is_split", False))
        declining = sum(
            1 for p in members
            if state_map.get(p, {}).get("drift_last", 0) <
               state_map.get(p, {}).get("drift_prev",  0)
        )
        stats[a] = {
            "n":                    n,
            "death_rate":           deaths   / n,
            "split_rate":           splits   / n,
            "survival_rate":        (n - deaths - splits) / n,
            "drift_declining_rate": declining / n,
            "example_pids":         members[:3],
        }
    return stats


def _name_archetype(stats, unknown_n):
    if stats["death_rate"] > 0.6:
        return "death_precursor"
    if stats["split_rate"] > 0.4:
        return "crystallization_precursor"
    if stats["survival_rate"] > 0.8 and stats["drift_declining_rate"] > 0.5:
        return "stabilizing"
    return f"unknown_{unknown_n}"


def _write_archetype_finding(conn, corpus_id, label, stats, mean_center,
                             feat_names, k, sil):
    behavioral_purity = max(
        stats["death_rate"], stats["split_rate"], stats["survival_rate"]
    )
    confidence = round(min(0.95, (sil + behavioral_purity) / 2.0), 3)

    n_feats = len(feat_names)
    # Feature names are flat; annotate by T-step
    def step_label(i):
        name = feat_names[i] if i < n_feats else "?"
        if "_t1" in name:  return "T-1"
        if "_t2" in name:  return "T-2"
        if "_t3" in name:  return "T-3"
        return "T0"

    top_feats = sorted(range(n_feats), key=lambda i: abs(float(mean_center[i])), reverse=True)[:12]
    center_lines = [
        f"  {feat_names[i]:<42} {float(mean_center[i]):+.3f}  ({step_label(i)})"
        for i in top_feats
    ]

    body_lines = [
        f"Corpus: {corpus_id} | KMeans k={k} (silhouette={sil:.3f})",
        f"Members: {stats['n']} clusters",
        f"Terminal death rate:  {stats['death_rate']:.1%}",
        f"Terminal split rate:  {stats['split_rate']:.1%}",
        f"Survival rate:        {stats['survival_rate']:.1%}",
        f"Drift declining:      {stats['drift_declining_rate']:.1%}",
        f"Example persistent_cluster_ids: "
        f"{', '.join(str(p) for p in stats['example_pids'])}",
        "",
        f"Top archetype features (mean center, {len(feat_names)}-dim fingerprint):",
    ] + center_lines

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO sci_findings
              (type, title, body, corpus_ids, n_observations,
               confidence, created_by, source, outcome)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            "principle",
            f"Trajectory Archetype: {label} — {corpus_id}",
            "\n".join(body_lines),
            [corpus_id],
            stats["n"],
            confidence,
            "arc_discovery",
            "arc_discovery.py",
            "pending",
        ))
        return cur.fetchone()[0]


def run_unsupervised(conn, corpus_id):
    """
    Returns list of archetype dicts (label, center vector, feat_names, stats)
    for cross-corpus matching.
    """
    print(f"\n── Unsupervised: {corpus_id} ────────────────────────────────────")

    pids, X, feat_names = _load_trajectory_matrix(conn, corpus_id)
    if len(pids) < MIN_TRAJ_PIDS:
        print(f"  Only {len(pids)} trajectory-eligible PIDs (< {MIN_TRAJ_PIDS}) — skip")
        return []

    print(f"  {len(pids)} PIDs, trajectory matrix {X.shape} ({len(feat_names)} features)")

    # Normalize each feature column across all PIDs
    X_norm = X.copy()
    for fi in range(X.shape[1]):
        vmin, vmax = X[:, fi].min(), X[:, fi].max()
        if vmax > vmin:
            X_norm[:, fi] = (X[:, fi] - vmin) / (vmax - vmin)

    # Select k by silhouette score
    k_max = min(9, len(pids) // 2)
    best_k, best_sil, best_labels = 3, -1.0, None
    sil_scores = {}
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
          ", ".join(f"k={k}:{s:.3f}" for k, s in sorted(sil_scores.items())))
    print(f"  Best k={best_k} (sil={best_sil:.3f})")

    if best_sil < MIN_SIL_SCORE:
        msg = (
            f"No stable trajectory archetypes found — best silhouette={best_sil:.3f} "
            f"(threshold {MIN_SIL_SCORE}). Corpus may lack recurring unlabeled patterns."
        )
        print(f"  {msg}")
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sci_findings
                  (type, title, body, corpus_ids, created_by, source, outcome)
                VALUES ('finding', %s, %s, %s, 'arc_discovery', 'arc_discovery.py', 'pending')
            """, (f"Discovery note — {corpus_id}", msg, [corpus_id]))
        conn.commit()
        return []

    km_final    = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    assignments = km_final.fit_predict(X_norm)
    centers     = km_final.cluster_centers_

    terminal   = _compute_terminal_states(conn, corpus_id, pids, assignments, best_k)
    now        = datetime.now(timezone.utc)
    ml_rows    = []
    archetypes = []
    unknown_n  = 0

    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM ml_results WHERE model_name LIKE 'trajectory_kmeans_%%' AND corpus_id=%s",
            (corpus_id,),
        )

    for a in range(best_k):
        stats = terminal.get(a)
        if stats is None:
            continue
        if "unknown" in _name_archetype(stats, 0):
            unknown_n += 1
        label = _name_archetype(stats, unknown_n)

        print(
            f"  Archetype {a}: {label:<30} "
            f"n={stats['n']}, death={stats['death_rate']:.1%}, "
            f"split={stats['split_rate']:.1%}, survive={stats['survival_rate']:.1%}"
        )

        fid = _write_archetype_finding(
            conn, corpus_id, label, stats, centers[a], feat_names, best_k, best_sil
        )
        model_name = f"trajectory_kmeans_{best_k}"
        ml_rows += [
            (model_name, corpus_id, "trajectory_archetype", "silhouette_score",
             best_sil, None, len(pids), f"k={best_k}", now),
            (model_name, corpus_id, "trajectory_archetype", "archetype_label",
             float(a), label, stats["n"], f"finding_id={fid}", now),
            (model_name, corpus_id, "trajectory_archetype", "n_clusters_matching",
             float(stats["n"]), label, len(pids), None, now),
        ]
        archetypes.append({
            "label":      label,
            "idx":        a,
            "center":     centers[a],
            "feat_names": feat_names,
            "stats":      stats,
        })

    _write_ml_results(conn, ml_rows)
    conn.commit()
    print(f"  Wrote {len(ml_rows)} ml_results rows, {len(archetypes)} archetypes, "
          f"{len(archetypes)} findings.")
    return archetypes


# ── Cross-corpus comparison ───────────────────────────────────────────────────

def cross_corpus_supervised(conn, corpus_ids):
    print(f"\n── Cross-corpus supervised comparison ───────────────────────────")
    per_corpus = {}
    for cid in corpus_ids:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT target, feature_name, metric_value
                FROM ml_results
                WHERE model_name LIKE 'lasso_discovery_%%'
                  AND metric_name = 'lasso_coefficient'
                  AND corpus_id = %s
                ORDER BY target, ABS(metric_value) DESC
            """, (cid,))
            per_corpus[cid] = {(r[0], r[1]): float(r[2]) for r in cur.fetchall()}

    all_keys = set()
    for d in per_corpus.values():
        all_keys |= set(d.keys())

    universal = sorted(
        (t, f) for (t, f) in all_keys
        if all((t, f) in per_corpus[cid] for cid in corpus_ids)
    )

    print(f"\n  Universal predictors (all {len(corpus_ids)} corpora) — law candidates:")
    last_t = None
    for t, f in universal:
        if t != last_t:
            print(f"  [{t}]")
            last_t = t
        vals = "  ".join(
            f"{cid.split('_')[0]}:{per_corpus[cid].get((t, f), 0):+.3f}"
            for cid in corpus_ids
        )
        print(f"    {f:<44} {vals}")

    unique_counts = {}
    for cid in corpus_ids:
        uniq = [(t, f) for (t, f) in per_corpus[cid]
                if not all((t, f) in per_corpus[c] for c in corpus_ids)]
        unique_counts[cid] = len(uniq)
    print(
        f"\n  Corpus-specific predictors: "
        + " | ".join(f"{cid.split('_')[0]}:{n}" for cid, n in unique_counts.items())
    )


def cross_corpus_unsupervised(conn, corpus_ids, all_archetypes):
    """
    Cosine similarity between mean trajectory vectors across corpora.
    all_archetypes: dict {corpus_id: [archetype_dict, ...]}
    """
    print(f"\n── Cross-corpus archetype comparison ────────────────────────────")

    active = {cid: archs for cid, archs in all_archetypes.items() if archs}
    if len(active) < 2:
        print("  Fewer than 2 corpora with archetypes — no comparison.")
        return

    cids  = list(active.keys())
    col_w = 30

    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            ca, cb = cids[i], cids[j]
            print(f"\n  {ca} × {cb}:")
            print(f"  {'archetype_A':<{col_w}}  {'best_match_B':<{col_w}}  sim   flag")
            print("  " + "─" * (col_w * 2 + 22))
            for arch_a in active[ca]:
                best_sim, best_lbl = 0.0, None
                va = arch_a["center"]
                for arch_b in active[cb]:
                    vb  = arch_b["center"]
                    # Align by min dimensionality (corpora may differ in feature count)
                    n   = min(len(va), len(vb))
                    sim = float(
                        np.dot(va[:n], vb[:n]) /
                        (np.linalg.norm(va[:n]) * np.linalg.norm(vb[:n]) + 1e-9)
                    )
                    if sim > best_sim:
                        best_sim, best_lbl = sim, arch_b["label"]
                universal = best_sim >= ARCHETYPE_SIM_THRESHOLD
                unnamed   = universal and "unknown" in arch_a["label"] and \
                            "unknown" in (best_lbl or "")
                flag = "UNIVERSAL LAW" if universal else ""
                if unnamed:
                    flag = "UNNAMED BUT REAL"
                print(
                    f"  {arch_a['label']:<{col_w}}  {best_lbl or '?':<{col_w}}  "
                    f"{best_sim:.3f}  {flag}"
                )


# ── Both mode synthesis ────────────────────────────────────────────────────────

def run_synthesis(conn, corpus_id, archetypes):
    """Cross-reference Lasso lag features with archetype boundaries."""
    print(f"\n── Synthesis: {corpus_id} ────────────────────────────────────────")

    with conn.cursor() as cur:
        cur.execute("""
            SELECT feature_name, metric_value
            FROM ml_results
            WHERE corpus_id = %s
              AND model_name LIKE 'lasso_discovery_%%'
              AND metric_name = 'lasso_coefficient'
              AND notes IN ('lag', 'lag_t1', 'lag_t2', 'lag_t3')
            ORDER BY ABS(metric_value) DESC
            LIMIT 10
        """, (corpus_id,))
        top_lag_feats = [(r[0], float(r[1])) for r in cur.fetchall()]

    # Fallback: top features regardless of group
    if not top_lag_feats:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT feature_name, metric_value
                FROM ml_results
                WHERE corpus_id = %s
                  AND model_name LIKE 'lasso_discovery_%%'
                  AND metric_name = 'lasso_coefficient'
                ORDER BY ABS(metric_value) DESC
                LIMIT 10
            """, (corpus_id,))
            top_lag_feats = [(r[0], float(r[1])) for r in cur.fetchall()]

    unknown_archs = [a for a in archetypes if "unknown" in a["label"]]

    lines = [
        f"Synthesis: supervised Lasso ↔ unsupervised archetypes — {corpus_id}",
        "",
        "Top Lasso-discovered lag/feature predictors:",
    ]
    for feat, coef in top_lag_feats[:5]:
        lines.append(f"  {feat:<44} coef={coef:+.4f}")

    lines += ["", "Archetype alignment with lag features:"]
    for arch in archetypes:
        stats   = arch["stats"]
        fn      = arch["feat_names"]
        center  = arch["center"]
        # drift delta T0 vs T-1
        t0_i = next((i for i, f in enumerate(fn) if f == "drift_magnitude"), -1)
        t1_i = next((i for i, f in enumerate(fn) if f == "drift_t1"), -1)
        drift_str = ""
        if t0_i >= 0 and t1_i >= 0:
            delta = float(center[t0_i]) - float(center[t1_i])
            drift_str = f", drift Δ(T0−T1)={delta:+.3f}"
        lines.append(
            f"  {arch['label']}: death={stats['death_rate']:.1%}{drift_str}"
        )

    if unknown_archs:
        lines += ["", "Unknown archetypes — inspect for emerging laws:"]
        for arch in unknown_archs:
            lines.append(
                f"  {arch['label']}: n={arch['stats']['n']}, "
                f"death={arch['stats']['death_rate']:.1%}, "
                f"split={arch['stats']['split_rate']:.1%}"
            )

    body = "\n".join(lines)
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO sci_findings
              (type, title, body, corpus_ids, created_by, source, outcome)
            VALUES ('principle', %s, %s, %s, 'arc_discovery', 'arc_discovery.py', 'pending')
            RETURNING id
        """, (
            f"Discovery Layer Synthesis — {corpus_id}",
            body,
            [corpus_id],
        ))
        fid = cur.fetchone()[0]
    conn.commit()
    print(f"  Synthesis finding id={fid}")


# ── View-based metrics (v_recovery_rate, v_surprise_hierarchy, v_temporal_period_stats)

def run_view_metrics(conn, corpus_id: str):
    """Query three Category-A views and write per-corpus metrics to ml_results.

    Views wired:
      v_recovery_rate          → arc_recovery_rate metrics (boundary resilience)
      v_surprise_hierarchy     → arc_surprise_signal metrics (anomaly detection)
      v_temporal_period_stats  → arc_temporal_delta metrics (phase-transition features)

    Results stored in ml_results under the respective model_names so that
    downstream scripts (arc_narrative.py, arc_ml_score.py) can consume them
    without re-running the expensive view queries.
    """
    now = datetime.now(timezone.utc)
    ml_rows = []

    # ── v_recovery_rate: cluster boundary-pressure resilience ─────────────────
    # For each corpus, computes mean/median periods to recover from a boundary
    # pressure spike. Low recovery_rate_periods = resilient; high = fragile.
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                                        AS n_spikes,
                    AVG(recovery_rate_periods)                      AS mean_recovery,
                    PERCENTILE_CONT(0.5) WITHIN GROUP
                        (ORDER BY recovery_rate_periods)            AS median_recovery,
                    COUNT(*) FILTER (WHERE recovery_rate_periods IS NULL) AS n_no_recovery
                FROM v_recovery_rate
                WHERE corpus_id = %s
            """, (corpus_id,))
            r = cur.fetchone()
        if r and r["n_spikes"]:
            n = int(r["n_spikes"])
            for metric, val in [
                ("mean_recovery_periods",   r["mean_recovery"]),
                ("median_recovery_periods", r["median_recovery"]),
                ("pct_no_recovery",
                 float(r["n_no_recovery"]) / n if n else None),
            ]:
                if val is not None:
                    ml_rows.append((
                        "arc_recovery_rate", corpus_id, "cluster_resilience",
                        metric, float(val), None, n,
                        "From v_recovery_rate; boundary-pressure recovery periods",
                        now,
                    ))
    except Exception as e:
        print(f"  run_view_metrics: v_recovery_rate failed for {corpus_id}: {e}")

    # ── v_surprise_hierarchy: mean anomaly signal by level ───────────────────
    # chunk_surprise = perturbation_score; cluster_surprise = centroid-prediction
    # residual; corpus_surprise = field_surprise_index from period_stats.
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT
                    COUNT(DISTINCT chunk_id)              AS n_chunks,
                    AVG(chunk_surprise)                   AS mean_chunk_surprise,
                    AVG(cluster_surprise)                 AS mean_cluster_surprise,
                    MAX(cluster_surprise)                 AS max_cluster_surprise,
                    AVG(corpus_surprise)                  AS mean_corpus_surprise
                FROM v_surprise_hierarchy
                WHERE corpus_id = %s
                  AND chunk_surprise IS NOT NULL
            """, (corpus_id,))
            r = cur.fetchone()
        if r and r["n_chunks"]:
            n = int(r["n_chunks"])
            for metric, val in [
                ("mean_chunk_surprise",   r["mean_chunk_surprise"]),
                ("mean_cluster_surprise", r["mean_cluster_surprise"]),
                ("max_cluster_surprise",  r["max_cluster_surprise"]),
                ("mean_corpus_surprise",  r["mean_corpus_surprise"]),
            ]:
                if val is not None:
                    ml_rows.append((
                        "arc_surprise_signal", corpus_id, "anomaly_detection",
                        metric, float(val), None, n,
                        "From v_surprise_hierarchy; multi-level surprise index",
                        now,
                    ))
    except Exception as e:
        print(f"  run_view_metrics: v_surprise_hierarchy failed for {corpus_id}: {e}")

    # ── v_temporal_period_stats: period-over-period delta features ────────────
    # Provides delta_entropy, delta_alg_conn etc. that are the primary signals
    # for phase-transition detection. Stores mean abs-delta per column as a
    # measure of structural volatility over the corpus history.
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Fetch available delta columns (view may vary across DB versions)
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'v_temporal_period_stats'
                  AND table_schema = 'public'
                  AND column_name LIKE 'delta_%'
            """)
            delta_cols = [row["column_name"] for row in cur.fetchall()]

        if delta_cols:
            select_parts = ", ".join(
                f"AVG(ABS({col})) AS {col}" for col in delta_cols
            )
            count_part = "COUNT(*) AS n_periods"
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(f"""
                    SELECT {count_part}, {select_parts}
                    FROM v_temporal_period_stats
                    WHERE corpus_id = %s
                """, (corpus_id,))
                r = cur.fetchone()
            if r and r["n_periods"]:
                n = int(r["n_periods"])
                for col in delta_cols:
                    val = r[col]
                    if val is not None:
                        ml_rows.append((
                            "arc_temporal_delta", corpus_id, "structural_volatility",
                            f"mean_abs_{col}", float(val), col, n,
                            "From v_temporal_period_stats; period-over-period structural change",
                            now,
                        ))
    except Exception as e:
        print(f"  run_view_metrics: v_temporal_period_stats failed for {corpus_id}: {e}")

    if not ml_rows:
        print(f"  run_view_metrics: no rows to write for {corpus_id}")
        return

    with conn.cursor() as cur:
        cur.execute("""
            DELETE FROM ml_results
            WHERE model_name IN (
                'arc_recovery_rate', 'arc_surprise_signal', 'arc_temporal_delta'
            ) AND corpus_id = %s
        """, (corpus_id,))
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO ml_results
              (model_name, corpus_id, target, metric_name, metric_value,
               feature_name, n_samples, notes, recorded_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, ml_rows)
    conn.commit()
    print(f"  run_view_metrics: wrote {len(ml_rows)} rows to ml_results for {corpus_id}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ARC cross-primitive pattern discovery.")
    parser.add_argument("--corpus-id", required=True,
                        help="corpus_id, or 'all' to run all active corpora sequentially")
    parser.add_argument("--mode", default="both",
                        choices=["supervised", "unsupervised", "both"],
                        help="Discovery mode (default: both)")
    parser.add_argument("--targets", default="default",
                        choices=["default", "hidden", "strategic", "all"],
                        help="Target set: default (is_dead/is_split_child/drift_next), "
                             "hidden (dark_matter_next/alg_conn_next/spectral_gap_next), "
                             "strategic (size_pct_next/prosecution_heat_next/"
                             "persistence_next/is_merger_next/void_fill_next), "
                             "all (all sets combined). Default: default.")
    args = parser.parse_args()

    conn = get_conn()
    t0   = datetime.now()

    if args.corpus_id == "all":
        corpora = _get_active_corpora(conn)
        print(f"Running {args.mode} on {len(corpora)} active corpora: "
              f"{', '.join(corpora)}")
    else:
        corpora = [args.corpus_id]

    all_archetypes = {}  # corpus_id -> list[archetype]

    for corpus_id in corpora:
        try:
            archetypes = []

            if args.mode in ("supervised", "both"):
                run_supervised(conn, corpus_id, target_set=args.targets)

            if args.mode in ("unsupervised", "both"):
                archetypes = run_unsupervised(conn, corpus_id)
                all_archetypes[corpus_id] = archetypes

            if args.mode == "both" and archetypes:
                run_synthesis(conn, corpus_id, archetypes)

            # Wire view-based metrics into ml_results for every discovery run.
            # Runs regardless of mode since recovery/surprise/temporal signals
            # are independent of supervised/unsupervised ML mode.
            run_view_metrics(conn, corpus_id)

        except Exception as exc:
            import traceback
            print(f"  ERROR on {corpus_id}: {exc}")
            traceback.print_exc()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO log_bugs
                          (severity, status, discovered_in, description)
                        VALUES ('low', 'open', 'arc_discovery.py', %s)
                    """, (f"arc_discovery failed for {corpus_id}: {exc}",))
                conn.commit()
            except Exception:
                pass

    # Cross-corpus comparisons
    if args.mode in ("supervised", "both") and len(corpora) >= 2:
        try:
            cross_corpus_supervised(conn, corpora)
        except Exception as exc:
            print(f"  Cross-corpus supervised comparison failed: {exc}")

    if args.mode in ("unsupervised", "both"):
        if sum(1 for v in all_archetypes.values() if v) >= 2:
            try:
                cross_corpus_unsupervised(conn, corpora, all_archetypes)
            except Exception as exc:
                print(f"  Cross-corpus unsupervised comparison failed: {exc}")

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nDone. Total time: {elapsed:.1f}s")
    conn.close()


if __name__ == "__main__":
    main()

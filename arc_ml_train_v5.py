#!/usr/bin/env python3
"""
arc_ml_train_v5.py — XGBoost death prediction model for arc_v5

Models trained:
  A) Cross-corpus (G06N + H01L + C30B combined)
  B) G06N-specific
  C) H01L-specific

Outputs:
  ml/models/death_cross_corpus_v5.pkl
  ml/models/death_G06N_quarterly_v5.pkl
  ml/models/death_H01L_quarterly_v5.pkl
  ml/models/shap_cross_corpus_v5.png
  ml_death_scores table in arc_v5 (all clusters, Model A scores + SHAP)
"""

import os
import sys
import pickle
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

DB_PARAMS = dict(
    host=os.environ.get("PGHOST", "/var/run/postgresql"),
    dbname=os.environ.get("PGDATABASE", "arc_v5"),
    user=os.environ.get("PGUSER", "jeff"),
    password=os.environ.get("PGPASSWORD", ""),
)

MODEL_DIR = Path(__file__).parent / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEATH_FEATURES = [
    # Original 12 features
    'cohesion',
    'size',
    'elongation_ratio',
    'drift_magnitude',
    'boundary_pressure_rate',
    'persistence_score',
    'convergence_score',
    'mean_betweenness',
    'cohesion_percentile',
    'drift_percentile',
    'size_percentile',
    'jerk',
    # New features from arc_discovery_v5 — significant across corpora
    'age_periods',                   # Universal signal — cluster age at prediction time
    'n_attractors',                  # Topology — multi-attractor structure = unstable
    'mean_triangle_count',           # cuGraph — triangles predict death across all 3 corpora
    'volume_estimate',               # Geometry — expanding volume precedes death
    'velocity_alignment',            # Matching — direction consistency of drift
    'max_adjacent_void_persistence', # Gap signal — persistent adjacent void = isolation
]

COALESCE_DEFAULTS = {
    'elongation_ratio':              1.5,
    'drift_magnitude':               0.0,
    'boundary_pressure_rate':        0.0,
    'persistence_score':             0.0,
    'convergence_score':             0.0,
    'mean_betweenness':              0.0,
    'cohesion_percentile':           0.5,
    'drift_percentile':              0.5,
    'size_percentile':               0.5,
    'jerk':                          0.0,
    'age_periods':                   1,
    'n_attractors':                  0,
    'mean_triangle_count':           0.0,
    'volume_estimate':               0.0,
    'velocity_alignment':            0.0,
    'max_adjacent_void_persistence': 0.0,
}

XGB_BASE_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc',
)

MODEL_VERSION = 'cross_corpus_v5_18feat_v2'

# ── DB helpers ────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(**DB_PARAMS)


def load_training_data(conn) -> pd.DataFrame:
    """Load training data from cluster_snapshot, excluding:
    - is_new=true rows (first appearance cannot die in same period)
    - Last 2 periods per corpus (death labels unreliable — patent lag)
    """
    query = """
    SELECT
      corpus_id,
      persistent_cluster_id,
      period_start,
      is_dead::int AS target,
      COALESCE(cohesion, NULL)                          AS cohesion,
      COALESCE(size, NULL)                              AS size,
      COALESCE(elongation_ratio, 1.5)                   AS elongation_ratio,
      COALESCE(drift_magnitude, 0.0)                    AS drift_magnitude,
      COALESCE(boundary_pressure_rate, 0.0)             AS boundary_pressure_rate,
      COALESCE(persistence_score, 0.0)                  AS persistence_score,
      COALESCE(convergence_score, 0.0)                  AS convergence_score,
      COALESCE(mean_betweenness, 0.0)                   AS mean_betweenness,
      COALESCE(cohesion_percentile, 0.5)                AS cohesion_percentile,
      COALESCE(drift_percentile, 0.5)                   AS drift_percentile,
      COALESCE(size_percentile, 0.5)                    AS size_percentile,
      COALESCE(jerk, 0.0)                               AS jerk,
      COALESCE(age_periods, 1)::real                    AS age_periods,
      COALESCE(n_attractors, 0)::real                   AS n_attractors,
      COALESCE(mean_triangle_count, 0.0)::real          AS mean_triangle_count,
      COALESCE(volume_estimate, 0.0)                    AS volume_estimate,
      COALESCE(velocity_alignment, 0.0)                 AS velocity_alignment,
      COALESCE(max_adjacent_void_persistence, 0)::real  AS max_adjacent_void_persistence
    FROM cluster_snapshot
    WHERE cohesion IS NOT NULL
      AND size IS NOT NULL
      AND is_new = false
      AND period_start < (
        SELECT MAX(period_start) - INTERVAL '6 months'
        FROM cluster_snapshot cs2
        WHERE cs2.corpus_id = cluster_snapshot.corpus_id
      )
    ORDER BY corpus_id, period_start;
    """
    log.info("Loading training data...")
    df = pd.read_sql(query, conn)
    log.info(f"  Loaded {len(df):,} rows across {df['corpus_id'].nunique()} corpora")
    return df


def load_score_data(conn) -> pd.DataFrame:
    """Load ALL cluster_snapshot rows for scoring (no exclusions)."""
    query = """
    SELECT
      corpus_id,
      persistent_cluster_id,
      period_start,
      COALESCE(cohesion, NULL)                          AS cohesion,
      COALESCE(size, NULL)                              AS size,
      COALESCE(elongation_ratio, 1.5)                   AS elongation_ratio,
      COALESCE(drift_magnitude, 0.0)                    AS drift_magnitude,
      COALESCE(boundary_pressure_rate, 0.0)             AS boundary_pressure_rate,
      COALESCE(persistence_score, 0.0)                  AS persistence_score,
      COALESCE(convergence_score, 0.0)                  AS convergence_score,
      COALESCE(mean_betweenness, 0.0)                   AS mean_betweenness,
      COALESCE(cohesion_percentile, 0.5)                AS cohesion_percentile,
      COALESCE(drift_percentile, 0.5)                   AS drift_percentile,
      COALESCE(size_percentile, 0.5)                    AS size_percentile,
      COALESCE(jerk, 0.0)                               AS jerk,
      COALESCE(age_periods, 1)::real                    AS age_periods,
      COALESCE(n_attractors, 0)::real                   AS n_attractors,
      COALESCE(mean_triangle_count, 0.0)::real          AS mean_triangle_count,
      COALESCE(volume_estimate, 0.0)                    AS volume_estimate,
      COALESCE(velocity_alignment, 0.0)                 AS velocity_alignment,
      COALESCE(max_adjacent_void_persistence, 0)::real  AS max_adjacent_void_persistence
    FROM cluster_snapshot
    WHERE cohesion IS NOT NULL
      AND size IS NOT NULL
    ORDER BY corpus_id, period_start;
    """
    log.info("Loading all rows for scoring...")
    df = pd.read_sql(query, conn)
    log.info(f"  Loaded {len(df):,} rows for scoring")
    return df


# ── Training helpers ──────────────────────────────────────────────────────────

def compute_scale_pos_weight(y: pd.Series) -> float:
    n_alive = (y == 0).sum()
    n_dead  = (y == 1).sum()
    if n_dead == 0:
        return 1.0
    return n_alive / n_dead


def train_model(X: pd.DataFrame, y: pd.Series,
                label: str, n_splits: int = 5):
    """Train XGBoost with cross-validation. Returns (model, cv_results)."""
    spw = compute_scale_pos_weight(y)
    log.info(f"  [{label}] n={len(y):,}  dead={y.sum():,}  alive={(y==0).sum():,}  "
             f"scale_pos_weight={spw:.2f}")

    params = dict(**XGB_BASE_PARAMS, scale_pos_weight=spw)
    model = xgb.XGBClassifier(**params)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    log.info(f"  [{label}] CV AUC: {aucs.mean():.4f} ± {aucs.std():.4f}  "
             f"(folds: {[f'{a:.3f}' for a in aucs]})")

    # Fit on full dataset for deployment
    model.fit(X, y)

    # Final metrics on full training set (optimistic — CV is the honest estimate)
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    train_auc = roc_auc_score(y, y_pred_proba)
    train_f1  = f1_score(y, y_pred, zero_division=0)
    cm        = confusion_matrix(y, y_pred)

    log.info(f"  [{label}] Train AUC={train_auc:.4f}  F1={train_f1:.4f}")
    log.info(f"  [{label}] Confusion matrix:\n{cm}")

    return model, {
        'cv_auc_mean': aucs.mean(),
        'cv_auc_std':  aucs.std(),
        'cv_aucs':     aucs.tolist(),
        'train_auc':   train_auc,
        'train_f1':    train_f1,
        'confusion':   cm,
        'n_total':     len(y),
        'n_dead':      int(y.sum()),
        'n_alive':     int((y == 0).sum()),
    }


def save_model(model, path: Path, results: dict, label: str):
    payload = {'model': model, 'features': DEATH_FEATURES,
               'results': results, 'label': label,
               'model_version': MODEL_VERSION}
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    log.info(f"  Saved → {path}")


# ── SHAP ──────────────────────────────────────────────────────────────────────

def run_shap(model, X: pd.DataFrame, out_path: Path):
    """Compute SHAP values and save bar chart."""
    log.info("Running SHAP TreeExplainer...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=DEATH_FEATURES
    ).sort_values(ascending=False)

    log.info("  Top 5 features by mean |SHAP|:")
    for feat, val in mean_abs.head(5).items():
        log.info(f"    {feat:<30s} {val:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    mean_abs.sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('Feature importance — Cross-corpus death model v5')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"  SHAP chart → {out_path}")

    return shap_values, mean_abs


# ── Scoring ───────────────────────────────────────────────────────────────────

DDL_SCORES = """
CREATE TABLE IF NOT EXISTS ml_death_scores (
  corpus_id               text NOT NULL,
  persistent_cluster_id   text NOT NULL,
  period_start            date NOT NULL,
  death_probability       real,
  model_version           text,
  scored_at               timestamptz DEFAULT NOW(),
  shap_cohesion           real,
  shap_size               real,
  shap_elongation         real,
  shap_drift              real,
  shap_boundary_pressure  real,
  shap_persistence        real,
  shap_convergence        real,
  shap_betweenness        real,
  shap_cohesion_pct       real,
  shap_drift_pct          real,
  shap_size_pct           real,
  shap_jerk               real,
  PRIMARY KEY (corpus_id, persistent_cluster_id, period_start, model_version)
);
"""

# SHAP feature → column name mapping (order matches DEATH_FEATURES)
SHAP_COLS = [
    'shap_cohesion',
    'shap_size',
    'shap_elongation',
    'shap_drift',
    'shap_boundary_pressure',
    'shap_persistence',
    'shap_convergence',
    'shap_betweenness',
    'shap_cohesion_pct',
    'shap_drift_pct',
    'shap_size_pct',
    'shap_jerk',
    # New v5 features
    'shap_age_periods',
    'shap_n_attractors',
    'shap_triangle_count',
    'shap_volume_estimate',
    'shap_velocity_alignment',
    'shap_void_persistence',
]

ALTER_SHAP_COLS = """
ALTER TABLE ml_death_scores
  ADD COLUMN IF NOT EXISTS shap_age_periods       real,
  ADD COLUMN IF NOT EXISTS shap_n_attractors      real,
  ADD COLUMN IF NOT EXISTS shap_triangle_count    real,
  ADD COLUMN IF NOT EXISTS shap_volume_estimate   real,
  ADD COLUMN IF NOT EXISTS shap_velocity_alignment real,
  ADD COLUMN IF NOT EXISTS shap_void_persistence  real;
"""


def write_scores(conn, score_df: pd.DataFrame, model, shap_values: np.ndarray):
    """Write death_probability + per-row SHAP values to ml_death_scores."""
    cur = conn.cursor()

    # Create table + add any new SHAP columns
    cur.execute(DDL_SCORES)
    cur.execute(ALTER_SHAP_COLS)
    conn.commit()

    # Delete existing scores for this model version
    cur.execute(
        "DELETE FROM ml_death_scores WHERE model_version = %s",
        (MODEL_VERSION,)
    )
    deleted = cur.rowcount
    if deleted:
        log.info(f"  Deleted {deleted:,} existing rows for {MODEL_VERSION}")
    conn.commit()

    X = score_df[DEATH_FEATURES]
    proba = model.predict_proba(X)[:, 1]

    rows = []
    for i, (_, row) in enumerate(score_df.iterrows()):
        sv = shap_values[i]  # length 12
        rows.append((
            row['corpus_id'],
            row['persistent_cluster_id'],
            row['period_start'],
            float(proba[i]),
            MODEL_VERSION,
            *[float(sv[j]) for j in range(len(SHAP_COLS))],
        ))

    insert_sql = f"""
    INSERT INTO ml_death_scores (
      corpus_id, persistent_cluster_id, period_start,
      death_probability, model_version,
      {', '.join(SHAP_COLS)}
    ) VALUES ({', '.join(['%s'] * (5 + len(SHAP_COLS)))})
    ON CONFLICT (corpus_id, persistent_cluster_id, period_start, model_version)
    DO UPDATE SET
      death_probability = EXCLUDED.death_probability,
      scored_at = NOW(),
      {', '.join(f'{c} = EXCLUDED.{c}' for c in SHAP_COLS)};
    """

    BATCH = 2000
    written = 0
    for i in range(0, len(rows), BATCH):
        cur.executemany(insert_sql, rows[i:i+BATCH])
        conn.commit()
        written += len(rows[i:i+BATCH])

    cur.close()
    log.info(f"  Wrote {written:,} score rows → ml_death_scores")


def verify_scores(conn):
    """Print verification summary."""
    query = """
    SELECT corpus_id,
      COUNT(*) as scored,
      ROUND(AVG(death_probability)::numeric, 3) as avg_dp,
      ROUND(MIN(death_probability)::numeric, 3) as min_dp,
      ROUND(MAX(death_probability)::numeric, 3) as max_dp,
      COUNT(*) FILTER (WHERE death_probability > 0.7) as high_risk
    FROM ml_death_scores
    WHERE model_version = %s
    GROUP BY corpus_id;
    """
    df = pd.read_sql(query, conn, params=(MODEL_VERSION,))
    log.info("\nScore verification:")
    log.info(df.to_string(index=False))
    return df


def high_risk_clusters(conn, threshold: float = 0.8):
    """Fetch high-risk clusters with labels for the latest period."""
    query = """
    SELECT s.corpus_id, s.persistent_cluster_id, s.period_start,
      ROUND(s.death_probability::numeric, 3) AS dp,
      cl.cluster_label
    FROM ml_death_scores s
    LEFT JOIN cluster_labels cl
      ON cl.corpus_id = s.corpus_id
      AND cl.persistent_cluster_id = s.persistent_cluster_id
    WHERE s.model_version = %s
      AND s.death_probability > %s
      AND s.period_start = (
        SELECT MAX(period_start) FROM ml_death_scores s2
        WHERE s2.corpus_id = s.corpus_id
          AND s2.model_version = s.model_version
      )
    ORDER BY s.corpus_id, s.death_probability DESC;
    """
    df = pd.read_sql(query, conn, params=(MODEL_VERSION, threshold))
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    conn = get_conn()

    # ── Load data ──────────────────────────────────────────────────────────────
    df = load_training_data(conn)

    # Per-corpus stats
    for cid, grp in df.groupby('corpus_id'):
        n_dead  = grp['target'].sum()
        n_alive = len(grp) - n_dead
        log.info(f"  {cid}: {len(grp):,} rows  dead={n_dead:,}  alive={n_alive:,}")

    X_all = df[DEATH_FEATURES]
    y_all = df['target']

    # ── Model A: Cross-corpus ──────────────────────────────────────────────────
    log.info("\n=== Model A: Cross-corpus ===")
    model_a, results_a = train_model(X_all, y_all, label='cross_corpus')
    save_model(model_a, MODEL_DIR / 'death_cross_corpus_v5_18feat_v2.pkl', results_a, 'cross_corpus')

    # ── Model B: G06N-specific ────────────────────────────────────────────────
    log.info("\n=== Model B: G06N_quarterly ===")
    mask_g06n = df['corpus_id'] == 'G06N_quarterly'
    if mask_g06n.sum() >= 50:
        model_b, results_b = train_model(
            X_all[mask_g06n], y_all[mask_g06n], label='G06N_quarterly')
        save_model(model_b, MODEL_DIR / 'death_G06N_quarterly_v5_18feat.pkl',
                   results_b, 'G06N_quarterly')
    else:
        log.warning("G06N data too small — skipping")
        model_b, results_b = None, {}

    # ── Model C: H01L-specific ────────────────────────────────────────────────
    log.info("\n=== Model C: H01L_quarterly ===")
    mask_h01l = df['corpus_id'] == 'H01L_quarterly'
    if mask_h01l.sum() >= 50:
        model_c, results_c = train_model(
            X_all[mask_h01l], y_all[mask_h01l], label='H01L_quarterly')
        save_model(model_c, MODEL_DIR / 'death_H01L_quarterly_v5_18feat.pkl',
                   results_c, 'H01L_quarterly')
    else:
        log.warning("H01L data too small — skipping")
        model_c, results_c = None, {}

    # ── SHAP (Model A) ────────────────────────────────────────────────────────
    log.info("\n=== SHAP analysis (Model A) ===")
    shap_values_train, shap_importance = run_shap(
        model_a, X_all, MODEL_DIR / 'shap_cross_corpus_v5.png')

    # ── Score all rows ────────────────────────────────────────────────────────
    log.info("\n=== Scoring all cluster_snapshot rows ===")
    score_df = load_score_data(conn)
    X_score = score_df[DEATH_FEATURES]

    log.info("Computing SHAP values for scoring set...")
    explainer   = shap.TreeExplainer(model_a)
    shap_scores = explainer.shap_values(X_score)

    write_scores(conn, score_df, model_a, shap_scores)
    verify_df = verify_scores(conn)

    # ── High-risk clusters ────────────────────────────────────────────────────
    log.info("\n=== High-risk clusters (DP > 0.8) at latest period ===")
    hr = high_risk_clusters(conn, threshold=0.8)
    if len(hr):
        log.info(hr.to_string(index=False))
    else:
        log.info("  None above 0.8 threshold")

    # ── Summary report ────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"\nTraining data:")
    for cid, grp in df.groupby('corpus_id'):
        log.info(f"  {cid}: {len(grp):,} rows  "
                 f"dead={grp['target'].sum():,}  alive={(grp['target']==0).sum():,}")

    log.info(f"\nModel A (cross-corpus): CV AUC = "
             f"{results_a['cv_auc_mean']:.4f} ± {results_a['cv_auc_std']:.4f}")
    log.info("  12-feat baseline: 0.8529  |  arc_v4 baseline: 0.868")

    if results_b:
        log.info(f"Model B (G06N): CV AUC = "
                 f"{results_b['cv_auc_mean']:.4f} ± {results_b['cv_auc_std']:.4f}")
        log.info("  12-feat baseline: 0.8176")
    if results_c:
        log.info(f"Model C (H01L): CV AUC = "
                 f"{results_c['cv_auc_mean']:.4f} ± {results_c['cv_auc_std']:.4f}")
        log.info("  12-feat baseline: 0.8796")

    log.info(f"\nTop 8 SHAP features (18-feat model):")
    for feat, val in shap_importance.head(8).items():
        is_new = feat not in {'cohesion', 'size', 'elongation_ratio', 'drift_magnitude',
                              'boundary_pressure_rate', 'persistence_score', 'convergence_score',
                              'mean_betweenness', 'cohesion_percentile', 'drift_percentile',
                              'size_percentile', 'jerk'}
        tag = '  [NEW v5]' if is_new else ''
        log.info(f"  {feat:<35s} {val:.4f}{tag}")

    log.info(f"\nScored: {len(score_df):,} rows → ml_death_scores "
             f"(model_version='{MODEL_VERSION}')")

    # ── Old vs new score comparison ───────────────────────────────────────────
    log.info("\n=== Score delta: 18-feat vs 12-feat ===")
    compare_sql = """
    SELECT
      new.corpus_id,
      ROUND(AVG(new.death_probability)::numeric, 3)                             AS avg_dp_new,
      ROUND(AVG(old.death_probability)::numeric, 3)                             AS avg_dp_old,
      ROUND(AVG(new.death_probability - old.death_probability)::numeric, 4)     AS avg_change,
      COUNT(*) FILTER (
        WHERE ABS(new.death_probability - old.death_probability) > 0.1
      )                                                                          AS large_changes
    FROM ml_death_scores new
    JOIN ml_death_scores old
      ON  old.corpus_id             = new.corpus_id
      AND old.persistent_cluster_id = new.persistent_cluster_id
      AND old.period_start          = new.period_start
    WHERE new.model_version = 'cross_corpus_v5_18feat'
      AND old.model_version = 'cross_corpus_v5_12feat'
    GROUP BY new.corpus_id;
    """
    compare_df = pd.read_sql(compare_sql, conn)
    if len(compare_df):
        log.info(compare_df.to_string(index=False))
    else:
        log.info("  No 12-feat scores found for comparison (run old model first?)")

    # ── Top 15 high-risk clusters at penultimate period ───────────────────────
    log.info("\n=== Top 15 highest-risk clusters (18-feat model) ===")
    top_risk_sql = """
    SELECT
      mds.corpus_id,
      mds.persistent_cluster_id,
      cl.cluster_label,
      ROUND(mds.death_probability::numeric, 3) AS dp,
      cs.size,
      cs.age_periods,
      mds.period_start
    FROM ml_death_scores mds
    JOIN cluster_snapshot cs
      ON  cs.corpus_id             = mds.corpus_id
      AND cs.persistent_cluster_id = mds.persistent_cluster_id
      AND cs.period_start          = mds.period_start
    LEFT JOIN cluster_labels cl
      ON  cl.corpus_id             = mds.corpus_id
      AND cl.persistent_cluster_id = mds.persistent_cluster_id
    WHERE mds.model_version = 'cross_corpus_v5_18feat'
      AND cs.period_start = (
        SELECT MAX(period_start) - INTERVAL '6 months'
        FROM cluster_snapshot cs2
        WHERE cs2.corpus_id = mds.corpus_id
      )
    ORDER BY mds.death_probability DESC
    LIMIT 15;
    """
    top_df = pd.read_sql(top_risk_sql, conn)
    if len(top_df):
        log.info(top_df.to_string(index=False))
    else:
        log.info("  No scored rows found at target period")

    conn.close()
    log.info("Done.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
arc_survival.py — Multi-horizon cluster survival analysis using Cox Proportional Hazards.

Fits a Cox PH model on cluster lifetime data to produce:
  - Concordance index (model quality)
  - Hazard ratios per entry feature (what structural properties kill clusters)
  - Per-cluster survival probabilities at T+1, T+2, T+3 periods ahead

Writes estimates to pipe_cluster_survival table.
Compares top-10 most-at-risk against existing XGBoost death_probability.

Usage:
  PGHOST=/var/run/postgresql PGDATABASE=arc_v4 PGUSER=jeff \\
    python3 ml/arc_survival.py [--corpus G06N_quarterly]
"""
import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

try:
    from lifelines import CoxPHFitter
except ImportError:
    raise SystemExit("lifelines not installed — run: pip install lifelines")


FEATURE_COLS = [
    'size_at_entry',    # size_percentile at cluster birth
    'drift_at_entry',   # drift_magnitude at cluster birth
    'conv_at_entry',    # convergence_score at cluster birth
    'betw_at_entry',    # mean_betweenness at cluster birth
    'persist_at_entry', # persistence_score at cluster birth
    'heat_at_entry',    # prosecution_heat_index at cluster birth
]


def get_conn():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "/var/run/postgresql"),
        dbname=os.environ.get("PGDATABASE", "arc_v4"),
        user=os.environ.get("PGUSER", "jeff"),
    )


def _load_survival_data(conn, corpus_id):
    """
    Build per-cluster survival dataset.
    One row per persistent_cluster_id:
      duration_periods = how many quarters the cluster existed
      observed_death   = 1 if it eventually died, 0 if still active (censored)
      *_at_entry       = feature values at the cluster's first appearance
    """
    sql = """
        WITH raw AS (
            SELECT
                c.persistent_cluster_id,
                c.period_start,
                c.is_dead,
                c.size_percentile,
                c.drift_magnitude,
                c.convergence_score,
                c.mean_betweenness,
                c.persistence_score,
                c.prosecution_heat_index,
                ROW_NUMBER() OVER (
                    PARTITION BY c.persistent_cluster_id
                    ORDER BY c.period_start
                ) AS rn
            FROM pipe_clusters c
            WHERE c.corpus_id = %(cid)s
              AND c.is_junk   = FALSE
              AND c.persistent_cluster_id IS NOT NULL
        ),
        entry AS (
            SELECT
                persistent_cluster_id,
                size_percentile        AS size_at_entry,
                drift_magnitude        AS drift_at_entry,
                convergence_score      AS conv_at_entry,
                mean_betweenness       AS betw_at_entry,
                persistence_score      AS persist_at_entry,
                prosecution_heat_index AS heat_at_entry
            FROM raw WHERE rn = 1
        ),
        summary AS (
            SELECT
                persistent_cluster_id,
                COUNT(*)          AS duration_periods,
                MAX(is_dead::int) AS observed_death
            FROM raw
            GROUP BY persistent_cluster_id
        )
        SELECT
            s.persistent_cluster_id,
            s.duration_periods,
            s.observed_death,
            e.size_at_entry,
            e.drift_at_entry,
            e.conv_at_entry,
            e.betw_at_entry,
            e.persist_at_entry,
            e.heat_at_entry
        FROM summary s
        JOIN entry e USING (persistent_cluster_id)
        ORDER BY s.persistent_cluster_id
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"cid": corpus_id})
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=cols)
    df["duration_periods"] = pd.to_numeric(df["duration_periods"]).clip(lower=1)
    df["observed_death"]   = pd.to_numeric(df["observed_death"])
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        med = df[col].median()
        df[col] = df[col].fillna(0.0 if pd.isna(med) else med)
    return df


def _load_active_clusters(conn, corpus_id, df_all):
    """
    Return the most-recent row for each currently active cluster,
    with entry features merged in for Cox prediction.
    """
    sql = """
        SELECT DISTINCT ON (c.persistent_cluster_id)
            c.persistent_cluster_id,
            c.cluster_label,
            c.death_probability AS xgb_death_prob
        FROM pipe_clusters c
        WHERE c.corpus_id = %(cid)s
          AND c.is_dead   = FALSE
          AND c.is_junk   = FALSE
          AND c.persistent_cluster_id IS NOT NULL
        ORDER BY c.persistent_cluster_id, c.period_start DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"cid": corpus_id})
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()

    df_active = pd.DataFrame(rows, columns=cols)
    # Merge entry features from full survival dataset (birth-period features)
    entry = df_all[["persistent_cluster_id"] + FEATURE_COLS].copy()
    df_active = df_active.merge(entry, on="persistent_cluster_id", how="left")
    for col in FEATURE_COLS:
        med = df_all[col].median()
        df_active[col] = df_active[col].fillna(0.0 if pd.isna(med) else med)
    return df_active


def run_survival(corpus_id: str):
    conn = get_conn()
    print(f"\n── Survival analysis: {corpus_id} ──────────────────────────")

    df = _load_survival_data(conn, corpus_id)
    if df.empty or df['duration_periods'].isna().all():
        print("  No deaths observed — skipping survival analysis")
        return
    n_deaths   = int(df["observed_death"].sum())
    n_censored = int((df["observed_death"] == 0).sum())
    print(f"  {len(df):,} clusters: {n_deaths} deaths observed, {n_censored} censored")
    print(f"  Duration range: {int(df['duration_periods'].min())}–"
          f"{int(df['duration_periods'].max())} periods")

    # ── Fit Cox PH model ────────────────────────────────────────────────────
    # Drop near-zero-variance covariates before fitting (lifelines can't handle them).
    # Threshold raised to 1e-2: betw_at_entry is ~0 on sparse corpora (std ~1e-5)
    # but still passes 1e-4, producing degenerate HR≈2e+15 (bug #67).
    active_features = [
        col for col in FEATURE_COLS
        if df[col].std() > 1e-2
    ]
    if len(active_features) < 1:
        print("  ERROR: no non-constant features — cannot fit Cox PH model")
        conn.close()
        return
    if len(active_features) < len(FEATURE_COLS):
        dropped = set(FEATURE_COLS) - set(active_features)
        print(f"  Dropped near-zero-variance columns: {dropped}")

    fit_cols = ["duration_periods", "observed_death"] + active_features
    # High penalizer handles near-perfect separation; step_size prevents NaN deltas
    cph = CoxPHFitter(penalizer=1.0)
    cph.fit(df[fit_cols], duration_col="duration_periods", event_col="observed_death",
            fit_options={"step_size": 0.1})

    print("\nCox PH Model Summary:")
    cph.print_summary()
    concordance = float(cph.concordance_index_)
    print(f"\n  Concordance index: {concordance:.4f}")
    if concordance > 0.7:
        print("  → Good discriminative ability (>0.7 = better than random)")
    elif concordance > 0.6:
        print("  → Moderate discriminative ability")
    else:
        print("  → Weak discriminative ability — entry features alone may not predict timing")

    # ── Predict for active clusters ─────────────────────────────────────────
    df_active = _load_active_clusters(conn, corpus_id, df)
    n_active  = len(df_active)
    print(f"\n  {n_active} active clusters — predicting T+1/T+2/T+3 survival ...")

    horizons = [1, 2, 3]
    if n_active == 0:
        print("  No active clusters found — skipping survival prediction.")
        conn.commit()
        return
    surv_fn = cph.predict_survival_function(df_active[active_features], times=horizons)
    # surv_fn: DataFrame with rows=times, cols=cluster index

    now = datetime.now(timezone.utc)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM pipe_cluster_survival WHERE corpus_id = %s",
                    (corpus_id,))

    rows_out = []
    for i, pid in enumerate(df_active["persistent_cluster_id"]):
        for h_idx, h in enumerate(horizons):
            sp = float(surv_fn.iloc[h_idx, i])
            rows_out.append((corpus_id, pid, now, h, sp, 1.0 - sp))

    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO pipe_cluster_survival
              (corpus_id, persistent_cluster_id, estimated_at, horizon_periods,
               survival_probability, death_probability)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, rows_out)
    conn.commit()
    print(f"  Wrote {len(rows_out)} survival estimates ({n_active} clusters × 3 horizons).")

    # ── Top 10 most at-risk ──────────────────────────────────────────────────
    print("\n  Top 10 clusters most likely to die within 3 periods:")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT pid, label, die_t1, die_t2, die_t3, xgb_death_prob
            FROM (
                SELECT DISTINCT ON (c.persistent_cluster_id)
                    c.persistent_cluster_id  AS pid,
                    c.cluster_label          AS label,
                    se1.death_probability    AS die_t1,
                    se2.death_probability    AS die_t2,
                    se3.death_probability    AS die_t3,
                    c.death_probability      AS xgb_death_prob
                FROM pipe_clusters c
                JOIN pipe_cluster_survival se1
                  ON se1.corpus_id              = c.corpus_id
                 AND se1.persistent_cluster_id  = c.persistent_cluster_id
                 AND se1.horizon_periods        = 1
                JOIN pipe_cluster_survival se2
                  ON se2.corpus_id              = c.corpus_id
                 AND se2.persistent_cluster_id  = c.persistent_cluster_id
                 AND se2.horizon_periods        = 2
                JOIN pipe_cluster_survival se3
                  ON se3.corpus_id              = c.corpus_id
                 AND se3.persistent_cluster_id  = c.persistent_cluster_id
                 AND se3.horizon_periods        = 3
                WHERE c.corpus_id = %s
                  AND c.is_dead   = FALSE
                  AND c.is_junk   = FALSE
                ORDER BY c.persistent_cluster_id, c.period_start DESC
            ) sub
            ORDER BY die_t3 DESC
            LIMIT 10
        """, (corpus_id,))
        top10 = cur.fetchall()

    print(f"  {'pid':<15} {'label':<45} {'T+1':>6} {'T+2':>6} {'T+3':>6} {'xgb':>6}")
    print("  " + "─" * 90)
    for pid, label, d1, d2, d3, xgb in top10:
        label_s = (label or "?")[:44]
        xgb_s   = f"{xgb:.3f}" if xgb is not None else "  N/A"
        print(f"  {pid:<15} {label_s:<45} {d1:.3f}  {d2:.3f}  {d3:.3f}  {xgb_s}")

    # ── Compare with XGBoost top-10 ──────────────────────────────────────────
    cox_top10_pids = {r[0] for r in top10}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT persistent_cluster_id FROM pipe_clusters
            WHERE corpus_id = %s AND is_dead = FALSE AND is_junk = FALSE
              AND death_probability IS NOT NULL
            ORDER BY death_probability DESC LIMIT 10
        """, (corpus_id,))
        xgb_top10_pids = {r[0] for r in cur.fetchall()}

    overlap = cox_top10_pids & xgb_top10_pids
    print(f"\n  XGBoost top-10 ∩ Cox top-10: {len(overlap)}/10")
    if len(overlap) >= 7:
        print("  → High agreement: same clusters flagged — "
              "calibration differs but signal is consistent.")
    elif len(overlap) >= 4:
        print("  → Moderate agreement: Cox capturing longitudinal signal "
              "beyond cross-sectional XGBoost.")
    else:
        print("  → Low agreement: Cox survival model capturing "
              "fundamentally different risk patterns.")

    # ── Register as pipeline step ────────────────────────────────────────────
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO sys_corpus_pipeline_steps (corpus_id, step_order, script, status, notes)
            SELECT %s, 11, 'ml/arc_survival.py', 'complete',
                   'Cox PH survival analysis — multi-horizon death probability'
            WHERE NOT EXISTS (
                SELECT 1 FROM sys_corpus_pipeline_steps
                WHERE corpus_id = %s AND script = 'ml/arc_survival.py'
            )
        """, (corpus_id, corpus_id))
    conn.commit()

    # ── File finding ─────────────────────────────────────────────────────────
    hazard_lines = "\n".join(
        f"  {feat:<30} HR={hr:.3f}"
        for feat, hr in cph.summary["exp(coef)"].sort_values(ascending=False).items()
    )
    overlap_str = (
        "High agreement — same clusters, calibration may differ."
        if len(overlap) >= 7
        else "Models diverge — Cox captures longitudinal signal XGBoost misses."
    )
    body = (
        f"Corpus: {corpus_id}\n"
        f"Model: Cox Proportional Hazards (penalizer=1.0)\n"
        f"Concordance index: {concordance:.4f}\n"
        f"N clusters: {len(df)}, deaths: {n_deaths}, censored: {n_censored}\n\n"
        f"Hazard ratios (HR>1 increases death risk):\n{hazard_lines}\n\n"
        f"XGBoost vs Cox top-10 overlap: {len(overlap)}/10 — {overlap_str}"
    )
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO sci_findings
              (type, title, body, corpus_ids, created_by, source, outcome)
            VALUES ('finding', %s, %s, %s, 'arc_survival', 'arc_survival.py', 'pending')
            RETURNING id
        """, (
            f"Multi-Horizon Survival Estimates — {corpus_id}",
            body,
            [corpus_id],
        ))
        fid = cur.fetchone()[0]
    conn.commit()
    print(f"\n  Filed finding id={fid}")
    conn.close()


def main():
    ap = argparse.ArgumentParser(
        description="Cox PH survival analysis for cluster lifetimes.")
    ap.add_argument("--corpus-id", default="G06N_quarterly",
                    help="corpus_id to analyse (default: G06N_quarterly)")
    args = ap.parse_args()
    run_survival(args.corpus_id)


if __name__ == "__main__":
    main()

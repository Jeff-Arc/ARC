#!/usr/bin/env python3
"""
arc_cross_corpus_score.py — Score patent corpora using the cross-corpus death model.

Loads death_patent_cross_corpus.pkl and updates f_cluster_period.death_probability
for all specified patent corpora.

Usage:
    PGHOST=/var/run/postgresql PGDATABASE=arc_v4 PGUSER=jeff \
    python3 ml/arc_cross_corpus_score.py --model-name arc_cluster_death_patent_cross_corpus
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras

MODELS_DIR = Path(__file__).parent / "models"

DEATH_FEATURES = [
    "cohesion", "size", "elongation_ratio", "drift_magnitude",
    "boundary_pressure_rate", "persistence_score", "convergence_score",
    "mean_betweenness", "marginal_entropy_impact",
    "cohesion_percentile", "drift_percentile", "size_percentile", "jerk",
]

PATENT_CORPORA = [
    "G06N_quarterly", "C23C_quarterly", "H01L_quarterly", "G06F_quarterly",
    "G01B_quarterly", "G01N_quarterly", "G02B_quarterly", "C30B_quarterly",
    "G01C_quarterly", "G01S_quarterly", "G02F_quarterly", "G03C_quarterly",
    "G03F_quarterly", "G04B_quarterly", "G05B_quarterly", "G05D_quarterly",
    "G06K_quarterly", "G06T_quarterly", "G07C_quarterly", "G09G_quarterly",
    "H01L_21_quarterly", "H01L_22_quarterly", "H01L_23_quarterly",
    "H01L_24_quarterly", "H01L_25_quarterly",
]

FEATURE_SQL = """
    cohesion,
    size::real,
    COALESCE(elongation_ratio,       1.5)  AS elongation_ratio,
    COALESCE(drift_magnitude,        0.0)  AS drift_magnitude,
    COALESCE(boundary_pressure_rate, 0.0)  AS boundary_pressure_rate,
    COALESCE(persistence_score,      0.0)  AS persistence_score,
    COALESCE(convergence_score,      0.0)  AS convergence_score,
    COALESCE(mean_betweenness,       0.0)  AS mean_betweenness,
    COALESCE(marginal_entropy_impact,0.0)  AS marginal_entropy_impact,
    COALESCE(cohesion_percentile,    0.5)  AS cohesion_percentile,
    COALESCE(drift_percentile,       0.5)  AS drift_percentile,
    COALESCE(size_percentile,        0.5)  AS size_percentile,
    COALESCE(jerk,                   0.0)  AS jerk
"""


def get_conn():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "/var/run/postgresql"),
        dbname=os.environ.get("PGDATABASE", "arc_v4"),
        user=os.environ.get("PGUSER", "jeff"),
    )


def score_corpus(conn, corpus_id, model, model_name):
    """Load features from f_cluster_period, predict, update death_probability."""
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT
                corpus_id, period_start, cluster_id,
                {FEATURE_SQL}
            FROM f_cluster_period
            WHERE corpus_id = %s
              AND cohesion IS NOT NULL
              AND is_junk IS NOT TRUE
            ORDER BY period_start, cluster_id
        """, (corpus_id,))
        rows = cur.fetchall()

    if not rows:
        print(f"  {corpus_id}: no scoreable rows — skip")
        return 0

    meta = [(r[0], r[1], r[2]) for r in rows]
    X = np.array([r[3:] for r in rows], dtype=np.float32)

    probs = model.predict_proba(X)[:, 1]

    update_rows = [(float(prob), model_name, corp, ps, cid)
                   for (corp, ps, cid), prob in zip(meta, probs)]

    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, """
            UPDATE f_cluster_period
            SET death_probability   = %s,
                event_model_version = %s
            WHERE corpus_id    = %s
              AND period_start = %s
              AND cluster_id   = %s
        """, update_rows)
        n_updated = cur.rowcount
    conn.commit()

    print(f"  {corpus_id}: {len(rows):,} scored, {n_updated:,} rows updated")
    return n_updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="arc_cluster_death_patent_cross_corpus",
                        help="Model name tag (stored in event_model_version)")
    parser.add_argument("--corpus", default=None,
                        help="Score a single corpus (default: all patent corpora)")
    args = parser.parse_args()

    # Derive pkl filename from model name: arc_cluster_death_X → death_X
    # Convention: model_name = arc_cluster_death_<corpus_id>
    #             pkl file   = death_<corpus_id>.pkl
    corpus_part = args.model_name.replace("arc_cluster_death_", "")
    model_path = MODELS_DIR / f"death_{corpus_part}.pkl"

    if not model_path.exists():
        sys.exit(f"Model file not found: {model_path}")

    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    print(f"Loaded model: {model_path}")
    print(f"Model name tag: {args.model_name}\n")

    corpora = [args.corpus] if args.corpus else PATENT_CORPORA

    conn = get_conn()
    total_updated = 0
    skipped = []
    for corpus_id in corpora:
        n = score_corpus(conn, corpus_id, model, args.model_name)
        total_updated += n
        if n == 0:
            skipped.append(corpus_id)

    conn.close()

    print(f"\nTotal rows updated: {total_updated:,}")
    if skipped:
        print(f"Skipped (no data): {skipped}")


if __name__ == "__main__":
    main()

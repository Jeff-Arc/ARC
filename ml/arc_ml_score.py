#!/usr/bin/env python3
"""
arc_ml_score.py — Pipeline step 7: ML scoring, SHAP attribution, subclass signal ranking.

Steps:
  A. Call stored procedures (score_cluster_death_prob, score_cluster_birth_prob,
     score_period_anomaly) for every period in each active corpus.
  B. Retrain Python XGBoost on 13-feature death data (G06N_quarterly), run
     SHAP TreeExplainer, write per-cluster rows to cluster_shap_values for
     G06N_quarterly + H01L_quarterly + H01L subclass corpora.
  C. Aggregate mean |SHAP| by (corpus_id, feature_name), write rankings to
     ml_results; print cross-corpus comparison table.

Usage:
    PGHOST=/var/run/postgresql PGDATABASE=arc_v4 PGUSER=jeff \\
    python3 ml/arc_ml_score.py [--corpus CORPUS_ID] [--step A|B|C|all]

Options:
    --corpus   Restrict Step A to a single corpus (default: all 15 active corpora)
    --step     Which steps to run: A, B, C, or all (default: all)
"""

import argparse
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras

MODELS_DIR = Path(__file__).parent / "models"

try:
    from xgboost import XGBClassifier
    import shap
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\nInstall with: pip install xgboost shap")


# ── Constants ────────────────────────────────────────────────────────────────

# 13 features matching score_cluster_death_prob stored procedure (authoritative).
# arc_ml_train.py now uses the same 13 features (mean_density removed, bug #48 fixed).
DEATH_FEATURES = [
    "cohesion", "size", "elongation_ratio", "drift_magnitude",
    "boundary_pressure_rate", "persistence_score", "convergence_score",
    "mean_betweenness", "marginal_entropy_impact",
    "cohesion_percentile", "drift_percentile", "size_percentile", "jerk",
]

# SHAP is computed for these 7 corpora only (G06N root + all H01L).
# SHAP_CORPORA is populated dynamically at the start of step_b() by querying
# v_run_lookup WHERE status = 'active'. Adding a new corpus to sys_run_config
# automatically includes it in SHAP scoring — no code change required.

# Model is trained on G06N_quarterly (matches what pgml arc_cluster_death_prob uses).
SHAP_TRAIN_CORPUS = "G06N_quarterly"

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)

SHAP_BATCH = 2000  # rows per batch for cluster_shap_values INSERT


# ── SHAP training corpus selection ───────────────────────────────────────────

def _select_train_corpus(conn, corpus_id):
    """Return the best XGBoost training corpus for SHAP background for a given corpus.

    Rules (in priority order):
      1. corpus has a parent_corpus_id   → use parent (richest, broadest feature distribution)
      2. corpus IS a parent (has children) → use itself
      3. standalone corpus (no parent, no children) → use itself
      4. fallback if self has no training data → SHAP_TRAIN_CORPUS (G06N_quarterly)

    This ensures SHAP attributions are computed relative to the correct feature
    distribution — H01L sub-corpora use H01L_quarterly, longevity sub-corpora use
    longevity_quarterly, G06N micro/multi-res corpora use G06N_quarterly, etc.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                (SELECT s2.legacy_name FROM sys_run_config s2 WHERE s2.run_id = c.parent_run_id) AS parent_corpus_id,
                EXISTS(
                    SELECT 1 FROM sys_run_config ch
                    WHERE ch.parent_run_id = c.run_id
                      AND ch.status = 'active'
                ) AS has_children
            FROM sys_run_config c
            WHERE c.legacy_name = %s
        """, (corpus_id,))
        row = cur.fetchone()

    if row is None:
        return SHAP_TRAIN_CORPUS  # corpus not in corpora table — safe fallback
    parent_corpus_id, _has_children = row
    if parent_corpus_id:
        return parent_corpus_id  # rule 1: child corpus → train on parent
    return corpus_id             # rules 2 & 3: root or standalone → train on self


# ── DB ───────────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "/var/run/postgresql"),
        dbname=os.environ.get("PGDATABASE", "arc_v4"),
        user=os.environ.get("PGUSER", "jeff"),
    )


# ── Step A: pgml scoring via stored procedures ───────────────────────────────

def _ensure_score_history_constraint(conn):
    """Add unique constraint on ml_score_history if not already present."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS ml_score_history_corpus_pcid_period_uidx
            ON ml_score_history (corpus_id, persistent_cluster_id, period_start)
        """)
    conn.commit()


def _write_score_history(conn, corpus_id):
    """INSERT scored clusters into ml_score_history; ON CONFLICT DO NOTHING."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ml_score_history
                (corpus_id, persistent_cluster_id, period_start,
                 death_probability, phase_transition_score)
            SELECT
                c.corpus_id,
                c.persistent_cluster_id,
                c.period_start,
                c.death_probability,
                ps.phase_transition_score
            FROM pipe_clusters c
            LEFT JOIN pipe_period_stats ps
                ON  ps.corpus_id   = c.corpus_id
                AND ps.period_start = c.period_start
            WHERE c.corpus_id              = %s
              AND c.persistent_cluster_id IS NOT NULL
              AND c.death_probability     IS NOT NULL
            ON CONFLICT (corpus_id, persistent_cluster_id, period_start)
                DO NOTHING
        """, (corpus_id,))
        n = cur.rowcount
    conn.commit()
    print(f"  Wrote {n} score history rows for {corpus_id}")


def step_a(conn, corpus_filter=None):
    """Call score_cluster_death_prob, score_cluster_birth_prob, score_period_anomaly
    for every period in every active corpus (or just corpus_filter if given).
    """
    _ensure_score_history_constraint(conn)
    print("\n── Step A: pgml scoring ─────────────────────────────────────────")
    with conn.cursor() as cur:
        if corpus_filter:
            cur.execute(
                "SELECT corpus_id FROM v_run_lookup WHERE status='active' AND corpus_id = %s",
                (corpus_filter,),
            )
        else:
            cur.execute("SELECT corpus_id FROM v_run_lookup WHERE status='active' ORDER BY corpus_id")
        corpora = [row[0] for row in cur.fetchall()]

    print(f"  Corpora to score: {len(corpora)}")

    total_periods = 0
    for corpus_id in corpora:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT DISTINCT period FROM pipe_clusters
                   WHERE corpus_id = %s AND period IS NOT NULL
                   ORDER BY period""",
                (corpus_id,),
            )
            periods = [row[0] for row in cur.fetchall()]

        if not periods:
            print(f"  {corpus_id}: no periods found — skip")
            continue

        print(f"  {corpus_id}: {len(periods)} periods ...", end="", flush=True)
        t0 = datetime.now()
        with conn.cursor() as cur:
            for period in periods:
                cur.execute(
                    "CALL score_cluster_death_prob(%s, %s)", (period, corpus_id)
                )
                cur.execute(
                    "CALL score_cluster_birth_prob(%s, %s)", (period, corpus_id)
                )
                cur.execute(
                    "CALL score_period_anomaly(%s, %s)", (period, corpus_id)
                )
        conn.commit()
        elapsed = (datetime.now() - t0).total_seconds()
        total_periods += len(periods)
        print(f" done ({elapsed:.1f}s)")
        _write_score_history(conn, corpus_id)

    print(f"  Total periods scored: {total_periods}")


# ── Step B: Python XGBoost + SHAP ────────────────────────────────────────────

_FEATURE_SQL = """
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


def _load_training_data(conn, corpus_id):
    """Return (X, y) arrays using the 13 features from score_cluster_death_prob."""
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT
                {_FEATURE_SQL},
                CASE WHEN is_dead THEN 1 ELSE 0 END AS will_die
            FROM pipe_clusters
            WHERE corpus_id = %s
              AND cohesion IS NOT NULL
              AND is_junk  = FALSE
        """, (corpus_id,))
        rows = cur.fetchall()

    if not rows:
        raise ValueError(f"No training data in clusters for {corpus_id}")

    arr = np.array(rows, dtype=np.float32)
    X = arr[:, :-1]
    y = arr[:, -1].astype(int)
    return X, y


def _load_score_data(conn, corpus_id):
    """Return (meta, X) for all non-junk clusters in corpus_id.
    meta: list of (corpus_id, period_start, cluster_id)
    """
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT
                corpus_id, period_start, cluster_id,
                {_FEATURE_SQL}
            FROM pipe_clusters
            WHERE corpus_id = %s
              AND cohesion IS NOT NULL
              AND is_junk  = FALSE
            ORDER BY period_start, cluster_id
        """, (corpus_id,))
        rows = cur.fetchall()

    if not rows:
        return [], np.empty((0, len(DEATH_FEATURES)), dtype=np.float32)

    meta = [(r[0], r[1], r[2]) for r in rows]
    X = np.array([r[3:] for r in rows], dtype=np.float32)
    return meta, X


def _write_shap_batch(conn, records):
    """Upsert a batch of (corpus_id, period_start, cluster_id, feature_name, shap_value) rows."""
    now = datetime.now(timezone.utc)
    rows = [(r[0], r[1], r[2], r[3], float(r[4]), now) for r in records]
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO pipe_cluster_shap_values
                (corpus_id, period_start, cluster_id, feature_name, shap_value, scored_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (corpus_id, period_start, cluster_id, feature_name)
                DO UPDATE SET shap_value = EXCLUDED.shap_value,
                              scored_at  = EXCLUDED.scored_at
        """, rows)
    conn.commit()


def step_b(conn, shap_corpora: list):
    """Retrain Python XGBoost per corpus family, run TreeExplainer, write
    cluster_shap_values for each active corpus.

    Each corpus is scored using a model trained on its own data (if root) or its
    parent corpus's data (if child). See _select_train_corpus() for the selection
    rules. This avoids cross-corpus SHAP attribution artifacts from using a single
    G06N-trained model for all corpora (bug #76).

    Corpora that need SHAP re-run after this fix (previously scored with the
    G06N_quarterly model but now use a different training corpus):
      - H01L_quarterly                  → now trains on itself
      - H01L_21/22/23/24/25_quarterly   → now trains on H01L_quarterly
      - openalex_cs_sample              → now trains on itself (if standalone)
      - openalex_ee_sample              → now trains on itself (if standalone)
      - longevity_quarterly             → now trains on itself
      - longevity_cardio/cellular/genetic/neuro/patents_quarterly
                                        → now trains on longevity_quarterly
      G06N multi-res and micro corpora (parent=G06N_quarterly) are unchanged.
    """
    print("\n── Step B: SHAP attribution ─────────────────────────────────────")
    print(f"  Active corpora for SHAP: {shap_corpora}")

    # Group corpora by training corpus to train one model per family.
    train_to_score: dict = {}
    for corpus_id in shap_corpora:
        train_corpus = _select_train_corpus(conn, corpus_id)
        train_to_score.setdefault(train_corpus, []).append(corpus_id)

    print("  Training corpus groups:")
    for train_corpus, score_list in sorted(train_to_score.items()):
        print(f"    {train_corpus} → scores: {score_list}")

    total_rows = 0
    for train_corpus, score_list in train_to_score.items():
        # 1. Train model on the appropriate training corpus.
        print(f"\n  Loading training data from {train_corpus}...")
        try:
            X_train, y_train = _load_training_data(conn, train_corpus)
            train_corpus_used = train_corpus
        except ValueError as e:
            print(f"  WARNING: {e} — falling back to {SHAP_TRAIN_CORPUS}")
            try:
                X_train, y_train = _load_training_data(conn, SHAP_TRAIN_CORPUS)
                train_corpus_used = SHAP_TRAIN_CORPUS
            except ValueError:
                print(f"  ERROR: fallback also failed — skipping group {score_list}")
                continue

        # 2. Load model from disk if saved by arc_ml_train, else retrain + save.
        model_path = MODELS_DIR / f"death_{train_corpus_used}.pkl"
        if model_path.exists():
            with open(model_path, "rb") as fh:
                model = pickle.load(fh)
            print(f"  Loaded cached model from {model_path}")
        else:
            pos_weight = float((y_train == 0).sum()) / max((y_train == 1).sum(), 1)
            print(f"  {len(X_train):,} rows, {y_train.sum()} positives "
                  f"({y_train.mean():.1%} death rate), pos_weight={pos_weight:.2f}")
            model = XGBClassifier(scale_pos_weight=pos_weight, **XGB_PARAMS)
            model.fit(X_train, y_train)
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            with open(model_path, "wb") as fh:
                pickle.dump(model, fh)
            print(f"  Trained + saved model → {model_path}")

        # 3. TreeExplainer for this family's model.
        explainer = shap.TreeExplainer(model)
        print(f"  Model on {train_corpus_used}. TreeExplainer ready.")

        # 3. Score each corpus in this family.
        for corpus_id in score_list:
            print(f"  {corpus_id}: loading clusters...", end="", flush=True)
            meta, X_score = _load_score_data(conn, corpus_id)
            if not meta:
                print(" no data — skip")
                continue
            print(f" {len(meta):,} clusters...", end="", flush=True)

            # Compute SHAP values (shape: n_clusters × n_features).
            # XGBClassifier binary returns (n, f) log-odds SHAP for class=1;
            # older shap versions return a list — take index [1].
            shap_vals = explainer.shap_values(X_score)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            batch = []
            for i, (corpus, period_start, cluster_id) in enumerate(meta):
                for j, feat in enumerate(DEATH_FEATURES):
                    batch.append((corpus, period_start, cluster_id, feat, shap_vals[i, j]))
                    if len(batch) >= SHAP_BATCH:
                        _write_shap_batch(conn, batch)
                        batch = []
            if batch:
                _write_shap_batch(conn, batch)

            rows_written = len(meta) * len(DEATH_FEATURES)
            total_rows += rows_written
            print(f" {rows_written:,} rows written")

    print(f"  Total cluster_shap_values rows: {total_rows:,}")


# ── Step C: subclass signal ranking ──────────────────────────────────────────

def step_c(conn, shap_corpora: list):
    """Aggregate mean |SHAP| by (corpus_id, feature_name), write to ml_results,
    print per-parent comparison table.
    """
    print("\n── Step C: subclass signal ranking ──────────────────────────────")
    SHAP_CORPORA = shap_corpora
    # Build training corpus map for notes (mirrors _select_train_corpus logic).
    train_corpus_map = {c: _select_train_corpus(conn, c) for c in SHAP_CORPORA}

    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute("""
            SELECT
                csv.corpus_id,
                (SELECT s2.legacy_name FROM sys_run_config s2 WHERE s2.run_id = c.parent_run_id) AS parent_corpus_id,
                csv.feature_name,
                AVG(ABS(csv.shap_value))  AS mean_abs_shap,
                STDDEV(csv.shap_value)    AS std_shap,
                COUNT(DISTINCT (csv.period_start, csv.cluster_id)) AS n_clusters
            FROM pipe_cluster_shap_values csv
            JOIN sys_run_config c ON c.legacy_name = csv.corpus_id
            WHERE csv.corpus_id = ANY(%s)
            GROUP BY csv.corpus_id, (SELECT s2.legacy_name FROM sys_run_config s2 WHERE s2.run_id = c.parent_run_id), csv.feature_name
            ORDER BY csv.corpus_id, mean_abs_shap DESC
        """, (SHAP_CORPORA,))
        rows = cur.fetchall()

    if not rows:
        print("  No cluster_shap_values rows found — skipping.")
        return

    # ── write to ml_results ──────────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    ml_rows = []
    for r in rows:
        n = int(r["n_clusters"])
        train_src = train_corpus_map.get(r["corpus_id"], SHAP_TRAIN_CORPUS)
        ml_rows.append((
            "arc_shap_signal_ranking",
            r["corpus_id"],
            "death_probability",
            "mean_abs_shap",
            float(r["mean_abs_shap"]),
            r["feature_name"],
            n,
            f"SHAP from {train_src}-trained XGBoost; {len(DEATH_FEATURES)}-feature death model",
            now,
        ))
        if r["std_shap"] is not None:
            ml_rows.append((
                "arc_shap_signal_ranking",
                r["corpus_id"],
                "death_probability",
                "std_shap",
                float(r["std_shap"]),
                r["feature_name"],
                n,
                None,
                now,
            ))

    # Delete prior run rows for these corpora before re-inserting
    with conn.cursor() as cur:
        cur.execute("""
            DELETE FROM ml_results
            WHERE model_name = 'arc_shap_signal_ranking'
              AND corpus_id  = ANY(%s)
        """, (SHAP_CORPORA,))
        deleted = cur.rowcount
        if deleted:
            print(f"  Replaced {deleted} prior arc_shap_signal_ranking rows.")

        psycopg2.extras.execute_batch(cur, """
            INSERT INTO ml_results
              (model_name, corpus_id, target, metric_name,
               metric_value, feature_name, n_samples, notes, recorded_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, ml_rows)
    conn.commit()
    print(f"  Wrote {len(ml_rows)} rows to ml_results.")

    # ── print comparison tables ───────────────────────────────────────────────
    # Build ranking per corpus
    rankings = {}
    for r in rows:
        cid = r["corpus_id"]
        if cid not in rankings:
            rankings[cid] = []
        rankings[cid].append((r["feature_name"], float(r["mean_abs_shap"])))

    # Group by parent
    with conn.cursor() as cur:
        cur.execute(
            "SELECT legacy_name AS name, (SELECT s2.legacy_name FROM sys_run_config s2 WHERE s2.run_id = c.parent_run_id) AS parent_corpus_id FROM sys_run_config c WHERE legacy_name = ANY(%s)",
            (SHAP_CORPORA,),
        )
        parent_map = {row[0]: row[1] for row in cur.fetchall()}

    parents = sorted({v for v in parent_map.values() if v})
    roots   = [c for c in SHAP_CORPORA if parent_map.get(c) is None]

    for root in roots:
        children = [c for c in SHAP_CORPORA if parent_map.get(c) == root]
        group = [root] + children
        group = [c for c in group if c in rankings]
        if not group:
            continue

        print(f"\n  Parent: {root}")
        # Header
        col_w = 30
        hdr = f"  {'feature':<{col_w}}"
        for c in group:
            short = c.replace("_quarterly", "").replace(root.replace("_quarterly","") + "_", "")
            hdr += f"  {short:>10}"
        print(hdr)
        print("  " + "─" * (col_w + len(group) * 12))

        # All features, ordered by root corpus mean_abs_shap
        feats = [f for f, _ in rankings.get(root, [])]
        # Add any missing features from children
        for c in children:
            for f, _ in rankings.get(c, []):
                if f not in feats:
                    feats.append(f)

        feat_scores = {
            c: {f: v for f, v in rankings.get(c, [])}
            for c in group
        }
        for feat in feats:
            row_str = f"  {feat:<{col_w}}"
            for c in group:
                v = feat_scores[c].get(feat, 0.0)
                row_str += f"  {v:>10.4f}"
            print(row_str)

    print()


# ── Registration ─────────────────────────────────────────────────────────────

def register_pipeline_step(conn):
    """Register arc_ml_score.py as step_order=10 for all active corpora.

    Status is 'complete' only if step 9 (ml_train) is also complete for that corpus.
    If step 9 is pending/missing, step 10 is registered as 'pending'.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT legacy_name AS name FROM sys_run_config WHERE status='active' ORDER BY legacy_name")
        corpora = [row[0] for row in cur.fetchall()]

    registered = 0
    with conn.cursor() as cur:
        for corpus_id in corpora:
            # Skip if already registered
            cur.execute("""
                SELECT id FROM sys_corpus_pipeline_steps
                WHERE corpus_id = %s AND step_order = 10
            """, (corpus_id,))
            if cur.fetchone():
                continue
            # Determine status based on step 9 completion
            cur.execute("""
                SELECT status FROM sys_corpus_pipeline_steps
                WHERE corpus_id = %s AND step_order = 9
            """, (corpus_id,))
            row = cur.fetchone()
            step9_complete = row is not None and row[0] == 'complete'
            status = 'complete' if step9_complete else 'pending'
            notes = ('Step A: pgml scoring; Step B: SHAP; Step C: subclass signal ranking'
                     if step9_complete else
                     'arc_ml_score.py ran (pgml scoring + SHAP + ranking); pending full '
                     'status until steps 8-9 complete')
            cur.execute("""
                INSERT INTO sys_corpus_pipeline_steps
                    (corpus_id, step_order, script, status, notes)
                VALUES (%s, 10, 'arc_ml_score.py', %s, %s)
            """, (corpus_id, status, notes))
            registered += 1
    conn.commit()
    if registered:
        print(f"  Registered step 10 for {registered} corpora.")
    else:
        print("  Step 10 already registered for all corpora.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ARC ML scoring, SHAP, signal ranking.")
    parser.add_argument("--corpus-id", default=None,
                        help="Restrict Step A to one corpus (default: all active)")
    parser.add_argument("--step", default="all",
                        choices=["A", "B", "C", "all"],
                        help="Steps to run (default: all)")
    parser.add_argument("--register", action="store_true",
                        help="Register pipeline step 7 for all active corpora and exit")
    args = parser.parse_args()

    conn = get_conn()

    if args.register:
        register_pipeline_step(conn)
        conn.close()
        return

    t_start = datetime.now()

    # Fetch active corpora once; passed to step_b and step_c so both operate
    # on the same list without step_c suffering a NameError when run standalone.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT corpus_id FROM v_run_lookup WHERE status = 'active' ORDER BY corpus_id"
        )
        shap_corpora = [row[0] for row in cur.fetchall()]

    # If a specific corpus was requested, include it in SHAP even if its
    # v_run_lookup status is not 'active' (e.g. 'pending' on first run).
    if args.corpus_id and args.corpus_id not in shap_corpora:
        shap_corpora.append(args.corpus_id)

    if args.step in ("A", "all"):
        step_a(conn, corpus_filter=args.corpus_id)

    if args.step in ("B", "all"):
        step_b(conn, shap_corpora)

    if args.step in ("C", "all"):
        step_c(conn, shap_corpora)

    # Self-registration removed: migration 142 owns step numbering.
    # Use --register flag only for manual backfill if needed.

    elapsed = (datetime.now() - t_start).total_seconds()
    print(f"\nDone. Total time: {elapsed:.1f}s")
    conn.close()


if __name__ == "__main__":
    main()

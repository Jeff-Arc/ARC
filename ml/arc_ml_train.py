#!/usr/bin/env python3
"""
arc_ml_train.py — Train cluster death and phase-transition ML models
for a specified corpus using XGBoost.

Reads features from v_cluster_event_training (death) and
v_period_stats_ml_features (phase_transition). Writes metrics and
feature importances to the ml_results table.  Results are corpus-
scoped: G06N thresholds / models are NOT imported or referenced.

Usage:
    PGHOST=/var/run/postgresql PGDATABASE=arc_v4 PGUSER=jeff \\
    python3 arc_ml_train.py --corpus-id H01L_quarterly

Options:
    --corpus-id       corpus_id to train on  (required)
    --model           death | phase_transition | all  (default: all)
    --death-only      shorthand for --model death
    --training-view   override the death training view
                      (default: v_cluster_event_training)
    --test-size       fraction held out for eval  (default: 0.25)
    --seed            random seed  (default: 42)
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

# ── optional imports (fail loudly if missing) ────────────────────────────────
try:
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                                  r2_score, mean_squared_error)
    from xgboost import XGBClassifier, XGBRegressor
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n"
             f"Install with: pip install scikit-learn xgboost")

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ── DB ───────────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "/var/run/postgresql"),
        dbname=os.environ.get("PGDATABASE", "arc_v4"),
        user=os.environ.get("PGUSER", "jeff"),
    )

# ── Death model ──────────────────────────────────────────────────────────────

DEATH_FEATURES = [
    "cohesion", "size", "elongation_ratio", "drift_magnitude",
    "boundary_pressure_rate", "persistence_score", "convergence_score",
    "mean_betweenness", "marginal_entropy_impact",
    "cohesion_percentile", "drift_percentile", "size_percentile", "jerk",
]  # 13 features — matches score_cluster_death_prob stored proc and arc_ml_score.py (bug #48 fix)


DEFAULT_DEATH_TRAINING_VIEW = "v_cluster_event_training"


def load_death_data(conn, corpus_id: str,
                    training_view: str = DEFAULT_DEATH_TRAINING_VIEW,
                    ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    with conn.cursor() as cur:
        feature_cols = ", ".join(DEATH_FEATURES)
        cur.execute(f"""
            SELECT period_start, {feature_cols}, will_die
            FROM {training_view}
            WHERE corpus_id = %s
              AND cohesion IS NOT NULL
        """, (corpus_id,))
        rows = cur.fetchall()

    if not rows:
        raise ValueError(f"No training data in {training_view} for {corpus_id}")

    periods = np.array([r[0] for r in rows])                  # date objects, shape (n,)
    arr = np.array([r[1:] for r in rows], dtype=np.float32)   # features + will_die
    X = arr[:, :-1]
    y = arr[:, -1].astype(int)
    return X, y, DEATH_FEATURES, periods


def train_death_model(X, y, test_size: float, seed: int,
                      periods: np.ndarray | None = None) -> dict:
    if periods is not None:
        X_tr, X_te, y_tr, y_te, _, periods_te = train_test_split(
            X, y, periods, test_size=test_size, stratify=y, random_state=seed
        )
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        periods_te = None

    pos_weight = float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=seed,
        verbosity=0,
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    auc_global = float(roc_auc_score(y_te, y_prob))

    # Per-period AUC: compute separately for each period_start in the test set.
    # Skips periods where only one class is present (roc_auc undefined).
    per_period_aucs = []
    if periods_te is not None:
        for period in np.unique(periods_te):
            mask = (periods_te == period)
            yp, pp = y_te[mask], y_prob[mask]
            if yp.sum() > 0 and (1 - yp).sum() > 0:
                per_period_aucs.append(float(roc_auc_score(yp, pp)))

    metrics = {
        "auc_global":             auc_global,
        "auc_within_period_mean": float(np.mean(per_period_aucs)) if per_period_aucs else None,
        "auc_within_period_std":  float(np.std(per_period_aucs))  if per_period_aucs else None,
        "f1":       float(f1_score(y_te, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "n_train":  int(len(y_tr)),
        "n_test":   int(len(y_te)),
        "n_pos":    int(y.sum()),
        "n_total":  int(len(y)),
        "pos_rate": float(y.mean()),
        "n_periods_evaluated": len(per_period_aucs),
    }
    importance = dict(zip(DEATH_FEATURES,
                          model.feature_importances_.tolist()))
    return {"metrics": metrics, "importance": importance, "model": model}


# ── Phase transition model ───────────────────────────────────────────────────

PT_FEATURES = [
    "n_clusters", "system_entropy", "algebraic_connectivity",
    "void_count", "leiden_modularity", "topology_reorganization",
    "birth_rate", "death_rate", "n_precursor_clusters",
    "corpus_centroid_drift", "field_surprise_index", "path_length",
]


def load_pt_data(conn, corpus_id: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with conn.cursor() as cur:
        feature_cols = ", ".join(PT_FEATURES)
        cur.execute(f"""
            SELECT {feature_cols}, phase_transition_score
            FROM v_period_stats_ml_features
            WHERE corpus_id = %s
              AND phase_transition_score IS NOT NULL
        """, (corpus_id,))
        rows = cur.fetchall()

    if not rows:
        raise ValueError(f"No PT training data in v_period_stats_ml_features for {corpus_id}")

    arr = np.array(rows, dtype=np.float32)
    X = arr[:, :-1]
    y = arr[:, -1]
    return X, y, PT_FEATURES


def train_pt_model(X, y, test_size: float, seed: int) -> dict:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="rmse",
        random_state=seed,
        verbosity=0,
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    r2  = float(r2_score(y_te, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))

    # 5-fold CV R² on full data (small dataset — CV is more reliable than single split)
    cv_scores = cross_val_score(
        XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=seed,
                     verbosity=0),
        X, y, cv=min(5, len(y) // 5), scoring="r2"
    )

    metrics = {
        "r2":      r2,
        "rmse":    rmse,
        "cv_r2":   float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "n_train": int(len(y_tr)),
        "n_test":  int(len(y_te)),
        "n_total": int(len(y)),
    }
    importance = dict(zip(PT_FEATURES,
                          model.feature_importances_.tolist()))
    return {"metrics": metrics, "importance": importance, "model": model}


# ── DB write ─────────────────────────────────────────────────────────────────

def write_results(conn, corpus_id: str, model_name: str, target: str,
                  metrics: dict, importance: dict, n_samples: int,
                  notes: str | None = None):
    now = datetime.now(timezone.utc)
    rows = []

    # Scalar metrics
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            rows.append((model_name, corpus_id, target, metric_name,
                         float(metric_value), None, n_samples, notes, now))

    # Feature importances
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        rows.append((model_name, corpus_id, target, "feature_importance",
                     float(imp), feat, n_samples, notes, now))

    with conn.cursor() as cur:
        # C-15: DELETE before INSERT to prevent duplicate rows on re-run.
        # The ml_results_unique_metric index (migration 144) enforces uniqueness
        # on (corpus_id, model_name, target, metric_name, COALESCE(feature_name,''));
        # clearing old rows first avoids ON CONFLICT silently dropping new data.
        cur.execute(
            "DELETE FROM ml_results WHERE corpus_id = %s AND model_name = %s AND target = %s",
            (corpus_id, model_name, target),
        )
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO ml_results
              (model_name, corpus_id, target, metric_name,
               metric_value, feature_name, n_samples, notes, recorded_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, rows)
    conn.commit()
    print(f"  Wrote {len(rows)} rows to ml_results for {model_name}.")


# ── Pretty print ─────────────────────────────────────────────────────────────

def print_death_results(corpus_id: str, result: dict):
    m = result["metrics"]
    imp = result["importance"]
    print(f"\n{'─'*60}")
    print(f"  arc_cluster_death — {corpus_id}")
    print(f"{'─'*60}")
    auc_wp  = m.get("auc_within_period_mean")
    auc_std = m.get("auc_within_period_std")
    n_per   = m.get("n_periods_evaluated", 0)
    wp_str  = (f"{auc_wp:.4f} ± {auc_std:.4f}  (n={n_per} periods)"
               if auc_wp is not None else "N/A")
    print(f"  N total:   {m['n_total']:>6}  (pos={m['n_pos']}, rate={m['pos_rate']:.1%})")
    print(f"  Train/Test:{m['n_train']:>6} / {m['n_test']}")
    print(f"  F1:        {m['f1']:.4f}")
    print(f"  ROC-AUC (global, inflated):  {m['auc_global']:.4f}")
    print(f"  ROC-AUC (within-period):     {wp_str}")
    print(f"  Accuracy:  {m['accuracy']:.4f}")
    print(f"\n  Feature importances (top 8):")
    for feat, v in sorted(imp.items(), key=lambda x: -x[1])[:8]:
        bar = "█" * int(v * 40)
        print(f"    {feat:<30}  {v:.4f}  {bar}")


def print_pt_results(corpus_id: str, result: dict):
    m = result["metrics"]
    imp = result["importance"]
    print(f"\n{'─'*60}")
    print(f"  arc_phase_transition — {corpus_id}")
    print(f"{'─'*60}")
    print(f"  N total:   {m['n_total']:>6}")
    print(f"  Train/Test:{m['n_train']:>6} / {m['n_test']}")
    print(f"  R²:        {m['r2']:.4f}")
    print(f"  RMSE:      {m['rmse']:.4f}")
    print(f"  CV R²:     {m['cv_r2']:.4f} ± {m['cv_r2_std']:.4f}")
    print(f"\n  Feature importances (top 8):")
    for feat, v in sorted(imp.items(), key=lambda x: -x[1])[:8]:
        bar = "█" * int(v * 40)
        print(f"    {feat:<30}  {v:.4f}  {bar}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ARC ML models for a corpus.")
    parser.add_argument("--corpus-id", required=True, help="corpus_id to train on")
    parser.add_argument("--model", default="all",
                        choices=["death", "phase_transition", "all"],
                        help="Which model to train (default: all)")
    parser.add_argument("--death-only", action="store_true",
                        help="Shorthand for --model death")
    parser.add_argument("--training-view", default=DEFAULT_DEATH_TRAINING_VIEW,
                        help="Override the death training view "
                             f"(default: {DEFAULT_DEATH_TRAINING_VIEW})")
    parser.add_argument("--test-size", type=float, default=0.25,
                        help="Test fraction (default: 0.25)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-models", action="store_true",
                        help="Save trained model objects to ml/models/ as .pkl files")
    args = parser.parse_args()

    if args.death_only:
        args.model = "death"

    corpus_id = args.corpus_id
    print(f"Training ML models for corpus: {corpus_id}")
    print(f"Training view: {args.training_view}")
    print(f"Test size: {args.test_size:.0%}  |  Seed: {args.seed}\n")

    conn = get_conn()

    # ── Death model ──────────────────────────────────────────────────────────
    if args.model in ("death", "all"):
        print("Loading death training data...")
        try:
            X, y, features, periods = load_death_data(conn, corpus_id,
                                                       training_view=args.training_view)
            print(f"  {len(X):,} rows, {y.sum()} positives ({y.mean():.1%} death rate)")
            result = train_death_model(X, y, args.test_size, args.seed, periods=periods)
            print_death_results(corpus_id, result)
            write_results(
                conn, corpus_id,
                model_name=f"arc_cluster_death_{corpus_id}",
                target="will_die",
                metrics=result["metrics"],
                importance=result["importance"],
                n_samples=len(X),
                notes=f"XGBoost, seed={args.seed}, test_size={args.test_size}",
            )
            if args.save_models:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                path = MODELS_DIR / f"death_{corpus_id}.pkl"
                with open(path, "wb") as fh:
                    pickle.dump(result["model"], fh)
                print(f"  Saved death model → {path}")
        except ValueError as e:
            print(f"  SKIP death model: {e}")

    # ── Phase transition model ───────────────────────────────────────────────
    if args.model in ("phase_transition", "all"):
        print("\nLoading phase transition training data...")
        try:
            X, y, features = load_pt_data(conn, corpus_id)
            print(f"  {len(X):,} rows")
            result = train_pt_model(X, y, args.test_size, args.seed)
            print_pt_results(corpus_id, result)
            write_results(
                conn, corpus_id,
                model_name=f"arc_phase_transition_{corpus_id}",
                target="phase_transition_score",
                metrics=result["metrics"],
                importance=result["importance"],
                n_samples=len(X),
                notes=f"XGBoost regression, seed={args.seed}, test_size={args.test_size}",
            )
            if args.save_models:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                path = MODELS_DIR / f"phase_transition_{corpus_id}.pkl"
                with open(path, "wb") as fh:
                    pickle.dump(result["model"], fh)
                print(f"  Saved phase_transition model → {path}")
        except ValueError as e:
            print(f"  SKIP phase transition model: {e}")

    conn.close()
    print(f"\nDone.")


if __name__ == "__main__":
    main()

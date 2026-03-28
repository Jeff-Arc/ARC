#!/usr/bin/env python3
"""
arc_ml.py — ML pipeline orchestrator for a single corpus.

Runs the full ML pipeline in sequence:
  1. arc_ml_train  — XGBoost death + phase_transition models → ml_results
                     (saves ml/models/death_{corpus_id}.pkl and phase_transition_{corpus_id}.pkl)
  2. arc_ml_score  — stored-proc scoring + SHAP attribution
                     → clusters.death_probability, cluster_shap_values
                     (loads saved model if present; retrains + saves if not)
  3. arc_backtest  — walk-forward law accuracy → prediction_cone, ml_results
  4. arc_survival  — Cox PH survival estimates → pipe_cluster_survival, findings

Note on Step 2 scope: arc_ml_score Steps B+C (SHAP) always operate on ALL
active corpora, not just the specified --corpus-id.  This keeps SHAP rankings
consistent across the corpus family.  Step A (stored-proc scoring) is restricted
to the specified corpus.

Usage:
  PGHOST=/var/run/postgresql PGDATABASE=arc_v4 PGUSER=jeff \\
    python3 ml/arc_ml.py --corpus-id G06N_quarterly

Exit codes: 0 = all steps succeeded, 1 = a step failed.
"""

import argparse
import os
import sys
import traceback
from datetime import datetime, timezone

import psycopg2

# ── Import sub-scripts (module-level code in each is safe: no main() auto-call) ─

_ML_DIR = os.path.dirname(os.path.abspath(__file__))
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

import arc_ml_train   # noqa: E402
import arc_ml_score   # noqa: E402
import arc_backtest   # noqa: E402
import arc_survival   # noqa: E402


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_conn():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "/var/run/postgresql"),
        dbname=os.environ.get("PGDATABASE", "arc_v4"),
        user=os.environ.get("PGUSER", "jeff"),
    )


def _find_running_task(conn, corpus_id: str) -> int | None:
    """Return id of a running sys_task_queue 'ml' task for this corpus, or None."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM sys_task_queue "
            "WHERE corpus_id = %s AND task_type = 'ml' AND status = 'running' "
            "ORDER BY started_at DESC LIMIT 1",
            (corpus_id,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def _count(conn, query: str, *params) -> int:
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchone()[0]


# ── Step runner ───────────────────────────────────────────────────────────────

def _run_step(step_name: str, argv: list[str], module) -> float:
    """
    Temporarily patch sys.argv and call module.main().

    Returns elapsed seconds.
    Raises RuntimeError (wrapping the original exception) on failure.
    SystemExit(0) is treated as success; any other SystemExit code is a failure.
    """
    t0 = datetime.now(timezone.utc)
    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        module.main()
    except SystemExit as e:
        if e.code is not None and e.code != 0:
            raise RuntimeError(
                f"{step_name} exited with code {e.code}"
            ) from e
    finally:
        sys.argv = old_argv
    return (datetime.now(timezone.utc) - t0).total_seconds()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ARC ML pipeline orchestrator: train → score → backtest → survival"
    )
    parser.add_argument("--corpus-id", required=True,
                        help="corpus_id to run ML pipeline for")
    args = parser.parse_args()
    corpus_id = args.corpus_id

    conn = _get_conn()
    task_id = _find_running_task(conn, corpus_id)

    # Step definitions: (display_name, sys.argv to set, module to call)
    steps = [
        (
            "ml_train",
            [arc_ml_train.__file__, "--corpus-id", corpus_id, "--save-models"],
            arc_ml_train,
        ),
        (
            "ml_score",
            [arc_ml_score.__file__, "--corpus-id", corpus_id],
            arc_ml_score,
        ),
        (
            "backtest",
            # --clear removes stale prediction_cone rows before inserting new ones,
            # preventing duplicate law-trigger rows on re-runs.
            [arc_backtest.__file__, "--corpus-id", corpus_id, "--clear"],
            arc_backtest,
        ),
        (
            "survival",
            [arc_survival.__file__, "--corpus-id", corpus_id],
            arc_survival,
        ),
    ]

    t_total_start = datetime.now(timezone.utc)
    print(f"\n{'='*70}")
    print(f"ARC ML PIPELINE — {corpus_id}")
    if task_id:
        print(f"sys_task_queue task_id: {task_id}")
    print(f"{'='*70}\n")

    step_results = []   # list of {"step", "elapsed", "rows_delta", "status"}
    total_rows   = 0

    for step_name, argv, module in steps:
        print(f"\n── {step_name} {'─' * max(1, 58 - len(step_name))}")

        # Snapshot primary output table counts before step.
        rows_before = _count(conn, "SELECT COUNT(*) FROM ml_results WHERE corpus_id = %s",
                             corpus_id)

        try:
            elapsed = _run_step(step_name, argv, module)
        except Exception as exc:
            elapsed = (datetime.now(timezone.utc) - t_total_start).total_seconds()
            tb = traceback.format_exc()
            err_msg = f"{step_name} failed after {elapsed:.1f}s: {exc}"

            print(f"\n  ✗ {err_msg}")
            print(tb)

            completed = [r["step"] for r in step_results]
            summary = (
                f"ML pipeline FAILED at step '{step_name}' for {corpus_id}. "
                f"Completed: {completed}. Error: {str(exc)[:300]}"
            )

            if task_id:
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT task_fail(%s, %s)", (task_id, err_msg[:1000]))
                    conn.commit()
                    print(f"  task_fail({task_id}) recorded.")
                except Exception as te:
                    print(f"  Warning: could not update task queue: {te}")

            conn.close()
            sys.exit(1)

        rows_after = _count(conn, "SELECT COUNT(*) FROM ml_results WHERE corpus_id = %s",
                            corpus_id)
        delta = max(0, rows_after - rows_before)
        total_rows += delta
        step_results.append({"step": step_name, "elapsed": round(elapsed, 1),
                              "rows_delta": delta, "status": "ok"})
        print(f"\n  ✓ {step_name} complete — {elapsed:.1f}s  (+{delta} ml_results rows)")

    # ── All steps succeeded ───────────────────────────────────────────────────
    total_elapsed = (datetime.now(timezone.utc) - t_total_start).total_seconds()

    # Supplementary counts for the task summary.
    scored = _count(
        conn,
        "SELECT COUNT(*) FROM pipe_clusters WHERE corpus_id = %s AND death_probability IS NOT NULL",
        corpus_id,
    )
    survival_rows = _count(
        conn,
        "SELECT COUNT(*) FROM pipe_cluster_survival WHERE corpus_id = %s",
        corpus_id,
    )
    shap_rows = _count(
        conn,
        "SELECT COUNT(*) FROM pipe_cluster_shap_values WHERE corpus_id = %s",
        corpus_id,
    )

    step_summary = "  |  ".join(
        f"{r['step']}: {r['elapsed']}s (+{r['rows_delta']})" for r in step_results
    )
    full_summary = (
        f"Complete in {total_elapsed:.0f}s. {step_summary}. "
        f"{scored} scored clusters, {survival_rows} survival rows, {shap_rows} SHAP rows."
    )

    print(f"\n{'='*70}")
    print(f"ML PIPELINE COMPLETE — {corpus_id}  ({total_elapsed:.0f}s)")
    for r in step_results:
        print(f"  {r['step']:<14}  {r['elapsed']:>6.1f}s   +{r['rows_delta']} ml_results rows")
    print(f"\n  death_probability scored : {scored} clusters")
    print(f"  pipe_cluster_survival    : {survival_rows} rows")
    print(f"  cluster_shap_values      : {shap_rows} rows")
    print(f"{'='*70}\n")

    if task_id:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT task_complete(%s, %s, %s, %s)",
                    (task_id, full_summary, total_rows, []),
                )
            conn.commit()
            print(f"  task_complete({task_id}) recorded.")
        except Exception as te:
            print(f"  Warning: could not update task queue: {te}")

    conn.close()
    sys.exit(0)


if __name__ == "__main__":
    main()

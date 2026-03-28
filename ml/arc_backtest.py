#!/usr/bin/env python3
"""
arc_backtest.py — ARC Prediction Law Backtesting

Applies 5 prediction laws retroactively to a corpus over a date range.
For each period T, generates predictions about T+1 outcomes, then scores
them against actual T+1 outcomes. Results stored in ml_prediction_cone.

Laws applied:
  L99  — Boundary Pressure Dissolution (death, bpr > P75)
  L100 — Large Cluster Cohesion Collapse (death, size > P75 & cohesion < P25)
  L102 — Load-Bearing Paradigm Persistence (survival, persistence_score < P25)
  L106 — High-Drift Small Cluster Dissolution (death, drift > P75 & size < P25)
  L107 — Anomalous High-Cohesion Small Cluster Mortality (death, cohesion > P75 & size < P25)
  L101 — Algebraic Connectivity Drop (phase_transition, alg_conn < P25)
  L104 — Entropy Plateau Stability (phase_transition, system_entropy > P75)

Usage:
    PGHOST=/var/run/postgresql PGDATABASE=arc_v4 PGUSER=jeff \\
    python3 arc_backtest.py --corpus G06N_quarterly [--dry-run] [--clear]
"""

import os
import sys
import json
import argparse
from datetime import date, datetime, timedelta, timezone

import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Backtest range: predict at T, observe at T+1
# T range: 2020_Q1 → 2024_Q3  (T+1 must exist, so last T = 2024_Q3→2024_Q4 obs)
# Confidence assignments per law (from findings.confidence)
LAW_CONFIDENCE = {99: 0.88, 100: 0.89, 102: 0.77, 106: 0.80, 107: 0.84,
                  101: 0.87, 104: 0.86}


def load_corpus_thresholds(conn, corpus_id: str) -> dict:
    """
    Compute per-corpus percentile thresholds from the actual DB distributions.

    Cluster-level metrics (bpr, cohesion, size, drift, persistence) come from
    v_cluster_event_training.  Period-level metrics (algebraic_connectivity,
    system_entropy, phase_transition_score) come from period_stats, because
    those columns are not present in the cluster-level view.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                percentile_cont(0.25) WITHIN GROUP (ORDER BY boundary_pressure_rate) AS bpr_p25,
                percentile_cont(0.75) WITHIN GROUP (ORDER BY boundary_pressure_rate) AS bpr_p75,
                percentile_cont(0.25) WITHIN GROUP (ORDER BY cohesion)               AS cohesion_p25,
                percentile_cont(0.75) WITHIN GROUP (ORDER BY cohesion)               AS cohesion_p75,
                percentile_cont(0.25) WITHIN GROUP (ORDER BY size)                   AS size_p25,
                percentile_cont(0.75) WITHIN GROUP (ORDER BY size)                   AS size_p75,
                percentile_cont(0.25) WITHIN GROUP (ORDER BY drift_magnitude)        AS drift_p25,
                percentile_cont(0.75) WITHIN GROUP (ORDER BY drift_magnitude)        AS drift_p75,
                percentile_cont(0.25) WITHIN GROUP (ORDER BY persistence_score)      AS persistence_p25
            FROM v_cluster_event_training
            WHERE corpus_id = %s
        """, (corpus_id,))
        cluster_row = dict(cur.fetchone())

        cur.execute("""
            SELECT
                percentile_cont(0.25) WITHIN GROUP (ORDER BY algebraic_connectivity) AS alg_conn_p25,
                percentile_cont(0.75) WITHIN GROUP (ORDER BY system_entropy)         AS entropy_p75,
                percentile_cont(0.75) WITHIN GROUP (ORDER BY phase_transition_score) AS pt_p75
            FROM pipe_period_stats
            WHERE corpus_id = %s
        """, (corpus_id,))
        period_row = dict(cur.fetchone())

    return {**cluster_row, **period_row}


def build_laws(t: dict) -> dict:
    """
    Build law definitions with trigger lambdas closed over corpus-specific
    percentile thresholds.  All threshold values come from load_corpus_thresholds().
    """
    bpr_p75      = t["bpr_p75"]
    coh_p25      = t["cohesion_p25"]
    coh_p75      = t["cohesion_p75"]
    size_p25     = t["size_p25"]
    size_p75     = t["size_p75"]
    drift_p75    = t["drift_p75"]
    persist_p25  = t["persistence_p25"]
    alg_conn_p25 = t["alg_conn_p25"]
    entropy_p75  = t["entropy_p75"]

    return {
        # Cluster-level death predictions
        99: {
            "title": "Boundary Pressure Dissolution Law",
            "prediction_type": "death",
            "target": "will_die",
            "scope": "quarterly_only",
            "threshold_doc": f"boundary_pressure_rate > P75 ({bpr_p75:.3f})",
            "trigger": lambda row, v=bpr_p75: (
                row["boundary_pressure_rate"] is not None
                and row["boundary_pressure_rate"] > v
            ),
        },
        100: {
            "title": "Large Cluster Cohesion Collapse Fragmentation Law",
            "prediction_type": "death",
            "target": "will_die",
            "scope": "universal",
            "threshold_doc": f"size > P75 ({size_p75:.0f}) AND cohesion < P25 ({coh_p25:.3f})",
            "trigger": lambda row, sp=size_p75, cp=coh_p25: (
                row["size"] is not None and row["size"] > sp
                and row["cohesion"] is not None and row["cohesion"] < cp
            ),
        },
        102: {
            "title": "Load-Bearing Paradigm Persistence Law",
            "prediction_type": "death",
            "target": "will_die",
            "scope": "universal",
            "threshold_doc": f"persistence_score < P25 ({persist_p25:.3f}) → low persistence → elevated death risk",
            "trigger": lambda row, v=persist_p25: (
                row["persistence_score"] is not None
                and row["persistence_score"] < v
            ),
        },
        106: {
            "title": "High-Drift Small Cluster Dissolution Law",
            "prediction_type": "death",
            "target": "will_die",
            "scope": "quarterly_only",
            "threshold_doc": f"drift_magnitude > P75 ({drift_p75:.3f}) AND size < P25 ({size_p25:.0f})",
            "trigger": lambda row, dp=drift_p75, sp=size_p25: (
                row["drift_magnitude"] is not None and row["drift_magnitude"] > dp
                and row["size"] is not None and row["size"] < sp
            ),
        },
        107: {
            "title": "Anomalous High-Cohesion Small Cluster Mortality Law",
            "prediction_type": "death",
            "target": "will_die",
            "scope": "universal",
            "threshold_doc": f"cohesion > P75 ({coh_p75:.3f}) AND size < P25 ({size_p25:.0f})",
            "trigger": lambda row, cp=coh_p75, sp=size_p25: (
                row["cohesion"] is not None and row["cohesion"] > cp
                and row["size"] is not None and row["size"] < sp
            ),
        },
        # Period-level phase transition predictions
        101: {
            "title": "Algebraic Connectivity Drop Fragmentation Law",
            "prediction_type": "phase_transition",
            "target": "phase_transition_score",
            "scope": "universal",
            "threshold_doc": f"algebraic_connectivity < P25 ({alg_conn_p25:.3f}) → structural fragmentation risk",
            "trigger": None,
            "ps_trigger": lambda ps, v=alg_conn_p25: (
                ps["algebraic_connectivity"] is not None
                and ps["algebraic_connectivity"] < v
            ),
        },
        104: {
            "title": "Entropy Plateau Structural Stability Law",
            "prediction_type": "phase_transition",
            "target": "phase_transition_score",
            "scope": "universal",
            "threshold_doc": f"system_entropy > P75 ({entropy_p75:.3f}) → entropy ceiling → structural stress",
            "trigger": None,
            "ps_trigger": lambda ps, v=entropy_p75: (
                ps["system_entropy"] is not None
                and ps["system_entropy"] > v
            ),
        },
    }


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------
def get_conn():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "/var/run/postgresql"),
        dbname=os.environ.get("PGDATABASE", "arc_v4"),
        user=os.environ.get("PGUSER", "jeff"),
    )


def next_period_start(conn, corpus_id: str, period_start: date) -> date | None:
    """Return the actual next period_start for this corpus from period_stats.

    Resolution-agnostic: works for quarterly, monthly, weekly, annual, and any
    other resolution stored in the DB.  Replaces the former next_quarter() which
    hardcoded 3-month arithmetic and silently returned wrong dates for non-quarterly
    corpora.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT period_start FROM pipe_period_stats "
            "WHERE corpus_id = %s AND period_start > %s "
            "ORDER BY period_start LIMIT 1",
            (corpus_id, period_start),
        )
        row = cur.fetchone()
        return row[0] if row else None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_periods(conn, corpus_id: str) -> list[dict]:
    """Load all periods for corpus in backtest range.

    Date range is derived dynamically from the corpus's own period_stats so that
    H01L (1900–2024), longevity, and other non-G06N corpora produce meaningful
    walk-forward folds rather than silently returning 0 periods (bug #74).

    Rules:
      - Skip the first 20% of periods (training window; model needs history).
      - Skip the last 2 periods (need T+1 confirmation window).
      - Require at least 10 periods; print a warning and return [] if fewer.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MIN(period_start), MAX(period_start), COUNT(*) "
            "FROM pipe_period_stats WHERE corpus_id = %s",
            (corpus_id,),
        )
        min_ps, max_ps, n_periods = cur.fetchone()

    if not min_ps or n_periods < 10:
        print(f"  load_periods: {corpus_id} has {n_periods} periods — "
              f"fewer than 10, skipping backtest")
        return []

    # Reserve first 20% as implicit training window; last 2 as confirmation window.
    skip_head = max(1, int(n_periods * 0.20))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT period_start FROM pipe_period_stats WHERE corpus_id=%s "
            "ORDER BY period_start OFFSET %s LIMIT 1",
            (corpus_id, skip_head),
        )
        row = cur.fetchone()
    backtest_start = row[0] if row else min_ps

    with conn.cursor() as cur:
        cur.execute(
            "SELECT period_start FROM pipe_period_stats WHERE corpus_id=%s "
            "ORDER BY period_start DESC OFFSET 2 LIMIT 1",
            (corpus_id,),
        )
        row = cur.fetchone()
    backtest_end = row[0] if row else max_ps

    print(f"  load_periods: {corpus_id} backtest window "
          f"{backtest_start} → {backtest_end} ({n_periods} total periods)")

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT period, period_start, period_end,
                   algebraic_connectivity, system_entropy,
                   phase_transition_score, n_births, n_deaths,
                   void_count, n_precursor_clusters
            FROM pipe_period_stats
            WHERE corpus_id = %s
              AND period_start >= %s AND period_start <= %s
            ORDER BY period_start
        """, (corpus_id, backtest_start, backtest_end))
        return [dict(r) for r in cur.fetchall()]


def load_period_stats_next(conn, period_start: date, corpus_id: str) -> dict | None:
    """Load period_stats for T+1 period (resolution-agnostic DB lookup)."""
    next_ps = next_period_start(conn, corpus_id, period_start)
    if next_ps is None:
        return None
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT period, period_start, phase_transition_score,
                   algebraic_connectivity, system_entropy,
                   n_births, n_deaths
            FROM pipe_period_stats
            WHERE corpus_id = %s AND period_start = %s
        """, (corpus_id, next_ps))
        row = cur.fetchone()
        return dict(row) if row else None


def load_clusters_at(conn, period_start: date, corpus_id: str) -> list[dict]:
    """Load cluster features for a period."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT cluster_id, cohesion, size, elongation_ratio,
                   drift_magnitude, boundary_pressure_rate,
                   persistence_score, convergence_score,
                   mean_betweenness, marginal_entropy_impact,
                   cohesion_percentile, drift_percentile, size_percentile,
                   jerk, will_die
            FROM v_cluster_event_training
            WHERE corpus_id = %s AND period_start = %s
        """, (corpus_id, period_start))
        return [dict(r) for r in cur.fetchall()]


def cluster_died_next(conn, cluster_id: int, next_period_start: date,
                      corpus_id: str) -> bool | None:
    """Check if cluster existed at T+1 (True = died between T and T+1).

    Uses will_die from v_cluster_event_training at period T (already computed).
    Also verifiable by checking T+1 cluster list.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM pipe_clusters
            WHERE corpus_id = %s AND period_start = %s AND cluster_id = %s
        """, (corpus_id, next_period_start, cluster_id))
        exists = cur.fetchone()[0]
        return exists == 0  # True if it died (not present at T+1)


# ---------------------------------------------------------------------------
# Prediction insertion
# ---------------------------------------------------------------------------
def clear_backtest_rows(conn, corpus_id: str):
    """Remove existing backtest rows from ml_prediction_cone."""
    with conn.cursor() as cur:
        cur.execute("""
            DELETE FROM ml_prediction_cone
            WHERE corpus_id = %s
              AND (predicted_properties->>'source') = 'arc_backtest'
        """, (corpus_id,))
    conn.commit()
    print(f"  Cleared existing arc_backtest rows for {corpus_id}.")


def insert_prediction(conn, *, corpus_id: str, period_predicted: str,
                      period_observed: str, confidence: float,
                      n_confirming: int, n_refuting: int,
                      status: str, predicted_properties: dict):
    """Insert one ml_prediction_cone row."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ml_prediction_cone
              (corpus_id, period_predicted, period_observed,
               predicted_properties, confidence_at_prediction, confidence_current,
               n_confirming_signals, n_refuting_signals, status)
            VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s)
        """, (
            corpus_id, period_predicted, period_observed,
            json.dumps(predicted_properties), confidence, confidence,
            n_confirming, n_refuting, status,
        ))


# ---------------------------------------------------------------------------
# Law evaluation
# ---------------------------------------------------------------------------
def eval_death_laws(conn, period: dict, clusters: list[dict], dry_run: bool,
                    corpus_id: str, laws: dict,
                    next_period: dict | None = None) -> list[dict]:
    """Apply cluster-level death laws at period T, observe T+1."""
    results = []
    period_label_t  = period["period"]
    # Use DB-derived T+1 period label to avoid hardcoded quarterly arithmetic
    period_label_t1 = next_period["period"] if next_period else "unknown"

    for law_id, law in laws.items():
        if law["prediction_type"] != "death":
            continue
        trigger_fn = law["trigger"]

        triggered = [c for c in clusters if trigger_fn(c)]
        if not triggered:
            continue

        n_predicted = len(triggered)
        n_confirmed = 0
        n_refuted   = 0

        for c in triggered:
            # Use will_die from T (pre-computed in view against T+1)
            actual_died = bool(c["will_die"])
            if actual_died:
                n_confirmed += 1
            else:
                n_refuted += 1

        status = "confirmed" if n_confirmed > n_refuted else "refuted"
        confidence = LAW_CONFIDENCE[law_id]

        props = {
            "source":          "arc_backtest",
            "prediction_type": "death",
            "law_id":          law_id,
            "law_title":       law["title"],
            "scope":           law["scope"],
            "threshold":       law["threshold_doc"],
            "n_predicted_clusters": n_predicted,
            "cluster_ids":     [c["cluster_id"] for c in triggered],
            "accuracy_this_period": round(n_confirmed / n_predicted, 3),
        }

        results.append({
            "period_predicted": period_label_t,
            "period_observed":  period_label_t1,
            "confidence":       confidence,
            "n_confirming":     n_confirmed,
            "n_refuting":       n_refuted,
            "status":           status,
            "props":            props,
            "law_id":           law_id,
            "n_predicted":      n_predicted,
        })

        if not dry_run:
            insert_prediction(conn,
                corpus_id=corpus_id,
                period_predicted=period_label_t,
                period_observed=period_label_t1,
                confidence=confidence,
                n_confirming=n_confirmed,
                n_refuting=n_refuted,
                status=status,
                predicted_properties=props,
            )

    return results


def eval_phase_transition_laws(conn, period: dict, next_period: dict | None,
                               dry_run: bool, corpus_id: str,
                               laws: dict, thresholds: dict) -> list[dict]:
    """Apply period-level phase transition laws."""
    if next_period is None:
        return []

    results = []
    period_label_t  = period["period"]
    period_label_t1 = next_period["period"]
    pt_score_next   = next_period.get("phase_transition_score") or 0.0
    PT_THRESHOLD    = thresholds["pt_p75"]   # corpus-specific P75

    for law_id, law in laws.items():
        if law["prediction_type"] != "phase_transition":
            continue
        ps_trigger = law.get("ps_trigger")
        if ps_trigger is None:
            continue

        triggered = ps_trigger(period)
        if not triggered:
            continue

        # Ground truth: did phase transition actually occur at T+1?
        pt_occurred = pt_score_next >= PT_THRESHOLD
        n_confirmed = 1 if pt_occurred else 0
        n_refuted   = 0 if pt_occurred else 1
        status      = "confirmed" if pt_occurred else "refuted"
        confidence  = LAW_CONFIDENCE[law_id]

        props = {
            "source":              "arc_backtest",
            "prediction_type":     "phase_transition",
            "law_id":              law_id,
            "law_title":           law["title"],
            "scope":               law["scope"],
            "threshold":           law["threshold_doc"],
            "trigger_value":       (period.get("algebraic_connectivity")
                                   if law_id == 101 else period.get("system_entropy")),
            "observed_pt_score":   round(float(pt_score_next), 4),
            "pt_threshold_used":   PT_THRESHOLD,
        }

        results.append({
            "period_predicted": period_label_t,
            "period_observed":  period_label_t1,
            "confidence":       confidence,
            "n_confirming":     n_confirmed,
            "n_refuting":       n_refuted,
            "status":           status,
            "props":            props,
            "law_id":           law_id,
            "n_predicted":      1,
        })

        if not dry_run:
            insert_prediction(conn,
                corpus_id=corpus_id,
                period_predicted=period_label_t,
                period_observed=period_label_t1,
                confidence=confidence,
                n_confirming=n_confirmed,
                n_refuting=n_refuted,
                status=status,
                predicted_properties=props,
            )

    return results


# ---------------------------------------------------------------------------
# Accuracy summary
# ---------------------------------------------------------------------------
def print_summary(all_results: list[dict], corpus_id: str):
    from collections import defaultdict

    by_law  = defaultdict(lambda: {"confirmed": 0, "refuted": 0,
                                   "n_predicted": 0, "title": ""})
    by_type = defaultdict(lambda: {"confirmed": 0, "refuted": 0, "n_predicted": 0})

    for r in all_results:
        law_id = r["law_id"]
        by_law[law_id]["confirmed"]  += r["n_confirming"]
        by_law[law_id]["refuted"]    += r["n_refuting"]
        by_law[law_id]["n_predicted"] += r["n_predicted"]
        by_law[law_id]["title"]       = r["props"].get("law_title", "")

        ptype = r["props"].get("prediction_type", "?")
        by_type[ptype]["confirmed"]  += r["n_confirming"]
        by_type[ptype]["refuted"]    += r["n_refuting"]
        by_type[ptype]["n_predicted"] += r["n_predicted"]

    print(f"\n{'='*72}")
    print(f"  ARC BACKTEST ACCURACY SUMMARY — {corpus_id}")
    result_periods = sorted({r["period_start"] for r in all_results if "period_start" in r})
    ps_range = (f"{result_periods[0]} → {result_periods[-1]}" if result_periods else "none")
    print(f"  Periods: {ps_range}  (T+1 observed)")
    print(f"{'='*72}")

    print(f"\n{'BY PREDICTION TYPE':}")
    print(f"  {'Type':<20} {'N':>6} {'Confirmed':>10} {'Refuted':>8} {'Accuracy':>10}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*8} {'-'*10}")
    for ptype, s in sorted(by_type.items()):
        n = s["confirmed"] + s["refuted"]
        acc = s["confirmed"] / n if n else 0
        print(f"  {ptype:<20} {s['n_predicted']:>6} {s['confirmed']:>10} "
              f"{s['refuted']:>8} {acc:>9.1%}")

    print(f"\n{'BY LAW':}")
    print(f"  {'Law':>5} {'N':>6} {'Conf':>6} {'Ref':>6} {'Acc':>7}  Title")
    print(f"  {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*7}  {'-'*40}")
    for law_id, s in sorted(by_law.items()):
        n = s["confirmed"] + s["refuted"]
        acc = s["confirmed"] / n if n else 0
        title = s["title"][:40]
        print(f"  #{law_id:<4} {s['n_predicted']:>6} {s['confirmed']:>6} "
              f"{s['refuted']:>6} {acc:>6.1%}  {title}")

    total_c = sum(s["confirmed"] for s in by_law.values())
    total_r = sum(s["refuted"]   for s in by_law.values())
    total_n = sum(s["n_predicted"] for s in by_law.values())
    total   = total_c + total_r
    overall_acc = total_c / total if total else 0
    print(f"\n  {'TOTAL':<5} {total_n:>6} {total_c:>6} {total_r:>6} {overall_acc:>6.1%}")
    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# Prediction accuracy → ml_results (v_prediction_confirmation_rate)
# ---------------------------------------------------------------------------

def write_prediction_accuracy(conn, corpus_id: str):
    """Aggregate ml_prediction_cone rows into per-corpus accuracy metrics and write
    to ml_results under model_name='arc_prediction_accuracy'.

    This operationalises v_prediction_confirmation_rate for a specific corpus:
    for each prediction_type (death_prediction, phase_transition_prediction etc.)
    it computes confirmation_rate and sample_size from the ml_prediction_cone table
    and stores them so downstream consumers can track law accuracy over time.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute("""
            SELECT
                predicted_properties->>'prediction_type' AS prediction_type,
                COUNT(*)                                AS sample_size,
                COUNT(*) FILTER (WHERE status = 'confirmed') AS n_confirmed,
                ROUND(
                    COUNT(*) FILTER (WHERE status = 'confirmed')::numeric
                    / NULLIF(COUNT(*), 0), 4
                )                                       AS confirmation_rate
            FROM ml_prediction_cone
            WHERE corpus_id = %s
              AND status IN ('confirmed', 'refuted')
            GROUP BY predicted_properties->>'prediction_type'
            ORDER BY sample_size DESC
        """, (corpus_id,))
        rows = cur.fetchall()

    if not rows:
        print("  write_prediction_accuracy: no confirmed/refuted rows in ml_prediction_cone — skip")
        return

    now = datetime.now(timezone.utc)
    ml_rows = []
    for r in rows:
        n = int(r["sample_size"])
        ml_rows.append((
            "arc_prediction_accuracy", corpus_id,
            "prediction_outcome", "confirmation_rate",
            float(r["confirmation_rate"] or 0),
            r["prediction_type"], n,
            f"Backtest accuracy from ml_prediction_cone; {r['n_confirmed']}/{n} confirmed",
            now,
        ))
        ml_rows.append((
            "arc_prediction_accuracy", corpus_id,
            "prediction_outcome", "sample_size",
            float(n),
            r["prediction_type"], n, None, now,
        ))

    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM ml_results WHERE model_name='arc_prediction_accuracy' AND corpus_id=%s",
            (corpus_id,),
        )
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO ml_results
              (model_name, corpus_id, target, metric_name, metric_value,
               feature_name, n_samples, notes, recorded_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, ml_rows)
    conn.commit()
    print(f"  Wrote {len(ml_rows)} arc_prediction_accuracy rows to ml_results.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ARC prediction law backtesting")
    parser.add_argument("--corpus-id", required=True,
                        help="Corpus ID to backtest (e.g. G06N_quarterly)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate predictions without writing to DB")
    parser.add_argument("--clear", action="store_true",
                        help="Delete existing arc_backtest rows before running")
    args = parser.parse_args()
    corpus_id = args.corpus_id

    conn = get_conn()
    if args.clear:
        clear_backtest_rows(conn, corpus_id)

    thresholds = load_corpus_thresholds(conn, corpus_id)
    laws       = build_laws(thresholds)

    periods = load_periods(conn, corpus_id)
    print(f"Corpus: {corpus_id}")
    ps_start = periods[0]["period_start"] if periods else "—"
    ps_end   = periods[-1]["period_start"] if periods else "—"
    print(f"Backtesting {len(periods)} periods ({ps_start} → {ps_end})")
    print(f"Dry run: {args.dry_run}")
    print(f"\nComputed thresholds for {corpus_id}:")
    print(f"  bpr_p75={thresholds['bpr_p75']:.3f}  "
          f"cohesion_p25={thresholds['cohesion_p25']:.3f}  "
          f"cohesion_p75={thresholds['cohesion_p75']:.3f}")
    print(f"  size_p25={thresholds['size_p25']:.0f}  "
          f"size_p75={thresholds['size_p75']:.0f}  "
          f"drift_p75={thresholds['drift_p75']:.3f}  "
          f"persistence_p25={thresholds['persistence_p25']:.3f}")
    print(f"  alg_conn_p25={thresholds['alg_conn_p25']:.3f}  "
          f"entropy_p75={thresholds['entropy_p75']:.3f}  "
          f"pt_p75={thresholds['pt_p75']:.3f}")
    print()

    all_results = []
    total_predictions = 0

    for period in periods:
        ps    = period["period_start"]
        label = period["period"]
        clusters   = load_clusters_at(conn, ps, corpus_id)
        next_stats = load_period_stats_next(conn, ps, corpus_id)

        # Cluster-level death laws
        death_results = eval_death_laws(conn, period, clusters, args.dry_run, corpus_id, laws,
                                        next_period=next_stats)
        # Period-level phase transition laws
        pt_results    = eval_phase_transition_laws(conn, period, next_stats, args.dry_run,
                                                   corpus_id, laws, thresholds)

        period_results = death_results + pt_results
        all_results.extend(period_results)
        total_predictions += sum(r["n_predicted"] for r in period_results)

        n_preds = len(period_results)
        if period_results:
            confirms = sum(1 for r in period_results if r["status"] == "confirmed")
            print(f"  {label}: {len(clusters)} clusters, "
                  f"{n_preds} law triggers, {confirms}/{n_preds} confirmed")
        else:
            print(f"  {label}: {len(clusters)} clusters, no law triggers")

    if not args.dry_run:
        conn.commit()

    print(f"\nTotal individual predictions: {total_predictions}")
    print_summary(all_results, corpus_id)

    if not args.dry_run:
        write_prediction_accuracy(conn, corpus_id)

    # ── Register as pipeline step (W4 fix — migration 140) ───────────────────
    if not args.dry_run:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sys_corpus_pipeline_steps
                    (corpus_id, step_order, script, status, notes)
                SELECT %s, 10, 'ml/arc_backtest.py', 'complete',
                       'Walk-forward backtest: 7 prediction laws scored'
                WHERE NOT EXISTS (
                    SELECT 1 FROM sys_corpus_pipeline_steps
                    WHERE corpus_id = %s AND step_order = 10
                )
            """, (corpus_id, corpus_id))
        conn.commit()

    conn.close()


if __name__ == "__main__":
    main()

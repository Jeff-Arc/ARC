#!/home/jeff/miniconda3/bin/python3
# BUG LOGGING: Any bugs found during development must be inserted
# into the bug_log table in arc_v4 DB. Use severity: critical/high/
# medium/low. Set patched_in_code=TRUE only if the fix is in this
# file or a DB function, not just in analysis scripts.
"""
arc_pipeline_v2.py — Thin orchestrator for the ARC v4 pipeline (v2 architecture).

Call sequence per period:
  1. Phase 1 — Leiden clustering (arc_leiden.run_period)
       knn_edges → leidenalg.find_partition → chunk_periods + clusters
  2. compute_centroids (plpgsql — for matching; also re-run inside arc_run_pipeline)
  3. Phase 4 — arc_run_pipeline (SQL master orchestrator, migration 129+130)
       Resolves period metadata, then calls:
         compute_centroids → arc_compute_period (SVD geometry, all geometry
         batches, temporal Stage 6, period_stats, spectral, scoring,
         law backtesting) → junk-edge cleanup → boundary_percentile →
         n_dark_matter_chunks update

After all periods: run_cluster_matching (plpython3u, migration 128)
  → fix_lifecycle_flags → compute_split_children → compute_merger_targets
  → compute_velocity_direction_stability
Usage:
  python arc_pipeline_v2.py --corpus-id G06N_quarterly --period-start 2023-01-01
  python arc_pipeline_v2.py --corpus-id G06N_quarterly --all-periods --write-back
"""

import argparse
import contextlib
import io
import json
import os
import math
import sys
import time
from datetime import date

import numpy as np
import psycopg2
import psycopg2.extras
import igraph
import leidenalg

# ── Connection ─────────────────────────────────────────────────────────────────

def get_conn():
    conn = psycopg2.connect(
        dbname=os.environ.get("PGDATABASE", "arc_v4"),
        host=os.environ.get("PGHOST", "/var/run/postgresql"),
        user=os.environ.get("PGUSER", "jeff"),
    )
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("LOAD 'age'")
    cur.execute('SET search_path = ag_catalog, "$user", public')
    cur.close()
    return conn


def fetch(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def run(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()


def call_fn(conn, sql, params=None) -> dict:
    rows = fetch(conn, sql, params)
    conn.commit()
    result = rows[0][0] if rows else {}
    return result if isinstance(result, dict) else {}


_REAL_MIN = 1.175494e-38  # PostgreSQL real minimum positive normal


def to_pg_real(v):
    """Flush subnormal floats to 0.0 so PostgreSQL real columns never overflow."""
    v = float(v)
    if abs(v) < _REAL_MIN:
        return 0.0
    return v


def safe_real(v):
    """Clamp a float to a valid PostgreSQL real range, returning None for bad values."""
    if v is None or not math.isfinite(v):
        return None
    if abs(v) < _REAL_MIN:
        return 0.0
    if abs(v) > 3.4e38:
        return None
    return float(v)


def fmt_time(s: float) -> str:
    return f"{s:.2f}s" if s < 60 else f"{s / 60:.1f}m"


# ── ANSI colour helpers (only when writing to a TTY) ───────────────────────────

_C_RED    = "\033[31m"
_C_YELLOW = "\033[33m"
_C_GREY   = "\033[90m"
_C_RESET  = "\033[0m"


def _clr(code: str, msg: str) -> str:
    return f"{code}{msg}{_C_RESET}" if IS_TTY else msg


def _contract_print(level: str, msg: str) -> None:
    """Print a contract message with appropriate colour."""
    if level == "ERROR":
        print(_clr(_C_RED,    f"  [ERROR]   {msg}"), file=sys.stderr)
    elif level == "WARNING":
        print(_clr(_C_YELLOW, f"  [WARNING] {msg}"))
    else:
        print(_clr(_C_GREY,   f"  [INFO]    {msg}"))


# Severity ordering: ERROR > WARNING > INFO
_SEV_RANK = {"ERROR": 2, "WARNING": 1, "INFO": 0}


def _should_raise(level: str, min_severity: str) -> bool:
    """Return True if level >= min_severity (caller should raise RuntimeError)."""
    return _SEV_RANK.get(level, 0) >= _SEV_RANK.get(min_severity, 2)


# ── Phase contracts ─────────────────────────────────────────────────────────────

def verify_phase_contract(conn, corpus_id, phase, period_start=None,
                          severity: str = "ERROR"):
    """
    Verify that the previous phase left data in a valid state.

    severity controls the minimum level that raises RuntimeError:
      'ERROR'   — only hard failures abort (default)
      'WARNING' — also abort on degraded-quality conditions
      'INFO'    — abort on any anomaly

    Contract numbering refers to what the phase REQUIRES (its prerequisite),
    not to the file's internal phase numbering:
      phase 2 — geometry requires chunk_periods (output of Leiden/Phase 1)
      phase 3 — leiden requires knn_edges (pre-pipeline prerequisite)
      phase 4 — period_stats requires clusters with cohesion (output of geometry)
      phase 5 — matching requires period_stats (output of SQL+stats phases)
      phase 6 — lifecycle requires persistent_cluster_id (output of matching)
    """
    with conn.cursor() as cur:

        if phase == 2:
            # Phase 2 (geometry) requires: chunk_periods exist for period
            cur.execute("""
                SELECT COUNT(*) FROM chunk_periods
                WHERE corpus_id = %s AND period_start = %s
            """, (corpus_id, period_start))
            n = cur.fetchone()[0]
            if n == 0:
                raise RuntimeError(
                    f"Phase 2 contract violated: no chunk_periods rows "
                    f"for {corpus_id} {period_start}. "
                    f"Phase 1 (ingestion) may not have completed."
                )

        elif phase == 3:
            # Phase 3 (leiden) requires: knn_edges exist for period
            cur.execute("""
                SELECT COUNT(*) FROM knn_edges
                WHERE corpus_id = %s AND period_start = %s
            """, (corpus_id, period_start))
            n = cur.fetchone()[0]
            if n == 0:
                raise RuntimeError(
                    f"Phase 3 contract violated: no knn_edges rows "
                    f"for {corpus_id} {period_start}. "
                    f"Phase 2 (kNN) may not have completed."
                )
            # Also verify knn_edges period_start values match chunk_periods
            # period_start values — guards against GENERATED ALWAYS AS label mismatch
            cur.execute("""
                SELECT COUNT(*) FROM knn_edges k
                WHERE k.corpus_id = %s
                  AND k.period_start = %s
                  AND NOT EXISTS (
                      SELECT 1 FROM chunk_periods cp
                      WHERE cp.corpus_id = k.corpus_id
                        AND cp.period_start = k.period_start
                  )
            """, (corpus_id, period_start))
            mismatched = cur.fetchone()[0]
            if mismatched > 0:
                raise RuntimeError(
                    f"Phase 3 contract violated: {mismatched} knn_edges rows "
                    f"for {corpus_id} {period_start} have no matching "
                    f"chunk_periods period_start — period label mismatch."
                )

        elif phase == 4:
            # Phase 4 (period_stats) requires: clusters exist with
            # non-NULL cohesion (geometry must have run)
            # Thresholds: >25% NULL → ERROR (abort), >10% → WARNING, >5% → INFO
            cur.execute("""
                SELECT COUNT(*),
                       SUM(CASE WHEN cohesion IS NULL THEN 1 ELSE 0 END)
                FROM clusters
                WHERE corpus_id = %s AND period_start = %s
                AND is_junk = FALSE
            """, (corpus_id, period_start))
            total, null_cohesion = cur.fetchone()
            total = total or 0
            null_cohesion = null_cohesion or 0
            if total == 0:
                raise RuntimeError(
                    f"Phase 4 contract violated: no non-junk clusters "
                    f"for {corpus_id} {period_start}. "
                    f"Phase 3 (Leiden) may not have completed."
                )
            null_rate = null_cohesion / total if total else 0.0
            if null_rate > 0.25:
                level = "ERROR"
            elif null_rate > 0.10:
                level = "WARNING"
            elif null_rate > 0.05:
                level = "INFO"
            else:
                level = None
            if level:
                msg = (f"Phase 4: {null_cohesion}/{total} clusters "
                       f"({null_rate:.0%}) have NULL cohesion "
                       f"for {corpus_id} {period_start}")
                _contract_print(level, msg)
                if _should_raise(level, severity):
                    raise RuntimeError(msg)

        elif phase == 6:
            # Phase 6 (lifecycle) requires: persistent_cluster_id set
            # Thresholds: <60% → ERROR, <80% → WARNING, <90% → INFO
            cur.execute("""
                SELECT COUNT(*),
                       SUM(CASE WHEN persistent_cluster_id IS NULL
                                THEN 1 ELSE 0 END)
                FROM clusters
                WHERE corpus_id = %s AND period_start = %s
                AND is_junk = FALSE
            """, (corpus_id, period_start))
            total, null_pid = cur.fetchone()
            total = total or 0
            null_pid = null_pid or 0
            coverage = 1.0 - (null_pid / total if total else 0.0)
            if coverage < 0.60:
                level = "ERROR"
            elif coverage < 0.80:
                level = "WARNING"
            elif coverage < 0.90:
                level = "INFO"
            else:
                level = None
            if level:
                msg = (f"Phase 6: {coverage:.0%} persistent_cluster_id "
                       f"coverage for {corpus_id} {period_start} "
                       f"({null_pid}/{total} NULL)")
                _contract_print(level, msg)
                if _should_raise(level, severity):
                    raise RuntimeError(msg)


# ── Table display ───────────────────────────────────────────────────────────────

VERBOSE = False          # set from --verbose in main()
IS_TTY  = sys.stdout.isatty()

_TBL_HEADER = (
    f"{'Period':<12} | {'Leiden':<14} | {'Geometry':<10} | "
    f"{'SQL':<10} | {'Stats':<10} | {'Match':<8} | {'Total':<8}"
)
_TBL_DIV = (
    "─" * 12 + "─┼─" + "─" * 14 + "─┼─" + "─" * 10 + "─┼─" +
    "─" * 10 + "─┼─" + "─" * 10 + "─┼─" + "─" * 8  + "─┼─" + "─" * 8
)


def _render_row(cells: dict) -> str:
    return (
        f"{cells.get('period',''):<12} | {cells.get('leiden',''):<14} | "
        f"{cells.get('geo',''):<10} | {cells.get('sql',''):<10} | "
        f"{cells.get('stats',''):<10} | {cells.get('match',''):<8} | "
        f"{cells.get('total',''):<8}"
    )


def _update_row(cells: dict) -> None:
    """Overwrite the last printed line using ANSI cursor-previous-line."""
    sys.stdout.write("\033[F" + _render_row(cells) + "\n")
    sys.stdout.flush()


def _print_tbl_header() -> None:
    print(_TBL_HEADER)
    print(_TBL_DIV)


def _phase_ctx():
    """Silence stdout inside a phase call when not in verbose mode."""
    return contextlib.redirect_stdout(io.StringIO()) if not VERBOSE else contextlib.nullcontext()


def period_label(d: date, resolution: str = "quarterly",
                 cutoff_year: int = 2010) -> str:
    """
    Generate a human-readable period label for a given date and resolution.

    quarterly : pre_YYYY | YYYY_Q1..Q4   (existing behaviour)
    monthly   : pre_YYYY | YYYY_M01..M12
    weekly    : pre_YYYY | YYYY_W01..W53  (ISO week)
    annual    : pre_YYYY | YYYY
    """
    if d.year < cutoff_year:
        return f"pre_{cutoff_year}"
    if resolution == "quarterly":
        return f"{d.year}_Q{(d.month - 1) // 3 + 1}"
    if resolution == "monthly":
        return f"{d.year}_M{d.month:02d}"
    if resolution == "weekly":
        iso = d.isocalendar()
        return f"{iso[0]}_W{iso[1]:02d}"   # iso[0]=ISO year, iso[1]=ISO week
    if resolution == "annual":
        return f"{d.year}"
    raise ValueError(f"Unknown resolution: {resolution!r}")


def period_end_for(period_start: date, resolution: str) -> date:
    """Return the last date of the period containing period_start."""
    from calendar import monthrange
    if resolution == "annual":
        return date(period_start.year, 12, 31)
    if resolution == "quarterly":
        q_end_month = ((period_start.month - 1) // 3 + 1) * 3
        last_day = monthrange(period_start.year, q_end_month)[1]
        return date(period_start.year, q_end_month, last_day)
    if resolution == "monthly":
        last_day = monthrange(period_start.year, period_start.month)[1]
        return date(period_start.year, period_start.month, last_day)
    if resolution == "weekly":
        from datetime import timedelta
        return period_start + timedelta(days=6)
    raise ValueError(f"Unknown resolution: {resolution!r}")


# ── Leiden helpers (inlined from arc_leiden.py) ────────────────────────────────

def print_section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _run_no_commit(conn, sql, params=None):
    """Execute SQL without committing (used inside leiden_run_period)."""
    with conn.cursor() as cur:
        cur.execute(sql, params)


def leiden_run_period(conn, period_start: date, period_end: date,
                      corpus_type: str, corpus_id: str,
                      write_back: bool, cfg: dict) -> dict:
    """
    Run Leiden community detection for one period.
    Returns a stats dict with n_clusters, n_junk, n_isolated, modularity, etc.
    Inlined from arc_leiden.py.
    """
    time_res       = cfg.get("_resolution", "quarterly")
    cutoff_yr      = int(cfg.get("_cutoff_year", 2010))
    period         = period_label(period_start, time_res, cutoff_yr)
    k_value        = int(cfg.get("KNN_K", 16))
    _corpus_prefix = corpus_id.split("_")[0].upper()
    resolution     = float(cfg.get(f"LEIDEN_RESOLUTION_{_corpus_prefix}",
                                   cfg.get("LEIDEN_RESOLUTION", 1.0)))
    seed           = int(cfg.get("LEIDEN_SEED", 42))

    with conn.cursor() as _cur:
        _cur.execute("SELECT resolution FROM v_run_lookup WHERE corpus_id = %s",
                     (corpus_id,))
        _row = _cur.fetchone()
        _resolution = (_row[0] if _row else 'quarterly').upper()

    _res_key = f"JUNK_SIZE_THRESHOLD_{_resolution}"
    junk_min = int(cfg.get(_res_key, cfg.get("JUNK_SIZE_THRESHOLD", 3)))
    print(f"  [junk] corpus={corpus_id} resolution={_resolution} key={_res_key} threshold={junk_min}")
    timings = {}

    print(f"\n{'=' * 60}")
    print(f"  leiden_run_period  —  {period} / {corpus_id}")
    print(f"  k={k_value}  γ={resolution}  seed={seed}  write-back={write_back}")
    print(f"{'=' * 60}")

    # ── Step 1: Load knn_edges ────────────────────────────────────────────────
    print_section("Step 1: Load knn_edges")
    t0 = time.time()

    edge_rows = fetch(conn, """
        SELECT chunk_id::text, neighbor_id::text, distance::float
        FROM knn_edges
        WHERE period_start = %s
          AND corpus_id    = %s
          AND k_value      = %s
    """, (period_start, corpus_id, k_value))

    cp_rows = fetch(conn, """
        SELECT chunk_id::text
        FROM chunk_periods
        WHERE period_start = %s AND corpus_id = %s
    """, (period_start, corpus_id))
    all_chunk_ids = [r[0] for r in cp_rows]
    chunk_set = set(all_chunk_ids)

    edge_weights: dict = {}
    for a, b, dist in edge_rows:
        if a not in chunk_set or b not in chunk_set:
            continue
        w = max(1.0 - dist, 0.0)
        key = (a, b) if a < b else (b, a)
        if key not in edge_weights or edge_weights[key] < w:
            edge_weights[key] = w

    n_directed   = len(edge_rows)
    n_undirected = len(edge_weights)

    nodes_with_edges: set = set()
    for a, b in edge_weights:
        nodes_with_edges.add(a)
        nodes_with_edges.add(b)

    isolated: set = chunk_set - nodes_with_edges
    n_isolated = len(isolated)

    print(f"  {len(all_chunk_ids):,} chunk_periods nodes")
    print(f"  {n_directed:,} directed edges → {n_undirected:,} undirected edges")
    print(f"  {n_isolated:,} isolated nodes (no knn_edges)")
    timings["load"] = time.time() - t0

    # ── Step 2: Build igraph + run Leiden ─────────────────────────────────────
    print_section("Step 2: Leiden Community Detection")
    t0 = time.time()

    connected_nodes = sorted(nodes_with_edges)
    node_idx = {c: i for i, c in enumerate(connected_nodes)}
    n_graph  = len(connected_nodes)

    g = igraph.Graph(n=n_graph, directed=False)
    elist = [(node_idx[a], node_idx[b]) for a, b in edge_weights]
    wlist = [edge_weights[(a, b)] for a, b in edge_weights]
    g.add_edges(elist)
    g.es["weight"] = wlist

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
        n_iterations=-1,
    )

    n_raw      = len(partition)
    modularity = partition.modularity
    print(f"  {n_raw} raw communities, modularity={modularity:.4f}  ({fmt_time(time.time() - t0)})")
    timings["leiden"] = time.time() - t0

    # ── Step 3: Assign cluster IDs ────────────────────────────────────────────
    print_section("Step 3: Assign Cluster IDs")
    t0 = time.time()

    membership = partition.membership
    comm_sizes: dict = {}
    for m in membership:
        comm_sizes[m] = comm_sizes.get(m, 0) + 1

    sorted_comms    = sorted(comm_sizes, key=lambda c: -comm_sizes[c])
    comm_to_cluster = {comm: idx + 1 for idx, comm in enumerate(sorted_comms)}

    chunk_cluster: dict = {}
    for i, node in enumerate(connected_nodes):
        chunk_cluster[node] = comm_to_cluster[membership[i]]
    for node in isolated:
        chunk_cluster[node] = 0

    final_sizes: dict = {}
    for cid in chunk_cluster.values():
        final_sizes[cid] = final_sizes.get(cid, 0) + 1

    junk_clusters = {cid for cid, sz in final_sizes.items()
                     if cid > 0 and sz < junk_min}
    n_clusters    = len([c for c in final_sizes if c > 0])
    n_junk        = len(junk_clusters)
    n_dark_matter = final_sizes.get(0, 0)
    largest_cid   = max((c for c in final_sizes if c > 0),
                        key=lambda c: final_sizes[c], default=0)
    largest_size  = final_sizes.get(largest_cid, 0)

    print(f"  {n_clusters} clusters  ({n_junk} junk, size<{junk_min})")
    print(f"  largest: cluster {largest_cid} = {largest_size} chunks"
          f" ({100 * largest_size / max(len(all_chunk_ids), 1):.1f}%)")
    print(f"  dark matter (isolated): {n_dark_matter}")
    timings["assign"] = time.time() - t0

    # ── Step 4: Write results ─────────────────────────────────────────────────
    if write_back:
        print_section("Step 4: Write chunk_periods + clusters")
        t0 = time.time()

        _run_no_commit(conn, """
            UPDATE embeddings.chunk_periods
            SET cluster_id = NULL, is_dark_matter = false
            WHERE period_start = %s AND corpus_id = %s
        """, (period_start, corpus_id))

        with conn.cursor() as cur:
            cur.execute("""
                CREATE TEMP TABLE _leiden_stage
                    (chunk_id text, cluster_id int)
                ON COMMIT DROP
            """)
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO _leiden_stage VALUES %s",
                list(chunk_cluster.items()),
                page_size=2000,
            )
            cur.execute("CREATE INDEX ON _leiden_stage (chunk_id)")
            cur.execute("""
                UPDATE embeddings.chunk_periods cp
                SET cluster_id     = s.cluster_id,
                    is_dark_matter = (s.cluster_id = 0)
                FROM _leiden_stage s
                WHERE cp.chunk_id::text = s.chunk_id
                  AND cp.period_start   = %s
                  AND cp.corpus_id      = %s
            """, (period_start, corpus_id))
            n_updated = cur.rowcount
            cur.execute("DROP TABLE _leiden_stage")

        period_str = period_label(period_start, time_res, cutoff_yr)
        _run_no_commit(conn, """
            INSERT INTO clusters
                (period, period_start, period_end,
                 cluster_id, size, corpus_type, corpus_id, is_junk)
            SELECT %s, %s, %s, cluster_id, COUNT(*), %s, %s,
                   (cluster_id = ANY(%s::int[]))
            FROM chunk_periods
            WHERE period_start = %s AND corpus_id = %s
              AND cluster_id IS NOT NULL
            GROUP BY cluster_id
            ON CONFLICT (period_start, cluster_id, corpus_id)
            DO UPDATE SET
                size    = EXCLUDED.size,
                is_junk = EXCLUDED.is_junk
        """, (period_str, period_start, period_end,
              corpus_type, corpus_id,
              list(junk_clusters),
              period_start, corpus_id))

        conn.commit()
        timings["write"] = time.time() - t0
        print(f"  {n_updated:,} chunk_periods rows written  ({fmt_time(timings['write'])})")
    else:
        print_section("Step 4: Write Results")
        print("  Skipped (pass --write-back to update chunk_periods)")

    return {
        "n_clusters":      n_clusters,
        "n_junk":          n_junk,
        "n_isolated":      n_isolated,
        "n_dark_matter":   n_dark_matter,
        "n_raw":           n_raw,
        "modularity":      round(modularity, 4),
        "largest_cluster": largest_size,
        "largest_pct":     round(100 * largest_size / max(len(all_chunk_ids), 1), 1),
        "timings":         {k: round(v, 3) for k, v in timings.items()},
    }


# ── Phase 1: Leiden clustering ─────────────────────────────────────────────────

def phase_1_leiden(conn, period_start: date, period_end: date,
                   corpus_type: str, corpus_id: str, cfg: dict) -> dict:
    """Run Leiden community detection for one period."""
    t0 = time.time()
    period = period_label(period_start, cfg.get('_resolution', 'quarterly'), cfg.get('_cutoff_year', 2010))
    stats = leiden_run_period(
        conn, period_start, period_end, corpus_type, corpus_id,
        write_back=True, cfg=cfg,
    )
    elapsed = time.time() - t0
    if VERBOSE:
        print(f"  [leiden]  {stats['n_clusters']} clusters  "
              f"largest={stats['largest_cluster']} ({stats['largest_pct']:.1f}%)  "
              f"dark={stats['n_dark_matter']}  "
              f"mod={stats['modularity']:.4f}  ({fmt_time(elapsed)})")

    # FIX 4: write leiden_modularity to pipe_period_stats for every period
    run(conn,
        "INSERT INTO pipe_period_stats "
        "  (period, period_start, period_end, corpus_id, leiden_modularity) "
        "VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (period_start, corpus_id) DO UPDATE "
        "  SET leiden_modularity = EXCLUDED.leiden_modularity",
        (period, period_start, period_end, corpus_id, stats.get('modularity')))

    return {"leiden": elapsed, "n_clusters": stats.get("n_clusters", 0)}


# ── Per-period runner ──────────────────────────────────────────────────────────

def run_period(conn, period_start: date, period_end: date,
               corpus_type: str, corpus_id: str,
               prev: "str | None", cfg: dict, write_back: bool,
               run_id: "int | None" = None,
               match_sym: str = "✓",
               dry_run: bool = False,
               phases: str = "all") -> dict:
    # phases controls which pipeline phases execute:
    #   "all" — Leiden + geometry + arc_compute_period + stats (default, single-pass)
    #   "12"  — Leiden + geometry only (Pass 1 of bulk two-pass)
    #   "34"  — arc_compute_period + stats only (Pass 2 of bulk two-pass)
    # See main() for when each is used.
    period = period_label(period_start, cfg.get('_resolution', 'quarterly'), cfg.get('_cutoff_year', 2010))

    if VERBOSE:
        print(f"\n{'=' * 60}")
        print(f"  arc_pipeline_v2.py  —  {period} / {corpus_id}")
        print(f"  prev={prev or 'None'}  write-back={write_back}")
        print(f"{'=' * 60}")

    all_timings: dict = {}
    wall_t0 = time.time()

    cells: dict = {}
    if not VERBOSE:
        cells = {"period": period, "leiden": "", "geo": "", "sql": "",
                 "stats": "", "match": match_sym, "total": ""}
        if IS_TTY:
            print(_render_row(cells))

    # ── Dry-run: check all per-period contracts, skip execution ───────────────
    if dry_run:
        contracts = [
            (3, "knn_edges present (Phase 1 Leiden prerequisite)"),
            (2, "chunk_periods present (Phase 2 geometry prerequisite)"),
        ]
        for contract_phase, label in contracts:
            verify_phase_contract(conn, corpus_id, contract_phase, period_start)
            print(f"  [contract {contract_phase} ✓] {label} — {corpus_id} {period_start}")
        return {}

    # Two-pass bulk mode Pass 2: skip Leiden+geometry (already done in Pass 1);
    # geometry writes (cohesion, spread, etc.) are already in the DB.
    if phases == "34":
        cells["leiden"] = "skip"
        cells["geo"]    = "skip"

    if phases != "34":
        # ── Phase 1: knn_edges validation checkpoint ───────────────────────────
        # Log knn_edges row count as a "phase 1" checkpoint before Leiden starts.
        ph1_id = _phase_log_start(conn, corpus_id, 1, "knn_validation", period_start)

    if phases != "34":
        # ── Phase 2: Leiden clustering + Phase 3: geometry ─────────────────────
        # ── Phase 2: Leiden clustering ─────────────────────────────────────────
        # Contract: knn_edges must exist before Leiden can partition the graph.
        verify_phase_contract(conn, corpus_id, 3, period_start)
        if VERBOSE:
            print("\n── Phase 2: Leiden Clustering ──────────────────────────────────")
        ph2_id = _phase_log_start(conn, corpus_id, 2, "leiden", period_start)
        t0 = time.time()
        try:
            with _phase_ctx():
                phase1_result = phase_1_leiden(
                    conn, period_start, period_end, corpus_type, corpus_id, cfg)
            p1_t = time.time() - t0
            # Checksum: COUNT(*) + SUM(size) of non-junk clusters written
            ph2_cksum = _count(conn,
                "SELECT COUNT(*) + COALESCE(SUM(size), 0) FROM clusters "
                "WHERE corpus_id=%s AND period_start=%s AND is_junk=FALSE",
                (corpus_id, period_start))
            ph2_rows = phase1_result.get("n_clusters", 0)
            _phase_log_complete(conn, ph2_id, ph2_rows, ph2_cksum, int(p1_t * 1000))
        except Exception as e:
            _phase_log_failed(conn, ph2_id, str(e))
            raise

        # Complete phase 1 checkpoint now that we know knn_edges count
        knn_cksum = _count(conn,
            "SELECT COUNT(*) FROM knn_edges WHERE corpus_id=%s AND period_start=%s",
            (corpus_id, period_start))
        _phase_log_complete(conn, ph1_id, knn_cksum, knn_cksum, 0)

        all_timings["phase_1_leiden"] = phase1_result
        if VERBOSE:
            print(f"  ✓ phase 2 (leiden)  {fmt_time(p1_t)}")
        else:
            cells["leiden"] = f"{phase1_result.get('n_clusters', 0)}cls {p1_t:.1f}s"
            if IS_TTY:
                _update_row(cells)

        # ── Compute centroids ──────────────────────────────────────────────────
        # Averages chunk embeddings per cluster and writes to clusters.centroid.
        # Must run after Leiden (cluster_id assigned in chunk_periods) and before
        # arc_compute_period (which uses centroid for drift_vector / drift_magnitude).
        # Also runs before the "12" early return so matching has centroids available.
        t0 = time.time()
        with conn.cursor() as cur:
            cur.execute("CALL compute_centroids(%s::date, %s)", (period_start, corpus_id))
        conn.commit()
        all_timings["centroids"] = time.time() - t0
        if VERBOSE:
            print(f"  ✓ centroids  ({fmt_time(all_timings['centroids'])})")
        else:
            cells["centroid"] = f"✓ {all_timings['centroids']:.1f}s"
            if IS_TTY:
                _update_row(cells)

        # Two-pass bulk mode: return after geometry so matching can assign
        # persistent_cluster_id before arc_compute_period calls compute_drift.
        if phases == "12":
            return all_timings

    # ── Phase 4: arc_run_pipeline (SQL master orchestrator) ───────────────────
    # Single call handles: centroids, arc_compute_period (SVD geometry, all
    # geometry batches, temporal Stage 6, period_stats, spectral, scoring,
    # law backtesting), junk-edge cleanup, boundary_percentile, and
    # n_dark_matter_chunks update.  Resolves period metadata (period_end,
    # period_key, prev period_key, corpus_type) internally from DB.
    # arc_run_pipeline issues its own COMMITs internally.
    if VERBOSE:
        print("\n── Phase 4: arc_run_pipeline (SQL) ─────────────────────────────")
    ph4_id = _phase_log_start(conn, corpus_id, 4, "sql_stats", period_start)
    t0 = time.time()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "CALL arc_run_pipeline(%s, %s::date)",
                (corpus_id, period_start),
            )
        conn.autocommit = False
        p3_t = round(time.time() - t0, 2)
        # Checksum: n_clusters from pipe_period_stats row written by arc_run_pipeline
        ph4_cksum = _count(conn,
            "SELECT COALESCE(n_clusters, 0) FROM pipe_period_stats "
            "WHERE corpus_id=%s AND period_start=%s",
            (corpus_id, period_start))
        _phase_log_complete(conn, ph4_id, ph4_cksum, ph4_cksum, int(p3_t * 1000))
    except Exception as e:
        if conn.autocommit:
            conn.autocommit = False
        _phase_log_failed(conn, ph4_id, str(e))
        raise

    all_timings["phase_3_sql"] = p3_t
    if VERBOSE:
        print(f"  ✓ phase 4 (arc_run_pipeline)  {fmt_time(p3_t)}")
    else:
        cells["sql"] = f"✓ {p3_t:.1f}s"
        if IS_TTY:
            _update_row(cells)

    all_timings["wall_seconds"] = round(time.time() - wall_t0, 2)
    wall_s = all_timings["wall_seconds"]
    if VERBOSE:
        print(f"\n  [{period}] total wall: {fmt_time(wall_s)}")
    else:
        cells["total"] = f"{wall_s:.1f}s"
        if IS_TTY:
            _update_row(cells)
        else:
            print(_render_row(cells))

    return all_timings


# ── pipeline_phase_runs recording ─────────────────────────────────────────────

def _phase_log_start(conn, corpus_id: str, phase: int, phase_name: str,
                     period_start=None) -> int:
    """Open a pipeline_phase_runs row; return its id. Uses autocommit-safe pattern."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT log_phase_start(%s, %s, %s, %s)",
            (corpus_id, phase, phase_name, period_start),
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id


def _phase_log_complete(conn, run_id: int, rows: int, checksum: int, ms: int) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT log_phase_complete(%s, %s, %s, %s)",
                    (run_id, rows, checksum, ms))
    conn.commit()


def _phase_log_failed(conn, run_id: int, error: str) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT log_phase_failed(%s, %s)", (run_id, error))
    conn.commit()


def _count(conn, sql, params) -> int:
    """Execute a COUNT query; return the integer result."""
    rows = fetch(conn, sql, params)
    return int(rows[0][0]) if rows and rows[0][0] is not None else 0


# ── pipeline_runs recording ────────────────────────────────────────────────────

def start_pipeline_run(conn, period, corpus_type, corpus_id, k) -> int:
    """Insert a pipeline_runs row with status='running' and return its id."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO pipeline_runs
                (period, corpus_type, corpus_id, k_value, status,
                 step_timings, duration_seconds)
            VALUES (%s, %s, %s, %s, 'running', '{}'::jsonb, 0)
            RETURNING id
        """, (period, corpus_type, corpus_id, k))
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id


def finish_pipeline_run(conn, run_id, timings, wall_s):
    """Update an existing pipeline_runs row to status='complete' with final timings."""
    flat = {}
    for phase, v in timings.items():
        if isinstance(v, dict):
            for step, secs in v.items():
                if isinstance(secs, (int, float)):
                    flat[f"{phase}.{step}"] = secs
        elif isinstance(v, (int, float)):
            flat[phase] = v
    run(conn, """
        UPDATE pipeline_runs
        SET status='complete', step_timings=%s::jsonb,
            duration_seconds=%s, finished_at=now()
        WHERE id=%s
    """, (json.dumps(flat), round(wall_s, 2), run_id))


# ── Main ───────────────────────────────────────────────────────────────────────
# NOTE: Cluster matching (run_matching / _PersistentIdRegistry / etc.) was
# removed in migration 128/129.  Matching now runs as a single DB call:
#   CALL run_cluster_matching(corpus_id)
# See migrations/128_run_cluster_matching_proc.sql.


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--period-start", help="Single period start YYYY-MM-DD")
    grp.add_argument("--all-periods",  action="store_true",
                     help="Run all periods with chunk_periods rows for this corpus")
    grp.add_argument("--from-period",  help="Run all periods >= YYYY-MM-DD in chronological order")

    ap.add_argument("--corpus-id",       required=True)
    ap.add_argument("--corpus-type",     default="patents")
    ap.add_argument("--write-back",      action="store_true",
                    help="Pass write-back flag to AGE LP; records to pipeline_runs")
    ap.add_argument("--no-pipeline-run", action="store_true",
                    help="Skip recording to pipeline_runs")
    ap.add_argument("--skip-matching",   action="store_true",
                    help="Skip Phase 5 arc_match_clusters (useful for single-period debug runs).")
    ap.add_argument("--verbose",         action="store_true",
                    help="Print per-phase detail (default: compact table display).")
    ap.add_argument("--dry-run",         action="store_true",
                    help="Check all phase contracts without executing any pipeline phases. "
                         "Exits 0 if all contracts pass, 1 on first violation.")

    args = ap.parse_args()
    global VERBOSE
    VERBOSE = args.verbose
    dry_run = args.dry_run
    if dry_run:
        args.skip_matching = True
        args.no_pipeline_run = True

    conn = get_conn()

    # Read config from DB (best-effort numeric coercion)
    cfg: dict = {}
    for k, v in fetch(conn, "SELECT key, value FROM pipeline_config"):
        try:
            cfg[k] = int(v)
        except (ValueError, TypeError):
            try:
                cfg[k] = float(v)
            except (ValueError, TypeError):
                cfg[k] = v

    k_value = int(cfg.get("KNN_K", 8))

    # Resolve cutoff year from corpus time_resolution → pipeline_config key.
    # Falls back to PERIOD_CUTOFF_YEAR (existing key, default 2010) if no row
    # or if the corpus has no time_resolution set.
    _res_rows = fetch(conn,
        "SELECT resolution FROM v_run_lookup WHERE corpus_id = %s", (args.corpus_id,))
    _time_res = _res_rows[0][0] if _res_rows and _res_rows[0][0] else None
    cfg['_resolution'] = _time_res if _time_res else 'quarterly'
    _cutoff_key = {
        'weekly':    'WEEKLY_CUTOFF_YEAR',
        'monthly':   'MONTHLY_CUTOFF_YEAR',
        'quarterly': 'QUARTERLY_CUTOFF_YEAR',
        'annual':    'ANNUAL_CUTOFF_YEAR',
    }.get(_time_res, 'PERIOD_CUTOFF_YEAR')
    # Check for corpus-specific overrides before falling back to the global key.
    # Resolution: QUARTERLY_CUTOFF_YEAR_G06N_QUARTERLY > QUARTERLY_CUTOFF_YEAR_G06N > QUARTERLY_CUTOFF_YEAR
    # Needed because quarterly corpora have different pre-bucket boundaries:
    #   H01L_quarterly=1990, G06N_quarterly=2010, G06N_3/5/7/10/20_quarterly=2000.
    _corpus_full_key   = f"{_cutoff_key}_{args.corpus_id.upper()}"
    _corpus_prefix_key = f"{_cutoff_key}_{args.corpus_id.split('_')[0].upper()}"
    cfg['_cutoff_year'] = int(cfg.get(_corpus_full_key,
                                      cfg.get(_corpus_prefix_key,
                                              cfg.get(_cutoff_key,
                                                      cfg.get('PERIOD_CUTOFF_YEAR', 2010)))))

    # Determine periods
    if args.all_periods:
        period_rows = fetch(conn, """
            SELECT DISTINCT period_start, period_end
            FROM chunk_periods
            WHERE corpus_id = %s
            ORDER BY period_start
        """, (args.corpus_id,))
    elif args.from_period:
        cutoff = date.fromisoformat(args.from_period)
        period_rows = fetch(conn, """
            SELECT DISTINCT period_start, period_end
            FROM chunk_periods
            WHERE corpus_id = %s AND period_start >= %s
            ORDER BY period_start
        """, (args.corpus_id, cutoff))
    else:
        ps = date.fromisoformat(args.period_start)
        pe_row = fetch(conn,
            "SELECT MAX(period_end) FROM chunk_periods "
            "WHERE period_start=%s AND corpus_id=%s",
            (ps, args.corpus_id))
        pe = pe_row[0][0] if pe_row and pe_row[0][0] else ps
        period_rows = [(ps, pe)]

    # Seed prev from the most recent completed period before this run so that
    # temporal metrics (drift, velocity, jerk, BPR, membership_churn, etc.)
    # are correctly computed for the first period processed.  Without seeding,
    # a re-run of --all-periods would see prev=None for the first period and
    # leave all temporal metrics NULL even when valid prior data exists.
    # For a fresh corpus with no prior pipeline_runs rows the query returns
    # nothing and prev stays None, which is correct.
    _pre_bucket = f"pre_{cfg['_cutoff_year']}"
    prev: "str | None" = None
    if args.all_periods and period_rows:
        first_label = period_label(period_rows[0][0], cfg.get('_resolution', 'quarterly'), cfg['_cutoff_year'])
        prev_row = fetch(conn, """
            SELECT period FROM pipeline_runs
            WHERE corpus_id = %s AND status = 'complete'
              AND period != %s
              AND period < %s
            ORDER BY period DESC
            LIMIT 1
        """, (args.corpus_id, _pre_bucket, first_label))
        if prev_row:
            prev = prev_row[0][0]
    elif args.from_period and period_rows:
        cutoff = date.fromisoformat(args.from_period)
        prev_row = fetch(conn, """
            SELECT period FROM pipeline_runs
            WHERE corpus_id = %s AND status = 'complete'
              AND period != %s
              AND period < %s
            ORDER BY period DESC
            LIMIT 1
        """, (args.corpus_id, _pre_bucket, period_label(cutoff, cfg.get('_resolution', 'quarterly'), cfg['_cutoff_year'])))
        if prev_row:
            prev = prev_row[0][0]

    wall_t0 = time.time()
    match_sym   = "-" if args.skip_matching else "✓"
    total_cls   = 0

    if not VERBOSE and period_rows:
        _print_tbl_header()

    # For bulk (--all-periods / --from-period) runs: use a two-pass approach so
    # persistent_cluster_id is populated by matching before arc_compute_period
    # calls compute_drift.  compute_drift requires persistent_cluster_id to join
    # consecutive cluster pairs — without it, every period writes zero drift rows.
    #
    # Single --period-start runs use the single-pass path because prior periods
    # already have persistent_cluster_id set from earlier matching runs.
    is_bulk = bool(period_rows
                   and (args.all_periods or args.from_period)
                   and not args.skip_matching)

    try:
        try:
            if is_bulk:
                # ── Pass 1: Leiden + geometry for all periods ──────────────────
                for period_start, period_end in period_rows:
                    run_period(
                        conn, period_start, period_end,
                        args.corpus_type, args.corpus_id,
                        None, cfg, args.write_back, run_id=None,
                        match_sym=match_sym, dry_run=dry_run, phases="12",
                    )

                # ── Matching (between passes — before arc_compute_period) ───────
                if not args.skip_matching:
                    ph6_id = _phase_log_start(conn, args.corpus_id, 6, "matching")
                    t0_match = time.time()
                    try:
                        conn.autocommit = True
                        with conn.cursor() as cur:
                            cur.execute("CALL run_cluster_matching(%s)", (args.corpus_id,))
                        conn.autocommit = False
                        ph6_rows = _count(conn,
                            "SELECT COUNT(*) FROM clusters "
                            "WHERE corpus_id=%s AND persistent_cluster_id IS NOT NULL",
                            (args.corpus_id,))
                        _phase_log_complete(conn, ph6_id, ph6_rows, ph6_rows,
                                            int((time.time() - t0_match) * 1000))
                    except Exception as e:
                        if conn.autocommit:
                            conn.autocommit = False
                        _phase_log_failed(conn, ph6_id, str(e))
                        print(f"  [phase6] WARNING: run_cluster_matching failed: {e}")
                        print(f"  [phase6] persistent_cluster_id may be incomplete.")
                        print(f"  [phase6] Re-run with --from-period to repair.")

                # ── Pass 2: arc_compute_period + stats for all periods ──────────
                # persistent_cluster_id is now set; compute_drift will find matches.
                for period_start, period_end in period_rows:
                    period = period_label(period_start, cfg.get('_resolution', 'quarterly'), cfg['_cutoff_year'])
                    run_id = None
                    if not args.no_pipeline_run and args.write_back:
                        run_id = start_pipeline_run(
                            conn, period, args.corpus_type, args.corpus_id, k_value)

                    timings = run_period(
                        conn, period_start, period_end,
                        args.corpus_type, args.corpus_id,
                        prev, cfg, args.write_back, run_id,
                        match_sym=match_sym, dry_run=dry_run, phases="34",
                    )

                    total_cls += timings.get("phase_1_leiden", {}).get("n_clusters", 0)
                    if run_id is not None:
                        finish_pipeline_run(conn, run_id, timings, timings.get("wall_seconds", 0))
                    prev = period

            else:
                # ── Single-pass: single period or --skip-matching ──────────────
                # Prior periods already have persistent_cluster_id from earlier
                # matching runs, so arc_compute_period computes drift correctly.
                for period_start, period_end in period_rows:
                    period = period_label(period_start, cfg.get('_resolution', 'quarterly'), cfg['_cutoff_year'])

                    # Create pipeline_runs row before Phase 4 so arc_compute_period
                    # can write step timings to pipeline_run_steps.
                    run_id = None
                    if not args.no_pipeline_run and args.write_back:
                        run_id = start_pipeline_run(
                            conn, period, args.corpus_type, args.corpus_id, k_value)

                    timings = run_period(
                        conn, period_start, period_end,
                        args.corpus_type, args.corpus_id,
                        prev, cfg, args.write_back, run_id,
                        match_sym=match_sym,
                        dry_run=dry_run,
                    )

                    total_cls += timings.get("phase_1_leiden", {}).get("n_clusters", 0)
                    if run_id is not None:
                        finish_pipeline_run(conn, run_id, timings, timings.get("wall_seconds", 0))
                    prev = period

        except RuntimeError as contract_err:
            conn.rollback()
            print(f"\n[CONTRACT VIOLATION] {contract_err}", file=sys.stderr)
            sys.exit(1)
        except Exception:
            conn.rollback()
            raise

        if not is_bulk:
            # ── Phase 6: Persistent cluster ID matching ───────────────────────
            # Single-pass path: matching runs after the period loop.
            # Use --skip-matching to suppress (e.g. single-period debug runs).
            if not args.skip_matching:
                ph6_id = _phase_log_start(conn, args.corpus_id, 6, "matching")
                t0_match = time.time()
                try:
                    conn.autocommit = True
                    with conn.cursor() as cur:
                        cur.execute("CALL run_cluster_matching(%s)", (args.corpus_id,))
                    conn.autocommit = False
                    ph6_rows = _count(conn,
                        "SELECT COUNT(*) FROM clusters "
                        "WHERE corpus_id=%s AND persistent_cluster_id IS NOT NULL",
                        (args.corpus_id,))
                    _phase_log_complete(conn, ph6_id, ph6_rows, ph6_rows,
                                        int((time.time() - t0_match) * 1000))
                except Exception as e:
                    if conn.autocommit:
                        conn.autocommit = False
                    _phase_log_failed(conn, ph6_id, str(e))
                    print(f"  [phase6] WARNING: run_cluster_matching failed: {e}")
                    print(f"  [phase6] persistent_cluster_id may be incomplete.")
                    print(f"  [phase6] Re-run with --from-period to repair.")

        if not args.skip_matching:
            # ── Phase 5.5: Split child detection ──────────────────────────────
            # Must run after matching so match_type and persistent_cluster_id
            # are populated.  Uses centroid proximity to flag BIRTH clusters
            # whose nearest prior cluster also has a CONTINUATION sibling in
            # the same period (= split event).
            if args.write_back:
                print("\n── Phase 5.5: Split child detection ────────────────────────────")
                t0_split = time.time()
                n_split_total = 0
                with conn.cursor() as cur:
                    for period_start, _period_end in period_rows:
                        cur.execute(
                            "SELECT compute_split_children(%s::date, %s)",
                            (period_start, args.corpus_id),
                        )
                        result = cur.fetchone()[0]
                        n_split_total += result.get("n_updated", 0)
                conn.commit()
                print(f"  split children flagged: {n_split_total} "
                      f"| {fmt_time(time.time() - t0_split)}")

                # ── Merger targets (migration 127) ─────────────────────────────────
                # compute_merger_targets uses pgvector LATERAL <=> to find stable
                # clusters nearest to dead clusters from the prior period.
                # Runs after fix_lifecycle_flags (needs accurate is_dead) and after
                # compute_split_children (counts is_split_child for n_splits).
                print("\n── Merger targets ───────────────────────────────────────────────")
                t0_merge = time.time()
                n_merge_total, n_split_proc_total = 0, 0
                with conn.cursor() as cur:
                    for period_start, _period_end in period_rows:
                        cur.execute(
                            "CALL compute_merger_targets(%s::date, %s, NULL::integer, NULL::integer)",
                            (period_start, args.corpus_id),
                        )
                        row = cur.fetchone()
                        if row:
                            n_merge_total     += row[0] or 0
                            n_split_proc_total += row[1] or 0
                conn.commit()
                print(f"  mergers={n_merge_total}  splits(proc)={n_split_proc_total} "
                      f"| {fmt_time(time.time() - t0_merge)}")

                # ── Velocity direction stability (requires persistent_cluster_id) ──
                # compute_velocity_direction_stability computes mean cosine similarity
                # of consecutive centroid displacement vectors over STABILITY_WINDOW
                # periods. Runs after matching so persistent_cluster_id is available.
                print("\n── Velocity direction stability ─────────────────────────────────")
                t0_vel = time.time()
                n_vel_total = 0
                with conn.cursor() as cur:
                    for period_start, _period_end in period_rows:
                        cur.execute(
                            "SELECT compute_velocity_direction_stability(%s::date, %s)",
                            (period_start, args.corpus_id),
                        )
                        result = cur.fetchone()[0]
                        if result:
                            n_vel_total += result.get("n_updated", 0)
                conn.commit()
                print(f"  velocity stability updated: {n_vel_total} clusters "
                      f"| {fmt_time(time.time() - t0_vel)}")

    finally:
        conn.close()

    wall_total = time.time() - wall_t0
    n_periods = len(period_rows)
    if dry_run:
        print(f"\n[DRY RUN] All phase contracts passed for {n_periods} period(s) of {args.corpus_id}")
    elif VERBOSE:
        print(f"\n  Total wall time: {fmt_time(wall_total)}")
    else:
        print(f"\nTotal: {n_periods} periods | {total_cls} clusters | wall {fmt_time(wall_total)}")


if __name__ == "__main__":
    main()

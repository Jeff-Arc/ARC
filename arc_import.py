#!/usr/bin/env python3
"""
arc_import.py — Import cloud script output into arc_v5.

Usage:
    # Full corpus import
    python3 arc_import.py --corpus G06N_quarterly \
        --input /home/jeff/arc/data/cloud_out_G06N

    # Incremental (new period only — skips existing rows via ON CONFLICT DO NOTHING)
    python3 arc_import.py --corpus G06N_quarterly \
        --input /home/jeff/arc/data/cloud_out_G06N --incremental

    # Export T-1 context file for cloud script's next incremental run
    python3 arc_import.py --corpus G06N_quarterly --export-context

TSV layout under --input:
    {input}/import/{corpus_id}/chunk_periods.tsv
    {input}/import/{corpus_id}/chunk_graph.tsv
    {input}/import/{corpus_id}/centroids.tsv
    {input}/import/{corpus_id}/f_edge.tsv
    {input}/import/{corpus_id}/f_void.tsv
    {input}/import/{corpus_id}/f_gap.tsv
    {input}/import/{corpus_id}/f_period.tsv
    {input}/import/{corpus_id}/matching.tsv

Context export writes to:
    /home/jeff/arc/data/cloud_in/{corpus_id}/context_{corpus_id}_prev.tsv
    Columns: chunk_id, cluster_id, core_number, period_start
"""

import argparse
import csv
import io
import os
import sys
import time
from datetime import datetime

import psycopg2

# ─── DB connection ────────────────────────────────────────────────────────────

DB_PARAMS = dict(
    host=os.environ.get("PGHOST", "/var/run/postgresql"),
    dbname=os.environ.get("PGDATABASE", "arc_v5"),
    user=os.environ.get("PGUSER", "jeff"),
    password=os.environ.get("PGPASSWORD", ""),
)

# ─── Table definitions ────────────────────────────────────────────────────────
# Maps TSV filename → (table_name, ordered_column_list, transform_fn_or_None)
# transform_fn receives a dict row and returns a list of values in column order.

def _row_chunk_periods(r):
    return [
        r["corpus_id"],
        r["period_start"],
        r["chunk_id"],
        _int(r["cluster_id"]),
        _real(r["point_density"]),
        _real(r["distance_to_centroid"]),
        _real(r["energy"]),
        _real(r["boundary_score"]),
        _real(r["intrinsic_dim"]),
        _real(r.get("uncertainty")),
        _real(r.get("boundary_proximity")),
        _bool(r.get("is_dark_matter")),
        _int(r.get("membership_volatility")),
        _real(r.get("belief_persistence_score")),
    ]

def _row_chunk_graph(r):
    return [
        r["corpus_id"],
        r["period_start"],
        r["chunk_id"],
        _real(r["betweenness_centrality"]),
        _int(r["core_number"]),
        _real(r["clustering_coeff"]),
        _int(r.get("triangle_count")),
        _int(r.get("degree")),
        _real(r.get("pagerank")),
        _real(r.get("eigenvector_centrality")),
        _real(r.get("katz_centrality")),
        _real(r.get("harmonic_centrality")),
        _real(r.get("in_degree_centrality")),
    ]

def _row_cloud_chunk_measures(r):
    return [
        r["corpus_id"],
        r["period_start"],
        r["chunk_id"],
        _int(r["cluster_id"]),
        _real(r.get("point_density")),
        _real(r.get("distance_to_centroid")),
        _real(r.get("energy")),
        _real(r.get("boundary_score")),
        _real(r.get("intrinsic_dim")),
        _real(r.get("uncertainty")),
        _real(r.get("boundary_proximity")),
        _bool(r.get("is_dark_matter")),
        _int(r.get("membership_volatility")),
        _real(r.get("belief_persistence_score")),
        _real(r.get("betweenness_centrality")),
        _int(r.get("core_number")),
        _real(r.get("clustering_coeff")),
        _int(r.get("triangle_count")),
        _int(r.get("degree")),
        _real(r.get("pagerank")),
        _real(r.get("eigenvector_centrality")),
        _real(r.get("katz_centrality")),
        _real(r.get("harmonic_centrality")),
        _real(r.get("in_degree_centrality")),
    ]

def _row_centroids(r):
    return [
        r["corpus_id"],
        r["period_start"],
        _int(r["cluster_id"]),
        _vec(r["centroid"]),
        _real(r.get("elongation_ratio")),
        _real(r.get("volume_estimate")),
        _real(r.get("skewness_pc1")),
        _real(r.get("kurtosis_pc1")),
        _real(r.get("skewness_pc2")),
        _real(r.get("kurtosis_pc2")),
        _real(r.get("mean_density")),
        _real(r.get("outlier_fraction")),
        _real(r.get("mean_uncertainty")),
        _real(r.get("boundary_sharpness")),
        _int(r.get("n_attractors")),
        _int(r.get("n_saddle_points")),
        _real(r.get("avg_path_length")),
        _real(r.get("propagation_speed")),
        _real(r.get("field_surprise_index")),
    ]

def _row_f_edge(r):
    return [
        r["corpus_id"],
        r["period_start"],
        _int(r["cluster_a"]),
        _int(r["cluster_b"]),
        _real(r["connection_weight"]),
        _real(r["semantic_overlap_a_to_b"]),
        _real(r["semantic_overlap_b_to_a"]),
        _int(r["n_shared_edges"]),
        _real(r.get("semantic_overlap_max")),
        _int(r.get("n_bridge_chunks")),
    ]

def _row_f_void(r):
    return [
        r["corpus_id"],
        r["period_start"],
        _int(r["cluster_a"]),
        _int(r["cluster_b"]),
        _real(r["centroid_distance"]),
    ]

def _row_f_gap(r):
    return [
        r["corpus_id"],
        r["period_start"],
        _int(r["cluster_a"]),
        _int(r["cluster_b"]),
        _real(r["centroid_distance"]),
        _real(r["boundary_distance"]),
        _real(r["void_depth"]),
        _int(r["n_dark_matter_near_midpoint"]),
        _real(r["dark_matter_density_near_void"]),
        _real(r["fringe_density_a"]),
        _real(r["fringe_density_b"]),
        _real(r["connection_weight"]),
        _real(r["connection_weight_ab"]),
        _real(r["connection_weight_ba"]),
        _int(r["size_a"]),
        _int(r["size_b"]),
        _bool(r["nearest_chunk_is_dark_matter"]),
        _null(r["gap_type"]),
    ]

def _row_f_period(r):
    return [
        r["corpus_id"],
        r["period_start"],
        _real(r["algebraic_connectivity"]),
        _real(r["spectral_gap"]),
        _int(r["n_clusters"]),
        _int(r["n_chunks"]),
        _real(r.get("phase_transition_score")),
        _real(r.get("leiden_modularity")),
        _int(r.get("n_dark_matter_chunks")),
        _int(r.get("n_births")),
        _int(r.get("n_deaths")),
        _int(r.get("n_reborn")),
        _int(r.get("n_split_child")),
    ]

def _row_matching(r):
    return [
        r["corpus_id"],
        r["period_start"],
        _int(r["cluster_id"]),
        _null(r["persistent_cluster_id"]),
        _null(r["match_type"]),
        _real(r["composite_score"]),
        _real(r["centroid_sim"]),
        _real(r["chunk_overlap"]),
        _real(r["core_continuity"]),
        _int(r["reborn_after_periods"]),
        _null(r["reborn_from_period"]),
        _real(r.get("membership_churn")),
        _vec(r.get("drift_vector")),
        _real(r.get("velocity_alignment")),
    ]

TABLE_MAP = {
    "chunk_periods.tsv": (
        "cloud_chunk_periods",
        ["corpus_id","period_start","chunk_id","cluster_id","point_density",
         "distance_to_centroid","energy","boundary_score","intrinsic_dim",
         "uncertainty","boundary_proximity","is_dark_matter",
         "membership_volatility","belief_persistence_score"],
        _row_chunk_periods,
    ),
    "chunk_graph.tsv": (
        "cloud_chunk_graph",
        ["corpus_id","period_start","chunk_id","betweenness_centrality",
         "core_number","clustering_coeff","triangle_count","degree","pagerank",
         "eigenvector_centrality","katz_centrality","harmonic_centrality",
         "in_degree_centrality"],
        _row_chunk_graph,
    ),
    "cloud_chunk_measures.tsv": (
        "cloud_chunk_measures",
        ["corpus_id","period_start","chunk_id","cluster_id","point_density",
         "distance_to_centroid","energy","boundary_score","intrinsic_dim",
         "uncertainty","boundary_proximity","is_dark_matter",
         "membership_volatility","belief_persistence_score",
         "betweenness_centrality","core_number","clustering_coeff",
         "triangle_count","degree","pagerank","eigenvector_centrality",
         "katz_centrality","harmonic_centrality","in_degree_centrality"],
        _row_cloud_chunk_measures,
    ),
    "centroids.tsv": (
        "cloud_centroids",
        ["corpus_id","period_start","cluster_id","centroid",
         "elongation_ratio","volume_estimate","skewness_pc1","kurtosis_pc1",
         "skewness_pc2","kurtosis_pc2","mean_density","outlier_fraction",
         "mean_uncertainty","boundary_sharpness","n_attractors","n_saddle_points",
         "avg_path_length","propagation_speed","field_surprise_index"],
        _row_centroids,
    ),
    "f_edge.tsv": (
        "cloud_f_edge",
        ["corpus_id","period_start","cluster_a","cluster_b","connection_weight",
         "semantic_overlap_a_to_b","semantic_overlap_b_to_a","n_shared_edges",
         "semantic_overlap_max","n_bridge_chunks"],
        _row_f_edge,
    ),
    "f_void.tsv": (
        "cloud_f_void",
        ["corpus_id","period_start","cluster_a","cluster_b","centroid_distance"],
        _row_f_void,
    ),
    "f_gap.tsv": (
        "cloud_f_gap",
        ["corpus_id","period_start","cluster_a","cluster_b",
         "centroid_distance","boundary_distance","void_depth",
         "n_dark_matter_near_midpoint","dark_matter_density_near_void",
         "fringe_density_a","fringe_density_b",
         "connection_weight","connection_weight_ab","connection_weight_ba",
         "size_a","size_b","nearest_chunk_is_dark_matter","gap_type"],
        _row_f_gap,
    ),
    "f_period.tsv": (
        "cloud_f_period",
        ["corpus_id","period_start","algebraic_connectivity","spectral_gap",
         "n_clusters","n_chunks","phase_transition_score","leiden_modularity",
         "n_dark_matter_chunks","n_births","n_deaths","n_reborn","n_split_child"],
        _row_f_period,
    ),
    "matching.tsv": (
        "cloud_matching",
        ["corpus_id","period_start","cluster_id","persistent_cluster_id",
         "match_type","composite_score","centroid_sim","chunk_overlap",
         "core_continuity","reborn_after_periods","reborn_from_period",
         "membership_churn","drift_vector","velocity_alignment"],
        _row_matching,
    ),
}

# ─── Type coercions ───────────────────────────────────────────────────────────

def _null(v):
    """Return None for NULL/empty string values."""
    if v in (None, "", "NULL", "\\N"):
        return None
    return v

def _real(v):
    if v in (None, "", "NULL", "\\N"):
        return None
    try:
        return float(v)
    except ValueError:
        return None

def _int(v):
    if v in (None, "", "NULL", "\\N"):
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return int(float(v))  # handles "125.0", "0.0" etc.
        except (ValueError, OverflowError):
            return None

def _vec(v):
    """Convert space-separated float string to pgvector text: '[v1,v2,...]'"""
    if v in (None, "", "NULL", "\\N"):
        return None
    return "[" + ",".join(v.strip().split()) + "]"

def _bool(v):
    """Convert 'true'/'false'/'1'/'0' to Python bool, None for NULL."""
    if v in (None, "", "NULL", "\\N"):
        return None
    return str(v).lower() in ("true", "1", "t", "yes")

# ─── COPY helpers ─────────────────────────────────────────────────────────────

def _to_tsv_value(v):
    """Encode a Python value as a TSV cell for psycopg2 copy_expert."""
    if v is None:
        return "\\N"
    s = str(v)
    # Escape backslash, tab, newline
    s = s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")
    return s

def rows_to_tsv_buffer(rows):
    """Convert a list of value-lists to a TSV StringIO for COPY FROM STDIN."""
    buf = io.StringIO()
    for row in rows:
        buf.write("\t".join(_to_tsv_value(v) for v in row) + "\n")
    buf.seek(0)
    return buf

def load_tsv(path, transform_fn):
    """Read TSV and return list of transformed value-lists."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(transform_fn(r))
    return rows

# ─── Import logic ─────────────────────────────────────────────────────────────

def import_corpus(corpus_id, input_dir, incremental=False, replace=False):
    tsv_dir = os.path.join(input_dir, "import", corpus_id)
    if not os.path.isdir(tsv_dir):
        print(f"ERROR: TSV directory not found: {tsv_dir}")
        sys.exit(1)

    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = False

    totals = {}
    errors = []

    if replace:
        mode_str = "replace (DELETE + INSERT)"
    elif incremental:
        mode_str = "incremental (ON CONFLICT DO NOTHING)"
    else:
        mode_str = "full (ON CONFLICT DO NOTHING)"

    print(f"\n{'='*60}")
    print(f"Importing corpus: {corpus_id}")
    print(f"Source dir:       {tsv_dir}")
    print(f"Mode:             {mode_str}")
    print(f"{'='*60}\n")

    for tsv_file, (table, columns, transform_fn) in TABLE_MAP.items():
        path = os.path.join(tsv_dir, tsv_file)
        if not os.path.exists(path):
            print(f"  WARN: {tsv_file} not found — skipping")
            continue

        t0 = time.time()
        try:
            rows = load_tsv(path, transform_fn)
            n = len(rows)

            if n == 0:
                print(f"  {table:<30} 0 rows (empty file)")
                totals[table] = 0
                continue

            buf = rows_to_tsv_buffer(rows)
            col_list = ", ".join(columns)

            # Stage through a temp table to safely handle TSV files that
            # contain duplicate PK rows (the cloud script can emit dupes in
            # chunk_periods).  LIKE without INCLUDING ALL copies only column
            # definitions — no PK/unique constraints on the temp table, so
            # COPY accepts all rows including intra-file duplicates.
            tmp = f"_tmp_{table}"
            with conn.cursor() as cur:
                cur.execute(
                    f"CREATE TEMP TABLE {tmp} (LIKE {table}) ON COMMIT DROP"
                )
                cur.copy_expert(
                    f"COPY {tmp} ({col_list}) FROM STDIN",
                    buf,
                )
                if replace:
                    print(f"  Replacing existing data for corpus {corpus_id} in {table}")
                    cur.execute(
                        f"DELETE FROM {table} WHERE corpus_id = %s",
                        (corpus_id,),
                    )
                    # No ON CONFLICT needed — DELETE already cleared this corpus.
                    cur.execute(f"""
                        INSERT INTO {table} ({col_list})
                        SELECT {col_list} FROM {tmp}
                    """)
                else:
                    print(f"  Appending to {table} (use --replace to overwrite)")
                    cur.execute(f"""
                        INSERT INTO {table} ({col_list})
                        SELECT {col_list} FROM {tmp}
                        ON CONFLICT DO NOTHING
                    """)
                inserted = cur.rowcount

            conn.commit()
            elapsed = time.time() - t0
            totals[table] = inserted
            print(f"  {table:<30} {inserted:>7,} rows  ({elapsed:.2f}s)")

        except Exception as e:
            conn.rollback()
            msg = f"  ERROR loading {tsv_file}: {e}"
            print(msg)
            errors.append(msg)

    # ── Refresh materialized views ────────────────────────────────────────────
    views = ["cluster_snapshot", "field_snapshot", "persistent_voids"]

    print()
    for view in views:
        t0 = time.time()
        try:
            # Try CONCURRENTLY first (requires unique index and existing data)
            with conn.cursor() as cur:
                cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}")
            conn.commit()
            elapsed = time.time() - t0
            print(f"  REFRESH {view:<30} ({elapsed:.2f}s)")
        except psycopg2.errors.ObjectNotInPrerequisiteState:
            # No unique index or first populate — fall back to non-concurrent
            conn.rollback()
            try:
                with conn.cursor() as cur:
                    cur.execute(f"REFRESH MATERIALIZED VIEW {view}")
                conn.commit()
                elapsed = time.time() - t0
                print(f"  REFRESH {view:<30} ({elapsed:.2f}s) [non-concurrent]")
            except Exception as e:
                conn.rollback()
                print(f"  FATAL: REFRESH {view} failed: {e}")
                conn.close()
                sys.exit(1)
        except Exception as e:
            conn.rollback()
            print(f"  FATAL: REFRESH {view} failed: {e}")
            conn.close()
            sys.exit(1)

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Import complete: {corpus_id}")
    total_rows = sum(totals.values())
    print(f"Total rows loaded: {total_rows:,}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    print(f"{'─'*60}\n")

# ─── Context export ───────────────────────────────────────────────────────────

def export_context(corpus_id):
    """
    Export the most recent period's chunk assignments for use as T-1 context
    in the next incremental cloud script run.

    Output: /home/jeff/arc/data/cloud_in/{corpus_id}/context_{corpus_id}_prev.tsv
    Columns: chunk_id, cluster_id, core_number, period_start
    """
    out_dir = f"/home/jeff/arc/data/cloud_in/{corpus_id}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"context_{corpus_id}_prev.tsv")

    conn = psycopg2.connect(**DB_PARAMS)
    query = """
        SELECT
            cm.chunk_id,
            cm.cluster_id,
            cm.core_number,
            cm.period_start
        FROM cloud_chunk_measures cm
        WHERE cm.corpus_id    = %s
          AND cm.period_start = (
            SELECT MAX(period_start)
            FROM cloud_chunk_measures
            WHERE corpus_id = %s
          )
        ORDER BY cm.chunk_id
    """
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(query, (corpus_id, corpus_id))
        rows = cur.fetchall()
    conn.close()

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["chunk_id", "cluster_id", "core_number", "period_start"])
        for row in rows:
            writer.writerow([
                row[0],
                row[1] if row[1] is not None else "",
                row[2] if row[2] is not None else "",
                row[3].isoformat() if row[3] is not None else "",
            ])

    elapsed = time.time() - t0
    print(f"Context export: {len(rows):,} rows → {out_path}  ({elapsed:.2f}s)")

# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Import cloud script output into arc_v5")
    parser.add_argument("--corpus", required=True, help="Corpus ID (e.g. G06N_quarterly)")
    parser.add_argument("--input", help="Input root dir containing import/{corpus}/*.tsv")
    parser.add_argument("--incremental", action="store_true",
                        help="Use ON CONFLICT DO NOTHING (skip existing rows)")
    parser.add_argument("--replace", action="store_true",
                        help="DELETE existing corpus rows before importing (full overwrite)")
    parser.add_argument("--export-context", action="store_true",
                        help="Export T-1 context TSV for incremental cloud script run")
    args = parser.parse_args()

    if args.export_context:
        export_context(args.corpus)
    elif args.input:
        import_corpus(args.corpus, args.input,
                      incremental=args.incremental, replace=args.replace)
    else:
        parser.error("--input is required unless --export-context is used")

if __name__ == "__main__":
    main()

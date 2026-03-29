#!/usr/bin/env python3
"""
arc_cloud_export.py — Export corpus data from arc_v5 to R2 for cloud pipeline

Exports data_documents to chunks TSV format and generates config.json,
then uploads both to Cloudflare R2 for arc_cloud_run.py consumption.

Usage:
    python3 arc_cloud_export.py --corpus C30B_quarterly
    python3 arc_cloud_export.py --corpus all
    python3 arc_cloud_export.py --corpus C30B_quarterly --force   # re-upload even if exists
    python3 arc_cloud_export.py --corpus C30B_quarterly --local-only  # export locally, skip R2

Output format (chunks TSV, no header):
    document_id\tidea_text\tcontent_date

Config.json fields:
    corpus_id, embedding_model, k, leiden_res, resolution, year_from,
    void_distance_threshold, junk_threshold, void_radius, fringe_radius

Environment:
    PGHOST, PGUSER, PGPASSWORD, PGDATABASE (or defaults to arc_v5 on Hetzner)
    R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET (from .env)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import boto3
import psycopg2

# ─── Paths ────────────────────────────────────────────────────────────────────

ARC_DIR    = Path("/root/arc")
ENV_FILE   = ARC_DIR / ".env"
EXPORT_DIR = Path("/tmp/cloud_in")

# ─── Corpus configs ──────────────────────────────────────────────────────────
# Sparse corpora get higher void_distance_threshold (wider gaps between clusters)

SPARSE_CORPORA = {
    "C23C_quarterly", "C30B_quarterly", "G01B_quarterly", "G01N_quarterly",
    "A01B_quarterly", "A01C_quarterly", "A01D_quarterly",
}

# Corpora with known custom leiden_res (from arc_v4 sys_run_config)
CUSTOM_LEIDEN_RES = {
    "H01L_quarterly": 2.0,
    "G06F_quarterly": 2.0,
    "G06N_quarterly": 1.5,
}

# Year-from defaults by corpus (from fallback configs)
YEAR_FROM_MAP = {
    "H01L_quarterly": 1976,
    "G06N_quarterly": 1976,
    "G06F_quarterly": 1990,
    "G01N_quarterly": 1990,
    "G01B_quarterly": 1990,
    "G02B_quarterly": 1990,
    "C23C_quarterly": 1990,
    "C30B_quarterly": 1990,
    "A61P9_quarterly": 1995,
    "A61P25_quarterly": 1995,
    "C12N15_quarterly": 1995,
    "A61K38_quarterly": 1995,
    "longevity_patents_quarterly": 1995,
}

DEFAULT_YEAR_FROM = 1976

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_env(path: Path = ENV_FILE) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export").strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip().strip("'\"")
    return env


def get_db_conn():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        dbname=os.environ.get("PGDATABASE", "arc_v5"),
        user=os.environ.get("PGUSER", "jeff"),
        password=os.environ.get("PGPASSWORD", ""),
    )


def make_r2_client():
    env = {**load_env(), **os.environ}
    return boto3.client(
        "s3",
        endpoint_url=env["R2_ENDPOINT"],
        aws_access_key_id=env["R2_ACCESS_KEY"],
        aws_secret_access_key=env["R2_SECRET_KEY"],
    ), env["R2_BUCKET"]


def r2_key_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ─── Export ───────────────────────────────────────────────────────────────────

def build_config(corpus_id: str, chunk_count: int) -> dict:
    """Generate config.json for a corpus."""
    is_sparse = corpus_id in SPARSE_CORPORA or chunk_count < 20_000
    year_from = YEAR_FROM_MAP.get(corpus_id, DEFAULT_YEAR_FROM)
    leiden_res = CUSTOM_LEIDEN_RES.get(corpus_id, 1.0)

    return {
        "corpus_id": corpus_id,
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "k": 16,
        "leiden_res": leiden_res,
        "leiden_seed": 42,
        "resolution": "quarterly",
        "year_from": year_from,
        "void_distance_threshold": 0.5 if is_sparse else 0.3,
        "junk_threshold": 2,
        "void_radius": 0.15,
        "fringe_radius": 0.20,
        "max_seq_length": 512,
        "exported_at": datetime.now().isoformat(),
        "chunk_count": chunk_count,
    }


def export_corpus(conn, corpus_id: str, output_dir: Path) -> tuple[Path, Path, int]:
    """
    Export chunks TSV and config.json for a corpus.
    Returns (chunks_path, config_path, chunk_count).
    """
    corpus_dir = output_dir / corpus_id
    corpus_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = corpus_dir / f"chunks_{corpus_id}.tsv"
    config_path = corpus_dir / "config.json"

    # Export chunks: document_id \t idea_text \t content_date
    print(f"[{ts()}] Exporting {corpus_id} from data_documents...")
    t0 = time.time()

    with conn.cursor(name="export_chunks") as cur:
        cur.itersize = 5000
        cur.execute("""
            SELECT document_id,
                   COALESCE(title, '') || ' ' || COALESCE(abstract, ''),
                   COALESCE(content_date, publication_date)::text
            FROM data_documents
            WHERE corpus_id = %s
              AND (title IS NOT NULL OR abstract IS NOT NULL)
            ORDER BY content_date
        """, (corpus_id,))

        count = 0
        with open(chunks_path, "w", encoding="utf-8") as fh:
            for row in cur:
                doc_id, text, date_str = row
                # Clean text: replace tabs/newlines with spaces
                text = (text or "").replace("\t", " ").replace("\n", " ").strip()
                if not text or not date_str:
                    continue
                fh.write(f"{doc_id}\t{text}\t{date_str}\n")
                count += 1

    elapsed = time.time() - t0
    size_mb = chunks_path.stat().st_size / 1_048_576
    print(f"[{ts()}] Exported {count:,} chunks ({size_mb:.1f} MB) in {elapsed:.1f}s")

    # Generate config.json
    config = build_config(corpus_id, count)
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)
    print(f"[{ts()}] Config: k={config['k']} leiden_res={config['leiden_res']} "
          f"year_from={config['year_from']} void_threshold={config['void_distance_threshold']}")

    return chunks_path, config_path, count


def upload_to_r2(s3, bucket: str, corpus_id: str,
                 chunks_path: Path, config_path: Path, force: bool) -> None:
    """Upload chunks TSV and config.json to R2."""
    chunks_key = f"{corpus_id}/chunks_{corpus_id}.tsv"
    config_key = f"{corpus_id}/config.json"

    if not force and r2_key_exists(s3, bucket, chunks_key):
        print(f"[{ts()}] R2 already has {chunks_key} — skipping (use --force to overwrite)")
        return

    print(f"[{ts()}] Uploading to R2...")
    t0 = time.time()
    s3.upload_file(str(chunks_path), bucket, chunks_key)
    sz = chunks_path.stat().st_size / 1_048_576
    print(f"[{ts()}] Uploaded chunks ({sz:.1f} MB)")

    s3.upload_file(str(config_path), bucket, config_key)
    print(f"[{ts()}] Uploaded config.json")
    print(f"[{ts()}] R2 upload done in {time.time() - t0:.1f}s")


# ─── List available corpora ──────────────────────────────────────────────────

def list_corpora(conn) -> list[tuple[str, int]]:
    """Return all patent corpora with their document counts."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT corpus_id, count(*) AS n
            FROM data_documents
            WHERE source_type = 'patents'
              AND (title IS NOT NULL OR abstract IS NOT NULL)
            GROUP BY corpus_id
            ORDER BY n DESC
        """)
        return cur.fetchall()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Export corpus data from arc_v5 to R2 for cloud pipeline")
    ap.add_argument("--corpus", required=True,
                    help="Corpus ID, or 'all' to export all patent corpora, "
                         "or 'list' to show available corpora")
    ap.add_argument("--force", action="store_true",
                    help="Re-upload even if chunks TSV already exists on R2")
    ap.add_argument("--local-only", action="store_true",
                    help="Export locally only, skip R2 upload")
    ap.add_argument("--output-dir", type=Path, default=EXPORT_DIR,
                    help=f"Local output directory (default: {EXPORT_DIR})")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel export workers (default: 4)")
    return ap.parse_args()


def _export_one(corpus_id: str, output_dir: Path, force: bool, local_only: bool) -> tuple:
    """Export + upload one corpus. Returns (corpus_id, count, error_or_None)."""
    try:
        conn = get_db_conn()
        chunks_path, config_path, count = export_corpus(conn, corpus_id, output_dir)
        conn.close()
        if not local_only:
            s3, bucket = make_r2_client()
            upload_to_r2(s3, bucket, corpus_id, chunks_path, config_path, force)
        return corpus_id, count, None
    except Exception as e:
        return corpus_id, 0, str(e)


def main():
    args = parse_args()

    if args.corpus == "list":
        conn = get_db_conn()
        corpora = list_corpora(conn)
        print(f"\n{'Corpus':<35} {'Docs':>10}")
        print("-" * 47)
        for cid, n in corpora:
            print(f"{cid:<35} {n:>10,}")
        print(f"\nTotal: {len(corpora)} corpora, {sum(n for _, n in corpora):,} docs")
        conn.close()
        return

    # Determine corpora to export
    if args.corpus == "all":
        conn = get_db_conn()
        corpora = [cid for cid, _ in list_corpora(conn)]
        conn.close()
    else:
        corpora = [c.strip() for c in args.corpus.split(",")]

    t0 = time.time()
    n_workers = min(args.workers, len(corpora))

    if n_workers <= 1:
        # Sequential fallback
        conn = get_db_conn()
        s3 = bucket = None
        if not args.local_only:
            s3, bucket = make_r2_client()
        total_exported = 0
        for i, cid in enumerate(corpora, 1):
            print(f"\n{'='*60}\n[{i}/{len(corpora)}] {cid}\n{'='*60}")
            chunks_path, config_path, count = export_corpus(conn, cid, args.output_dir)
            total_exported += count
            if s3:
                upload_to_r2(s3, bucket, cid, chunks_path, config_path, args.force)
        conn.close()
    else:
        # Parallel export — each worker gets its own DB connection
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"Exporting {len(corpora)} corpora with {n_workers} workers...")
        total_exported = 0
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_export_one, cid, args.output_dir, args.force, args.local_only): cid
                for cid in corpora
            }
            for fut in as_completed(futures):
                cid, count, err = fut.result()
                if err:
                    print(f"  FAILED {cid}: {err}")
                else:
                    total_exported += count
                    print(f"  Done {cid}: {count:,} chunks")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done: {len(corpora)} corpora, {total_exported:,} chunks in {elapsed:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

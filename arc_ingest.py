#!/usr/bin/env python3
"""
arc_ingest.py — Unified ingest for all ARC corpora (patents + journals).

Replaces:
  ingest/arc_h01l_ingest.py      (H01L-specific, old schema)
  ingest/arc_openalex_ingest.py  (OpenAlex journals, old schema)
  ingest/arc_patent_ingest.py    (generic patents, data_documents/data_chunks)

Writes to: data_documents, data_chunks (real flat tables).
Does NOT write: chunk_periods (pipeline step handles that), embeddings.

Usage:
  python3 arc_ingest.py --corpus-id H01L_quarterly
  python3 arc_ingest.py --corpus-id openalex_cs_sample
  python3 arc_ingest.py --corpus-id H01L_quarterly --limit 10000
  python3 arc_ingest.py --corpus-id H01L_quarterly --dry-run
  python3 arc_ingest.py --corpus-id H01L_quarterly --force-reingest
"""

import argparse
import csv
import gzip
import json
import os
import pickle
import re
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from glob import glob
from pathlib import Path

import psycopg2
import psycopg2.extras


# ── Constants ────────────────────────────────────────────────────────────────────

DOC_BATCH       = 500        # documents per commit
LOG_EVERY       = 5_000      # print progress every N docs
MIN_CHUNK_CHARS = 20
MAX_CHUNKS      = 5          # max sentence chunks per abstract

_DATE_MIN     = date(1800, 1, 1)
_DATE_MAX     = date(2030, 12, 31)
PRE1990_START = date(1900, 1, 1)
PRE1990_END   = date(1989, 12, 31)

# Sentence boundary: split on ./?/! followed by capital letter, digit, or quote
_SENT_RE = re.compile(r'(?<=[.?!])\s+(?=[A-Z0-9"\(\[])')

# ── Fallback corpus configs for arc_v5 (no sys_run_config) ──────────────────────
# Sourced from arc_v4 sys_run_config. Used when target DB has no sys_run_config.
_FALLBACK_CORPUS_CONFIGS: dict = {
    "H01L_quarterly":             {"source_type": "patents", "source_filter": "H01L",  "resolution": "quarterly", "year_from": None, "year_to": None, "embedding_model": "Qwen/Qwen3-Embedding-0.6B", "domain": "semiconductors", "label": "H01L Semiconductor Devices", "concepts": [], "leiden_res": None, "status": "active"},
    "G06N_quarterly":             {"source_type": "patents", "source_filter": "G06N",  "resolution": "quarterly", "year_from": None, "year_to": None, "embedding_model": "Qwen/Qwen3-Embedding-0.6B", "domain": "ai_ml",         "label": "G06N Machine Learning",      "concepts": [], "leiden_res": None, "status": "active"},
    "C23C_quarterly":             {"source_type": "patents", "source_filter": "C23C",  "resolution": "quarterly", "year_from": 1990, "year_to": None, "embedding_model": "Qwen/Qwen3-Embedding-0.6B", "domain": None,             "label": "C23C Coatings",              "concepts": [], "leiden_res": None, "status": "active"},
    "C30B_quarterly":             {"source_type": "patents", "source_filter": "C30B",  "resolution": "quarterly", "year_from": 1990, "year_to": None, "embedding_model": "Qwen/Qwen3-Embedding-0.6B", "domain": None,             "label": "C30B Crystal Growth",        "concepts": [], "leiden_res": None, "status": "active"},
    "G01B_quarterly":             {"source_type": "patents", "source_filter": "G01B",  "resolution": "quarterly", "year_from": 1990, "year_to": None, "embedding_model": "Qwen/Qwen3-Embedding-0.6B", "domain": None,             "label": "G01B Measurement",           "concepts": [], "leiden_res": None, "status": "active"},
    "G01N_quarterly":             {"source_type": "patents", "source_filter": "G01N",  "resolution": "quarterly", "year_from": 1990, "year_to": None, "embedding_model": "Qwen/Qwen3-Embedding-0.6B", "domain": None,             "label": "G01N Analysis",              "concepts": [], "leiden_res": None, "status": "active"},
    "G02B_quarterly":             {"source_type": "patents", "source_filter": "G02B",  "resolution": "quarterly", "year_from": 1990, "year_to": None, "embedding_model": "Qwen/Qwen3-Embedding-0.6B", "domain": None,             "label": "G02B Optics",                "concepts": [], "leiden_res": None, "status": "active"},
    "G06F_quarterly":             {"source_type": "patents", "source_filter": "G06F",  "resolution": "quarterly", "year_from": 1990, "year_to": None, "embedding_model": "Qwen/Qwen3-Embedding-0.6B", "domain": "ai_ml",         "label": "G06F Computing",             "concepts": [], "leiden_res": None, "status": "active"},
    "longevity_cardio_quarterly": {"source_type": "patents", "source_filter": "A61P9", "resolution": "quarterly", "year_from": 1995, "year_to": None, "embedding_model": "Qwen/Qwen3-Embedding-0.6B", "domain": "longevity",     "label": "Longevity Cardio",           "concepts": [], "leiden_res": None, "status": "active"},
    "A61P9_quarterly":             {"source_type": "patents", "source_filter": "A61P9",  "resolution": "quarterly", "year_from": 1995, "year_to": None, "embedding_model": "Qwen/Qwen3-Embedding-0.6B", "domain": "longevity",     "label": "A61P9 Cardiovascular",       "concepts": [], "leiden_res": None, "status": "active"},
    "A61P25_quarterly":            {"source_type": "patents", "source_filter": "A61P25", "resolution": "quarterly", "year_from": 1995, "year_to": None, "embedding_model": "Qwen/Qwen3-Embedding-0.6B", "domain": "longevity",     "label": "A61P25 Neurology",           "concepts": [], "leiden_res": None, "status": "active"},
    "C12N15_quarterly":            {"source_type": "patents", "source_filter": "C12N15", "resolution": "quarterly", "year_from": 1995, "year_to": None, "embedding_model": "Qwen/Qwen3-Embedding-0.6B", "domain": "longevity",     "label": "C12N15 Genetics",            "concepts": [], "leiden_res": None, "status": "active"},
    "A61K38_quarterly":            {"source_type": "patents", "source_filter": "A61K38", "resolution": "quarterly", "year_from": 1995, "year_to": None, "embedding_model": "Qwen/Qwen3-Embedding-0.6B", "domain": "longevity",     "label": "A61K38 Peptides",            "concepts": [], "leiden_res": None, "status": "active"},
    "longevity_patents_quarterly": {"source_type": "patents", "source_filter": "A61K38", "resolution": "quarterly", "year_from": 1995, "year_to": None, "embedding_model": "Qwen/Qwen3-Embedding-0.6B", "domain": "longevity",     "label": "Longevity Patents",           "concepts": [], "leiden_res": None, "status": "active"},
}

PIPELINE_STEPS = [
    (1,  "ingest"),
    (2,  "embed"),
    (3,  "knn"),
    (4,  "pipeline"),
    (5,  "match"),
    (6,  "void"),
    (7,  "label"),
    (8,  "ml_train"),
    (9,  "ml_score"),
    (10, "backtest"),
    (11, "survival"),
    (12, "discovery"),
    (13, "narrative"),
]


# ── DB connection ────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "/var/run/postgresql"),
        dbname=os.environ.get("PGDATABASE", "arc_v4"),
        user=os.environ.get("PGUSER", "jeff"),
    )


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ── Corpus config ────────────────────────────────────────────────────────────────

def load_corpus_config(conn, corpus_id: str) -> dict:
    """Load corpus metadata from sys_run_config. Falls back to _FALLBACK_CORPUS_CONFIGS
    when the target DB has no sys_run_config (e.g. arc_v5)."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT legacy_name, source_type, source_filter, resolution,
                       leiden_res, embedding_model, year_from, year_to,
                       domain, status, label, concepts
                FROM sys_run_config WHERE legacy_name = %s
            """, (corpus_id,))
            row = cur.fetchone()
    except Exception:
        conn.rollback()
        row = None

    if row is not None:
        keys = ["legacy_name", "source_type", "source_filter", "resolution",
                "leiden_res", "embedding_model", "year_from", "year_to",
                "domain", "status", "label", "concepts"]
        return dict(zip(keys, row))

    # sys_run_config missing or corpus not found — try fallback (arc_v5 mode)
    if corpus_id in _FALLBACK_CORPUS_CONFIGS:
        cfg = dict(_FALLBACK_CORPUS_CONFIGS[corpus_id])
        cfg["legacy_name"] = corpus_id
        print(f"  [arc_v5 mode] Using fallback config for {corpus_id!r}")
        return cfg

    print(f"ERROR: corpus_id {corpus_id!r} not found in sys_run_config "
          f"and has no fallback config.", file=sys.stderr)
    print("       Either register in sys_run_config or add to _FALLBACK_CORPUS_CONFIGS.",
          file=sys.stderr)
    sys.exit(1)


# ── Provisioning ─────────────────────────────────────────────────────────────────

def provision_if_needed(conn, corpus_id: str, cfg: dict) -> None:
    """
    Idempotent: ensure x_corpora row, embeddings partitions, and
    sys_corpus_pipeline_steps rows exist for this corpus.
    data_documents and data_chunks are flat tables — no partitions needed there.
    Skips silently when running against arc_v5 (no sys_run_config / embeddings schema).
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema='public' AND table_name='sys_run_config'
        """)
        if cur.fetchone() is None:
            print("  [arc_v5 mode] Skipping provisioning (no sys_run_config)")
            return
    source_type     = cfg["source_type"]
    resolution      = cfg["resolution"] or "quarterly"
    embedding_model = cfg["embedding_model"]
    label           = cfg.get("label") or corpus_id

    class_map = {
        "patents":  "patent_cpc_class",
        "openalex": "research_paper",
        "journals": "research_paper",
        "narrative":"narrative_family",
    }
    class_code = class_map.get(source_type, "other")

    part_safe = corpus_id.lower().replace("-", "_")
    cp_part   = f"chunk_periods_{part_safe}"
    knn_part  = f"knn_edges_{part_safe}"

    with conn.cursor() as cur:
        # x_corpora row
        cur.execute("SELECT 1 FROM sys_run_config WHERE legacy_name = %s", (corpus_id,))
        if cur.fetchone() is None:
            cur.execute("""
                INSERT INTO sys_run_config
                  (legacy_name, embedding_model, status, resolution)
                VALUES (%s, %s, 'active', %s)
                ON CONFLICT (legacy_name) DO NOTHING
            """, (corpus_id, embedding_model, resolution))
            print(f"  [created] sys_run_config: {corpus_id}")
        else:
            print(f"  [exists]  sys_run_config: {corpus_id}")

        # embeddings.chunk_periods partition
        cur.execute("""
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'embeddings' AND c.relname = %s
        """, (cp_part,))
        if cur.fetchone() is None:
            cur.execute(
                f"CREATE TABLE embeddings.{cp_part} "
                f"PARTITION OF embeddings.chunk_periods FOR VALUES IN (%s)",
                (corpus_id,)
            )
            print(f"  [created] partition embeddings.{cp_part}")
        else:
            print(f"  [exists]  partition embeddings.{cp_part}")

        # embeddings.knn_edges partition
        cur.execute("""
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'embeddings' AND c.relname = %s
        """, (knn_part,))
        if cur.fetchone() is None:
            cur.execute(
                f"CREATE TABLE embeddings.{knn_part} "
                f"PARTITION OF embeddings.knn_edges FOR VALUES IN (%s)",
                (corpus_id,)
            )
            print(f"  [created] partition embeddings.{knn_part}")
        else:
            print(f"  [exists]  partition embeddings.{knn_part}")

        # sys_corpus_pipeline_steps (idempotent)
        for step_order, script in PIPELINE_STEPS:
            cur.execute("""
                INSERT INTO sys_corpus_pipeline_steps
                  (corpus_id, step_order, script, status)
                VALUES (%s, %s, %s, 'pending')
                ON CONFLICT (corpus_id, step_order) DO NOTHING
            """, (corpus_id, step_order, script))

    conn.commit()
    print(f"  Provisioning complete.")


# ── Batch insert ─────────────────────────────────────────────────────────────────

def _has_data_chunks(conn) -> bool:
    """Return True if data_chunks table exists in the target DB."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema='public' AND table_name='data_chunks'
        """)
        return cur.fetchone() is not None


def flush_batch(conn, doc_batch: list, chunk_batch: list) -> None:
    """Bulk-insert a batch of documents and chunks. Commits after each call.
    Skips data_chunks insert when the table is absent (arc_v5 mode)."""
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, """
                INSERT INTO data_documents
                  (document_id, external_id, corpus_id, corpus_type,
                   source_type, source_api, title, abstract,
                   assignee, filing_date, publication_date,
                   cpc_codes, content_date, n_ideas, venue, url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id) DO UPDATE SET
                    abstract = EXCLUDED.abstract
                    WHERE data_documents.abstract IS NULL
            """, doc_batch, page_size=DOC_BATCH)

            if chunk_batch and _has_data_chunks(conn):
                psycopg2.extras.execute_batch(cur, """
                    INSERT INTO data_chunks
                      (chunk_id, document_id, idea_index, idea_text,
                       embedding_model, type, subtype, content_date, filing_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO NOTHING
                """, chunk_batch, page_size=DOC_BATCH * MAX_CHUNKS)

        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ── Common helpers ────────────────────────────────────────────────────────────────

def _valid_date(d) -> "date | None":
    """
    Reject garbled USPTO dates (e.g. '9176-06-01' for patent 4041523).
    Accepts 1800-01-01 to 2030-12-31 only.
    """
    if d is None:
        return None
    return d if _DATE_MIN <= d <= _DATE_MAX else None


def assign_period(d: date, resolution: str) -> "tuple[date, date]":
    """
    Return (period_start, period_end) for a date.
    Any date before 1990-01-01 maps to the pre-1990 bucket.
    """
    if d < date(1990, 1, 1):
        return PRE1990_START, PRE1990_END

    if resolution == "quarterly":
        month = ((d.month - 1) // 3) * 3 + 1
        ps = date(d.year, month, 1)
        pe = date(d.year, month + 3, 1) - timedelta(days=1) if month < 10 \
             else date(d.year, 12, 31)
        return ps, pe

    if resolution == "monthly":
        import calendar
        ps = date(d.year, d.month, 1)
        last = calendar.monthrange(d.year, d.month)[1]
        return ps, date(d.year, d.month, last)

    if resolution == "annual":
        return date(d.year, 1, 1), date(d.year, 12, 31)

    if resolution == "weekly":
        ps = d - timedelta(days=d.weekday())   # ISO Monday
        return ps, ps + timedelta(days=6)

    # Default: quarterly
    return assign_period(d, "quarterly")


# ── Patent helpers ────────────────────────────────────────────────────────────────

def build_cpc_index(source_filter: str, corpus_id: str, data_dir: Path) -> "tuple[set, dict]":
    """
    Scan g_cpc_current.tsv for patents matching CPC group prefixes.
    source_filter: comma-separated prefixes e.g. 'H01L' or 'A61K38,C12N5'.
    Matches at GROUP level (col 5) — more precise than subclass.
    Caches result to /tmp/arc_ingest_{corpus_id}_cpc.pkl for reruns.
    Returns (matching_ids: set[str], cpc_map: dict[patent_id, list[str]]).
    """
    cache_path = Path(f"/tmp/arc_ingest_{corpus_id}_cpc.pkl")
    if cache_path.exists():
        print(f"  Loading CPC index from cache {cache_path}...")
        with open(cache_path, "rb") as f:
            result = pickle.load(f)
        print(f"  Loaded {len(result[0]):,} patent IDs from cache.")
        return result

    prefixes = [p.strip() for p in source_filter.split(",") if p.strip()]
    cpc_file = data_dir / "g_cpc_current.tsv"
    print(f"[{ts()}] Scanning {cpc_file} for CPC prefixes: {prefixes}")

    matching_ids: set = set()
    cpc_map: dict = {}
    row_count = 0

    with open(cpc_file, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quotechar='"')
        next(reader)  # skip header
        for row in reader:
            row_count += 1
            if row_count % 5_000_000 == 0:
                print(f"  [{ts()}] {row_count/1_000_000:.0f}M rows | "
                      f"{len(matching_ids):,} matched", flush=True)
            # cols: patent_id(0), cpc_sequence(1), cpc_section(2),
            #       cpc_class(3), cpc_subclass(4), cpc_group(5)
            if len(row) < 6:
                continue
            patent_id = row[0]
            cpc_group = row[5]
            if any(cpc_group.startswith(p) for p in prefixes):
                matching_ids.add(patent_id)
                if patent_id not in cpc_map:
                    cpc_map[patent_id] = []
                if cpc_group not in cpc_map[patent_id]:
                    cpc_map[patent_id].append(cpc_group)

    print(f"  [{ts()}] CPC scan done — {row_count:,} rows, {len(matching_ids):,} patents matched")
    with open(cache_path, "wb") as f:
        pickle.dump((matching_ids, cpc_map), f)
    print(f"  Cached to {cache_path}")
    return matching_ids, cpc_map


def load_filing_dates(patent_ids: set, data_dir: Path) -> dict:
    """
    Load g_application.tsv. Keeps earliest valid filing date per patent.
    Rejects garbled USPTO dates (e.g. '9176-06-01' for patent 4041523).
    Returns empty dict (gracefully) if g_application.tsv is missing —
    ingest then falls back to pub_date as content_date.
    """
    app_file = data_dir / "g_application.tsv"
    if not app_file.exists():
        print(f"[{ts()}] g_application.tsv not found — filing dates unavailable, "
              f"using patent grant date as content_date")
        return {}
    print(f"[{ts()}] Loading filing dates from {app_file}...")
    filing: dict = {}

    with open(app_file, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quotechar='"')
        next(reader)  # header: application_id, patent_id, patent_application_type, filing_date, ...
        for row in reader:
            if len(row) < 4:
                continue
            pid = row[1]
            if pid not in patent_ids:
                continue
            try:
                fd = _valid_date(date.fromisoformat(row[3]) if row[3] else None)
            except ValueError:
                fd = None
            if fd is None:
                continue
            if pid not in filing or fd < filing[pid]:
                filing[pid] = fd

    print(f"  Loaded filing dates for {len(filing):,} patents.")
    return filing


def load_patent_metadata(
    patent_ids: set,
    year_from: int,
    year_to: int,
    data_dir: Path,
) -> dict:
    """
    Load g_patent.tsv. Filters: utility patents only, not withdrawn,
    publication year in [year_from, year_to].
    Returns {patent_id: {'title': str, 'pub_date': date}}.
    """
    patent_file = data_dir / "g_patent.tsv"
    print(f"[{ts()}] Loading patent metadata from {patent_file}...")
    patents: dict = {}
    skipped = {"type": 0, "withdrawn": 0, "date": 0, "bad_date": 0}
    row_count = 0

    with open(patent_file, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quotechar='"')
        next(reader)  # header: patent_id, patent_type, patent_date, patent_title, wipo_kind, num_claims, withdrawn
        for row in reader:
            row_count += 1
            if row_count % 1_000_000 == 0:
                print(f"  [{ts()}] {row_count/1_000_000:.1f}M rows | {len(patents):,} kept", flush=True)
            if len(row) < 7:
                continue
            pid = row[0]
            if pid not in patent_ids:
                continue
            if row[1].lower() != "utility":
                skipped["type"] += 1
                continue
            if row[6].strip() == "1":
                skipped["withdrawn"] += 1
                continue
            try:
                pub_date = date.fromisoformat(row[2]) if row[2] else None
            except ValueError:
                skipped["bad_date"] += 1
                continue
            if pub_date is None or pub_date.year < year_from or pub_date.year > year_to:
                skipped["date"] += 1
                continue
            patents[pid] = {"title": row[3].strip(), "pub_date": pub_date}

    print(f"  [{ts()}] Metadata done — {len(patents):,} kept | "
          f"{skipped['type']} non-utility | {skipped['withdrawn']} withdrawn | "
          f"{skipped['date']} out-of-range | {skipped['bad_date']} bad-date")
    return patents


def load_abstracts(patent_ids: set, data_dir: Path) -> dict:
    """
    Load patent abstracts. Tries g_patent_abstract.tsv first, then g_abstract.tsv.
    Returns {patent_id: abstract_text}.
    """
    for name in ("g_patent_abstract.tsv", "g_abstract.tsv"):
        abstract_file = data_dir / name
        if abstract_file.exists():
            break
    else:
        print(f"ERROR: No abstract file found in {data_dir} "
              f"(tried g_patent_abstract.tsv, g_abstract.tsv)", file=sys.stderr)
        sys.exit(1)

    print(f"[{ts()}] Loading abstracts from {abstract_file}...")
    abstracts: dict = {}
    row_count = 0

    with open(abstract_file, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quotechar='"')
        next(reader)  # header: patent_id, patent_abstract
        for row in reader:
            row_count += 1
            if row_count % 1_000_000 == 0:
                print(f"  [{ts()}] {row_count/1_000_000:.1f}M rows | "
                      f"{len(abstracts):,} collected", flush=True)
            if len(row) < 2:
                continue
            pid = row[0]
            if pid not in patent_ids:
                continue
            text = row[1].strip()
            if text:
                abstracts[pid] = text

    print(f"  [{ts()}] Abstracts done — {len(abstracts):,} loaded")
    return abstracts


# ── Patent ingest ─────────────────────────────────────────────────────────────────

def run_patent_ingest(
    conn, corpus_id: str, cfg: dict, limit: int, dry_run: bool
) -> "tuple[int, int]":
    """
    Ingest USPTO patent data into data_documents + data_chunks.
    Returns (docs_written, chunks_written).
    """
    data_dir = Path(os.environ.get("ARC_DATA_DIR", "/home/jeff/data/patents/"))
    # Fallback to /home/jeff/data if the configured dir doesn't have the CPC file
    if not (data_dir / "g_cpc_current.tsv").exists():
        alt = Path("/home/jeff/data")
        if (alt / "g_cpc_current.tsv").exists():
            data_dir = alt
            print(f"  Using data dir: {data_dir}")

    source_filter   = cfg["source_filter"] or ""
    resolution      = cfg["resolution"] or "quarterly"
    embedding_model = cfg["embedding_model"]
    year_from       = cfg["year_from"] or 1800
    year_to         = cfg["year_to"] or 2030

    if not source_filter:
        print("ERROR: sys_run_config.source_filter is empty. "
              "Set comma-separated CPC prefixes (e.g. 'H01L').", file=sys.stderr)
        sys.exit(1)

    if dry_run:
        print(f"[dry-run] Would scan {data_dir} for CPC prefixes: {source_filter}")
        print(f"[dry-run] Year range: {year_from}–{year_to}, resolution: {resolution}")
        print(f"[dry-run] Limit: {limit:,}")
        return 0, 0

    # ── Load source data ──────────────────────────────────────────────────────────
    matching_ids, cpc_map = build_cpc_index(source_filter, corpus_id, data_dir)
    filing_dates          = load_filing_dates(matching_ids, data_dir)
    patents               = load_patent_metadata(matching_ids, year_from, year_to, data_dir)
    abstracts             = load_abstracts(set(patents.keys()), data_dir)

    writable = [pid for pid in patents if pid in abstracts]
    skipped_no_abstract = len(patents) - len(writable)
    print(f"\n[{ts()}] {len(writable):,} patents have both metadata and abstract "
          f"({skipped_no_abstract:,} skipped — no abstract)")
    print(f"[{ts()}] Writing to DB (limit={limit:,})...")

    docs_written  = 0
    chunks_written = 0
    doc_batch  = []
    chunk_batch = []
    t0 = time.time()

    for patent_id in writable:
        if docs_written >= limit:
            break

        meta          = patents[patent_id]
        abstract_text = abstracts[patent_id]
        pub_date      = meta["pub_date"]
        filing_date   = filing_dates.get(patent_id)
        content_date  = filing_date or pub_date
        idea_text     = " ".join(filter(None, [meta["title"], abstract_text])).strip()

        document_id = f"{corpus_id}_{patent_id}"
        chunk_id    = f"{corpus_id}_{patent_id}_chunk_0"

        doc_batch.append((
            document_id,
            patent_id,                    # external_id
            corpus_id,
            "patents",                    # corpus_type
            "patents",                    # source_type
            "uspto",                      # source_api
            meta["title"],
            abstract_text,
            None,                         # assignee
            filing_date,                  # filing_date
            pub_date,                     # publication_date
            cpc_map.get(patent_id, []),   # cpc_codes text[]
            content_date,                 # content_date (used for period assignment)
            1,                            # n_ideas
            None,                         # venue
            None,                         # url
        ))

        chunk_batch.append((
            chunk_id,
            document_id,
            0,                            # idea_index
            idea_text,                    # idea_text = title + abstract
            embedding_model,
            "patent",                     # type
            corpus_id,                    # subtype
            content_date,                 # content_date
            filing_date,                  # filing_date (→ prosecution_duration_days)
        ))

        docs_written  += 1
        chunks_written += 1

        if len(doc_batch) >= DOC_BATCH:
            flush_batch(conn, doc_batch, chunk_batch)
            doc_batch   = []
            chunk_batch = []
            if docs_written % LOG_EVERY == 0:
                elapsed = time.time() - t0
                rate = docs_written / elapsed if elapsed > 0 else 0
                pct  = docs_written / len(writable) * 100
                print(f"  Ingested {docs_written:,}/{len(writable):,} "
                      f"({pct:.1f}%) — {rate:.0f} docs/sec", flush=True)

    if doc_batch:
        flush_batch(conn, doc_batch, chunk_batch)

    return docs_written, chunks_written


# ── OpenAlex helpers ──────────────────────────────────────────────────────────────

def reconstruct_abstract(aii: dict) -> str:
    """Reconstruct plain text from OpenAlex abstract_inverted_index."""
    words: dict = {}
    try:
        for word, positions in aii.items():
            for pos in positions:
                words[pos] = word
    except (AttributeError, TypeError):
        return ""
    return " ".join(words[k] for k in sorted(words))


def split_into_chunks(abstract: str) -> list:
    """
    Split abstract into 1–MAX_CHUNKS sentence chunks (min MIN_CHUNK_CHARS each).
    Fallback: full abstract[:2000] as single chunk if splitting yields nothing.
    """
    raw = _SENT_RE.split(abstract.strip())
    out = []
    for sent in raw:
        sent = sent.strip()
        if len(sent) >= MIN_CHUNK_CHARS:
            out.append(sent)
        if len(out) >= MAX_CHUNKS:
            break
    if not out and len(abstract.strip()) >= MIN_CHUNK_CHARS:
        out = [abstract.strip()[:2000]]
    return out


# ── OpenAlex ingest ───────────────────────────────────────────────────────────────

def run_openalex_ingest(
    conn, corpus_id: str, cfg: dict, limit: int, dry_run: bool
) -> "tuple[int, int]":
    """
    Ingest OpenAlex journal papers into data_documents + data_chunks.
    Returns (docs_written, chunks_written).
    """
    base_dir = Path(os.environ.get("ARC_OPENALEX_DIR", "/home/jeff/data/openalex/works/"))
    if not base_dir.exists():
        alt = Path("/mnt/c/Users/jeff/data/openalex/works")
        if alt.exists():
            base_dir = alt
            print(f"  Using OpenAlex dir: {base_dir}")

    partitions = sorted(glob(str(base_dir / "updated_date=*/part_*.gz")), reverse=True)

    source_filter   = cfg.get("source_filter")        # field_id or None
    year_from       = cfg.get("year_from")
    year_to         = cfg.get("year_to")
    concepts        = cfg.get("concepts") or []        # text[] from sys_run_config
    embedding_model = cfg["embedding_model"]

    if dry_run:
        print(f"[dry-run] Found {len(partitions)} OpenAlex partition files under {base_dir}")
        print(f"[dry-run] field_id filter : {source_filter or '(all fields)'}")
        print(f"[dry-run] year range      : {year_from or 'any'}–{year_to or 'any'}")
        print(f"[dry-run] concepts        : {concepts or '(none)'}")
        print(f"[dry-run] Limit           : {limit:,}")
        return 0, 0

    concept_lower = [c.lower() for c in concepts] if concepts else []
    docs_written   = 0
    chunks_written = 0
    doc_batch      = []
    chunk_batch    = []
    t0             = time.time()
    n_partitions   = 0

    for part_path in partitions:
        if docs_written >= limit:
            break
        n_partitions += 1

        try:
            fh = gzip.open(part_path, "rt", encoding="utf-8")
        except Exception as e:
            print(f"  WARN: cannot open {part_path}: {e}", file=sys.stderr)
            continue

        with fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    rec = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                pt = rec.get("primary_topic") or {}

                # Field filter (from source_filter)
                if source_filter is not None:
                    rec_field_id = (pt.get("field") or {}).get("id", "")
                    if rec_field_id != source_filter:
                        continue

                # Year range filters
                pub_year = rec.get("publication_year") or 0
                if year_from is not None and pub_year < year_from:
                    continue
                if year_to is not None and pub_year > year_to:
                    continue

                # Topic score gate (always applied)
                if (pt.get("score") or 0.0) < 0.8:
                    continue

                # Abstract required
                aii = rec.get("abstract_inverted_index")
                if not aii:
                    continue

                # Concept filter (optional)
                if concept_lower:
                    rec_concepts = [
                        (c.get("display_name") or "").lower()
                        for c in (rec.get("concepts") or [])
                    ]
                    if not any(
                        any(kw in rc for rc in rec_concepts)
                        for kw in concept_lower
                    ):
                        continue

                raw_id   = rec.get("id", "")
                short_id = raw_id.split("/")[-1]    # 'W4401406562'
                if not short_id.startswith("W"):
                    continue

                abstract = reconstruct_abstract(aii)
                if len(abstract) < MIN_CHUNK_CHARS:
                    continue

                title       = (rec.get("title") or rec.get("display_name") or "").strip()
                pub_date_str = rec.get("publication_date")
                try:
                    content_date = date.fromisoformat(pub_date_str) if pub_date_str else None
                except ValueError:
                    content_date = None

                doi   = rec.get("doi")
                loc   = rec.get("primary_location") or {}
                src   = loc.get("source") or {}
                venue = src.get("display_name")

                sentences = split_into_chunks(abstract)
                if not sentences:
                    continue

                document_id = f"{corpus_id}_{short_id}"

                doc_batch.append((
                    document_id,
                    short_id,               # external_id
                    corpus_id,
                    "papers",               # corpus_type
                    "openalex",             # source_type
                    "openalex",             # source_api
                    title,
                    abstract,
                    None,                   # assignee
                    None,                   # filing_date
                    None,                   # publication_date (content_date used)
                    None,                   # cpc_codes
                    content_date,           # content_date
                    len(sentences),         # n_ideas
                    venue,                  # venue
                    doi,                    # url
                ))

                for idx, sent in enumerate(sentences):
                    chunk_id = f"{corpus_id}_{short_id}_chunk_{idx}"
                    chunk_batch.append((
                        chunk_id,
                        document_id,
                        idx,                # idea_index
                        sent,               # idea_text
                        embedding_model,
                        "journal",          # type
                        corpus_id,          # subtype
                        content_date,       # content_date
                        None,               # filing_date
                    ))
                    chunks_written += 1

                docs_written += 1

                if len(doc_batch) >= DOC_BATCH:
                    flush_batch(conn, doc_batch, chunk_batch)
                    doc_batch   = []
                    chunk_batch = []
                    if docs_written % LOG_EVERY == 0:
                        elapsed = time.time() - t0
                        rate = docs_written / elapsed if elapsed > 0 else 0
                        print(f"  Ingested {docs_written:,} docs "
                              f"| {n_partitions} partitions scanned "
                              f"| {rate:.0f} docs/sec", flush=True)

                if docs_written >= limit:
                    break

    if doc_batch:
        flush_batch(conn, doc_batch, chunk_batch)

    return docs_written, chunks_written


# ── Completion ────────────────────────────────────────────────────────────────────

def mark_step_complete(conn, corpus_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema='public' AND table_name='sys_corpus_pipeline_steps'
        """)
        if cur.fetchone() is None:
            return  # arc_v5 mode — no pipeline steps table
        cur.execute("""
            UPDATE sys_corpus_pipeline_steps
            SET status = 'complete', completed_at = NOW()
            WHERE corpus_id = %s AND script = 'ingest'
        """, (corpus_id,))
    conn.commit()


def enqueue_next(corpus_id: str) -> None:
    """
    Enqueue next pipeline task via psql. Checks for existing embeddings to
    decide between enqueue_corpus_chain (full) and enqueue_pipeline_only (fast).
    Handles missing enqueue functions gracefully — warns but does not fail.
    """
    pghost = os.environ.get("PGHOST", "/var/run/postgresql")
    pgdb   = os.environ.get("PGDATABASE", "arc_v4")
    pguser = os.environ.get("PGUSER", "jeff")
    psql   = ["psql", f"--host={pghost}", f"--dbname={pgdb}",
              f"--username={pguser}", "--tuples-only", "--no-align"]

    # Check if embeddings already exist (re-ingest case)
    check = subprocess.run(
        psql + ["-c",
            f"SELECT COUNT(*) FROM data_chunks "
            f"WHERE chunk_id LIKE '{corpus_id}_%' AND embedding IS NOT NULL"],
        capture_output=True, text=True
    )
    embedding_count = int((check.stdout.strip() or "0").split()[0])

    if embedding_count > 0:
        fn  = f"enqueue_pipeline_only('{corpus_id}')"
        msg = f"Embeddings exist ({embedding_count:,}) — enqueued pipeline only"
    else:
        fn  = f"enqueue_corpus_chain('{corpus_id}')"
        msg = "Enqueued full chain: embed_knn → pipeline → label → cartographer"

    result = subprocess.run(
        psql + ["-c", f"SELECT {fn}"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"\n{msg}")
    else:
        stderr = result.stderr.strip()
        print(f"\nWARN: Could not call {fn}: {stderr}")
        print("      The enqueue functions may not exist yet in this schema.")
        print(f"      Manually run: psql -c \"SELECT {fn}\"")


# ── Main ──────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Unified ARC ingest — patents (USPTO) and journals (OpenAlex).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 arc_ingest.py --corpus-id H01L_quarterly
  python3 arc_ingest.py --corpus-id openalex_cs_sample --limit 50000
  python3 arc_ingest.py --corpus-id G06N_quarterly --dry-run
  python3 arc_ingest.py --corpus-id H01L_quarterly --force-reingest
        """)
    ap.add_argument("--corpus-id", required=True,
                    help="Corpus ID (must exist in sys_run_config)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after N documents (default: unlimited)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print stats only — no DB writes")
    ap.add_argument("--force-reingest", action="store_true",
                    help="Re-insert even if documents already exist")
    return ap.parse_args()


def main():
    args      = parse_args()
    corpus_id = args.corpus_id
    t0        = time.time()

    conn = get_conn()
    cfg  = load_corpus_config(conn, corpus_id)

    source_type = cfg["source_type"]
    print(f"\n=== arc_ingest — {corpus_id} ===")
    print(f"  source_type   : {source_type}")
    print(f"  source_filter : {cfg['source_filter'] or '(none)'}")
    print(f"  resolution    : {cfg['resolution']}")
    print(f"  year range    : {cfg['year_from'] or 'any'}–{cfg['year_to'] or 'any'}")
    print(f"  embed model   : {cfg['embedding_model']}")
    print(f"  status        : {cfg['status']}")
    print()

    # Check existing document count
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM data_documents WHERE corpus_id = %s", (corpus_id,))
        existing = int(cur.fetchone()[0])

    if args.dry_run:
        print(f"[dry-run] Existing documents: {existing:,}")
        print(f"[dry-run] source_type={source_type!r} — would dispatch to {source_type} loader")
        print()

    elif existing > 0 and not args.force_reingest:
        print(f"Corpus already has {existing:,} documents. "
              f"Use --force-reingest to re-ingest.")
        conn.close()
        return

    # Provision DB resources (skip in dry-run)
    if not args.dry_run:
        print("Provisioning DB resources...")
        provision_if_needed(conn, corpus_id, cfg)
        print()

    limit = args.limit if args.limit is not None else 2_147_483_647

    # ── Dispatch to source-specific loader ───────────────────────────────────────
    if source_type == "patents":
        docs_written, chunks_written = run_patent_ingest(
            conn, corpus_id, cfg, limit, args.dry_run
        )
    elif source_type in ("openalex", "journals"):
        docs_written, chunks_written = run_openalex_ingest(
            conn, corpus_id, cfg, limit, args.dry_run
        )
    else:
        print(f"ERROR: Unknown source_type {source_type!r}. "
              f"Expected 'patents' or 'openalex'.", file=sys.stderr)
        conn.close()
        sys.exit(1)

    if args.dry_run:
        print("[dry-run] Complete — no data written.")
        conn.close()
        return

    elapsed = time.time() - t0
    rate    = docs_written / elapsed if elapsed > 0 and docs_written > 0 else 0

    print(f"\n{'='*58}")
    print(f"  Documents written:  {docs_written:>10,}")
    print(f"  Chunks written:     {chunks_written:>10,}")
    print(f"  Duration:           {int(elapsed//60)}m {int(elapsed%60)}s")
    print(f"  Throughput:         {rate:>9.0f} docs/sec")
    print(f"{'='*58}")

    if docs_written > 0:
        mark_step_complete(conn, corpus_id)

    conn.close()

    if docs_written > 0:
        enqueue_next(corpus_id)

    print(f"\nNext step: python3 arc_embed.py --corpus-id {corpus_id}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
arc_europepmc_backfill.py — Backfill new schema fields for already-ingested PMC articles.

Targets the 1.4M+ PMC articles ingested before 2026-04-08 (which have NULL values in
the 18 new document columns and empty 5 new child tables). Does NOT touch existing
child-table rows (author, reference, body_section, etc.) — only UPDATEs document and
INSERTs into the 5 new tables.

What it does per article:
  1. Re-parse the JATS XML with the current extract_article()
  2. UPDATE europepmc.document SET <new_columns> = … WHERE doc_id = <pk>
  3. INSERT into europepmc.corresponding_author, footnote, trans_title,
     trans_abstract, supplementary_material (these are currently empty, so no
     conflict handling needed)

What it does NOT do:
  - DOES NOT re-insert author/reference/body_section/abstract rows (no duplicates)
  - DOES NOT touch ingest_progress table
  - DOES NOT modify source/source_file on existing rows

Usage:
  python3 arc_europepmc_backfill.py --pmc-dir /data/downloads/pmc_fulltext/
  python3 arc_europepmc_backfill.py --pmc-dir /data/downloads/pmc_fulltext/ --source oa_comm
  python3 arc_europepmc_backfill.py --dir /data/downloads/europepmc/  # for xml.gz format
  python3 arc_europepmc_backfill.py --dry-run  # parse but don't execute SQL
  python3 arc_europepmc_backfill.py --limit 100  # process only first 100 articles

Progress is logged via a separate tracker table: europepmc.backfill_progress.
Resumable: skip files already marked done in backfill_progress.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tarfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import psycopg2
import psycopg2.extras
from lxml import etree

# Reuse the patched extraction logic from the main ingest script
sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "arc_europepmc_ingest", str(Path(__file__).parent / "arc_europepmc_ingest.py")
)
_aei = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_aei)
extract_article = _aei.extract_article
_strip_entities = _aei._strip_entities


# ── Config ─────────────────────────────────────────────────────────────────────

DB_PARAMS = dict(
    host="/var/run/postgresql",
    dbname="arc_v5",
    user=os.environ.get("PGUSER", "arc"),
)

SCHEMA = "europepmc"
BATCH_SIZE = 1000
ABORT_FLAG = Path("/tmp/ARC_ABORT_FLAG")

# Columns to update on europepmc.document — only the NEW ones (not pre-existing).
# pmcid/pmid/doi/article_title/etc already populated; don't overwrite.
# source_file is also updated (was NULL for old rows).
UPDATE_COLUMNS = [
    "source_file",
    "issn_ppub", "issn_epub",
    "journal_id_iso_abbrev", "journal_id_publisher",
    "publisher_loc",
    "pub_date_epub", "pub_date_ppub", "pub_date_collection",
    "elocation_id",
    "copyright_year",
    "restricted_by",
    "word_count", "fig_count", "table_count", "ref_count", "page_count", "equation_count",
]

# New child tables (currently empty — safe to INSERT without conflict risk)
NEW_CHILD_TABLES = {
    "corresponding_author": ["doc_id", "corresp_id", "label", "full_text", "email", "source_file"],
    "footnote": ["doc_id", "fn_id", "fn_type", "fn_text", "sequence", "source_file"],
    "trans_title": ["doc_id", "language", "trans_title", "source_file"],
    "trans_abstract": ["doc_id", "language", "abstract_text", "source_file"],
    "supplementary_material": ["doc_id", "supp_id", "label", "caption", "mimetype",
                               "media_type", "href", "source_file"],
}

# Child tables we ALSO want to backfill source_file on (already-populated tables
# with NULL source_file — use a separate single-column UPDATE approach)
SOURCE_FILE_CHILD_TABLES = [
    "abstract", "abstract_section", "acknowledgment", "article_category",
    "article_id", "author", "author_affiliation", "body_section", "custom_meta",
    "funding", "keyword", "pub_history", "reference",
]


# ── SQL templates ──────────────────────────────────────────────────────────────

def _build_update_sql():
    set_clause = ", ".join(f"{c} = %s" for c in UPDATE_COLUMNS)
    return f"UPDATE {SCHEMA}.document SET {set_clause} WHERE doc_id = %s"

UPDATE_DOC_SQL = _build_update_sql()

UPDATE_CHILD_SOURCE_FILE_SQL = (
    # Filled in per-table at runtime
    "UPDATE {SCHEMA}.{table} SET source_file = %s WHERE doc_id = %s AND source_file IS NULL"
)

BACKFILL_PROGRESS_DDL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.backfill_progress (
    file_name      text PRIMARY KEY,
    status         text NOT NULL,
    started_at     timestamptz,
    finished_at    timestamptz,
    articles_seen  bigint DEFAULT 0,
    docs_updated   bigint DEFAULT 0,
    error_detail   text
)
"""


def ensure_progress_table(conn):
    with conn.cursor() as cur:
        cur.execute(BACKFILL_PROGRESS_DDL)
    conn.commit()


def check_backfill_done(conn, file_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT status FROM {SCHEMA}.backfill_progress WHERE file_name = %s",
            (file_name,),
        )
        row = cur.fetchone()
        return bool(row and row[0] == "done")


def mark_backfill(conn, file_name: str, status: str, **kwargs):
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {SCHEMA}.backfill_progress
                (file_name, status, started_at, articles_seen, docs_updated, error_detail)
            VALUES (%s, %s, now(), %s, %s, %s)
            ON CONFLICT (file_name) DO UPDATE SET
                status = EXCLUDED.status,
                finished_at = CASE WHEN EXCLUDED.status IN ('done','error')
                                   THEN now() ELSE NULL END,
                articles_seen = COALESCE(EXCLUDED.articles_seen,
                                         {SCHEMA}.backfill_progress.articles_seen),
                docs_updated = COALESCE(EXCLUDED.docs_updated,
                                        {SCHEMA}.backfill_progress.docs_updated),
                error_detail = EXCLUDED.error_detail
            """,
            (file_name, status, kwargs.get("articles_seen", 0),
             kwargs.get("docs_updated", 0), kwargs.get("error_detail")),
        )
    conn.commit()


# ── Per-article backfill ──────────────────────────────────────────────────────

def backfill_one_article(cur, article_el, source_file: str, dry_run: bool = False) -> bool:
    """Parse one JATS article and apply backfill UPDATEs/INSERTs.
    Returns True if the document row existed and was updated."""
    data = extract_article(article_el, source="backfill", source_file=source_file)
    if data is None:
        return False
    doc = data["document"]
    doc_id = doc["doc_id"]

    # UPDATE europepmc.document for the new columns only
    update_values = tuple(doc.get(c) for c in UPDATE_COLUMNS) + (doc_id,)
    if not dry_run:
        cur.execute(UPDATE_DOC_SQL, update_values)
        if cur.rowcount == 0:
            # doc_id not present in europepmc.document — not a backfill target
            return False

    # INSERT into the 5 new tables (currently empty, no conflict risk)
    for table_name, columns in NEW_CHILD_TABLES.items():
        rows = data.get(table_name, [])
        if not rows:
            continue
        if not dry_run:
            cols = ", ".join(columns)
            sql = f"INSERT INTO {SCHEMA}.{table_name} ({cols}) VALUES %s"
            values = [tuple(r[c] for c in columns) for r in rows]
            psycopg2.extras.execute_values(cur, sql, values)

    # Backfill source_file on pre-existing child tables (only rows with NULL source_file)
    if not dry_run:
        for table in SOURCE_FILE_CHILD_TABLES:
            cur.execute(
                f"UPDATE {SCHEMA}.{table} SET source_file = %s "
                f"WHERE doc_id = %s AND source_file IS NULL",
                (source_file, doc_id),
            )
    return True


# ── Tar.gz iteration ──────────────────────────────────────────────────────────

def backfill_tar(conn, tar_path: Path, dry_run: bool = False, limit: int | None = None) -> dict:
    file_name = tar_path.name
    mark_backfill(conn, file_name, "processing")
    t0 = time.time()
    seen = 0
    updated = 0
    errors = 0
    with conn.cursor() as cur:
        try:
            with tarfile.open(str(tar_path), "r:gz") as tf:
                for member in tf:
                    if not member.isfile() or not member.name.endswith(".xml"):
                        continue
                    if ABORT_FLAG.exists():
                        print("Aborted by monitor — stopping")
                        break
                    if limit is not None and seen >= limit:
                        break
                    try:
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        xml_bytes = f.read()
                        xml_bytes = _strip_entities(xml_bytes)
                        tree = etree.fromstring(xml_bytes, parser=etree.XMLParser(
                            huge_tree=True, recover=True))
                        seen += 1
                        if backfill_one_article(cur, tree, file_name, dry_run=dry_run):
                            updated += 1
                        if seen % BATCH_SIZE == 0:
                            if not dry_run:
                                conn.commit()
                            print(f"  {file_name}: {seen:,} seen, {updated:,} updated")
                    except Exception as e:
                        errors += 1
                        traceback.print_exc()
                        continue
        except Exception as e:
            print(f"TAR ERROR: {e}")
            mark_backfill(conn, file_name, "error",
                          articles_seen=seen, docs_updated=updated,
                          error_detail=str(e)[:500])
            conn.rollback()
            return {"seen": seen, "updated": updated, "errors": errors + 1,
                    "elapsed": time.time() - t0}
    if not dry_run:
        conn.commit()
    mark_backfill(conn, file_name, "done",
                  articles_seen=seen, docs_updated=updated)
    return {"seen": seen, "updated": updated, "errors": errors,
            "elapsed": time.time() - t0}


# ── Xml.gz iteration (for legacy --dir paths with PMC*_PMC*.xml.gz files) ─────

def backfill_gz(conn, gz_path: Path, dry_run: bool = False, limit: int | None = None) -> dict:
    file_name = gz_path.name
    mark_backfill(conn, file_name, "processing")
    t0 = time.time()
    with gzip.open(gz_path, "rb") as f:
        raw = f.read()
    raw = _strip_entities(raw)
    parser = etree.XMLParser(huge_tree=True, recover=True)
    tree = etree.fromstring(raw, parser=parser)
    if tree.tag == "articles":
        articles = list(tree.findall("article"))
    elif tree.tag == "article":
        articles = [tree]
    else:
        articles = list(tree.findall(".//article"))
    seen = 0
    updated = 0
    errors = 0
    with conn.cursor() as cur:
        for article_el in articles:
            if ABORT_FLAG.exists():
                break
            if limit is not None and seen >= limit:
                break
            try:
                seen += 1
                if backfill_one_article(cur, article_el, file_name, dry_run=dry_run):
                    updated += 1
                if seen % BATCH_SIZE == 0 and not dry_run:
                    conn.commit()
            except Exception:
                errors += 1
                traceback.print_exc()
                continue
    if not dry_run:
        conn.commit()
    mark_backfill(conn, file_name, "done",
                  articles_seen=seen, docs_updated=updated)
    return {"seen": seen, "updated": updated, "errors": errors,
            "elapsed": time.time() - t0}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Backfill new schema fields for already-ingested europepmc rows")
    ap.add_argument("--pmc-dir", type=Path,
                    help="Directory of PMC OA tar.gz files (oa_comm / oa_noncomm)")
    ap.add_argument("--dir", type=Path,
                    help="Directory of PMC*_PMC*.xml.gz files (legacy format)")
    ap.add_argument("--source", default="oa_comm",
                    choices=["oa_comm", "oa_noncomm", "all"],
                    help="Filter tar files by source prefix (default: oa_comm)")
    ap.add_argument("--file", help="Process only files matching this substring")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse but don't execute SQL")
    ap.add_argument("--limit", type=int,
                    help="Process at most N articles per file (for smoke testing)")
    args = ap.parse_args()

    if not args.pmc_dir and not args.dir:
        ap.error("Must specify --pmc-dir or --dir")

    conn = psycopg2.connect(**DB_PARAMS)
    ensure_progress_table(conn)

    # Collect target files
    if args.pmc_dir:
        if args.source == "oa_comm":
            targets = sorted(args.pmc_dir.glob("oa_comm*.tar.gz"))
        elif args.source == "oa_noncomm":
            targets = sorted(args.pmc_dir.glob("oa_noncomm*.tar.gz"))
        else:
            targets = sorted(args.pmc_dir.glob("*.tar.gz"))
        runner = backfill_tar
    else:
        targets = sorted(args.dir.glob("PMC*_PMC*.xml.gz"))
        runner = backfill_gz

    if args.file:
        targets = [f for f in targets if args.file in f.name]

    if not targets:
        print(f"No files found")
        sys.exit(1)

    print(f"[{datetime.utcnow().isoformat()}Z] europepmc backfill")
    print(f"  Files: {len(targets)}")
    print(f"  Source filter: {args.source}")
    print(f"  Dry-run: {args.dry_run}")

    grand_seen = 0
    grand_updated = 0
    grand_errors = 0
    skipped = 0
    t_all = time.time()

    for i, path in enumerate(targets, 1):
        if check_backfill_done(conn, path.name) and not args.dry_run:
            skipped += 1
            continue
        if ABORT_FLAG.exists():
            print("Aborted by monitor — stopping")
            break
        print(f"[{i:>3}/{len(targets)}] {path.name} …")
        stats = runner(conn, path, dry_run=args.dry_run, limit=args.limit)
        grand_seen += stats["seen"]
        grand_updated += stats["updated"]
        grand_errors += stats["errors"]
        elapsed_all = time.time() - t_all
        print(f"  [{i}/{len(targets)}] {path.name}: "
              f"{stats['seen']:,} seen, {stats['updated']:,} updated, "
              f"{stats['errors']} err, {stats['elapsed']:.1f}s "
              f"(total: {grand_updated:,} updated, {elapsed_all/3600:.1f}h)")

    print(f"\n[{datetime.utcnow().isoformat()}Z] BACKFILL COMPLETE: "
          f"{grand_seen:,} seen, {grand_updated:,} updated, "
          f"{grand_errors} errors, {skipped} skipped, "
          f"{(time.time()-t_all)/3600:.1f}h")
    conn.close()


if __name__ == "__main__":
    main()

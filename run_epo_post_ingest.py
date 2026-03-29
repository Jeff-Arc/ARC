#!/usr/bin/env python3
"""
run_epo_post_ingest.py — Post-ingest tasks for normalized EPO DOCDB tables.

Run after run_epo_ingest.py completes all 162 files:
  1. Create indexes (sequential, one at a time)
  2. Add UNIQUE constraint on epo_person
  3. Add FK constraints (NOT VALID, then VALIDATE)
  4. ANALYZE all tables
  5. Create dedup view
  6. Report final counts and field fill rates

Usage:
  python3 run_epo_post_ingest.py
  python3 run_epo_post_ingest.py --step indexes    # run only index creation
  python3 run_epo_post_ingest.py --step analyze     # run only ANALYZE
  python3 run_epo_post_ingest.py --step counts      # report counts only
"""

import argparse
import sys
import time
from pathlib import Path

import psycopg2

PROGRESS_FILE = Path("/tmp/epo_post_ingest_progress.txt")


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(PROGRESS_FILE, "a") as f:
        f.write(line + "\n")


# ── Index definitions ────────────────────────────────────────────────────────

INDEXES = [
    # Core document
    ("idx_epo_doc_country",    "arc_v5.epo_document", "(country)"),
    ("idx_epo_doc_family",     "arc_v5.epo_document", "(family_id)"),
    ("idx_epo_doc_pubdate",    "arc_v5.epo_document", "(date_publ)"),
    ("idx_epo_doc_lang",       "arc_v5.epo_document", "(language_of_publication)"),
    ("idx_epo_doc_zip",        "arc_v5.epo_document", "(zip_source)"),
    # Title
    ("idx_epo_title_docid",    "arc_v5.epo_title", "(doc_id)"),
    ("idx_epo_title_lang",     "arc_v5.epo_title", "(lang)"),
    # Abstract
    ("idx_epo_abs_docid",      "arc_v5.epo_abstract", "(doc_id)"),
    # IPCR — doc_id + parsed fields for IPC-to-CPC matching
    ("idx_epo_ipcr_docid",     "arc_v5.epo_classification_ipcr", "(doc_id)"),
    ("idx_epo_ipcr_subclass",  "arc_v5.epo_classification_ipcr", "(ipc_subclass)"),
    ("idx_epo_ipcr_symbol",    "arc_v5.epo_classification_ipcr", "(ipc_symbol)"),
    # Patent classification (CPC) — doc_id + parsed fields
    ("idx_epo_patcls_docid",   "arc_v5.epo_patent_classification", "(doc_id)"),
    ("idx_epo_patcls_symbol",  "arc_v5.epo_patent_classification",
     "(classification_symbol text_pattern_ops)"),
    ("idx_epo_patcls_subclass", "arc_v5.epo_patent_classification", "(cpc_subclass)"),
    # IPC (old)
    ("idx_epo_ipc_docid",      "arc_v5.epo_classification_ipc", "(doc_id)"),
    # Person
    ("idx_epo_person_name",    "arc_v5.epo_person", "(name_docdb)"),
    # Document-applicant
    ("idx_epo_docapp_docid",   "arc_v5.epo_document_applicant", "(doc_id)"),
    ("idx_epo_docapp_person",  "arc_v5.epo_document_applicant", "(person_id)"),
    # Document-inventor
    ("idx_epo_docinv_docid",   "arc_v5.epo_document_inventor", "(doc_id)"),
    ("idx_epo_docinv_person",  "arc_v5.epo_document_inventor", "(person_id)"),
    # Priority claim
    ("idx_epo_prio_docid",     "arc_v5.epo_priority_claim", "(doc_id)"),
    # Patent citation
    ("idx_epo_citpat_docid",   "arc_v5.epo_citation_patent", "(doc_id)"),
    ("idx_epo_citpat_cited",   "arc_v5.epo_citation_patent",
     "(cited_country, cited_doc_number)"),
    # NPL citation
    ("idx_epo_citnpl_docid",   "arc_v5.epo_citation_npl", "(doc_id)"),
    # National classification
    ("idx_epo_natcls_docid",   "arc_v5.epo_classification_national", "(doc_id)"),
    # Designated state
    ("idx_epo_desig_docid",    "arc_v5.epo_designated_state", "(doc_id)"),
    # Related document
    ("idx_epo_reldoc_docid",   "arc_v5.epo_related_document", "(doc_id)"),
]

# Person unique constraint (added post-load)
PERSON_UNIQUE = (
    "uq_epo_person_name_country",
    "arc_v5.epo_person",
    "(name_docdb, country)",
)


# ── Steps ────────────────────────────────────────────────────────────────────

def step_indexes(conn):
    """Create all indexes, one at a time."""
    log("=== STEP: CREATE INDEXES ===")
    cur = conn.cursor()
    cur.execute("SET maintenance_work_mem = '1GB'")
    cur.execute("SET max_parallel_maintenance_workers = 0")
    conn.commit()

    for idx_name, table, idx_def in INDEXES:
        log(f"  Creating {idx_name} on {table}...")
        t0 = time.time()
        try:
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} {idx_def}")
            conn.commit()
            elapsed = time.time() - t0
            log(f"  {idx_name} done ({elapsed:.1f}s)")
        except Exception as e:
            conn.rollback()
            log(f"  ERROR creating {idx_name}: {e}")

    # Person unique constraint
    log(f"  Creating {PERSON_UNIQUE[0]}...")
    t0 = time.time()
    try:
        cur.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {PERSON_UNIQUE[0]} "
            f"ON {PERSON_UNIQUE[1]} {PERSON_UNIQUE[2]}")
        conn.commit()
        log(f"  {PERSON_UNIQUE[0]} done ({time.time() - t0:.1f}s)")
    except Exception as e:
        conn.rollback()
        log(f"  ERROR: {e}")

    cur.close()
    log("=== INDEXES COMPLETE ===")


def step_analyze(conn):
    """ANALYZE all epo_ tables."""
    log("=== STEP: ANALYZE ===")
    cur = conn.cursor()
    cur.execute("""
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'arc_v5' AND tablename LIKE 'epo_%'
        ORDER BY tablename
    """)
    tables = [r[0] for r in cur.fetchall()]

    for t in tables:
        log(f"  ANALYZE arc_v5.{t}...")
        t0 = time.time()
        cur.execute(f"ANALYZE arc_v5.{t}")
        conn.commit()
        log(f"  done ({time.time() - t0:.1f}s)")

    cur.close()
    log("=== ANALYZE COMPLETE ===")


def step_dedup_view(conn):
    """Create the dedup view."""
    log("=== STEP: CREATE DEDUP VIEW ===")
    cur = conn.cursor()
    cur.execute("""
        CREATE OR REPLACE VIEW arc_v5.epo_document_dedup AS
        SELECT DISTINCT ON (d.family_id) d.*,
            t.title_text AS title_en,
            a.abstract_text AS abstract_en
        FROM arc_v5.epo_document d
        LEFT JOIN arc_v5.epo_title t
            ON t.doc_id = d.doc_id AND t.lang = 'en'
        LEFT JOIN arc_v5.epo_abstract a
            ON a.doc_id = d.doc_id AND a.lang = 'en'
        WHERE EXISTS (
            SELECT 1 FROM arc_v5.epo_abstract a2 WHERE a2.doc_id = d.doc_id
        )
        ORDER BY d.family_id,
            CASE d.country
                WHEN 'EP' THEN 1 WHEN 'US' THEN 2
                WHEN 'WO' THEN 3 ELSE 4
            END,
            d.date_publ DESC
    """)
    conn.commit()
    cur.close()
    log("=== DEDUP VIEW CREATED ===")


def step_counts(conn):
    """Report final counts for all tables."""
    log("=== STEP: FINAL COUNTS ===")
    cur = conn.cursor()
    cur.execute("""
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'arc_v5' AND tablename LIKE 'epo_%'
        ORDER BY tablename
    """)
    tables = [r[0] for r in cur.fetchall()]

    for t in tables:
        cur.execute(f"SELECT count(*) FROM arc_v5.{t}")
        cnt = cur.fetchone()[0]
        cur.execute(
            f"SELECT pg_size_pretty(pg_total_relation_size('arc_v5.{t}'))")
        size = cur.fetchone()[0]
        log(f"  {t:<35} {cnt:>15,} rows  {size:>10}")

    # Total size
    cur.execute("""
        SELECT pg_size_pretty(sum(pg_total_relation_size(
            schemaname || '.' || tablename)))
        FROM pg_tables
        WHERE schemaname = 'arc_v5' AND tablename LIKE 'epo_%'
    """)
    total = cur.fetchone()[0]
    log(f"  {'TOTAL':<35} {'':>15}  {total:>10}")

    # Field fill rates on core document
    cur.execute("""
        SELECT count(*) as total,
               count(family_id) as has_family,
               count(date_publ) as has_pubdate,
               count(language_of_publication) as has_lang,
               count(originating_office) as has_office
        FROM arc_v5.epo_document
    """)
    r = cur.fetchone()
    total = r[0]
    if total > 0:
        log(f"\n  Document fill rates:")
        labels = ['family_id', 'date_publ', 'language', 'originating_office']
        for label, val in zip(labels, r[1:]):
            log(f"    {label}: {val:,} ({100*val/total:.1f}%)")

    # Parsed classification fill rates
    cur.execute("""
        SELECT count(*) as total,
               count(ipc_section) as parsed
        FROM arc_v5.epo_classification_ipcr
    """)
    r = cur.fetchone()
    if r[0] > 0:
        log(f"  IPCR parsed: {r[1]:,}/{r[0]:,} ({100*r[1]/r[0]:.1f}%)")

    cur.execute("""
        SELECT count(*) as total,
               count(cpc_section) as parsed
        FROM arc_v5.epo_patent_classification
    """)
    r = cur.fetchone()
    if r[0] > 0:
        log(f"  CPC parsed: {r[1]:,}/{r[0]:,} ({100*r[1]/r[0]:.1f}%)")

    cur.close()
    log("=== COUNTS COMPLETE ===")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Post-ingest tasks for EPO DOCDB tables")
    ap.add_argument("--step", choices=["indexes", "analyze", "view",
                                        "counts", "all"],
                    default="all",
                    help="Run specific step (default: all)")
    args = ap.parse_args()

    conn = psycopg2.connect(
        host="/var/run/postgresql", dbname="arc_v4", user="jeff")
    conn.autocommit = True

    steps = {
        "indexes": step_indexes,
        "analyze": step_analyze,
        "view": step_dedup_view,
        "counts": step_counts,
    }

    if args.step == "all":
        for name, func in steps.items():
            func(conn)
    else:
        steps[args.step](conn)

    conn.close()
    log("Done.")


if __name__ == "__main__":
    main()

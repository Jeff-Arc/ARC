#!/usr/bin/env python3
"""
arc_rosetta_build.py — Phase 1 ID linkage + metadata enrichment for arc_v5.rosetta_stone

Builds a canonical cross-source mapping table linking records across:
  - openalex.work         (DOI-seeded primary, largest DOI coverage)
  - pubmed.document       (PMID linkage)
  - semanticscholar.paper (DOI then PMID linkage)
  - europepmc.document    (DOI then PMID then PMCID linkage)
  - crossref.document     (DOI linkage)

Build passes (resumable per-pass via arc_v5.rosetta_stone_progress):

  Phase 1a — ID linkage (rows created/linked, source-specific columns set):
    Pass  1 : Seed OpenAlex DOI-bearing rows
    Pass  2 : Seed OpenAlex PMID/PMCID-only rows (no DOI)   [sub-pass of 1]
    Pass  3 : Link PubMed      (DOI match → insert remaining)
    Pass  4 : Link SemanticScholar (DOI → PMID → insert)
    Pass  5 : Link EuropePMC   (DOI → PMID → PMCID → insert)
    Pass  6 : Link Crossref    (DOI only)

  Phase 1b — Metadata enrichment (highest priority first, "set only if NULL"):
    Pass  7 : OpenAlex metadata (highest abstract priority — openalex.work.abstract_text)
    Pass  8 : PubMed metadata
    Pass  9 : EuropePMC metadata
    Pass 10 : SemanticScholar metadata
    Pass 11 : Crossref metadata (publisher/type, abstract if still NULL)

  Abstract priority chain (first non-null wins):
    1. openalex.work.abstract_text          (271M, already plain text post-ingest)
    2. pubmed.abstract_section              (string_agg via 'PMID:' || pmid)
    3. europepmc.abstract                   (string_agg via 'PMC:' || europepmc_pmcid)
    4. semanticscholar.paper.abstract       (direct column)
    5. crossref.document.abstract           (JATS-XML, last resort)

  Phase 1c — Fulltext detection:
    Pass 12 : Set has_fulltext where europepmc.body_section has rows for the PMCID

Indexes are built lazily between passes to avoid maintaining empty indexes during
the big Pass 1 seed insert (~250M rows). See PASS_INDEX_BUILDS below.

Usage:
    python arc_rosetta_build.py                      # run all pending passes
    python arc_rosetta_build.py --only-pass 3        # run a single pass
    python arc_rosetta_build.py --from-pass 7        # run passes 7..12
    python arc_rosetta_build.py --dry-run            # print plan, execute nothing
    python arc_rosetta_build.py --reset-progress N   # clear progress row for pass N

Logs to /tmp/rosetta_build.log plus stderr.

Phase 1 scope NOTES:
  * epo_doc_id / epo_family_id columns exist in the schema but are not populated
    by any pass — EPO patents don't carry DOI/PMID and require Phase 2 title
    matching. Columns stay NULL.
  * s2_tldr column exists but is NOT populated (S2 schema doesn't carry TLDR).
    Could be backfilled later from the S2 API.
  * Abstract priority is openalex > pubmed > europepmc > s2 > crossref.
    openalex.work.abstract_text is plain text (inverted index was expanded at
    ingest) with ~271M rows of coverage — by far the largest source.
  * Authors will live in a separate arc_v5.rosetta_authors table (not in Phase 1).
  * arxiv fulltext is not currently linked to has_fulltext. arxiv_fulltext joins
    by arxiv_id and rosetta has no arxiv_id column. Future phase.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional

import psycopg2
import psycopg2.extensions


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DSN = "dbname=arc_v5"
LOG_PATH = "/tmp/rosetta_build.log"
TABLESPACE = "raid_ts"

# Per-session tunings for faster bulk ops
SESSION_TUNINGS = [
    "SET maintenance_work_mem = '4GB'",
    "SET work_mem = '512MB'",
    "SET synchronous_commit = off",         # speeds up bulk INSERTs
    "SET temp_tablespaces = 'raid_ts'",     # root disk is 85% full; use RAID for temps
    "SET statement_timeout = 0",
]


# ---------------------------------------------------------------------------
# Normalization SQL fragments (used inside CREATE TEMP TABLE AS queries)
# ---------------------------------------------------------------------------
# Findings from format sampling (2026-04-08):
#   - No source uses URL-prefixed DOIs; LOWER(TRIM) alone is sufficient.
#   - SemanticScholar has 47% uppercase DOIs; LOWER is MANDATORY for cross-source join.
#   - PMIDs are all bare digit strings across sources.
#   - PMCIDs in europepmc.document are "PMC12345" form (uppercase, prefixed).
#   - openalex.work.doc_id is "https://openalex.org/W..."; strip prefix to bare W-id.
#   - openalex.work.abstract_text IS plain text (inverted index expanded at ingest),
#     so OpenAlex sits at the top of the abstract priority chain.

NORM_DOI   = "NULLIF(LOWER(TRIM({col})), '')"
NORM_PMID  = "NULLIF(TRIM({col}), '')"
NORM_PMCID = "NULLIF(UPPER(TRIM({col})), '')"
# Strip OpenAlex URL prefix to get bare W-id
NORM_OAID  = "NULLIF(REPLACE({col}, 'https://openalex.org/', ''), '')"


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

DDL_ROSETTA = f"""
CREATE TABLE IF NOT EXISTS arc_v5.rosetta_stone (
    canonical_id             bigserial PRIMARY KEY,

    -- Cross-source identifiers (canonical, normalized)
    doi                      text,
    pmid                     text,
    pmcid                    text,
    openalex_id              text,
    s2_id                    text,

    -- EPO patent identifiers (Phase 2 — left NULL in Phase 1)
    epo_doc_id               text,
    epo_family_id            text,

    -- Universal metadata (populated by Phase 1b, priority-based)
    title                    text,
    abstract                 text,
    pub_year                 int,
    journal                  text,
    has_fulltext             boolean DEFAULT false,

    -- OpenAlex-specific
    openalex_type            text,
    openalex_cited_by_count  int,

    -- PubMed-specific
    pubmed_mesh_terms        text[],

    -- Semantic Scholar-specific
    s2_fields_of_study       text[],
    s2_tldr                  text,   -- not populated in Phase 1 (no source column)

    -- EuropePMC-specific
    europepmc_pmcid          text,   -- set when EPMC record present (join key to body_section)

    -- Crossref-specific
    crossref_publisher       text,
    crossref_type            text,

    -- Linkage quality
    source_count             smallint DEFAULT 1,
    confidence               float DEFAULT 1.0,
    created_at               timestamptz DEFAULT now()
) TABLESPACE {TABLESPACE};
"""

DDL_PROGRESS = f"""
CREATE TABLE IF NOT EXISTS arc_v5.rosetta_stone_progress (
    pass_num        smallint PRIMARY KEY,
    pass_name       text NOT NULL,
    started_at      timestamptz,
    finished_at     timestamptz,
    rows_inserted   bigint DEFAULT 0,
    rows_updated    bigint DEFAULT 0,
    notes           text
) TABLESPACE {TABLESPACE};
"""

# Partial indexes — created lazily, see PASS_INDEX_BUILDS
INDEXES = {
    "idx_rosetta_doi":
        "CREATE INDEX IF NOT EXISTS idx_rosetta_doi "
        "ON arc_v5.rosetta_stone(doi) WHERE doi IS NOT NULL",
    "idx_rosetta_pmid":
        "CREATE INDEX IF NOT EXISTS idx_rosetta_pmid "
        "ON arc_v5.rosetta_stone(pmid) WHERE pmid IS NOT NULL",
    "idx_rosetta_pmcid":
        "CREATE INDEX IF NOT EXISTS idx_rosetta_pmcid "
        "ON arc_v5.rosetta_stone(pmcid) WHERE pmcid IS NOT NULL",
    "idx_rosetta_openalex":
        "CREATE INDEX IF NOT EXISTS idx_rosetta_openalex "
        "ON arc_v5.rosetta_stone(openalex_id) WHERE openalex_id IS NOT NULL",
    "idx_rosetta_s2":
        "CREATE INDEX IF NOT EXISTS idx_rosetta_s2 "
        "ON arc_v5.rosetta_stone(s2_id) WHERE s2_id IS NOT NULL",
    "idx_rosetta_europepmc_pmcid":
        "CREATE INDEX IF NOT EXISTS idx_rosetta_europepmc_pmcid "
        "ON arc_v5.rosetta_stone(europepmc_pmcid) WHERE europepmc_pmcid IS NOT NULL",
}

# Which indexes to build after which pass. Rationale:
#   - doi built right after Pass 2 (OpenAlex PMID-only seed done): passes 3-6 all need doi lookup
#   - pmid built after Pass 3 (PubMed linked): pass 4 S2 and pass 5 EPMC need pmid fallback
#   - s2 built after Pass 4 (S2 linked): pass 9 metadata needs s2 lookup
#   - pmcid + europepmc_pmcid built after Pass 5 (EPMC linked): pass 12 has_fulltext needs pmcid
#   - openalex built after Pass 6 (all linkage done): pass 10 metadata needs openalex lookup
PASS_INDEX_BUILDS: dict[int, list[str]] = {
    2:  ["idx_rosetta_doi"],
    3:  ["idx_rosetta_pmid"],
    4:  ["idx_rosetta_s2"],
    5:  ["idx_rosetta_pmcid", "idx_rosetta_europepmc_pmcid"],
    6:  ["idx_rosetta_openalex"],
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("rosetta_build")


def setup_logging(verbose: bool = False) -> None:
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    fh = logging.FileHandler(LOG_PATH)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    log.addHandler(sh)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

@contextmanager
def connect(dsn: str = DSN):
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            for stmt in SESSION_TUNINGS:
                cur.execute(stmt)
        conn.commit()
        yield conn
    finally:
        conn.close()


def exec_step(conn, sql: str, label: str, dry_run: bool = False) -> int:
    """Run a single SQL statement, log timing + rowcount. Returns rowcount."""
    log.info("  ▸ %s", label)
    if dry_run:
        log.info("    [dry-run] %s", _short(sql))
        return 0
    t0 = time.monotonic()
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.rowcount if cur.rowcount is not None else -1
    elapsed = time.monotonic() - t0
    log.info("    ✓ %s rows=%s in %.1fs", label, f"{rows:,}" if rows >= 0 else "-", elapsed)
    return rows


def _short(sql: str, n: int = 160) -> str:
    s = " ".join(sql.split())
    return s if len(s) <= n else s[:n] + "…"


def ensure_index(conn, name: str, dry_run: bool = False) -> None:
    """Create a single index if absent. Uses plain CREATE INDEX (not CONCURRENTLY)
    because we're already in a maintenance window — no live writers — and
    CONCURRENTLY would require autocommit mode which we don't want mid-pass."""
    ddl = INDEXES[name]
    log.info("  ◆ ensuring %s", name)
    if dry_run:
        log.info("    [dry-run] %s", _short(ddl))
        return
    t0 = time.monotonic()
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    log.info("    ✓ %s built in %.1fs", name, time.monotonic() - t0)


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------

def ensure_schema(conn, dry_run: bool = False) -> None:
    log.info("Ensuring schema arc_v5.rosetta_stone + progress table…")
    if dry_run:
        log.info("  [dry-run] CREATE TABLE arc_v5.rosetta_stone (...) TABLESPACE raid_ts")
        log.info("  [dry-run] CREATE TABLE arc_v5.rosetta_stone_progress (...)")
        return
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_tablespace WHERE spcname = %s", (TABLESPACE,))
        if cur.fetchone() is None:
            raise RuntimeError(f"Tablespace {TABLESPACE} does not exist")
        cur.execute(DDL_ROSETTA)
        cur.execute(DDL_PROGRESS)
    conn.commit()
    log.info("  ✓ schema ready")


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def pass_status(conn, pass_num: int) -> Optional[str]:
    """Returns 'done', 'running', 'failed', or None."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT CASE WHEN finished_at IS NOT NULL THEN 'done' "
            "ELSE 'running' END "
            "FROM arc_v5.rosetta_stone_progress WHERE pass_num = %s",
            (pass_num,),
        )
        row = cur.fetchone()
    return row[0] if row else None


def mark_started(conn, pass_num: int, pass_name: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO arc_v5.rosetta_stone_progress (pass_num, pass_name, started_at) "
            "VALUES (%s, %s, now()) "
            "ON CONFLICT (pass_num) DO UPDATE "
            "SET started_at = now(), finished_at = NULL, pass_name = EXCLUDED.pass_name",
            (pass_num, pass_name),
        )
    conn.commit()


def mark_done(conn, pass_num: int, inserted: int = 0, updated: int = 0, notes: str = "") -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE arc_v5.rosetta_stone_progress "
            "SET finished_at = now(), rows_inserted = %s, rows_updated = %s, notes = %s "
            "WHERE pass_num = %s",
            (inserted, updated, notes, pass_num),
        )
    conn.commit()


def reset_pass_progress(conn, pass_num: int) -> None:
    log.warning("Resetting progress marker for pass %d", pass_num)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM arc_v5.rosetta_stone_progress WHERE pass_num = %s", (pass_num,))
    conn.commit()


# ---------------------------------------------------------------------------
# Phase 1a — ID linkage passes
# ---------------------------------------------------------------------------

def pass_1_openalex_doi_seed(conn, dry_run: bool = False) -> dict:
    """Seed rosetta with OpenAlex works that have a DOI.
    Columns set: doi, openalex_id. source_count=1.
    """
    log.info("==== Pass 1: OpenAlex DOI seed ====")
    inserted = exec_step(
        conn,
        f"""
        INSERT INTO arc_v5.rosetta_stone (doi, openalex_id, source_count)
        SELECT DISTINCT ON ({NORM_DOI.format(col='doi')})
            {NORM_DOI.format(col='doi')}   AS doi,
            {NORM_OAID.format(col='doc_id')} AS openalex_id,
            1
        FROM openalex.work
        WHERE doi IS NOT NULL AND doi <> ''
        ORDER BY {NORM_DOI.format(col='doi')}, doc_id
        """,
        "INSERT seed from openalex.work (DOI-bearing)",
        dry_run=dry_run,
    )
    if not dry_run:
        conn.commit()
    return {"inserted": max(inserted, 0), "updated": 0}


def pass_2_openalex_pmid_pmcid_seed(conn, dry_run: bool = False) -> dict:
    """Seed OpenAlex rows that have NO DOI but have PMID or PMCID.
    These would otherwise be lost (Pass 1 filters them out)."""
    log.info("==== Pass 2: OpenAlex PMID/PMCID seed (no-DOI rows) ====")
    inserted = exec_step(
        conn,
        f"""
        INSERT INTO arc_v5.rosetta_stone (pmid, pmcid, openalex_id, source_count)
        SELECT DISTINCT ON ({NORM_OAID.format(col='doc_id')})
            {NORM_PMID.format(col='pmid')}   AS pmid,
            {NORM_PMCID.format(col='pmcid')} AS pmcid,
            {NORM_OAID.format(col='doc_id')} AS openalex_id,
            1
        FROM openalex.work
        WHERE (doi IS NULL OR doi = '')
          AND (pmid IS NOT NULL OR pmcid IS NOT NULL)
        ORDER BY {NORM_OAID.format(col='doc_id')}
        """,
        "INSERT seed from openalex.work (PMID/PMCID-only)",
        dry_run=dry_run,
    )
    if not dry_run:
        conn.commit()
    return {"inserted": max(inserted, 0), "updated": 0}


def pass_3_pubmed_link(conn, dry_run: bool = False) -> dict:
    """Link PubMed records via DOI; insert remaining by PMID."""
    log.info("==== Pass 3: PubMed linkage ====")

    exec_step(
        conn,
        f"""
        CREATE TEMP TABLE _stage_pubmed ON COMMIT DROP AS
        SELECT DISTINCT ON (pmid)
               {NORM_PMID.format(col='pmid')} AS pmid,
               {NORM_DOI.format(col='doi')}   AS doi
        FROM pubmed.document
        WHERE pmid IS NOT NULL
        ORDER BY pmid, doi NULLS LAST
        """,
        "stage pubmed.document → _stage_pubmed",
        dry_run=dry_run,
    )
    exec_step(
        conn,
        "CREATE INDEX ON _stage_pubmed(doi) WHERE doi IS NOT NULL",
        "index _stage_pubmed(doi)",
        dry_run=dry_run,
    )
    exec_step(conn, "ANALYZE _stage_pubmed", "ANALYZE _stage_pubmed", dry_run=dry_run)

    updated = exec_step(
        conn,
        """
        UPDATE arc_v5.rosetta_stone r
        SET pmid = s.pmid,
            source_count = source_count + 1
        FROM _stage_pubmed s
        WHERE r.doi = s.doi
          AND s.doi IS NOT NULL
          AND r.pmid IS NULL
        """,
        "UPDATE rosetta by DOI → set pmid",
        dry_run=dry_run,
    )
    inserted = exec_step(
        conn,
        """
        INSERT INTO arc_v5.rosetta_stone (doi, pmid, source_count)
        SELECT s.doi, s.pmid, 1
        FROM _stage_pubmed s
        WHERE NOT EXISTS (
            SELECT 1 FROM arc_v5.rosetta_stone r
            WHERE r.pmid = s.pmid
        )
        """,
        "INSERT new rosetta rows for unmatched pubmed",
        dry_run=dry_run,
    )
    if not dry_run:
        conn.commit()
    return {"inserted": max(inserted, 0), "updated": max(updated, 0)}


def pass_4_s2_link(conn, dry_run: bool = False) -> dict:
    """Link Semantic Scholar via DOI, then PMID fallback, then insert remaining."""
    log.info("==== Pass 4: SemanticScholar linkage ====")

    exec_step(
        conn,
        f"""
        CREATE TEMP TABLE _stage_s2 ON COMMIT DROP AS
        SELECT DISTINCT ON (s2_id)
               s2_id,
               {NORM_DOI.format(col='doi')}  AS doi,
               {NORM_PMID.format(col='pmid')} AS pmid
        FROM semanticscholar.paper
        WHERE s2_id IS NOT NULL
        ORDER BY s2_id
        """,
        "stage semanticscholar.paper → _stage_s2",
        dry_run=dry_run,
    )
    exec_step(
        conn,
        "CREATE INDEX ON _stage_s2(doi) WHERE doi IS NOT NULL",
        "index _stage_s2(doi)",
        dry_run=dry_run,
    )
    exec_step(
        conn,
        "CREATE INDEX ON _stage_s2(pmid) WHERE pmid IS NOT NULL",
        "index _stage_s2(pmid)",
        dry_run=dry_run,
    )
    exec_step(conn, "ANALYZE _stage_s2", "ANALYZE _stage_s2", dry_run=dry_run)

    upd_doi = exec_step(
        conn,
        """
        UPDATE arc_v5.rosetta_stone r
        SET s2_id = s.s2_id,
            source_count = source_count + 1
        FROM _stage_s2 s
        WHERE r.doi = s.doi
          AND s.doi IS NOT NULL
          AND r.s2_id IS NULL
        """,
        "UPDATE rosetta by DOI → set s2_id",
        dry_run=dry_run,
    )
    upd_pmid = exec_step(
        conn,
        """
        UPDATE arc_v5.rosetta_stone r
        SET s2_id = s.s2_id,
            source_count = source_count + 1
        FROM _stage_s2 s
        WHERE r.pmid = s.pmid
          AND s.pmid IS NOT NULL
          AND r.s2_id IS NULL
        """,
        "UPDATE rosetta by PMID fallback → set s2_id",
        dry_run=dry_run,
    )
    inserted = exec_step(
        conn,
        """
        INSERT INTO arc_v5.rosetta_stone (doi, pmid, s2_id, source_count)
        SELECT s.doi, s.pmid, s.s2_id, 1
        FROM _stage_s2 s
        WHERE NOT EXISTS (
            SELECT 1 FROM arc_v5.rosetta_stone r WHERE r.s2_id = s.s2_id
        )
        """,
        "INSERT new rosetta rows for unmatched s2",
        dry_run=dry_run,
    )
    if not dry_run:
        conn.commit()
    return {"inserted": max(inserted, 0), "updated": max(upd_doi, 0) + max(upd_pmid, 0)}


def pass_5_europepmc_link(conn, dry_run: bool = False) -> dict:
    """Link EuropePMC via DOI → PMID → PMCID cascade; insert remaining."""
    log.info("==== Pass 5: EuropePMC linkage ====")

    exec_step(
        conn,
        f"""
        CREATE TEMP TABLE _stage_epmc ON COMMIT DROP AS
        SELECT DISTINCT ON (doc_id)
               doc_id,
               {NORM_DOI.format(col='doi')}   AS doi,
               {NORM_PMID.format(col='pmid')}  AS pmid,
               {NORM_PMCID.format(col='pmcid')} AS pmcid
        FROM europepmc.document
        WHERE pmcid IS NOT NULL
        ORDER BY doc_id
        """,
        "stage europepmc.document → _stage_epmc",
        dry_run=dry_run,
    )
    exec_step(
        conn,
        "CREATE INDEX ON _stage_epmc(doi) WHERE doi IS NOT NULL",
        "index _stage_epmc(doi)",
        dry_run=dry_run,
    )
    exec_step(
        conn,
        "CREATE INDEX ON _stage_epmc(pmid) WHERE pmid IS NOT NULL",
        "index _stage_epmc(pmid)",
        dry_run=dry_run,
    )
    exec_step(
        conn,
        "CREATE INDEX ON _stage_epmc(pmcid)",
        "index _stage_epmc(pmcid)",
        dry_run=dry_run,
    )
    exec_step(conn, "ANALYZE _stage_epmc", "ANALYZE _stage_epmc", dry_run=dry_run)

    upd_doi = exec_step(
        conn,
        """
        UPDATE arc_v5.rosetta_stone r
        SET pmid            = COALESCE(r.pmid, s.pmid),
            pmcid           = COALESCE(r.pmcid, s.pmcid),
            europepmc_pmcid = s.pmcid,
            source_count    = source_count + 1
        FROM _stage_epmc s
        WHERE r.doi = s.doi
          AND s.doi IS NOT NULL
          AND r.europepmc_pmcid IS NULL
        """,
        "UPDATE rosetta by DOI → set epmc ids",
        dry_run=dry_run,
    )
    upd_pmid = exec_step(
        conn,
        """
        UPDATE arc_v5.rosetta_stone r
        SET pmcid           = COALESCE(r.pmcid, s.pmcid),
            europepmc_pmcid = s.pmcid,
            source_count    = source_count + 1
        FROM _stage_epmc s
        WHERE r.pmid = s.pmid
          AND s.pmid IS NOT NULL
          AND r.europepmc_pmcid IS NULL
        """,
        "UPDATE rosetta by PMID → set epmc ids",
        dry_run=dry_run,
    )
    upd_pmcid = exec_step(
        conn,
        """
        UPDATE arc_v5.rosetta_stone r
        SET europepmc_pmcid = s.pmcid,
            source_count    = source_count + 1
        FROM _stage_epmc s
        WHERE r.pmcid = s.pmcid
          AND r.europepmc_pmcid IS NULL
        """,
        "UPDATE rosetta by PMCID → set europepmc_pmcid",
        dry_run=dry_run,
    )
    inserted = exec_step(
        conn,
        """
        INSERT INTO arc_v5.rosetta_stone (doi, pmid, pmcid, europepmc_pmcid, source_count)
        SELECT s.doi, s.pmid, s.pmcid, s.pmcid, 1
        FROM _stage_epmc s
        WHERE NOT EXISTS (
            SELECT 1 FROM arc_v5.rosetta_stone r
            WHERE r.europepmc_pmcid = s.pmcid
        )
        """,
        "INSERT new rosetta rows for unmatched epmc",
        dry_run=dry_run,
    )
    if not dry_run:
        conn.commit()
    return {
        "inserted": max(inserted, 0),
        "updated": max(upd_doi, 0) + max(upd_pmid, 0) + max(upd_pmcid, 0),
    }


def pass_6_crossref_link(conn, dry_run: bool = False) -> dict:
    """Link Crossref by DOI only. No PMID/PMCID cascade.
    Crossref rows only contribute source_count++ on UPDATE and new-row inserts;
    metadata (publisher/type) is added in Pass 11."""
    log.info("==== Pass 6: Crossref linkage ====")

    exec_step(
        conn,
        f"""
        CREATE TEMP TABLE _stage_crossref ON COMMIT DROP AS
        SELECT DISTINCT ON ({NORM_DOI.format(col='doi')})
               {NORM_DOI.format(col='doi')} AS doi
        FROM crossref.document
        WHERE doi IS NOT NULL AND doi <> ''
        ORDER BY {NORM_DOI.format(col='doi')}
        """,
        "stage crossref.document → _stage_crossref",
        dry_run=dry_run,
    )
    exec_step(
        conn,
        "CREATE INDEX ON _stage_crossref(doi)",
        "index _stage_crossref(doi)",
        dry_run=dry_run,
    )
    exec_step(conn, "ANALYZE _stage_crossref", "ANALYZE _stage_crossref", dry_run=dry_run)

    # NOTE: Pass 6 increments source_count on every match. This is idempotent
    # only at the pass-level (via the progress table) — re-running this pass
    # manually without resetting progress would double-count. Don't bypass.
    updated = exec_step(
        conn,
        """
        UPDATE arc_v5.rosetta_stone r
        SET source_count = source_count + 1
        FROM _stage_crossref s
        WHERE r.doi = s.doi
        """,
        "UPDATE rosetta by DOI → source_count++",
        dry_run=dry_run,
    )
    inserted = exec_step(
        conn,
        """
        INSERT INTO arc_v5.rosetta_stone (doi, source_count)
        SELECT s.doi, 1
        FROM _stage_crossref s
        WHERE NOT EXISTS (
            SELECT 1 FROM arc_v5.rosetta_stone r WHERE r.doi = s.doi
        )
        """,
        "INSERT new rosetta rows for crossref-only DOIs",
        dry_run=dry_run,
    )
    if not dry_run:
        conn.commit()
    return {"inserted": max(inserted, 0), "updated": max(updated, 0)}


# ---------------------------------------------------------------------------
# Phase 1b — Metadata enrichment passes
# ---------------------------------------------------------------------------
# Semantics: "set only if NULL" per universal column (first-source-wins).
# Each pass also sets its own source-specific columns via COALESCE so re-runs
# are idempotent.
#
# Pass order is abstract-priority-high-to-low:
#     openalex > pubmed > europepmc > s2 > crossref
# This way the first pass to fill a universal column wins — the remaining
# passes only fill NULLs.
#
# Metadata enrichment must NOT change source_count (that's Phase 1a's job).
#
# Namespaced doc_id joins for child tables:
#   pubmed.abstract_section.doc_id = 'PMID:' || r.pmid
#   pubmed.mesh_heading.doc_id     = 'PMID:' || r.pmid
#   europepmc.abstract.doc_id      = 'PMC:'  || r.europepmc_pmcid

def pass_8_pubmed_metadata(conn, dry_run: bool = False) -> dict:
    """Enrich rows linked to PubMed with title, abstract, journal, pub_year, mesh_terms.
    All universal fields set only if NULL (OpenAlex ran first as Pass 7, so the
    dominant case is that OpenAlex already filled title/abstract)."""
    log.info("==== Pass 8: PubMed metadata enrichment ====")

    # Stage: pubmed metadata keyed by pmid. Abstract is aggregated from abstract_section.
    exec_step(
        conn,
        """
        CREATE TEMP TABLE _enrich_pubmed ON COMMIT DROP AS
        SELECT
            d.pmid,
            d.article_title                    AS title,
            d.journal_title                    AS journal,
            d.journal_pub_year                 AS pub_year,
            ab.abstract                        AS abstract,
            mh.mesh_terms                      AS mesh_terms
        FROM pubmed.document d
        LEFT JOIN (
            SELECT doc_id, string_agg(abstract_text, E'\\n\\n' ORDER BY sequence) AS abstract
            FROM pubmed.abstract_section
            WHERE abstract_text IS NOT NULL
            GROUP BY doc_id
        ) ab ON ab.doc_id = 'PMID:' || d.pmid
        LEFT JOIN (
            SELECT doc_id, array_agg(DISTINCT descriptor_name ORDER BY descriptor_name) AS mesh_terms
            FROM pubmed.mesh_heading
            WHERE descriptor_name IS NOT NULL
            GROUP BY doc_id
        ) mh ON mh.doc_id = 'PMID:' || d.pmid
        WHERE d.pmid IS NOT NULL
        """,
        "stage pubmed enrichment → _enrich_pubmed",
        dry_run=dry_run,
    )
    exec_step(
        conn,
        "CREATE INDEX ON _enrich_pubmed(pmid)",
        "index _enrich_pubmed(pmid)",
        dry_run=dry_run,
    )
    exec_step(conn, "ANALYZE _enrich_pubmed", "ANALYZE _enrich_pubmed", dry_run=dry_run)

    updated = exec_step(
        conn,
        """
        UPDATE arc_v5.rosetta_stone r
        SET title             = COALESCE(r.title, e.title),
            abstract          = COALESCE(r.abstract, e.abstract),
            journal           = COALESCE(r.journal, e.journal),
            pub_year          = COALESCE(r.pub_year, e.pub_year),
            pubmed_mesh_terms = COALESCE(r.pubmed_mesh_terms, e.mesh_terms)
        FROM _enrich_pubmed e
        WHERE r.pmid = e.pmid
        """,
        "UPDATE rosetta with PubMed metadata",
        dry_run=dry_run,
    )
    if not dry_run:
        conn.commit()
    return {"inserted": 0, "updated": max(updated, 0)}


def pass_9_europepmc_metadata(conn, dry_run: bool = False) -> dict:
    """Enrich rows linked to EuropePMC (has europepmc_pmcid) with title, abstract,
    journal, pub_year. All fields are 'set only if NULL' — openalex/pubmed took priority."""
    log.info("==== Pass 9: EuropePMC metadata enrichment ====")

    exec_step(
        conn,
        f"""
        CREATE TEMP TABLE _enrich_epmc ON COMMIT DROP AS
        SELECT
            d.pmcid,
            d.article_title  AS title,
            d.journal_title  AS journal,
            d.pub_year       AS pub_year,
            ab.abstract      AS abstract
        FROM europepmc.document d
        LEFT JOIN (
            SELECT doc_id,
                   string_agg(abstract_text, E'\\n\\n' ORDER BY id) AS abstract
            FROM europepmc.abstract
            WHERE abstract_text IS NOT NULL
            GROUP BY doc_id
        ) ab ON ab.doc_id = 'PMC:' || d.pmcid
        WHERE d.pmcid IS NOT NULL
        """,
        "stage europepmc enrichment → _enrich_epmc",
        dry_run=dry_run,
    )
    exec_step(
        conn,
        "CREATE INDEX ON _enrich_epmc(pmcid)",
        "index _enrich_epmc(pmcid)",
        dry_run=dry_run,
    )
    exec_step(conn, "ANALYZE _enrich_epmc", "ANALYZE _enrich_epmc", dry_run=dry_run)

    updated = exec_step(
        conn,
        """
        UPDATE arc_v5.rosetta_stone r
        SET title    = COALESCE(r.title, e.title),
            abstract = COALESCE(r.abstract, e.abstract),
            journal  = COALESCE(r.journal, e.journal),
            pub_year = COALESCE(r.pub_year, e.pub_year)
        FROM _enrich_epmc e
        WHERE r.europepmc_pmcid = e.pmcid
        """,
        "UPDATE rosetta with EuropePMC metadata",
        dry_run=dry_run,
    )
    if not dry_run:
        conn.commit()
    return {"inserted": 0, "updated": max(updated, 0)}


def pass_10_s2_metadata(conn, dry_run: bool = False) -> dict:
    """Enrich rows linked to Semantic Scholar with title, abstract, venue as journal,
    year, and fields_of_study array. Set only if NULL."""
    log.info("==== Pass 10: SemanticScholar metadata enrichment ====")

    exec_step(
        conn,
        """
        CREATE TEMP TABLE _enrich_s2 ON COMMIT DROP AS
        SELECT
            p.s2_id,
            p.title,
            p.abstract,
            COALESCE(p.journal_name, p.venue) AS journal,
            p.year                            AS pub_year,
            pf.fields                         AS fields_of_study
        FROM semanticscholar.paper p
        LEFT JOIN (
            SELECT paper_id,
                   array_agg(DISTINCT field_of_study ORDER BY field_of_study) AS fields
            FROM semanticscholar.paper_field
            WHERE field_of_study IS NOT NULL
            GROUP BY paper_id
        ) pf ON pf.paper_id = p.s2_id
        WHERE p.s2_id IS NOT NULL
        """,
        "stage s2 enrichment → _enrich_s2",
        dry_run=dry_run,
    )
    exec_step(
        conn,
        "CREATE INDEX ON _enrich_s2(s2_id)",
        "index _enrich_s2(s2_id)",
        dry_run=dry_run,
    )
    exec_step(conn, "ANALYZE _enrich_s2", "ANALYZE _enrich_s2", dry_run=dry_run)

    updated = exec_step(
        conn,
        """
        UPDATE arc_v5.rosetta_stone r
        SET title              = COALESCE(r.title, e.title),
            abstract           = COALESCE(r.abstract, e.abstract),
            journal            = COALESCE(r.journal, e.journal),
            pub_year           = COALESCE(r.pub_year, e.pub_year),
            s2_fields_of_study = COALESCE(r.s2_fields_of_study, e.fields_of_study)
        FROM _enrich_s2 e
        WHERE r.s2_id = e.s2_id
        """,
        "UPDATE rosetta with S2 metadata",
        dry_run=dry_run,
    )
    if not dry_run:
        conn.commit()
    return {"inserted": 0, "updated": max(updated, 0)}


def pass_7_openalex_metadata(conn, dry_run: bool = False) -> dict:
    """Enrich rows linked to OpenAlex with abstract, title, pub_year, openalex_type,
    openalex_cited_by_count.

    Runs FIRST in Phase 1b because openalex.work.abstract_text is the highest-
    priority abstract source (~271M rows of plain text, largest coverage by far).
    All subsequent metadata passes use COALESCE so they only fill NULLs.
    """
    log.info("==== Pass 7: OpenAlex metadata enrichment ====")

    exec_step(
        conn,
        f"""
        CREATE TEMP TABLE _enrich_openalex ON COMMIT DROP AS
        SELECT
            {NORM_OAID.format(col='doc_id')} AS openalex_id,
            COALESCE(display_name, title)    AS title,
            abstract_text                    AS abstract,
            publication_year                 AS pub_year,
            type                             AS openalex_type,
            cited_by_count                   AS cited_by_count
        FROM openalex.work
        WHERE doc_id IS NOT NULL
        """,
        "stage openalex enrichment → _enrich_openalex",
        dry_run=dry_run,
    )
    exec_step(
        conn,
        "CREATE INDEX ON _enrich_openalex(openalex_id)",
        "index _enrich_openalex(openalex_id)",
        dry_run=dry_run,
    )
    exec_step(conn, "ANALYZE _enrich_openalex", "ANALYZE _enrich_openalex", dry_run=dry_run)

    updated = exec_step(
        conn,
        """
        UPDATE arc_v5.rosetta_stone r
        SET title                   = COALESCE(r.title, e.title),
            abstract                = COALESCE(r.abstract, e.abstract),
            pub_year                = COALESCE(r.pub_year, e.pub_year),
            openalex_type           = COALESCE(r.openalex_type, e.openalex_type),
            openalex_cited_by_count = COALESCE(r.openalex_cited_by_count, e.cited_by_count)
        FROM _enrich_openalex e
        WHERE r.openalex_id = e.openalex_id
        """,
        "UPDATE rosetta with OpenAlex metadata",
        dry_run=dry_run,
    )
    if not dry_run:
        conn.commit()
    return {"inserted": 0, "updated": max(updated, 0)}


def pass_11_crossref_metadata(conn, dry_run: bool = False) -> dict:
    """Enrich rows linked to Crossref with publisher, type, and abstract as last-
    resort fallback (Crossref stores abstracts as JATS-XML snippets — messy but
    better than nothing). Title is also set only if NULL."""
    log.info("==== Pass 11: Crossref metadata enrichment ====")

    exec_step(
        conn,
        f"""
        CREATE TEMP TABLE _enrich_crossref ON COMMIT DROP AS
        SELECT
            {NORM_DOI.format(col='doi')}                        AS doi,
            title                                               AS title,
            abstract                                            AS abstract,
            publisher                                           AS publisher,
            type                                                AS crossref_type,
            container_title                                     AS journal,
            EXTRACT(year FROM COALESCE(
                issued_date, published_online_date, published_print_date))::int AS pub_year
        FROM crossref.document
        WHERE doi IS NOT NULL
        """,
        "stage crossref enrichment → _enrich_crossref",
        dry_run=dry_run,
    )
    exec_step(
        conn,
        "CREATE INDEX ON _enrich_crossref(doi)",
        "index _enrich_crossref(doi)",
        dry_run=dry_run,
    )
    exec_step(conn, "ANALYZE _enrich_crossref", "ANALYZE _enrich_crossref", dry_run=dry_run)

    updated = exec_step(
        conn,
        """
        UPDATE arc_v5.rosetta_stone r
        SET title              = COALESCE(r.title, e.title),
            abstract           = COALESCE(r.abstract, e.abstract),
            journal            = COALESCE(r.journal, e.journal),
            pub_year           = COALESCE(r.pub_year, e.pub_year),
            crossref_publisher = COALESCE(r.crossref_publisher, e.publisher),
            crossref_type      = COALESCE(r.crossref_type, e.crossref_type)
        FROM _enrich_crossref e
        WHERE r.doi = e.doi
        """,
        "UPDATE rosetta with Crossref metadata",
        dry_run=dry_run,
    )
    if not dry_run:
        conn.commit()
    return {"inserted": 0, "updated": max(updated, 0)}


# ---------------------------------------------------------------------------
# Phase 1c — Fulltext flag
# ---------------------------------------------------------------------------

def pass_12_has_fulltext(conn, dry_run: bool = False) -> dict:
    """Set has_fulltext = TRUE where europepmc.body_section has rows for the PMCID.

    NOTE: arxiv_fulltext is NOT checked in Phase 1 because rosetta has no arxiv_id
    column and the DOI pattern match for arXiv (10.48550/arXiv.XXXX) is unreliable.
    Future phase will add arxiv_id column + proper linkage.
    """
    log.info("==== Pass 12: has_fulltext flag ====")

    # Stage the set of pmcids known to have body_section content
    exec_step(
        conn,
        """
        CREATE TEMP TABLE _fulltext_pmcids ON COMMIT DROP AS
        SELECT DISTINCT d.pmcid
        FROM europepmc.document d
        WHERE d.pmcid IS NOT NULL
          AND EXISTS (
              SELECT 1 FROM europepmc.body_section bs
              WHERE bs.doc_id = d.doc_id
          )
        """,
        "stage pmcids with body_section rows → _fulltext_pmcids",
        dry_run=dry_run,
    )
    exec_step(
        conn,
        "CREATE INDEX ON _fulltext_pmcids(pmcid)",
        "index _fulltext_pmcids(pmcid)",
        dry_run=dry_run,
    )
    exec_step(conn, "ANALYZE _fulltext_pmcids", "ANALYZE _fulltext_pmcids", dry_run=dry_run)

    updated = exec_step(
        conn,
        """
        UPDATE arc_v5.rosetta_stone r
        SET has_fulltext = TRUE
        FROM _fulltext_pmcids f
        WHERE r.europepmc_pmcid = f.pmcid
          AND r.has_fulltext = FALSE
        """,
        "UPDATE rosetta set has_fulltext via europepmc.body_section",
        dry_run=dry_run,
    )
    if not dry_run:
        conn.commit()
    return {"inserted": 0, "updated": max(updated, 0)}


# ---------------------------------------------------------------------------
# Pass registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pass:
    num: int
    name: str
    fn: Callable[..., dict]


PASSES: tuple[Pass, ...] = (
    Pass(1,  "openalex_doi_seed",           pass_1_openalex_doi_seed),
    Pass(2,  "openalex_pmid_pmcid_seed",    pass_2_openalex_pmid_pmcid_seed),
    Pass(3,  "pubmed_link",                 pass_3_pubmed_link),
    Pass(4,  "s2_link",                     pass_4_s2_link),
    Pass(5,  "europepmc_link",              pass_5_europepmc_link),
    Pass(6,  "crossref_link",               pass_6_crossref_link),
    Pass(7,  "openalex_metadata",           pass_7_openalex_metadata),
    Pass(8,  "pubmed_metadata",             pass_8_pubmed_metadata),
    Pass(9,  "europepmc_metadata",          pass_9_europepmc_metadata),
    Pass(10, "s2_metadata",                 pass_10_s2_metadata),
    Pass(11, "crossref_metadata",           pass_11_crossref_metadata),
    Pass(12, "has_fulltext_flag",           pass_12_has_fulltext),
)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def log_summary(conn) -> None:
    log.info("==== FINAL SUMMARY ====")
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM arc_v5.rosetta_stone")
        total = cur.fetchone()[0]
        log.info("  total rows: %s", f"{total:,}")

        cur.execute(
            """
            SELECT
              COUNT(*) FILTER (WHERE doi             IS NOT NULL) AS with_doi,
              COUNT(*) FILTER (WHERE pmid            IS NOT NULL) AS with_pmid,
              COUNT(*) FILTER (WHERE pmcid           IS NOT NULL) AS with_pmcid,
              COUNT(*) FILTER (WHERE openalex_id     IS NOT NULL) AS with_openalex,
              COUNT(*) FILTER (WHERE s2_id           IS NOT NULL) AS with_s2,
              COUNT(*) FILTER (WHERE europepmc_pmcid IS NOT NULL) AS with_epmc,
              COUNT(*) FILTER (WHERE title           IS NOT NULL) AS with_title,
              COUNT(*) FILTER (WHERE abstract        IS NOT NULL) AS with_abstract,
              COUNT(*) FILTER (WHERE has_fulltext)                AS with_fulltext
            FROM arc_v5.rosetta_stone
            """
        )
        cols = [d.name for d in cur.description]
        vals = cur.fetchone()
        for c, v in zip(cols, vals):
            log.info("  %-16s %s (%.1f%%)", c, f"{v:,}", 100.0 * v / max(total, 1))

        cur.execute(
            """
            SELECT source_count, COUNT(*)
            FROM arc_v5.rosetta_stone
            GROUP BY source_count
            ORDER BY source_count
            """
        )
        log.info("  source_count distribution:")
        for sc, n in cur.fetchall():
            log.info("    %d sources: %s rows", sc, f"{n:,}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_pass(conn, p: Pass, dry_run: bool) -> None:
    status = pass_status(conn, p.num)
    if status == "done":
        log.info("Pass %2d (%s): already done, skipping", p.num, p.name)
        return
    if status == "running":
        log.warning(
            "Pass %2d (%s): found 'running' state (previous crash?), re-executing",
            p.num, p.name,
        )

    mark_started(conn, p.num, p.name)
    t0 = time.monotonic()
    try:
        stats = p.fn(conn, dry_run=dry_run)
    except Exception:
        log.exception("Pass %2d (%s) FAILED", p.num, p.name)
        raise
    elapsed = time.monotonic() - t0
    notes = f"elapsed={elapsed:.1f}s"
    if not dry_run:
        mark_done(
            conn, p.num,
            inserted=stats.get("inserted", 0),
            updated=stats.get("updated", 0),
            notes=notes,
        )
    log.info(
        "Pass %2d (%s) done: inserted=%s updated=%s in %.1fs",
        p.num, p.name,
        f"{stats.get('inserted', 0):,}",
        f"{stats.get('updated', 0):,}",
        elapsed,
    )

    # Lazy index builds after certain passes
    for idx_name in PASS_INDEX_BUILDS.get(p.num, []):
        ensure_index(conn, idx_name, dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 rosetta_stone builder")
    p.add_argument("--only-pass", type=int, help="Run only this pass number (1-12)")
    p.add_argument("--from-pass", type=int, help="Run from this pass onward")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan without executing any SQL")
    p.add_argument("--reset-progress", type=int, metavar="N",
                   help="Delete progress row for pass N (destructive — use with care)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(verbose=args.verbose)
    log.info("==== arc_rosetta_build.py starting ====")
    log.info("DSN=%s log=%s dry_run=%s", DSN, LOG_PATH, args.dry_run)

    with connect() as conn:
        ensure_schema(conn, dry_run=args.dry_run)

        if args.reset_progress is not None:
            reset_pass_progress(conn, args.reset_progress)

        if args.only_pass is not None:
            passes_to_run = [p for p in PASSES if p.num == args.only_pass]
            if not passes_to_run:
                log.error("No pass with num=%d", args.only_pass)
                return 2
        elif args.from_pass is not None:
            passes_to_run = [p for p in PASSES if p.num >= args.from_pass]
        else:
            passes_to_run = list(PASSES)

        log.info("Running passes: %s", [p.num for p in passes_to_run])
        for p in passes_to_run:
            run_pass(conn, p, dry_run=args.dry_run)

        if not args.dry_run:
            log.info("Running final ANALYZE…")
            with conn.cursor() as cur:
                cur.execute("ANALYZE arc_v5.rosetta_stone")
            conn.commit()
            log_summary(conn)

    log.info("==== arc_rosetta_build.py complete ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())

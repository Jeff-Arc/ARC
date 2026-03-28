#!/usr/bin/env python3
"""
arc_docdb_ingest.py — Ingest DOCDB back-file ZIPs into docdb_patents table.

ZIP structure:
  outer .zip → Root/DOC/DOCDB-202607-NNN-CC-NNNN.zip (per-country inner ZIPs)
             → DOCDB-202607-NNN-CC-NNNN.xml (one XML per inner ZIP)
             → <exch:exchange-document> elements (10K–42K per XML)

Usage:
  python3 arc_docdb_ingest.py [--zip PATH] [--dir DIR] [--create-table] [--no-raw-xml]
  python3 arc_docdb_ingest.py --zip ~/data/docdb_backfile/docdb_xml_bck_202607_001_A.zip
  python3 arc_docdb_ingest.py --dir ~/data/docdb_backfile --create-table
"""

import argparse
import io
import json
import os
import re as _re
import sys
import time
import traceback
import zipfile
from datetime import date
from pathlib import Path
from typing import Iterator

import psycopg2
import psycopg2.extras

# ── XML namespace ────────────────────────────────────────────────────────────
try:
    from lxml import etree
    HAVE_LXML = True
except ImportError:
    import xml.etree.ElementTree as etree  # fallback, no recovery
    HAVE_LXML = False

NS = "http://www.epo.org/exchange"
NS_TAG = f"{{{NS}}}"

# ── DB ────────────────────────────────────────────────────────────────────────
DB_PARAMS = dict(
    host="/var/run/postgresql",
    dbname="arc_v4",
    user=os.environ.get("PGUSER", "jeff"),
)

CREATE_TABLE_SQL = """
CREATE UNLOGGED TABLE IF NOT EXISTS arc_v5.docdb_patents (
    doc_id              text        NOT NULL,
    country             text,
    doc_number          text,
    kind                text,
    family_id           text,
    pub_date            date,
    title               text,
    abstract            text,
    cpc_codes           text[]      NOT NULL DEFAULT '{}',
    ipc_codes           text[]      NOT NULL DEFAULT '{}',
    ipcr_codes          text[]      NOT NULL DEFAULT '{}',
    applicants          text[]      NOT NULL DEFAULT '{}',
    applicant_countries text[]      NOT NULL DEFAULT '{}',
    inventors           text[]      NOT NULL DEFAULT '{}',
    inventor_countries  text[]      NOT NULL DEFAULT '{}',
    language            text,
    filing_date         date,
    application_number  text,
    priority_countries  text[]      NOT NULL DEFAULT '{}',
    priority_numbers    text[]      NOT NULL DEFAULT '{}',
    priority_dates      date[]      NOT NULL DEFAULT '{}',
    is_representative   boolean,
    date_added_docdb    date,
    citations           text[]      NOT NULL DEFAULT '{}',
    citation_categories text[]      NOT NULL DEFAULT '{}',
    npl_citations       text[]      NOT NULL DEFAULT '{}',
    raw_xml             text,
    ingested_at         timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id)
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS docdb_patents_cpc_gin
    ON arc_v5.docdb_patents USING GIN (cpc_codes);
CREATE INDEX IF NOT EXISTS docdb_patents_ipc_gin
    ON arc_v5.docdb_patents USING GIN (ipc_codes);
CREATE INDEX IF NOT EXISTS docdb_patents_country_idx
    ON arc_v5.docdb_patents (country);
CREATE INDEX IF NOT EXISTS docdb_patents_pub_date_idx
    ON arc_v5.docdb_patents (pub_date);
CREATE INDEX IF NOT EXISTS docdb_patents_family_idx
    ON arc_v5.docdb_patents (family_id);
CREATE INDEX IF NOT EXISTS docdb_patents_language_idx
    ON arc_v5.docdb_patents (language);
CREATE INDEX IF NOT EXISTS docdb_patents_filing_date_idx
    ON arc_v5.docdb_patents (filing_date);
CREATE INDEX IF NOT EXISTS docdb_patents_app_num_idx
    ON arc_v5.docdb_patents (application_number);
"""

# Column order must match doc_to_row() output
INSERT_SQL = """
INSERT INTO arc_v5.docdb_patents
    (doc_id, country, doc_number, kind, family_id, pub_date,
     title, abstract,
     cpc_codes, ipc_codes, ipcr_codes,
     applicants, applicant_countries,
     inventors, inventor_countries,
     language, filing_date, application_number,
     priority_countries, priority_numbers, priority_dates,
     is_representative, date_added_docdb,
     citations, citation_categories, npl_citations,
     raw_xml)
VALUES %s
ON CONFLICT (doc_id) DO NOTHING
"""

BATCH_SIZE = 10000


# ── XML helpers ───────────────────────────────────────────────────────────────

def _text_or_none(el):
    """Return stripped itertext or None."""
    if el is None:
        return None
    txt = "".join(el.itertext()).strip()
    return txt if txt else None


def _parse_date(s: str | None) -> date | None:
    """Parse YYYYMMDD string to date, or None."""
    if not s or len(s) != 8:
        return None
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except (ValueError, TypeError):
        return None


def extract_doc(doc_el, include_raw: bool = True) -> dict:
    """Extract one exchange-document element into a dict with all fields."""
    doc_id     = doc_el.get("doc-id")
    country    = doc_el.get("country")
    doc_number = doc_el.get("doc-number")
    kind       = doc_el.get("kind")
    family_id  = doc_el.get("family-id")
    pub_date   = _parse_date(doc_el.get("date-publ"))

    # is-representative attribute
    is_rep_str = doc_el.get("is-representative")
    is_representative = True if is_rep_str == "YES" else (False if is_rep_str == "NO" else None)

    # date-added-docdb attribute
    date_added_docdb = _parse_date(doc_el.get("date-added-docdb"))

    bib = doc_el.find(f"{NS_TAG}bibliographic-data")

    # ── Title — prefer English, fall back to first ──
    title = None
    if bib is not None:
        en_titles = [t for t in bib.findall(f"{NS_TAG}invention-title") if t.get("lang") == "en"]
        all_titles = bib.findall(f"{NS_TAG}invention-title")
        for t in (en_titles or all_titles)[:1]:
            title = _text_or_none(t)

    # ── Abstract — prefer English, fall back to first ──
    abstract = None
    en_abs = [a for a in doc_el.iter(f"{NS_TAG}abstract") if a.get("lang") == "en"]
    all_abs = list(doc_el.iter(f"{NS_TAG}abstract"))
    for ab in (en_abs or all_abs)[:1]:
        # Try exch:p first, then bare p, then the abstract element itself
        p = ab.find(f"{NS_TAG}p")
        if p is None:
            p = ab.find("p")
        abstract = _text_or_none(p if p is not None else ab)

    # ── CPC codes — patent-classifications ──
    cpc_codes = []
    if bib is not None:
        pc = bib.find(f"{NS_TAG}patent-classifications")
        if pc is not None:
            seen = set()
            for sym in pc.findall("patent-classification/classification-symbol"):
                code = (sym.text or "").strip()
                if code and code not in seen:
                    seen.add(code)
                    cpc_codes.append(code)

    # ── IPC codes (old format) — classification-ipc ──
    ipc_codes = []
    if bib is not None:
        ipc = bib.find(f"{NS_TAG}classification-ipc")
        if ipc is not None:
            seen = set()
            # main-classification + further-classification
            for el in ipc.findall("main-classification"):
                code = (el.text or "").strip()
                if code and code not in seen:
                    seen.add(code)
                    ipc_codes.append(code)
            for el in ipc.findall("further-classification"):
                code = (el.text or "").strip()
                if code and code not in seen:
                    seen.add(code)
                    ipc_codes.append(code)

    # ── IPCR codes (reformed IPC) — classifications-ipcr ──
    ipcr_codes = []
    if bib is not None:
        ipcr = bib.find(f"{NS_TAG}classifications-ipcr")
        if ipcr is not None:
            seen = set()
            for el in ipcr.findall("classification-ipcr/text"):
                code = (el.text or "").strip()
                if code and code not in seen:
                    seen.add(code)
                    ipcr_codes.append(code)

    # ── Applicants — docdb format names + residence countries ──
    applicants = []
    applicant_countries = []
    if bib is not None:
        seen_names = set()
        for app_el in bib.findall(f".//{NS_TAG}applicant[@data-format='docdb']"):
            name_el = app_el.find(f"{NS_TAG}applicant-name/name")
            name = (name_el.text or "").strip() if name_el is not None else ""
            if name and name not in seen_names:
                seen_names.add(name)
                applicants.append(name)
                # Residence country
                res = app_el.find(f"residence/country")
                if res is None:
                    res = app_el.find(f"{NS_TAG}residence/{NS_TAG}country")
                applicant_countries.append((res.text or "").strip() if res is not None else "")

    # ── Inventors — docdb format names + residence countries ──
    inventors = []
    inventor_countries = []
    if bib is not None:
        seen_names = set()
        for inv_el in bib.findall(f".//{NS_TAG}inventor[@data-format='docdb']"):
            name_el = inv_el.find(f"{NS_TAG}inventor-name/name")
            name = (name_el.text or "").strip() if name_el is not None else ""
            if name and name not in seen_names:
                seen_names.add(name)
                inventors.append(name)
                res = inv_el.find(f"residence/country")
                if res is None:
                    res = inv_el.find(f"{NS_TAG}residence/{NS_TAG}country")
                inventor_countries.append((res.text or "").strip() if res is not None else "")

    # ── Language of publication ──
    language = None
    if bib is not None:
        lang_el = bib.find(f"{NS_TAG}language-of-publication")
        if lang_el is not None:
            language = (lang_el.text or "").strip() or None

    # ── Application reference — filing date + application number ──
    filing_date = None
    application_number = None
    if bib is not None:
        # Prefer docdb format
        app_ref = bib.find(f"{NS_TAG}application-reference[@data-format='docdb']")
        if app_ref is None:
            app_ref = bib.find(f"{NS_TAG}application-reference")
        if app_ref is not None:
            did = app_ref.find("document-id")
            if did is None:
                did = app_ref.find(f"{NS_TAG}document-id")
            if did is not None:
                dn = did.find("doc-number")
                if dn is None:
                    dn = did.find(f"{NS_TAG}doc-number")
                if dn is not None:
                    application_number = (dn.text or "").strip() or None
                dt = did.find("date")
                if dt is None:
                    dt = did.find(f"{NS_TAG}date")
                if dt is not None:
                    filing_date = _parse_date((dt.text or "").strip())

    # ── Priority claims ──
    priority_countries = []
    priority_numbers = []
    priority_dates = []
    if bib is not None:
        pc_el = bib.find(f"{NS_TAG}priority-claims")
        if pc_el is not None:
            seen_prio = set()
            for claim in pc_el.findall(f"{NS_TAG}priority-claim[@data-format='docdb']"):
                did = claim.find("document-id")
                if did is None:
                    did = claim.find(f"{NS_TAG}document-id")
                if did is None:
                    continue
                cc = did.find("country")
                if cc is None:
                    cc = did.find(f"{NS_TAG}country")
                dn = did.find("doc-number")
                if dn is None:
                    dn = did.find(f"{NS_TAG}doc-number")
                dt = did.find("date")
                if dt is None:
                    dt = did.find(f"{NS_TAG}date")

                cc_text = (cc.text or "").strip() if cc is not None else ""
                dn_text = (dn.text or "").strip() if dn is not None else ""
                prio_key = f"{cc_text}{dn_text}"
                if prio_key in seen_prio:
                    continue
                seen_prio.add(prio_key)

                priority_countries.append(cc_text)
                priority_numbers.append(dn_text)
                pdate = _parse_date((dt.text or "").strip()) if dt is not None else None
                priority_dates.append(pdate)

    # ── Citations — patent citations with categories + NPL citations ──
    citations = []
    citation_categories = []
    npl_citations = []
    seen_cites = set()
    for cite in doc_el.iter(f"{NS_TAG}citation"):
        # Patent citations
        patcit = cite.find("patcit")
        if patcit is None:
            patcit = cite.find(f"{NS_TAG}patcit")
        if patcit is not None:
            for ref in patcit.findall("document-id"):
                cc = ref.find("country")
                cn = ref.find("doc-number")
                ck = ref.find("kind")
                if cc is not None and cn is not None:
                    key = f"{cc.text}{cn.text}{ck.text if ck is not None else ''}"
                    if key not in seen_cites:
                        seen_cites.add(key)
                        citations.append(key)
                        # Category for this citation
                        cat_el = cite.find("category")
                        if cat_el is None:
                            cat_el = cite.find(f"{NS_TAG}category")
                        cat = (cat_el.text or "").strip() if cat_el is not None else ""
                        citation_categories.append(cat)
                break  # one ref per patcit

        # NPL citations
        nplcit = cite.find("nplcit")
        if nplcit is None:
            nplcit = cite.find(f"{NS_TAG}nplcit")
        if nplcit is not None:
            npl_text = _text_or_none(nplcit)
            if npl_text:
                npl_citations.append(npl_text[:2000])  # cap length

    # ── Raw XML ──
    raw = None
    if include_raw and HAVE_LXML:
        raw = etree.tostring(doc_el, encoding="unicode")

    return {
        "doc_id":              doc_id,
        "country":             country,
        "doc_number":          doc_number,
        "kind":                kind,
        "family_id":           family_id,
        "pub_date":            pub_date,
        "title":               title,
        "abstract":            abstract,
        "cpc_codes":           cpc_codes,
        "ipc_codes":           ipc_codes,
        "ipcr_codes":          ipcr_codes,
        "applicants":          applicants,
        "applicant_countries": applicant_countries,
        "inventors":           inventors,
        "inventor_countries":  inventor_countries,
        "language":            language,
        "filing_date":         filing_date,
        "application_number":  application_number,
        "priority_countries":  priority_countries,
        "priority_numbers":    priority_numbers,
        "priority_dates":      priority_dates,
        "is_representative":   is_representative,
        "date_added_docdb":    date_added_docdb,
        "citations":           citations,
        "citation_categories": citation_categories,
        "npl_citations":       npl_citations,
        "raw_xml":             raw,
    }


_DOCTYPE_RE  = _re.compile(rb'<!DOCTYPE\s[^[>]*(?:\[[^\]]*\])?\s*>', _re.DOTALL)
_ENTITY_RE   = _re.compile(rb'&(?!amp;|lt;|gt;|apos;|quot;)[a-zA-Z][a-zA-Z0-9]*;')


def _strip_entities(xml_bytes: bytes) -> bytes:
    """Remove DOCTYPE declaration and unknown named entities so iterparse works."""
    xml_bytes = _DOCTYPE_RE.sub(b"", xml_bytes, count=1)
    xml_bytes = _ENTITY_RE.sub(b"", xml_bytes)
    return xml_bytes


def parse_xml_stream(xml_bytes: bytes, include_raw: bool = True) -> Iterator[dict]:
    """Stream-parse exchange-document elements from DOCDB XML bytes."""
    if not HAVE_LXML:
        raise RuntimeError("lxml required for DOCDB XML parsing")

    xml_bytes = _strip_entities(xml_bytes)
    tag = f"{NS_TAG}exchange-document"

    context = etree.iterparse(
        io.BytesIO(xml_bytes),
        events=("end",),
        tag=tag,
        huge_tree=True,
    )
    for _event, doc in context:
        yield extract_doc(doc, include_raw=include_raw)
        doc.clear()
        while doc.getprevious() is not None:
            del doc.getparent()[0]


def iter_docdb_zip(outer_zip_path: str, include_raw: bool = True,
                   error_sink: list | None = None) -> Iterator[dict]:
    """Yield all patent dicts from a DOCDB outer ZIP file."""
    outer = zipfile.ZipFile(outer_zip_path, "r")
    inner_zips = sorted(
        n for n in outer.namelist()
        if n.startswith("Root/DOC/") and n.endswith(".zip")
    )
    for inner_path in inner_zips:
        try:
            inner_data = outer.read(inner_path)
            inner = zipfile.ZipFile(io.BytesIO(inner_data), "r")
            xml_names = [n for n in inner.namelist() if n.endswith(".xml")]
            for xml_name in xml_names:
                xml_bytes = inner.read(xml_name)
                try:
                    yield from parse_xml_stream(xml_bytes, include_raw=include_raw)
                except Exception as exc:
                    err = {
                        "type": "xml_parse_exception",
                        "inner_zip": inner_path,
                        "xml_file": xml_name,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    if error_sink is not None:
                        error_sink.append(err)
                    else:
                        raise
        except zipfile.BadZipFile as exc:
            err = {
                "type": "bad_zip",
                "inner_zip": inner_path,
                "error": str(exc),
            }
            if error_sink is not None:
                error_sink.append(err)
            else:
                raise


# ── DB operations ─────────────────────────────────────────────────────────────

def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    print("Table docdb_patents ensured (UNLOGGED).")


def create_indexes(conn):
    print("Creating indexes (this may take a while on large tables)...")
    with conn.cursor() as cur:
        cur.execute(CREATE_INDEX_SQL)
    conn.commit()
    print("Indexes created.")


def set_logged(conn):
    """Convert UNLOGGED table to LOGGED after bulk ingest."""
    print("Setting table to LOGGED...")
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE arc_v5.docdb_patents SET LOGGED;")
    conn.commit()
    print("Table is now LOGGED.")


def doc_to_row(d: dict):
    return (
        d["doc_id"],
        d["country"],
        d["doc_number"],
        d["kind"],
        d["family_id"],
        d["pub_date"],
        d["title"],
        d["abstract"],
        d["cpc_codes"],
        d["ipc_codes"],
        d["ipcr_codes"],
        d["applicants"],
        d["applicant_countries"],
        d["inventors"],
        d["inventor_countries"],
        d["language"],
        d["filing_date"],
        d["application_number"],
        d["priority_countries"],
        d["priority_numbers"],
        d["priority_dates"],
        d["is_representative"],
        d["date_added_docdb"],
        d["citations"],
        d["citation_categories"],
        d["npl_citations"],
        d["raw_xml"],
    )


def ingest_zip(conn, zip_path: str, include_raw: bool = True,
               error_log: Path | None = None,
               delete_after: bool = False,
               progress_file: Path | None = None) -> dict:
    """Ingest one outer ZIP. Returns stats dict."""
    t0 = time.time()
    inserted = 0
    skipped = 0
    errors = 0
    error_samples = []
    batch = []
    total_seen = 0
    parse_errors = []

    zip_name = Path(zip_path).name
    log_fh = open(error_log, "a") if error_log else None
    print(f"Processing {zip_name} ...")

    def _record_error(err_dict: dict):
        nonlocal errors
        errors += 1
        if len(error_samples) < 20:
            error_samples.append(err_dict)
        if log_fh:
            log_fh.write(json.dumps(err_dict) + "\n")
            log_fh.flush()

    def _log_progress():
        if progress_file:
            elapsed = time.time() - t0
            rate = total_seen / elapsed if elapsed > 0 else 0
            with open(progress_file, "a") as pf:
                pf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                         f"{zip_name}: {total_seen:,} docs, {inserted:,} inserted, "
                         f"{errors:,} errors, {rate:,.0f} docs/s\n")

    for doc in iter_docdb_zip(zip_path, include_raw=include_raw, error_sink=parse_errors):
        for pe in parse_errors:
            _record_error(pe)
        parse_errors.clear()

        total_seen += 1
        if not doc.get("doc_id"):
            _record_error({
                "type": "null_doc_id",
                "country": doc.get("country"),
                "doc_number": doc.get("doc_number"),
                "kind": doc.get("kind"),
                "family_id": doc.get("family_id"),
                "date_publ": str(doc.get("pub_date")) if doc.get("pub_date") else None,
                "title": (doc.get("title") or "")[:120],
            })
            continue

        batch.append(doc_to_row(doc))
        if len(batch) >= BATCH_SIZE:
            n = _flush_batch(conn, batch)
            inserted += n
            skipped += len(batch) - n
            batch = []
            if total_seen % 50000 == 0:
                elapsed = time.time() - t0
                rate = total_seen / elapsed
                print(f"  {total_seen:>8,} docs  {inserted:>8,} inserted  "
                      f"{errors:>5,} errors  {rate:,.0f} docs/s")
                _log_progress()

    for pe in parse_errors:
        _record_error(pe)

    if batch:
        n = _flush_batch(conn, batch)
        inserted += n
        skipped += len(batch) - n

    if log_fh:
        log_fh.close()

    elapsed = time.time() - t0
    msg = (f"  Done: {total_seen:,} docs  {inserted:,} inserted  "
           f"{skipped:,} skipped (conflict)  {errors:,} errors  "
           f"{elapsed:.1f}s")
    print(msg)
    if error_log and errors:
        print(f"  Error log: {error_log}")

    if progress_file:
        with open(progress_file, "a") as pf:
            pf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                     f"COMPLETED {zip_name}: {total_seen:,} docs, {inserted:,} inserted, "
                     f"{skipped:,} skipped, {errors:,} errors, {elapsed:.1f}s\n")

    if delete_after and total_seen > 0:
        try:
            Path(zip_path).unlink()
            print(f"  Deleted {zip_name}")
        except OSError as e:
            print(f"  WARNING: could not delete {zip_path}: {e}")

    return {"total": total_seen, "inserted": inserted, "skipped": skipped,
            "errors": errors, "error_samples": error_samples, "elapsed": elapsed}


def _flush_batch(conn, batch: list) -> int:
    """Bulk-insert batch, return count of rows actually inserted."""
    valid = [row for row in batch if row[0] is not None and row[0] != ""]
    if not valid:
        return 0
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, INSERT_SQL, valid, page_size=BATCH_SIZE)
        n = cur.rowcount
    conn.commit()
    return max(n, 0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Ingest DOCDB ZIPs into docdb_patents")
    ap.add_argument("--zip",          help="Single outer ZIP file to ingest")
    ap.add_argument("--dir",          help="Directory of outer ZIP files")
    ap.add_argument("--create-table", action="store_true",
                    help="Create table + indexes before ingesting")
    ap.add_argument("--create-indexes", action="store_true",
                    help="Create GIN + other indexes only (run after bulk load)")
    ap.add_argument("--set-logged", action="store_true",
                    help="Convert UNLOGGED table to LOGGED")
    ap.add_argument("--no-raw-xml",   action="store_true",
                    help="Skip storing raw_xml (saves ~10x storage)")
    ap.add_argument("--dry-run",      action="store_true",
                    help="Parse only, no DB writes. Reports counts per ZIP.")
    ap.add_argument("--scan-errors",  action="store_true",
                    help="Scan ZIP(s) and report all error records (no insert).")
    ap.add_argument("--error-log",    default=None,
                    help="JSONL file to append error records during ingest.")
    ap.add_argument("--delete-after", action="store_true",
                    help="Delete each ZIP after successful ingest to free disk space.")
    ap.add_argument("--progress-file", default="/tmp/docdb_ingest_progress.txt",
                    help="File to log progress updates (default: /tmp/docdb_ingest_progress.txt)")
    args = ap.parse_args()

    include_raw = not args.no_raw_xml

    if not HAVE_LXML:
        print("ERROR: lxml is required. Install with: pip install lxml")
        sys.exit(1)

    conn = psycopg2.connect(**DB_PARAMS)

    if args.create_table:
        ensure_table(conn)

    if args.create_indexes:
        create_indexes(conn)
        conn.close()
        return

    if args.set_logged:
        set_logged(conn)
        conn.close()
        return

    # Gather ZIP files to process
    zips = []
    if args.zip:
        zips = [args.zip]
    elif args.dir:
        zips = sorted(Path(args.dir).glob("docdb_xml_bck_*.zip"))
    else:
        ap.print_help()
        sys.exit(0)

    if not zips:
        print("No ZIP files found.")
        sys.exit(1)

    print(f"Files to process: {len(zips)}")
    if not args.no_raw_xml:
        print("  NOTE: raw_xml=ON (use --no-raw-xml to skip XML storage)")

    progress_file = Path(args.progress_file) if args.progress_file else None
    error_log_path = Path(args.error_log) if args.error_log else None

    grand_total = 0
    grand_inserted = 0
    grand_errors = 0
    t_all = time.time()

    if progress_file:
        with open(progress_file, "a") as pf:
            pf.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] === INGEST SESSION START === "
                     f"Files: {len(zips)}, raw_xml={'ON' if not args.no_raw_xml else 'OFF'}\n")

    for zip_path in zips:
        if args.scan_errors:
            t0 = time.time()
            all_errors = []
            parse_errors = []
            n = 0
            for doc in iter_docdb_zip(str(zip_path), include_raw=False,
                                      error_sink=parse_errors):
                for pe in parse_errors:
                    all_errors.append(pe)
                parse_errors.clear()
                n += 1
                if not doc.get("doc_id"):
                    all_errors.append({
                        "type": "null_doc_id",
                        "country": doc.get("country"),
                        "doc_number": doc.get("doc_number"),
                        "kind": doc.get("kind"),
                        "family_id": doc.get("family_id"),
                        "date_publ": str(doc.get("pub_date")) if doc.get("pub_date") else None,
                        "title": (doc.get("title") or "")[:120],
                    })
            for pe in parse_errors:
                all_errors.append(pe)
            elapsed = time.time() - t0

            print(f"\n{Path(zip_path).name}")
            print(f"  Docs seen: {n:,}  Errors: {len(all_errors):,}  "
                  f"elapsed: {elapsed:.1f}s")

            by_type: dict[str, int] = {}
            for e in all_errors:
                by_type[e["type"]] = by_type.get(e["type"], 0) + 1
            print("  By type:", by_type)

            print("  --- 5 error samples ---")
            for e in all_errors[:5]:
                print(f"  {json.dumps(e, default=str)}")

            if error_log_path:
                with open(error_log_path, "a") as fh:
                    for e in all_errors:
                        fh.write(json.dumps(e, default=str) + "\n")
                print(f"  Written {len(all_errors):,} errors → {error_log_path}")

            grand_total += n
            grand_errors += len(all_errors)

        elif args.dry_run:
            t0 = time.time()
            n = 0
            countries = {}
            abstracts = 0
            cpcs = 0
            ipcs = 0
            inventors_n = 0
            for doc in iter_docdb_zip(str(zip_path), include_raw=False):
                n += 1
                c = doc["country"]
                countries[c] = countries.get(c, 0) + 1
                if doc["abstract"]:
                    abstracts += 1
                if doc["cpc_codes"]:
                    cpcs += 1
                if doc["ipc_codes"] or doc["ipcr_codes"]:
                    ipcs += 1
                if doc["inventors"]:
                    inventors_n += 1
            elapsed = time.time() - t0
            print(f"\n{Path(zip_path).name}")
            print(f"  Patents: {n:,}  elapsed: {elapsed:.1f}s")
            print(f"  Abstracts: {abstracts:,} ({100*abstracts/n:.1f}%)")
            print(f"  With CPC:  {cpcs:,} ({100*cpcs/n:.1f}%)")
            print(f"  With IPC:  {ipcs:,} ({100*ipcs/n:.1f}%)")
            print(f"  With inventors: {inventors_n:,} ({100*inventors_n/n:.1f}%)")
            print("  Countries:", dict(sorted(countries.items(), key=lambda x: -x[1])[:10]))
            grand_total += n

        else:
            stats = ingest_zip(conn, str(zip_path), include_raw=include_raw,
                               error_log=error_log_path,
                               delete_after=args.delete_after,
                               progress_file=progress_file)
            grand_total += stats["total"]
            grand_inserted += stats["inserted"]
            grand_errors += stats["errors"]

    elapsed_all = time.time() - t_all
    if args.scan_errors:
        print(f"\nTotal: {grand_total:,} docs  {grand_errors:,} errors  "
              f"({elapsed_all:.1f}s)")
    elif args.dry_run:
        print(f"\nTotal: {grand_total:,} patents across {len(zips)} ZIP(s)  "
              f"({elapsed_all:.1f}s)")
    else:
        summary = (f"\nAll done: {grand_total:,} total  {grand_inserted:,} inserted  "
                   f"{grand_errors:,} errors  {elapsed_all:.1f}s")
        print(summary)
        if progress_file:
            with open(progress_file, "a") as pf:
                pf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] === SESSION COMPLETE === "
                         f"{grand_total:,} total, {grand_inserted:,} inserted, "
                         f"{grand_errors:,} errors, {elapsed_all:.1f}s\n")

    conn.close()


if __name__ == "__main__":
    main()

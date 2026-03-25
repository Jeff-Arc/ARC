#!/usr/bin/env python3
"""
arc_patent_api.py — USPTO Open Data Portal API client for ARC.

PURPOSE:
  1. On-demand full text: fetch claims + description for specific patents
  2. Setup: create patent_full_text_cache table + get_full_text() SQL function in arc_v5

USPTO API: https://api.uspto.gov
Auth:      X-Api-Key header (camel-case — X-API-KEY fails silently with 403)
Key env:   USPTO_API_KEY

RATE LIMITS (per https://data.uspto.gov/apis/rate-limits):
  - Metadata endpoint: ~4-15 req/sec (burst=1) — SEQUENTIAL only
  - Weekly quota: ~5,000,000 metadata calls (resets Sunday midnight UTC)
  - XML file downloads: 20 downloads/year per unique file URL per API key
  - Always hit the cache before calling the API — avoid burning download quota

IMPORTANT: The download limit (20/year/file) is per UNIQUE FILE PATH. Each
individual patent XML has its own 20-request annual budget. Use
patent_full_text_cache to avoid re-fetching.

Usage:
  # Create cache table and SQL function in arc_v5
  USPTO_API_KEY=... python3 arc_patent_api.py --mode setup

  # Fetch full text for a specific patent application number
  USPTO_API_KEY=... python3 arc_patent_api.py --mode fulltext --patent-id 16123456
"""

# ── USPTO API rate limits ──────────────────────────────────────────────────────
#
# The USPTO Open Data Portal exposes two distinct API tiers with very different
# limits. It is critical to know which tier each mode in this script uses:
#
#  TIER 1 — Metadata API  (search + application lookup)
#    Endpoint pattern:  /api/v1/patent/applications/...
#    Weekly quota:      ~5,000,000 calls (resets Sunday midnight UTC)
#    Used by:           --mode incremental (search), --mode backfill-abstracts
#                       (application lookup to resolve fileLocationURI)
#
#  TIER 2 — Documents API  (individual patent XML file fetches)
#    Endpoint pattern:  /api/v1/datasets/products/files/...  (the fileLocationURI)
#    Weekly quota:      ~1,200,000 calls/week
#    Used by:           --mode backfill-abstracts (XML download for abstract text)
#                       --mode fulltext (full claims + description download)
#    NOTE: This is NOT the Bulk Datasets download tier — the 20/year limit
#          does NOT apply here. See the Bulk Datasets note below.
#
#  NOT USED — Bulk Datasets Downloads  (zip files from bulkdata.uspto.gov)
#    Limit:             20 downloads/year per unique file URL per API key
#    Example files:     ipg111213.zip, ipa100415.zip  (weekly grant/pgpub zips)
#    This script does NOT download bulk zip files via the API. The fileLocationURI
#    returned by the metadata endpoint resolves to individual patent XMLs via
#    TIER 2 (Documents API), not to bulk zip files. The 20/year cap is irrelevant.
#
#  Burst limit (all tiers): 1 — calls must be SEQUENTIAL, never parallel.
#    REQUEST_DELAY_SEC = 0.25 enforces ~4 req/sec.
#    HTTP 429 is retried up to 3× with a 5-second back-off.
#
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import csv
import json
import os
import pickle
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime
from pathlib import Path

import requests

import psycopg2

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE          = "https://api.uspto.gov/api/v1"
REQUEST_DELAY_SEC = 0.25   # 4 req/sec max (burst=1 enforced by sequential calls)
RETRY_DELAY_SEC   = 5.0    # wait on HTTP 429
API_KEY           = os.environ.get("USPTO_API_KEY", "")

# ── CPC prefix mapping ─────────────────────────────────────────────────────────
# Maps corpus_id → USPTO CPC search prefix(es).
# Longevity corpora use comma-separated prefixes (OR-queried).

CORPUS_CPC_MAP = {
    'H01L_quarterly':    'H01L',
    'G06N_quarterly':    'G06N',
    'G06F_quarterly':    'G06F',
    'G01N_quarterly':    'G01N',
    'G01B_quarterly':    'G01B',
    'G02B_quarterly':    'G02B',
    'C23C_quarterly':    'C23C',
    'C30B_quarterly':    'C30B',
    'A61P9_quarterly':   'A61P9',
    'A61P25_quarterly':  'A61P25',
    'C12N15_quarterly':  'C12N15,A61K48,C12N9',
    'A61K38_quarterly':  'A61K38,A61K35,C12N5',
    'longevity_patents_quarterly': 'A61K38,A61K35,C12N5,C12N15,A61K48,C12N9,A61P9',
}


# ── HTTP ──────────────────────────────────────────────────────────────────────

def _get(path: str, api_key: str, retries: int = 3) -> dict:
    """
    Sequential GET to USPTO API with X-Api-Key header.
    NEVER call from parallel threads with the same API key (burst=1).
    Retries up to 3× with 5-second delay on HTTP 429.

    Uses requests (not urllib) — urllib.add_header() capitalizes keys, which
    breaks AWS API Gateway auth (X-Api-Key → X-api-key → 403 silently).
    """
    url = f"{API_BASE}{path}"
    headers = {"X-Api-Key": api_key}
    for attempt in range(retries):
        time.sleep(REQUEST_DELAY_SEC)
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 429 and attempt < retries - 1:
            print(f"  [429] Rate limit — waiting {RETRY_DELAY_SEC}s", file=sys.stderr)
            time.sleep(RETRY_DELAY_SEC)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Failed after {retries} attempts: {url}")


# ── Patent data ───────────────────────────────────────────────────────────────

def fetch_application_metadata(app_number: str, api_key: str) -> dict:
    """
    Fetch patent file wrapper via /api/v1/patent/applications/{appNum}.

    Returns the full patentFileWrapperDataBag entry, which includes:
      - applicationMetaData: title, CPC codes, inventors, assignees, dates
      - eventDataBag: prosecution history (25+ events)
      - grantDocumentMetaData: fileLocationURI for full-text XML download
      - assignmentBag: assignee names and addresses
      - foreignPriorityBag: priority claims

    Rate: counts against 5M/week metadata quota.
    """
    data = _get(f"/patent/applications/{app_number}", api_key)
    bag = data.get("patentFileWrapperDataBag", [])
    return bag[0] if bag else {}


def fetch_full_text_xml(file_location_uri: str, api_key: str) -> bytes:
    """
    Download full patent XML (abstract + claims + description) from USPTO.

    This calls the Documents API (Tier 2, ~1.2M calls/week) — NOT the Bulk
    Datasets download tier. The fileLocationURI points to an individual patent
    XML file, not a bulk zip. The 20-downloads/year-per-file limit applies ONLY
    to bulk zip downloads from bulkdata.uspto.gov and does NOT apply here.

    The API returns HTTP 302 → signed S3 URL. requests follows redirects
    automatically, so the final response is the raw XML bytes.
    """
    # fileLocationURI is returned as an absolute URL by the USPTO API
    url = file_location_uri if file_location_uri.startswith("http") else f"{API_BASE}{file_location_uri}"
    time.sleep(REQUEST_DELAY_SEC)
    resp = requests.get(url, headers={"X-Api-Key": api_key}, timeout=60)
    resp.raise_for_status()
    return resp.content


def parse_patent_xml(xml_bytes: bytes) -> dict:
    """
    Parse USPTO grant XML into {abstract, claims, description}.
    Root tag: <us-patent-grant> with children:
      <abstract>, <claims> (N × <claim>), <description>
    """
    root = ET.fromstring(xml_bytes)

    def get_text(tag: str) -> str:
        el = root.find(f".//{tag}")
        return " ".join(el.itertext()).strip() if el is not None else ""

    claims = [
        " ".join(c.itertext()).strip()
        for c in root.findall(".//claim")
        if " ".join(c.itertext()).strip()
    ]

    return {
        "abstract":    get_text("abstract"),
        "claims":      "\n".join(claims),
        "description": get_text("description"),
    }


# ── Cache ─────────────────────────────────────────────────────────────────────

def get_or_fetch_full_text(conn, external_id: str, api_key: str) -> dict:
    """
    Safe entry point for full-text retrieval. Always checks cache first.
    Only calls USPTO API on a cache miss (preserving the 20/year download limit).

    Returns: {abstract, claims, description, source}
    source = 'cache' | 'api_fresh' | 'error'
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT abstract, claims, description FROM patent_full_text_cache "
            "WHERE external_id = %s",
            (external_id,),
        )
        row = cur.fetchone()
    if row:
        return {"abstract": row[0], "claims": row[1], "description": row[2], "source": "cache"}

    print(f"  Cache miss — fetching from USPTO API for {external_id}...")

    try:
        meta = fetch_application_metadata(external_id, api_key)
        grant_meta = meta.get("grantDocumentMetaData", {})
        file_uri = grant_meta.get("fileLocationURI")
        if not file_uri:
            return {"abstract": None, "claims": None, "description": None, "source": "no_file_uri"}

        xml_bytes = fetch_full_text_xml(file_uri, api_key)
        result = parse_patent_xml(xml_bytes)
        result["source"] = "api_fresh"

    except Exception as e:
        print(f"  [ERROR] USPTO API fetch failed for {external_id}: {e}", file=sys.stderr)
        return {"abstract": None, "claims": None, "description": None, "source": f"error: {e}"}

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO patent_full_text_cache
              (external_id, abstract, claims, description)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (external_id) DO NOTHING
            """,
            (external_id, result["abstract"], result["claims"], result["description"]),
        )
    conn.commit()
    return result


# ── Incremental update (TSV-based) ────────────────────────────────────────────
#
# NOTE: The USPTO Open Data Portal API does not support CPC-filtered search.
# POST /patent/applications/search returns 404; GET ignores all filter params.
# Incremental update uses PatentsView bulk TSV files instead — the same data
# source arc_ingest.py uses. Fresh zip files are auto-extracted when newer
# than the existing TSVs.
#
# Data files (PatentsView bulk download — /home/jeff/data/):
#   g_cpc_current.tsv      patent_id → cpc_group (3.1 GB; use cached pickle)
#   g_patent.tsv           patent_id → title, pub_date, type, withdrawn
#   g_patent_abstract.tsv  patent_id → abstract text
#
# CPC index cache: /tmp/arc_ingest_{corpus_id}_cpc.pkl  (same as arc_ingest.py)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR_CANDIDATES = [
    Path(os.environ.get("ARC_DATA_DIR", "/home/jeff/data/patents/")),
    Path("/home/jeff/data"),
]


def get_data_dir() -> Path:
    """Return the directory containing g_cpc_current.tsv (or its zip)."""
    for d in DATA_DIR_CANDIDATES:
        if (d / "g_cpc_current.tsv").exists():
            return d
    for d in DATA_DIR_CANDIDATES:
        if (d / "g_cpc_current.tsv.zip").exists():
            return d
    return DATA_DIR_CANDIDATES[0]


def extract_zip_if_needed(data_dir: Path, name: str) -> bool:
    """
    Extract {name}.zip → {name} if the zip is newer than the extracted file.
    Returns True if extraction happened.
    """
    zip_path = data_dir / f"{name}.zip"
    tsv_path = data_dir / name
    if not zip_path.exists():
        return False
    if tsv_path.exists() and zip_path.stat().st_mtime <= tsv_path.stat().st_mtime:
        return False
    print(f"  Extracting {zip_path.name} ({zip_path.stat().st_size / 1e6:.0f} MB)...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    size_gb = tsv_path.stat().st_size / 1e9
    print(f"  Done: {tsv_path.name} ({size_gb:.1f} GB)")
    return True


def refresh_bulk_data(data_dir: Path) -> None:
    """Extract any zip files that are newer than their extracted counterparts."""
    refreshed = []
    for name in ("g_cpc_current.tsv", "g_patent.tsv", "g_patent_abstract.tsv"):
        if extract_zip_if_needed(data_dir, name):
            refreshed.append(name)
    if refreshed:
        print(f"  Refreshed: {', '.join(refreshed)}")
    else:
        print("  Bulk data files are up to date.")


def get_last_ingest_date(conn, corpus_id: str) -> str:
    """
    Return the most recent publication_date (= patent grant date) in
    data_documents for this corpus. This is the correct cutoff for
    finding new grants in the bulk TSV files.

    NOTE: content_date = filing_date in most records (arc_ingest.py sets
    content_date = filing_date or pub_date), so MAX(content_date) would
    return the most recent FILING date, not GRANT date — wrong for this purpose.

    Falls back to '2020-01-01' if no records exist.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT MAX(publication_date)::text
            FROM data_documents
            WHERE corpus_id = %s
              AND source_type = 'patents'
              AND publication_date IS NOT NULL
        """, (corpus_id,))
        row = cur.fetchone()
        return row[0] if row and row[0] else '2020-01-01'


def _load_cpc_index(corpus_id: str, cpc_prefix: str, data_dir: Path) -> "tuple[set, dict]":
    """
    Load or build the CPC index for this corpus.
    Reuses /tmp/arc_ingest_{corpus_id}_cpc.pkl if it exists and TSV hasn't
    been refreshed since the cache was written (same cache as arc_ingest.py).

    Returns (matching_ids: set[str], cpc_map: dict[patent_id, list[str]]).
    """
    cache_path = Path(f"/tmp/arc_ingest_{corpus_id}_cpc.pkl")
    cpc_file   = data_dir / "g_cpc_current.tsv"

    if cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        tsv_mtime   = cpc_file.stat().st_mtime if cpc_file.exists() else 0
        if cache_mtime >= tsv_mtime:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            print(f"  CPC index: loaded from cache ({len(cached[0]):,} patents)")
            return cached

    prefixes = [p.strip() for p in cpc_prefix.split(',')]
    print(f"  Building CPC index from {cpc_file} (may take ~1 min)...")
    matching_ids: set = set()
    cpc_map: dict = {}

    with open(cpc_file, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t", quotechar='"')
        next(reader)  # header
        for row in reader:
            if len(row) < 6:
                continue
            patent_id = row[0].strip().strip('"')
            cpc_group = row[5].strip().strip('"')
            if any(cpc_group.startswith(p) for p in prefixes):
                matching_ids.add(patent_id)
                if patent_id not in cpc_map:
                    cpc_map[patent_id] = []
                if cpc_group not in cpc_map[patent_id]:
                    cpc_map[patent_id].append(cpc_group)

    with open(cache_path, "wb") as f:
        pickle.dump((matching_ids, cpc_map), f)
    print(f"  CPC index: built {len(matching_ids):,} matching patents → cached")
    return matching_ids, cpc_map


def _iter_new_patents(matching_ids: set, cutoff: date, data_dir: Path):
    """
    Yield (patent_id, title, pub_date) for patents in matching_ids
    with pub_date > cutoff. Reads g_patent.tsv sequentially.
    Skips withdrawn patents (withdrawn column = 1).
    """
    patent_file = data_dir / "g_patent.tsv"
    with open(patent_file, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t", quotechar='"')
        next(reader)  # header: patent_id, patent_type, patent_date, patent_title, ...
        for row in reader:
            if len(row) < 4:
                continue
            patent_id = row[0].strip().strip('"')
            if patent_id not in matching_ids:
                continue
            withdrawn = row[6].strip().strip('"') if len(row) > 6 else '0'
            if withdrawn == '1':
                continue
            try:
                pub_date = datetime.strptime(
                    row[2].strip().strip('"'), '%Y-%m-%d'
                ).date()
            except ValueError:
                continue
            if pub_date > cutoff:
                title = row[3].strip().strip('"')
                yield patent_id, title, pub_date


def _load_abstracts_for_ids(patent_ids: set, data_dir: Path) -> dict:
    """
    Read g_patent_abstract.tsv and return {patent_id: abstract_text}
    for the given set of IDs. Reads the full file once.
    """
    abstract_file = data_dir / "g_patent_abstract.tsv"
    abstracts = {}
    with open(abstract_file, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t", quotechar='"')
        next(reader)  # header
        for row in reader:
            if len(row) < 2:
                continue
            pid  = row[0].strip().strip('"')
            if pid not in patent_ids:
                continue
            text = row[1].strip().strip('"')
            if text:
                abstracts[pid] = text
    return abstracts


def _build_cpc_query(cpc_prefix: str) -> str:
    """Build Lucene query for one or more CPC prefixes (comma-separated).

    Correct field: applicationMetaData.cpcClassificationBag (flat string array).
    NOTE: ...cpcClassificationBag.cpcClassificationCode returns 404 — wrong path.
    """
    prefixes = [c.strip() for c in cpc_prefix.split(',')]
    if len(prefixes) == 1:
        return f"applicationMetaData.cpcClassificationBag:{_api_cpc_prefix(prefixes[0])}*"
    return ' OR '.join(
        f"applicationMetaData.cpcClassificationBag:{_api_cpc_prefix(c)}*" for c in prefixes
    )


def _api_cpc_prefix(prefix: str) -> str:
    """
    Convert a condensed CPC prefix (e.g. 'A61P9', 'C12N15') to the double-space
    format used by the USPTO API's cpcClassificationBag index.

    The USPTO stores CPC codes as 'CCCC  GG/NNNN' with two spaces between the
    4-character class and the group number.  A bare 4-char class prefix (e.g.
    'H01L') is returned unchanged since the double-space only appears when a
    group number is present.

    Examples:
        'H01L'   → 'H01L'       (4 chars — no group, unchanged)
        'A61P9'  → 'A61P  9'    (5 chars — insert double space after class)
        'C12N15' → 'C12N  15'   (6 chars — insert double space after class)
        'A61K38' → 'A61K  38'
    """
    if len(prefix) <= 4:
        return prefix
    return prefix[:4] + '  ' + prefix[4:]


def _count_new_patents_single(cpc_prefix: str, date_from: str, date_to: str) -> int:
    """
    Count new granted patents for a single CPC prefix via USPTO POST search API.
    Uses limit=1 (minimum allowed) and reads totalCount from response.
    Costs one metadata API call against the 5M/week quota.
    """
    payload = {
        "q": f"applicationMetaData.cpcClassificationBag:{_api_cpc_prefix(cpc_prefix)}*",
        "filters": [
            {
                "name": "applicationMetaData.applicationStatusDescriptionText",
                "value": ["Patented Case"],
            }
        ],
        "rangeFilters": [
            {
                "field": "applicationMetaData.grantDate",
                "valueFrom": date_from,
                "valueTo": date_to,
            }
        ],
        "pagination": {"offset": 0, "limit": 1},
    }
    time.sleep(REQUEST_DELAY_SEC)
    resp = requests.post(
        f"{API_BASE}/patent/applications/search",
        headers={"X-Api-Key": API_KEY, "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    # 404 "No matching records found" means zero results — not a routing error
    if resp.status_code == 404:
        return 0
    resp.raise_for_status()
    return resp.json().get('count', 0)


def count_new_patents(cpc_prefix: str, date_from: str, date_to: str) -> int:
    """
    Count new granted patents for one or more CPC prefixes (comma-separated).

    For multi-CPC corpora the USPTO OR query returns 0 or errors, so each
    prefix is queried separately and the counts are summed. This may
    double-count patents that carry multiple CPC codes — acceptable for
    estimating new-patent volumes.
    """
    prefixes = [p.strip() for p in cpc_prefix.split(',')]
    total = 0
    for prefix in prefixes:
        total += _count_new_patents_single(prefix, date_from, date_to)
    return total


def _normalize_cpc(cpc: str) -> str:
    """Strip internal whitespace from a CPC code string.

    The USPTO API returns codes as 'H01L  21/486' (double space between class
    and group).  PatentsView bulk TSVs and data_documents store them without
    spaces: 'H01L21/486'.  This normalizes API-sourced codes to match the
    bulk format.
    """
    return ''.join(cpc.split())


API_PAGE_SIZE = 25   # USPTO search endpoint; conservative — increase if API allows more


def _fetch_patents_api(api_prefix: str, date_from: str, date_to: str) -> list:
    """
    Paginate through the USPTO POST search API for a single (formatted) CPC
    prefix and return a list of patent dicts:
      {'patent_id': str, 'title': str, 'grant_date': str, 'cpc_codes': list[str]}

    api_prefix must already be in double-space format (e.g. 'A61P  9').
    Calls are sequential with REQUEST_DELAY_SEC between each page.
    """
    results = []
    offset  = 0

    while True:
        payload = {
            "q": f"applicationMetaData.cpcClassificationBag:{api_prefix}*",
            "filters": [
                {
                    "name":  "applicationMetaData.applicationStatusDescriptionText",
                    "value": ["Patented Case"],
                }
            ],
            "rangeFilters": [
                {
                    "field":     "applicationMetaData.grantDate",
                    "valueFrom": date_from,
                    "valueTo":   date_to,
                }
            ],
            "pagination": {"offset": offset, "limit": API_PAGE_SIZE},
        }

        time.sleep(REQUEST_DELAY_SEC)
        resp = requests.post(
            f"{API_BASE}/patent/applications/search",
            headers={"X-Api-Key": API_KEY, "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )

        if resp.status_code == 404:
            break   # no (more) results
        resp.raise_for_status()

        data = resp.json()
        bag  = data.get("patentFileWrapperDataBag", [])
        if not bag:
            break

        for item in bag:
            meta       = item.get("applicationMetaData", {})
            patent_num = meta.get("patentNumber")
            if not patent_num:
                continue   # application not yet granted — skip
            # Abstract is intentionally absent: the search endpoint returns only
            # bibliographic metadata (title, CPC codes, dates, inventors).
            # Abstract text lives in the grant/pgpub XML and must be fetched
            # separately via --mode backfill-abstracts (Documents API, Tier 2).
            results.append({
                "patent_id":  str(patent_num),
                "title":      meta.get("inventionTitle") or "",
                "grant_date": meta.get("grantDate"),
                "cpc_codes":  [_normalize_cpc(c) for c in meta.get("cpcClassificationBag", [])],
            })

        total_count = data.get("count", 0)
        offset += len(bag)
        if offset >= total_count:
            break

        if offset % (API_PAGE_SIZE * 20) == 0:   # progress every 500 records
            print(f"    fetched {offset:,}/{total_count:,} for {api_prefix}...",
                  flush=True)

    return results


def run_incremental_api(
    conn,
    corpus_id: str,
    dry_run: bool = False,
    since: str = None,
) -> int:
    """
    Insert new patents from the USPTO POST search API for one corpus.

    Used when bulk TSV files are stale (no patents past the cutoff date).
    Stores title, grant_date, and CPC codes; abstract is NULL because the
    search endpoint does not return abstract text.  Abstract will be backfilled
    on the next bulk-TSV incremental run once PatentsView publishes updated files.

    For multi-CPC corpora each prefix is fetched separately; results are
    deduplicated by patentNumber before insert.

    Rate: ~4 req/sec (burst=1 — sequential only).
    """
    cpc_prefix = CORPUS_CPC_MAP.get(corpus_id)
    if not cpc_prefix:
        print(f"ERROR: No CPC mapping for corpus {corpus_id}", file=sys.stderr)
        return 0

    last_date = since or get_last_ingest_date(conn, corpus_id)
    today     = date.today().isoformat()

    print(f"\n  [API mode] Fetching patents {last_date} → {today}")

    prefixes   = [p.strip() for p in cpc_prefix.split(',')]
    all_patents: dict = {}   # patent_id → {title, grant_date, cpc_codes}

    for prefix in prefixes:
        api_prefix = _api_cpc_prefix(prefix)
        print(f"  Querying API: {api_prefix}* ...", flush=True)
        records    = _fetch_patents_api(api_prefix, last_date, today)
        print(f"    → {len(records):,} records")
        for r in records:
            pid = r["patent_id"]
            if pid not in all_patents:
                all_patents[pid] = r
            else:
                # Merge CPC codes from additional prefix hits
                existing = set(all_patents[pid]["cpc_codes"])
                all_patents[pid]["cpc_codes"] = list(existing | set(r["cpc_codes"]))

    print(f"  Unique patents after dedup: {len(all_patents):,}")

    if not all_patents:
        print("  Nothing to insert.")
        return 0

    if dry_run:
        print(f"  [dry-run] Would insert up to {len(all_patents):,} patents")
        return 0

    inserted = 0
    t0       = time.time()

    for patent_id, data in all_patents.items():
        if not patent_id or (not data["title"] and not data["cpc_codes"]):
            continue

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO data_documents (
                    document_id, corpus_id, external_id,
                    title, abstract,
                    publication_date, content_date,
                    cpc_codes,
                    source_type, source_api, corpus_type
                ) VALUES (
                    %s, %s, %s,
                    %s, NULL,
                    %s::date, %s::date,
                    %s,
                    'patents', 'uspto_api', 'patents'
                )
                ON CONFLICT (document_id) DO NOTHING
            """, (
                f"{corpus_id}_{patent_id}",
                corpus_id,
                patent_id,
                data["title"],
                data["grant_date"],
                data["grant_date"],
                data["cpc_codes"] or None,
            ))
            if cur.rowcount > 0:
                inserted += 1

        if inserted % 500 == 0 and inserted > 0:
            conn.commit()
            elapsed = time.time() - t0
            print(f"  Progress: {inserted:,} inserted ({elapsed:.0f}s)")

    conn.commit()
    elapsed = time.time() - t0
    print(f"  API ingest complete: {inserted:,} inserted, "
          f"{len(all_patents) - inserted:,} already present. "
          f"Elapsed: {elapsed:.0f}s")
    return inserted


def run_incremental_update(
    conn,
    corpus_id: str,
    dry_run: bool = False,
    since: str = None,
):
    """
    Insert all new patents for a corpus since last ingest date.

    Process:
    1. Auto-extract zip files if fresher than existing TSVs
    2. Get last content_date from data_documents (or use --since override)
    3. Build / reuse CPC index from g_cpc_current.tsv (cached in /tmp)
    4. Scan g_patent.tsv for new patents (pub_date > cutoff)
    5. Load abstracts from g_patent_abstract.tsv for the new set
    6. Insert into data_documents (ON CONFLICT DO NOTHING)
    7. If bulk files have no new patents (stale), fall back to USPTO API

    Performance: scanning g_patent.tsv (~1.1 GB) takes ~30 sec; g_patent_abstract.tsv
    (~5.8 GB) takes ~3 min. The CPC index is cached so subsequent runs are faster.
    API fallback: ~4 req/sec; abstracts not available (stored NULL, backfilled
    on next bulk run once PatentsView publishes updated files).
    """
    cpc_prefix = CORPUS_CPC_MAP.get(corpus_id)
    if not cpc_prefix:
        print(f"ERROR: No CPC mapping for corpus {corpus_id}", file=sys.stderr)
        return

    data_dir  = get_data_dir()
    last_date = since or get_last_ingest_date(conn, corpus_id)

    try:
        cutoff = datetime.strptime(last_date, '%Y-%m-%d').date()
    except ValueError:
        cutoff = date(2020, 1, 1)

    print(f"\n{'='*60}")
    print(f"Corpus:     {corpus_id}")
    print(f"CPC prefix: {cpc_prefix}")
    print(f"Cutoff:     {last_date} (patents with pub_date > this)")
    print(f"Data dir:   {data_dir}")

    # Step 1: Try bulk-file path; fall back to API if files are missing or stale
    try:
        refresh_bulk_data(data_dir)
        matching_ids, cpc_map = _load_cpc_index(corpus_id, cpc_prefix, data_dir)

        print(f"  Scanning g_patent.tsv for patents after {last_date}...")
        new_patents = []   # list of (patent_id, title, pub_date)
        for patent_id, title, pub_date in _iter_new_patents(matching_ids, cutoff, data_dir):
            new_patents.append((patent_id, title, pub_date))

        print(f"  New patents found: {len(new_patents):,}")
    except FileNotFoundError as e:
        print(f"  Bulk files not available ({e.filename}). Falling back to API mode.")
        new_patents = []

    if not new_patents:
        # Bulk files are stale or missing — fall back to USPTO POST search API.
        if not API_KEY:
            print("  ERROR: USPTO_API_KEY not set — cannot use API fallback.",
                  file=sys.stderr)
            return 0
        return run_incremental_api(conn, corpus_id, dry_run=dry_run, since=last_date)

    if dry_run:
        print(f"  [dry-run] Would insert up to {len(new_patents):,} patents")
        return 0

    # Step 4: load abstracts for only the new IDs (one pass through abstract TSV)
    new_ids = {pid for pid, _, _ in new_patents}
    print(f"  Loading abstracts for {len(new_ids):,} patents from g_patent_abstract.tsv...")
    abstracts = _load_abstracts_for_ids(new_ids, data_dir)
    print(f"  Abstracts found: {len(abstracts):,}")

    # Step 5: insert
    inserted           = 0
    skipped_no_abstract = 0
    t0 = time.time()

    for patent_id, title, pub_date in new_patents:
        abstract = abstracts.get(patent_id, '')
        if not abstract and not title:
            skipped_no_abstract += 1
            continue

        codes = cpc_map.get(patent_id, [])

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO data_documents (
                    document_id, corpus_id, external_id,
                    title, abstract,
                    publication_date, content_date,
                    cpc_codes,
                    source_type, source_api, corpus_type
                ) VALUES (
                    %s, %s, %s,
                    %s, %s,
                    %s::date, %s::date,
                    %s,
                    'patents', 'uspto', 'patents'
                )
                ON CONFLICT (document_id) DO NOTHING
            """, (
                f"{corpus_id}_{patent_id}",
                corpus_id,
                patent_id,
                title,
                abstract,
                pub_date,
                pub_date,    # content_date = pub_date (filing dates not loaded here)
                codes or None,
            ))
            if cur.rowcount > 0:
                inserted += 1
        if inserted % 500 == 0:
            conn.commit()
            elapsed = time.time() - t0
            rate = inserted / elapsed if elapsed > 0 else 0
            print(f"  Progress: {inserted:,}/{len(new_patents):,} "
                  f"({rate:.0f}/sec)")

    conn.commit()
    elapsed = time.time() - t0
    print(f"\n  Complete: {inserted:,} inserted, "
          f"{skipped_no_abstract} skipped (no title/abstract). "
          f"Elapsed: {elapsed:.0f}s")
    return inserted


# ── Backfill abstracts helpers ────────────────────────────────────────────────

def _fetch_file_uri_by_patent_number(patent_number: str, api_key: str) -> "tuple[str|None, str]":
    """
    Search for a patent by its grant number and return (fileLocationURI, source).

    external_id in data_documents stores the patent GRANT number (e.g. 12454767),
    not the application number. The GET /patent/applications/{id} endpoint expects
    an application number, so direct lookup always 404s for recent patents.
    This function uses POST search on applicationMetaData.patentNumber instead,
    which returns the full file wrapper including grantDocumentMetaData.fileLocationURI
    in a single call.

    Returns (uri, 'grant'|'pgpub'|'none').
    """
    payload = {
        "q": f"applicationMetaData.patentNumber:{patent_number}",
        "pagination": {"offset": 0, "limit": 1},
    }
    time.sleep(REQUEST_DELAY_SEC)
    resp = requests.post(
        f"{API_BASE}/patent/applications/search",
        headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    if resp.status_code == 404:
        return None, "none"
    resp.raise_for_status()

    bag  = resp.json().get("patentFileWrapperDataBag", [])
    item = bag[0] if bag else {}

    grant_uri = item.get("grantDocumentMetaData", {}).get("fileLocationURI")
    if grant_uri:
        return grant_uri, "grant"

    pgpub_uri = item.get("pgpubDocumentMetaData", {}).get("fileLocationURI")
    if pgpub_uri:
        return pgpub_uri, "pgpub"

    return None, "none"


# ── Backfill abstracts ────────────────────────────────────────────────────────

def run_backfill_abstracts(
    conn,
    api_key: str,
    corpus_id: str = None,
    dry_run: bool = False,
    limit: int = None,
) -> None:
    """
    Backfill abstract text for data_documents rows where abstract IS NULL
    and source_api='uspto_api'.

    For each patent (sequential — USPTO burst=1):
      1. Fetch metadata → grantDocumentMetaData.fileLocationURI
         (falls back to pgpubDocumentMetaData.fileLocationURI)
      2. Download full-text XML from that URI
      3. Parse <abstract> tag via parse_patent_xml()
      4. UPDATE data_documents SET abstract = ... WHERE external_id = ... AND abstract IS NULL

    Uses DISTINCT ON (external_id) so patents shared across corpora are fetched
    once; the single UPDATE covers all matching rows automatically.
    Resume-safe: WHERE abstract IS NULL skips already-filled rows on re-run.
    Commits every 100 patents processed.
    """
    where_parts = ["abstract IS NULL", "source_api = 'uspto_api'"]
    params: list = []
    if corpus_id:
        where_parts.append("corpus_id = %s")
        params.append(corpus_id)
    where_clause = " AND ".join(where_parts)
    limit_clause = "LIMIT %s" if limit else ""

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT DISTINCT ON (external_id) external_id, corpus_id, publication_date
            FROM data_documents
            WHERE {where_clause}
            ORDER BY external_id, corpus_id, publication_date
            {limit_clause}
            """,
            params + ([limit] if limit else []),
        )
        rows = cur.fetchall()

    total = len(rows)
    print(f"\n{'='*60}")
    print(f"Backfill abstracts: {total:,} distinct patents to process")
    if corpus_id:
        print(f"Corpus filter:      {corpus_id}")
    if limit:
        print(f"Limit:              {limit}")
    if dry_run:
        print("[dry-run] No updates will be written.")
    print(f"{'='*60}\n")

    if not rows:
        print("Nothing to do.")
        return

    done = skipped = errors = updated_rows = 0
    t0 = time.time()
    pending_commit = 0

    for i, (ext_id, sample_corpus, sample_date) in enumerate(rows, 1):

        # ── Progress every 100 ───────────────────────────────────────────────
        if i % 100 == 0:
            elapsed = time.time() - t0
            rate    = i / elapsed if elapsed > 0 else 0
            remaining = (total - i) / rate if rate > 0 else 0
            print(
                f"  [{i:>6}/{total}]  done={done}  skip={skipped}  err={errors}"
                f"  ({rate:.1f}/sec, ~{remaining/60:.1f} min remaining)"
            )

        # ── Step 1: search by patent number → fileLocationURI ───────────────
        # external_id is a patent GRANT number (e.g. 12454767), not an
        # application number. Must search by patentNumber field — direct
        # GET /patent/applications/{id} always 404s for recent patents.
        try:
            file_uri, uri_source = _fetch_file_uri_by_patent_number(ext_id, api_key)
        except Exception as e:
            print(f"  [ERROR] patent number search failed for {ext_id}: {e}",
                  file=sys.stderr)
            errors += 1
            continue

        if not file_uri:
            skipped += 1
            continue

        # ── Dry-run: show what would be fetched ──────────────────────────────
        if dry_run:
            print(f"  [{i}/{total}] {ext_id}  ({sample_corpus})  [{uri_source}]")
            print(f"         uri:    {file_uri}")
            done += 1
            continue

        # ── Step 2: download XML ──────────────────────────────────────────────
        try:
            xml_bytes = fetch_full_text_xml(file_uri, api_key)
        except Exception as e:
            print(f"  [ERROR] XML download failed for {ext_id}: {e}",
                  file=sys.stderr)
            errors += 1
            continue

        # ── Step 3: parse abstract ────────────────────────────────────────────
        try:
            parsed = parse_patent_xml(xml_bytes)
        except Exception as e:
            print(f"  [ERROR] XML parse failed for {ext_id}: {e}",
                  file=sys.stderr)
            errors += 1
            continue

        abstract = parsed.get("abstract", "").strip()
        if not abstract:
            skipped += 1
            continue

        # ── Step 4: update all corpora sharing this external_id ───────────────
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE data_documents SET abstract = %s"
                " WHERE external_id = %s AND abstract IS NULL",
                (abstract, ext_id),
            )
            updated_rows += cur.rowcount

        done += 1
        pending_commit += 1
        if pending_commit >= 100:
            conn.commit()
            pending_commit = 0

    conn.commit()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Backfill complete.")
    print(f"  Patents processed:         {done + skipped + errors:,} / {total:,}")
    print(f"  Abstracts written:         {done:,}  ({updated_rows:,} rows across corpora)")
    print(f"  Skipped (no URI/abstract): {skipped:,}")
    print(f"  Errors:                    {errors:,}")
    print(f"  Elapsed:                   {elapsed/60:.1f} min")
    print(f"{'='*60}")


# ── Bulk URI fetch ────────────────────────────────────────────────────────────

BULK_URI_CACHE_PATH = Path.home() / "arc" / "data" / "ptfw_uri_cache.tsv"
BULK_URI_DATE_FROM  = "2025-10-01"
BULK_URI_FIELDS     = [
    "applicationNumberText",
    "applicationMetaData.grantDate",
    "applicationMetaData.inventionTitle",
    "applicationMetaData.cpcClassificationBag",
    "grantDocumentMetaData.fileLocationURI",
]


def _load_uri_cache_ids(path: Path) -> set:
    """Return the set of external_ids already written to ptfw_uri_cache.tsv."""
    seen: set = set()
    if not path.exists():
        return seen
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader, None)   # skip header row
        for row in reader:
            if row:
                seen.add(row[0])
    return seen


def _post_search_download_page(
    api_prefix: str,
    date_from: str,
    date_to: str,
    offset: int,
    api_key: str,
    retries: int = 3,
) -> dict:
    """
    One page from POST /api/v1/patent/applications/search (full record).

    NOTE: /search/download rejects grantDocumentMetaData.fileLocationURI as an
    invalid download field and returns an empty bag even with valid fields.
    The standard /search endpoint returns full file-wrapper records including
    grantDocumentMetaData.fileLocationURI, which the caller extracts.

    Returns {} on HTTP 404 (no results).
    Retries up to `retries` times on HTTP 429 with RETRY_DELAY_SEC back-off.
    Calls are sequential (never parallel) — USPTO burst limit = 1.
    """
    payload = {
        "q": f"applicationMetaData.cpcClassificationBag:{api_prefix}*",
        "filters": [
            {
                "name":  "applicationMetaData.applicationStatusDescriptionText",
                "value": ["Patented Case"],
            }
        ],
        "rangeFilters": [
            {
                "field":     "applicationMetaData.grantDate",
                "valueFrom": date_from,
                "valueTo":   date_to,
            }
        ],
        "pagination": {"offset": offset, "limit": API_PAGE_SIZE},
    }
    url     = f"{API_BASE}/patent/applications/search"
    headers = {"X-Api-Key": api_key, "Content-Type": "application/json"}

    for attempt in range(retries):
        time.sleep(REQUEST_DELAY_SEC)
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 429 and attempt < retries - 1:
            print(f"  [429] Rate limit — waiting {RETRY_DELAY_SEC}s", file=sys.stderr)
            time.sleep(RETRY_DELAY_SEC)
            continue
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        return resp.json()

    raise RuntimeError(f"Failed after {retries} attempts on {url} ({api_prefix})")


def run_bulk_uri_fetch(
    corpus_id: str,
    dry_run: bool = False,
    date_from: str = BULK_URI_DATE_FROM,
) -> int:
    """
    Fetch fileLocationURIs for all granted patents in a corpus since date_from.

    Calls POST /api/v1/patent/applications/search/download (Tier 1 metadata
    quota — does NOT consume the document XML download budget).

    Output TSV at BULK_URI_CACHE_PATH, columns:
      external_id  corpus_id  title  grant_date  cpc_codes  file_uri

    external_id = patent grant number (matches data_documents.external_id).
    cpc_codes   = pipe-delimited normalised codes (e.g. H01L21/486|H01L21/768).

    Resume-safe: rows already in the TSV are skipped (keyed by external_id).
    --dry-run: prints per-corpus counts via the search count endpoint; no writes.
    """
    if not API_KEY:
        print("ERROR: USPTO_API_KEY env var not set", file=sys.stderr)
        return 0

    corpus_ids = (
        list(CORPUS_CPC_MAP.keys()) if corpus_id == "all" else [corpus_id]
    )
    date_to = date.today().isoformat()

    print(f"\n{'='*60}")
    print(f"Bulk URI fetch  {date_from} → {date_to}")
    print(f"{'='*60}")

    # ── dry-run: count only, no file writes ──────────────────────────────────
    if dry_run:
        total = 0
        for cid in corpus_ids:
            cpc = CORPUS_CPC_MAP.get(cid)
            if not cpc:
                print(f"  WARNING: no CPC mapping for {cid}", file=sys.stderr)
                continue
            n = count_new_patents(cpc, date_from, date_to)
            print(f"  {cid:<42} {n:>8,}")
            total += n
        print(f"{'='*60}")
        print(f"  {'TOTAL':<42} {total:>8,}")
        return total

    # ── live fetch ────────────────────────────────────────────────────────────
    cache_path = BULK_URI_CACHE_PATH
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids     = _load_uri_cache_ids(cache_path)
    write_header = not cache_path.exists() or cache_path.stat().st_size == 0
    print(f"  Cache: {len(seen_ids):,} existing external_ids ({cache_path})")

    total_written = 0

    with open(cache_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        if write_header:
            writer.writerow(["external_id", "corpus_id", "title",
                             "grant_date", "cpc_codes", "file_uri"])

        for cid in corpus_ids:
            cpc_prefix = CORPUS_CPC_MAP.get(cid)
            if not cpc_prefix:
                print(f"  WARNING: no CPC mapping for {cid}", file=sys.stderr)
                continue

            prefixes       = [p.strip() for p in cpc_prefix.split(',')]
            corpus_written = 0
            corpus_skipped = 0

            for prefix in prefixes:
                api_prefix = _api_cpc_prefix(prefix)
                print(f"\n  [{cid}] {api_prefix}*  ({date_from} → {date_to})",
                      flush=True)
                offset = 0

                while True:
                    data = _post_search_download_page(
                        api_prefix, date_from, date_to, offset, API_KEY
                    )
                    if not data:
                        break

                    bag = data.get("patentFileWrapperDataBag", [])
                    if not bag:
                        break

                    for item in bag:
                        meta      = item.get("applicationMetaData", {})
                        grant_num = str(meta.get("patentNumber") or "").strip()
                        if not grant_num:
                            # Fallback: application number for records that slipped
                            # through the Patented Case filter without a grant number
                            grant_num = str(
                                item.get("applicationNumberText", "")
                            ).strip()
                        if not grant_num:
                            continue

                        if grant_num in seen_ids:
                            corpus_skipped += 1
                            continue

                        title      = meta.get("inventionTitle") or ""
                        grant_date = meta.get("grantDate") or ""
                        cpc_codes  = "|".join(
                            _normalize_cpc(c)
                            for c in meta.get("cpcClassificationBag", [])
                        )
                        file_uri   = (
                            item.get("grantDocumentMetaData", {})
                                .get("fileLocationURI") or ""
                        )

                        writer.writerow([grant_num, cid, title, grant_date,
                                         cpc_codes, file_uri])
                        seen_ids.add(grant_num)
                        corpus_written += 1

                    total_count = data.get("count", 0)
                    offset     += len(bag)

                    if offset % (API_PAGE_SIZE * 20) == 0:
                        print(f"    …{offset:,}/{total_count:,}", flush=True)

                    if offset >= total_count:
                        break

                print(f"    → {corpus_written:,} written, "
                      f"{corpus_skipped:,} skipped (already cached)")
                fh.flush()

            total_written += corpus_written

    print(f"\n  Done: {total_written:,} new rows → {cache_path}")
    return total_written


# ── Setup ─────────────────────────────────────────────────────────────────────

def create_cache_table_and_function(conn) -> None:
    """
    Create patent_full_text_cache table and get_full_text() SQL function in arc_v5.
    Safe to run multiple times (CREATE TABLE IF NOT EXISTS, CREATE OR REPLACE).
    """
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS patent_full_text_cache (
              external_id   text        PRIMARY KEY,
              abstract      text,
              claims        text,
              description   text,
              fetched_at    timestamptz DEFAULT now()
            );

            COMMENT ON TABLE patent_full_text_cache IS
            'Cached full patent text (abstract, claims, description) fetched from
USPTO Open Data Portal API. IMPORTANT: each patent XML file has a 20-download/year
limit per API key. Always read from this cache; never call the API directly for
already-cached patents. See: https://data.uspto.gov/apis/rate-limits';
        """)

        # SQL function callable from cartographer sessions
        cur.execute("""
            CREATE OR REPLACE FUNCTION get_full_text(p_external_id text)
            RETURNS jsonb
            LANGUAGE plpython3u
            AS $func$
                # imports needed for both cache and API paths
                import json, time

                # ── Cache lookup ──────────────────────────────────────────────
                # plpy.execute(query, args) requires a prepared plan — use plpy.prepare()
                _sel = plpy.prepare(
                    "SELECT abstract, claims, description "
                    "FROM patent_full_text_cache "
                    "WHERE external_id = $1",
                    ["text"]
                )
                rv = plpy.execute(_sel, [p_external_id])
                if len(rv) > 0:
                    # PostgreSQL 14 plpython3u needs json.dumps() for RETURNS jsonb
                    return json.dumps({
                        "abstract":    rv[0]["abstract"],
                        "claims":      rv[0]["claims"],
                        "description": rv[0]["description"],
                        "source":      "cache",
                    })

                # ── API fetch ─────────────────────────────────────────────────
                # IMPORTANT: 20 downloads/year limit per unique patent XML file
                import requests, xml.etree.ElementTree as ET

                # Read from PostgreSQL GUC — os.environ is the postgres daemon's env,
                # not the calling shell's env. Set per-session with:
                #   SET app.uspto_api_key = '...';
                try:
                    api_key = plpy.execute(
                        "SELECT current_setting('app.uspto_api_key', true) AS k"
                    )[0]["k"] or ""
                except Exception:
                    api_key = ""
                if not api_key:
                    return json.dumps({"error": "app.uspto_api_key GUC not set — run: SET app.uspto_api_key = '...';", "source": "config_error"})

                base = "https://api.uspto.gov/api/v1"
                hdrs = {"X-Api-Key": api_key}

                # Step 1: metadata → fileLocationURI
                try:
                    time.sleep(0.25)
                    r = requests.get(
                        f"{base}/patent/applications/{p_external_id}",
                        headers=hdrs, timeout=30,
                    )
                    r.raise_for_status()
                    bag = r.json().get("patentFileWrapperDataBag", [])
                    file_uri = (bag[0].get("grantDocumentMetaData", {}).get("fileLocationURI")
                                if bag else None)
                except Exception as e:
                    return json.dumps({"error": str(e), "source": "metadata_fetch_error"})

                if not file_uri:
                    return json.dumps({"error": "no fileLocationURI in response", "source": "api_missing"})

                # Step 2: download XML (fileLocationURI is an absolute URL)
                xml_url = file_uri if file_uri.startswith("http") else f"{base}{file_uri}"
                try:
                    time.sleep(0.25)
                    r2 = requests.get(xml_url, headers=hdrs, timeout=60)
                    r2.raise_for_status()
                    xml_bytes = r2.content
                except Exception as e:
                    return json.dumps({"error": str(e), "source": "xml_fetch_error"})

                # Step 3: parse
                root = ET.fromstring(xml_bytes)

                def get_text(tag):
                    el = root.find(f".//{tag}")
                    return " ".join(el.itertext()).strip() if el is not None else ""

                claims = [
                    " ".join(c.itertext()).strip()
                    for c in root.findall(".//claim")
                    if " ".join(c.itertext()).strip()
                ]

                result = {
                    "abstract":    get_text("abstract"),
                    "claims":      chr(10).join(claims),
                    "description": get_text("description"),
                    "source":      "api_fresh",
                }

                # Step 4: cache for future calls
                _ins = plpy.prepare(
                    "INSERT INTO patent_full_text_cache "
                    "(external_id, abstract, claims, description) "
                    "VALUES ($1, $2, $3, $4) "
                    "ON CONFLICT (external_id) DO NOTHING",
                    ["text", "text", "text", "text"]
                )
                plpy.execute(_ins, [p_external_id, result["abstract"],
                                    result["claims"], result["description"]])
                return json.dumps(result)
            $func$;

            COMMENT ON FUNCTION get_full_text(text) IS
            'Fetch full patent text (abstract + claims + description) for a given
external_id (USPTO application number). Checks patent_full_text_cache first;
calls USPTO API only on cache miss.

IMPORTANT: USPTO API has a 20 downloads/year limit per unique patent XML file
per API key. patent_full_text_cache prevents redundant fetches.
Requires USPTO_API_KEY environment variable in the PostgreSQL process environment.

Usage:
  SELECT get_full_text(''16123456'') ->> ''claims'' AS claims;
  SELECT get_full_text(d.external_id) FROM data_documents d WHERE ...;

Returns JSONB: {abstract, claims, description, source}
  source = ''cache'' | ''api_fresh'' | ''api_missing'' | ''*_error''';
        """)

    conn.commit()
    print("Created patent_full_text_cache table and get_full_text() function.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="USPTO API client for ARC arc_v5")
    ap.add_argument("--mode",
                    choices=["setup", "fulltext", "incremental", "count-new",
                             "backfill-abstracts", "bulk-uri-fetch"],
                    required=True,
                    help="setup: create cache table + SQL function | "
                         "fulltext: fetch claims for one patent | "
                         "count-new: show new patent counts per corpus | "
                         "incremental: fetch and insert new patents since last ingest | "
                         "backfill-abstracts: fill NULL abstracts via full-text XML download | "
                         "bulk-uri-fetch: fetch fileLocationURIs → ptfw_uri_cache.tsv")
    ap.add_argument("--patent-id",
                    help="Patent application number for --mode fulltext")
    ap.add_argument("--corpus",
                    help='Corpus ID, or "all" for all corpora in CORPUS_CPC_MAP')
    ap.add_argument("--since",
                    help="Override start date (YYYY-MM-DD). Default: auto from DB")
    ap.add_argument("--dry-run", action="store_true",
                    help="count-new / incremental / backfill-abstracts: report only, no writes")
    ap.add_argument("--limit", type=int,
                    help="backfill-abstracts: max number of patents to process")
    ap.add_argument("--db", default="arc_v5",
                    help="Target database (default: arc_v5)")
    return ap.parse_args()


def main():
    args = parse_args()

    conn = psycopg2.connect(
        host=os.environ.get("PGHOST", "/var/run/postgresql"),
        dbname=args.db,
        user=os.environ.get("PGUSER", "jeff"),
    )

    if args.mode == "setup":
        create_cache_table_and_function(conn)

    elif args.mode == "fulltext":
        if not args.patent_id:
            print("ERROR: --patent-id required for --mode fulltext", file=sys.stderr)
            sys.exit(1)
        if not API_KEY:
            print("ERROR: USPTO_API_KEY env var not set", file=sys.stderr)
            sys.exit(1)
        result = get_or_fetch_full_text(conn, args.patent_id, API_KEY)
        print(json.dumps(result, indent=2))

    elif args.mode == "count-new":
        if not API_KEY:
            print("ERROR: USPTO_API_KEY env var not set", file=sys.stderr)
            sys.exit(1)
        corpus_ids = (list(CORPUS_CPC_MAP.keys())
                      if args.corpus == 'all'
                      else [args.corpus])
        today = date.today().isoformat()
        print(f"\n{'='*60}")
        print(f"New patents since last ingest  (as of {today})")
        print(f"{'='*60}")
        total = 0
        for cid in corpus_ids:
            cpc = CORPUS_CPC_MAP.get(cid)
            if not cpc:
                print(f"  WARNING: no CPC mapping for {cid}", file=sys.stderr)
                continue
            since = args.since or get_last_ingest_date(conn, cid)
            n = count_new_patents(cpc, since, today)
            print(f"  {cid:<42} since {since}:  {n:>8,}")
            total += n
        print(f"{'='*60}")
        print(f"  {'TOTAL':<42}          {total:>8,}")

    elif args.mode == "incremental":
        if not args.corpus:
            print("ERROR: --corpus required for --mode incremental", file=sys.stderr)
            sys.exit(1)
        corpus_ids = (list(CORPUS_CPC_MAP.keys())
                      if args.corpus == 'all'
                      else [args.corpus])
        t0 = time.time()
        for cid in corpus_ids:
            run_incremental_update(conn, cid,
                                   dry_run=args.dry_run,
                                   since=args.since)
        elapsed = time.time() - t0
        print(f"\nTotal elapsed: {elapsed/60:.1f} min")

    elif args.mode == "backfill-abstracts":
        if not API_KEY:
            print("ERROR: USPTO_API_KEY env var not set", file=sys.stderr)
            sys.exit(1)
        run_backfill_abstracts(
            conn,
            api_key=API_KEY,
            corpus_id=args.corpus if args.corpus and args.corpus != "all" else None,
            dry_run=args.dry_run,
            limit=args.limit,
        )

    elif args.mode == "bulk-uri-fetch":
        if not args.corpus:
            print("ERROR: --corpus required for --mode bulk-uri-fetch", file=sys.stderr)
            sys.exit(1)
        run_bulk_uri_fetch(
            corpus_id=args.corpus,
            dry_run=args.dry_run,
            date_from=args.since or BULK_URI_DATE_FROM,
        )

    conn.close()


if __name__ == "__main__":
    main()

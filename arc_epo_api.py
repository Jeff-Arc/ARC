#!/usr/bin/env python3
"""
arc_epo_api.py — EPO Open Patent Services (OPS) API client for ARC.

PURPOSE:
  Fetch worldwide patent bibliographic data (including abstracts) from the
  EPO OPS REST API and insert into data_documents.

EPO OPS API: https://ops.epo.org/3.2/
Auth:         OAuth2 client_credentials — token expires in ~20 min (1199s)
Credentials:  EPO_CONSUMER_ID, EPO_CONSUMER_SECRET from .env

RATE LIMITS (EPO OPS):
  Search:        100 results per page maximum.
  Result cap:    10,000 total per query (EPO hard cap).
                 Adaptive windowing: quarterly → monthly → weekly auto-fallback.
  Throttle:      Server broadcasts its load state in x-throttling-control:
                   idle:       0.5s between requests (~2 req/sec)
                   busy:       2.0s
                   overloaded: 10.0s  (search quota drops from 30 → 5)
  Quota headers: x-individualquotaperhour-used (unit-based, not request-based)
                 x-registeredquotaperweek-used

DATA NOTES:
  - Abstract IS included in /search/biblio XML for EP, US, CN, KR patents.
    No separate fetch is needed for the majority of records.
  - Non-major-office patents (TW, JP, some others) may lack an English abstract
    in the search response; --mode backfill-abstracts fetches them per-doc.
  - CPC codes reconstructed from structured patent-classification XML elements
    (section/class/subclass/main-group/subgroup → e.g. "H01L21/683").
  - IPC classification text used as fallback when CPC elements are absent.
  - Raw XML per document stored in patent_raw_api table for audit/reprocessing.

CQL index keys (EPO OPS v3.2):
  cl=    IPC + CPC combined (use this — 'cpc=' is NOT indexed)
  ic=    IPC only
  pd=    publication date (YYYYMMDD)  pd>=20250101  pd within "20250101 20251231"
  pn=    country/office filter (pn=EP)

Usage:
  python3 arc_epo_api.py --mode count-new --corpus H01L_quarterly
  python3 arc_epo_api.py --mode count-new --corpus all --since 2024-01-01
  python3 arc_epo_api.py --mode incremental --corpus H01L_quarterly --dry-run
  python3 arc_epo_api.py --mode incremental --corpus H01L_quarterly
  python3 arc_epo_api.py --mode backfill-abstracts --corpus H01L_quarterly --limit 500
"""

import argparse
import calendar
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from typing import Iterator

import psycopg2
import requests

# ── Config ─────────────────────────────────────────────────────────────────────

EPO_AUTH_URL   = "https://ops.epo.org/3.2/auth/accesstoken"
EPO_SEARCH_URL = "https://ops.epo.org/3.2/rest-services/published-data/search/biblio"
EPO_ABSTRACT_BASE = "https://ops.epo.org/3.2/rest-services/published-data/publication/epodoc"

EPO_NS = "http://www.epo.org/exchange"   # exchange-document namespace
OPS_NS = "http://ops.epo.org"            # ops:biblio-search namespace

CONSUMER_ID     = os.environ.get("EPO_CONSUMER_ID",     "")
CONSUMER_SECRET = os.environ.get("EPO_CONSUMER_SECRET", "")

PAGE_SIZE = 100   # EPO OPS maximum results per page

# Delay (seconds) between requests by throttle state
THROTTLE_DELAYS = {
    "idle":       0.5,
    "busy":       2.0,
    "overloaded": 10.0,
}
DEFAULT_DELAY = 1.0


# ── CPC prefix mapping ─────────────────────────────────────────────────────────
# Same corpora as arc_patent_api.py — EPO 'cl=' searches both IPC and CPC codes.

CORPUS_CPC_MAP = {
    "H01L_quarterly":              "H01L",
    "G06N_quarterly":              "G06N",
    "G06F_quarterly":              "G06F",
    "G01N_quarterly":              "G01N",
    "G01B_quarterly":              "G01B",
    "G02B_quarterly":              "G02B",
    "C23C_quarterly":              "C23C",
    "C30B_quarterly":              "C30B",
    "A61P9_quarterly":             "A61P9",
    "A61P25_quarterly":            "A61P25",
    "C12N15_quarterly":            "C12N15,A61K48,C12N9",
    "A61K38_quarterly":            "A61K38,A61K35,C12N5",
    "longevity_patents_quarterly": "A61K38,A61K35,C12N5,C12N15,A61K48,C12N9,A61P9",
}


# ── OAuth2 token ───────────────────────────────────────────────────────────────

class EpoToken:
    """
    EPO OPS OAuth2 client_credentials token with auto-refresh.

    Tokens expire in 1199 seconds (~20 min).  This class refreshes
    automatically when fewer than 60 seconds remain, so long-running
    incremental runs that span multiple token lifetimes work without
    any caller changes.
    """

    def __init__(self, consumer_id: str, consumer_secret: str):
        self._id       = consumer_id
        self._secret   = consumer_secret
        self._token    = ""
        self._expires  = 0.0   # absolute epoch time of expiry

    def get(self) -> str:
        """Return a valid Bearer token string, refreshing if < 60s remain."""
        if time.time() >= self._expires - 60:
            self._refresh()
        return self._token

    def _refresh(self):
        resp = requests.post(
            EPO_AUTH_URL,
            data={"grant_type": "client_credentials"},
            auth=(self._id, self._secret),
            timeout=15,
        )
        resp.raise_for_status()
        data         = resp.json()
        self._token  = data["access_token"]
        expires_in   = int(data.get("expires_in", 1199))
        self._expires = time.time() + expires_in
        print(f"  [EPO auth] Token refreshed (expires in {expires_in}s)", file=sys.stderr)


# ── Throttle tracker ───────────────────────────────────────────────────────────

class EpoThrottle:
    """
    Parses x-throttling-control headers and enforces appropriate delays.

    EPO broadcasts its current load as the first word in the header:
      "idle (images=green:200, ..., search=green:30)"
      "busy (images=green:100, ..., search=green:15)"
      "overloaded (images=green:50, ..., search=green:5)"

    The search=green:N value is remaining capacity in the current window.
    We slow down extra if that number drops very low (≤ 3).
    """

    def __init__(self):
        self.state             = "idle"
        self.search_remaining  = 30
        self._last_request_at  = 0.0

    def update(self, header: str):
        """Parse throttle state from x-throttling-control response header."""
        if not header:
            return
        m = re.match(r"^(\w+)", header.strip())
        if m:
            self.state = m.group(1).lower()
        m = re.search(r"search=\w+:(\d+)", header)
        if m:
            self.search_remaining = int(m.group(1))

    def wait(self):
        """Sleep the appropriate duration before the next API call."""
        delay = THROTTLE_DELAYS.get(self.state, DEFAULT_DELAY)
        if self.search_remaining <= 3:
            delay = max(delay, 5.0)   # emergency brake — nearly out of search quota
        elapsed = time.time() - self._last_request_at
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_at = time.time()


# ── Date windowing ─────────────────────────────────────────────────────────────

def weekly_windows(
    date_from: date,
    date_to:   date,
) -> Iterator[tuple[date, date]]:
    """Yield (start, end) 7-day windows spanning date_from..date_to."""
    current = date_from
    while current <= date_to:
        week_end = min(current + timedelta(days=6), date_to)
        yield current, week_end
        current = week_end + timedelta(days=1)


def monthly_windows(
    date_from: date,
    date_to:   date,
) -> Iterator[tuple[date, date]]:
    """Yield (start, end) monthly windows spanning date_from..date_to."""
    current = date(date_from.year, date_from.month, 1)
    while current <= date_to:
        m_end = date(current.year, current.month,
                     calendar.monthrange(current.year, current.month)[1])
        yield max(current, date_from), min(m_end, date_to)
        next_month = current.month + 1
        if next_month > 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, next_month, 1)


def quarterly_windows(
    date_from: date,
    date_to:   date,
) -> Iterator[tuple[date, date]]:
    """Yield (start, end) quarterly date windows spanning date_from..date_to."""
    q_start_month = ((date_from.month - 1) // 3) * 3 + 1
    current = date(date_from.year, q_start_month, 1)

    while current <= date_to:
        q_end_month = current.month + 2
        q_end_day   = calendar.monthrange(current.year, q_end_month)[1]
        q_end = date(current.year, q_end_month, q_end_day)

        yield max(current, date_from), min(q_end, date_to)

        next_month = q_end_month + 1
        if next_month > 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, next_month, 1)


# ── XML helpers ────────────────────────────────────────────────────────────────

def _tag(name: str) -> str:
    """Return Clark-notation tag for the EPO exchange namespace."""
    return f"{{{EPO_NS}}}{name}"


def _find_text(el, tag: str) -> str:
    """Find first descendant with exchange-namespace tag; return stripped text."""
    found = el.find(f".//{_tag(tag)}")
    return found.text.strip() if found is not None and found.text else ""


def _extract_title(doc_el) -> str:
    """
    Extract invention title preferring English (lang='en').
    Falls back to first available language if no English title exists.
    """
    titles: dict[str, str] = {}
    for t in doc_el.findall(f".//{_tag('invention-title')}"):
        lang = t.get("lang", "xx")
        text = (t.text or "").strip()
        if text:
            titles[lang] = text
    return titles.get("en") or next(iter(titles.values()), "")


def _extract_abstract(doc_el) -> str:
    """
    Extract English abstract from exchange-document.
    EPO biblio includes abstracts for most major offices (EP, US, CN, KR).
    lang='en' is preferred; lang='ol' (original language) used as fallback.
    """
    abstracts: dict[str, str] = {}
    for ab in doc_el.findall(f".//{_tag('abstract')}"):
        lang  = ab.get("lang", "xx")
        parts = [" ".join(p.itertext()).strip()
                 for p in ab.findall(_tag("p"))]
        text  = " ".join(p for p in parts if p)
        if text:
            abstracts[lang] = text
    return abstracts.get("en") or next(iter(abstracts.values()), "")


def _extract_cpc_codes(doc_el) -> list[str]:
    """
    Extract CPC codes from structured patent-classification elements.

    Each patent-classification has section/class/subclass/main-group/subgroup
    children from which we reconstruct the standard form (e.g. 'H01L21/683').
    Falls back to IPC classification-ipcr text (e.g. 'H01L  21/   683' →
    'H01L21/683') when no structured CPC elements are present.
    """
    codes: list[str] = []

    for pc in doc_el.findall(f".//{_tag('patent-classification')}"):
        section    = _find_text(pc, "section")
        cls        = _find_text(pc, "class")
        subclass   = _find_text(pc, "subclass")
        main_group = _find_text(pc, "main-group")
        subgroup   = _find_text(pc, "subgroup")
        if section and cls and subclass:
            code = f"{section}{cls}{subclass}{main_group}/{subgroup}".rstrip("/")
            if code not in codes:
                codes.append(code)

    if not codes:
        # Fallback: IPC text with internal whitespace stripped
        for ipc in doc_el.findall(f".//{_tag('classification-ipcr')}"):
            text_el = ipc.find(_tag("text"))
            if text_el is not None and text_el.text:
                code = "".join(text_el.text.split())
                # Normalise: 'H01L21/683A' → keep only alphanumeric + '/'
                code = re.sub(r"[^A-Z0-9/]", "", code)[:15]
                if code and code not in codes:
                    codes.append(code)

    return codes


def _extract_applicant(doc_el) -> str | None:
    """
    Return the primary applicant name (sequence=1, data-format=epodoc).
    Strips the country suffix '[KR]' added by epodoc formatting.
    Returns None if no applicant found.
    """
    for appl in doc_el.findall(f".//{_tag('applicant')}"):
        if appl.get("data-format") != "epodoc":
            continue
        if appl.get("sequence") != "1":
            continue
        name_el = appl.find(f".//{_tag('name')}")
        if name_el is not None and name_el.text:
            return re.sub(r"\s*\[[A-Z]{2}\]\s*$", "", name_el.text.strip())
    return None


def _extract_pub_date(doc_el) -> str:
    """
    Extract publication date (YYYYMMDD string) from the docdb publication-reference.
    Returns empty string if not found.
    """
    for pub_ref in doc_el.findall(f".//{_tag('publication-reference')}"):
        for doc_id in pub_ref.findall(_tag("document-id")):
            if doc_id.get("document-id-type") == "docdb":
                d_el = doc_id.find(_tag("date"))
                if d_el is not None and d_el.text:
                    return d_el.text.strip()
    return ""


def _parse_biblio_xml(xml_text: str) -> tuple[int, list[dict]]:
    """
    Parse the EPO /search/biblio XML response body.

    Returns (total_result_count, [doc_dict, ...]).

    Each doc_dict contains:
      external_id  str         epodoc form: country + doc_number (e.g. 'EP4672314')
      country      str
      doc_number   str
      kind         str         grant/application kind code
      family_id    str
      pub_date     str         YYYYMMDD
      title        str
      abstract     str         English preferred; empty if absent in search result
      cpc_codes    list[str]
      applicant    str | None  primary applicant (epodoc, country suffix stripped)
      raw_xml      str         serialised exchange-document element
    """
    root = ET.fromstring(xml_text)

    bs_el = root.find(f"{{{OPS_NS}}}biblio-search")
    total = int(bs_el.get("total-result-count", 0)) if bs_el is not None else 0

    docs: list[dict] = []

    for doc_el in root.findall(f".//{_tag('exchange-document')}"):
        country    = doc_el.get("country", "")
        doc_number = doc_el.get("doc-number", "")
        if not country or not doc_number:
            continue

        docs.append({
            "external_id": f"{country}{doc_number}",
            "country":     country,
            "doc_number":  doc_number,
            "kind":        doc_el.get("kind", ""),
            "family_id":   doc_el.get("family-id", ""),
            "pub_date":    _extract_pub_date(doc_el),
            "title":       _extract_title(doc_el),
            "abstract":    _extract_abstract(doc_el),
            "cpc_codes":   _extract_cpc_codes(doc_el),
            "applicant":   _extract_applicant(doc_el),
            "raw_xml":     ET.tostring(doc_el, encoding="unicode"),
        })

    return total, docs


# ── EPO API calls ──────────────────────────────────────────────────────────────

def _search_page(
    token:       EpoToken,
    throttle:    EpoThrottle,
    cql:         str,
    range_start: int,
    range_end:   int,
    retries:     int = 5,
) -> tuple[int, list[dict]]:
    """
    Fetch one page of /search/biblio results.

    Returns (total_result_count, [doc_dict, ...]).
    Returns (0, []) on 404 (no results) or 400 (range > 10K or bad query).

    403 handling — EPO OPS enforces two distinct quota limits:
      1. Short throttle window (search=green:N in x-throttling-control): recovers
         within ~60s when N hits 0.
      2. Hourly unit quota (x-individualquotaperhour-used): each 100-result page
         costs ~2.5M units; the hourly budget is finite.  When exhausted, 403 persists
         for up to 60 minutes.
    We use exponential backoff (120s → 300s → 600s → 900s → 1200s) to handle both.
    """
    # Exponential backoff waits for 403: 120, 300, 600, 900, 1200 seconds
    _403_waits = [120, 300, 600, 900, 1200]

    for attempt in range(retries):
        throttle.wait()
        resp = requests.get(
            EPO_SEARCH_URL,
            params={"q": cql, "Range": f"{range_start}-{range_end}"},
            headers={"Authorization": f"Bearer {token.get()}"},
            timeout=30,
        )
        throttle.update(resp.headers.get("x-throttling-control", ""))

        if resp.status_code in (404, 400):
            return 0, []

        if resp.status_code == 403 and attempt < retries - 1:
            wait = _403_waits[min(attempt, len(_403_waits) - 1)]
            print(f"  [403] EPO quota exhausted — waiting {wait}s "
                  f"(attempt {attempt+1}/{retries})", file=sys.stderr)
            time.sleep(wait)
            continue

        if resp.status_code == 429 and attempt < retries - 1:
            wait = 10 * (2 ** attempt)
            print(f"  [429] EPO throttle — waiting {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue

        if resp.status_code >= 500 and attempt < retries - 1:
            time.sleep(5)
            continue

        resp.raise_for_status()
        return _parse_biblio_xml(resp.text)

    print(f"  [SKIP] Quota exhausted after {retries} retries for "
          f"Range={range_start}-{range_end} — skipping this page.", file=sys.stderr)
    return 0, []


def _count_window_adaptive(
    token:    EpoToken,
    throttle: EpoThrottle,
    prefix:   str,
    start:    date,
    end:      date,
    depth:    int = 0,
) -> int:
    """
    Count patents for one (prefix, date range), recursively subdividing
    if the result count hits the 10,000 cap.

    depth 0 → try the given window; cap → subdivide into months
    depth 1 → try a monthly window;  cap → subdivide into weeks
    depth 2 → try a weekly window;   cap → warn and return 10,000
    """
    from_str = start.strftime("%Y%m%d")
    to_str   = end.strftime("%Y%m%d")
    cql      = f'cl={prefix} AND pd within "{from_str} {to_str}"'
    count, _ = _search_page(token, throttle, cql, 1, 1)

    if count < 10000 or depth >= 2:
        if count >= 10000 and depth >= 2:
            print(f"  WARNING: {start}–{end} [{prefix}] still at 10K cap "
                  f"at weekly level — some records will be missed.", file=sys.stderr)
        return count

    subdivider = monthly_windows if depth == 0 else weekly_windows
    total = 0
    for sub_start, sub_end in subdivider(start, end):
        total += _count_window_adaptive(token, throttle, prefix, sub_start, sub_end, depth + 1)
    return total


def _fetch_window_adaptive(
    token:    EpoToken,
    throttle: EpoThrottle,
    prefix:   str,
    start:    date,
    end:      date,
    depth:    int = 0,
    label:    str = "",
) -> dict[str, dict]:
    """
    Fetch all docs for one (prefix, date range), recursively subdividing
    if the result count hits the 10,000 cap:
      depth 0 → quarterly window; cap → subdivide into months
      depth 1 → monthly window;   cap → subdivide into weeks
      depth 2 → weekly window;    cap → warn and paginate up to 10K

    Returns {external_id: doc_dict} (deduped).
    """
    from_str = start.strftime("%Y%m%d")
    to_str   = end.strftime("%Y%m%d")
    cql      = f'cl={prefix} AND pd within "{from_str} {to_str}"'

    total, first_page = _search_page(token, throttle, cql, 1, PAGE_SIZE)

    if total == 0:
        return {}

    window_label = label or f"{start}–{end} [{prefix}]"

    if total >= 10000 and depth < 2:
        # Subdivide instead of paginating past the cap
        subdivider = monthly_windows if depth == 0 else weekly_windows
        sub_name   = "monthly" if depth == 0 else "weekly"
        print(f"  {window_label}: {total:,} (cap hit) → subdividing to {sub_name}",
              flush=True)
        all_docs: dict[str, dict] = {}
        for sub_start, sub_end in subdivider(start, end):
            sub_label = f"  {sub_start}–{sub_end} [{prefix}]"
            docs = _fetch_window_adaptive(
                token, throttle, prefix, sub_start, sub_end, depth + 1, sub_label
            )
            all_docs.update(docs)
        return all_docs

    if total >= 10000:
        print(f"  WARNING: {window_label} still at 10K cap at weekly level "
              f"— some records will be missed.", file=sys.stderr)
    else:
        indent = "    " * depth
        print(f"  {indent}{window_label}: {total:,}", flush=True)

    # Collect all pages
    docs: dict[str, dict] = {d["external_id"]: d for d in first_page}
    range_start = PAGE_SIZE + 1
    while range_start <= min(total, 10000):
        range_end = min(range_start + PAGE_SIZE - 1, 10000)
        _, page   = _search_page(token, throttle, cql, range_start, range_end)
        for d in page:
            docs[d["external_id"]] = d
        range_start = range_end + 1

        fetched = min(range_end, min(total, 10000))
        if fetched % 1000 < PAGE_SIZE:
            indent = "    " * depth
            print(f"    {indent}... {fetched}/{min(total, 10000)}", flush=True)

    return docs


def _fetch_abstract_single(
    token:    EpoToken,
    throttle: EpoThrottle,
    epodoc:   str,
) -> str:
    """
    Fetch abstract for one document via the per-doc endpoint.
    Used by backfill-abstracts for patents whose abstract was absent
    in the original search response (common for TW, JP, etc.).

    URL: /published-data/publication/epodoc/{epodoc}/abstract
    Omitting the kind suffix (e.g. 'A2') lets EPO resolve the latest publication.
    """
    url = f"{EPO_ABSTRACT_BASE}/{epodoc}/abstract"
    throttle.wait()
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {token.get()}"},
        timeout=15,
    )
    throttle.update(resp.headers.get("x-throttling-control", ""))
    if resp.status_code != 200:
        return ""

    try:
        root = ET.fromstring(resp.text)
        abstracts: dict[str, str] = {}
        for ab in root.findall(f".//{_tag('abstract')}"):
            lang  = ab.get("lang", "xx")
            parts = [" ".join(p.itertext()).strip()
                     for p in ab.findall(_tag("p"))]
            text  = " ".join(p for p in parts if p).strip()
            if text:
                abstracts[lang] = text
        return abstracts.get("en") or next(iter(abstracts.values()), "")
    except ET.ParseError:
        return ""


# ── DB helpers ─────────────────────────────────────────────────────────────────

def ensure_raw_table(conn) -> None:
    """
    Create patent_raw_api if it doesn't already exist.

    Stores one raw XML blob per (external_id, source_api) pair so we can
    re-parse field extraction without re-fetching from the API.  The
    ON CONFLICT DO NOTHING in inserts means the first fetch is preserved.
    """
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS patent_raw_api (
                external_id  text        NOT NULL,
                source_api   text        NOT NULL DEFAULT 'epo_ops',
                fetched_at   timestamptz NOT NULL DEFAULT now(),
                raw_xml      text,
                PRIMARY KEY (external_id, source_api)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_patent_raw_api_source_fetched
            ON patent_raw_api (source_api, fetched_at)
        """)
    conn.commit()


def get_last_epo_ingest_date(conn, corpus_id: str) -> str:
    """
    Return the most recent publication_date for EPO documents in this corpus.
    Falls back to '2020-01-01' for a corpus with no EPO records yet.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT MAX(publication_date)::text
            FROM data_documents
            WHERE corpus_id   = %s
              AND source_api  = 'epo_ops'
              AND publication_date IS NOT NULL
        """, (corpus_id,))
        row = cur.fetchone()
        return row[0] if row and row[0] else "2020-01-01"


def _pub_date_to_sql(raw: str) -> str | None:
    """Convert 'YYYYMMDD' → 'YYYY-MM-DD' for PostgreSQL. Returns None on failure."""
    if not raw or len(raw) != 8:
        return None
    try:
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    except Exception:
        return None


def _insert_doc(conn, corpus_id: str, doc: dict) -> bool:
    """
    Insert one document into data_documents.
    Returns True if a new row was written; False if it already existed.
    """
    pub_date_sql = _pub_date_to_sql(doc["pub_date"])
    title        = doc["title"]   or None
    abstract     = doc["abstract"] or None

    if not title and not abstract:
        return False

    document_id = f"{corpus_id}_{doc['external_id']}"

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO data_documents (
                document_id, corpus_id, external_id,
                title, abstract,
                assignee,
                publication_date, content_date,
                cpc_codes,
                source_type, source_api, corpus_type
            ) VALUES (
                %s, %s, %s,
                %s, %s,
                %s,
                %s::date, %s::date,
                %s,
                'patents', 'epo_ops', 'patents'
            )
            ON CONFLICT (document_id) DO NOTHING
        """, (
            document_id,
            corpus_id,
            doc["external_id"],
            title,
            abstract,
            doc["applicant"],         # primary applicant as assignee
            pub_date_sql,
            pub_date_sql,             # content_date = publication_date
            doc["cpc_codes"] or None,
        ))
        return cur.rowcount > 0


def _upsert_raw(conn, doc: dict) -> bool:
    """
    Store raw XML in patent_raw_api.  ON CONFLICT DO NOTHING preserves the
    first-fetch timestamp; returns True if a new row was written.
    """
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO patent_raw_api (external_id, source_api, raw_xml)
            VALUES (%s, 'epo_ops', %s)
            ON CONFLICT (external_id, source_api) DO NOTHING
        """, (doc["external_id"], doc["raw_xml"]))
        return cur.rowcount > 0


# ── count-new ──────────────────────────────────────────────────────────────────

def count_new_epo(
    conn,
    corpus_id: str,
    since:     str | None = None,
) -> int:
    """
    Count EPO patents available for this corpus since the last ingest date.

    Uses adaptive windowing: starts at quarterly granularity, and when a
    window hits the 10K cap it recursively subdivides into monthly then weekly
    windows to get an accurate count.  This ensures H01L and other high-volume
    classes are not under-counted.

    Makes only Range=1-1 calls (just reading total-result-count) so it is
    lightweight — no patent documents are fetched.
    """
    cpc_prefix = CORPUS_CPC_MAP.get(corpus_id)
    if not cpc_prefix:
        print(f"  ERROR: No CPC mapping for {corpus_id}", file=sys.stderr)
        return 0

    last_date_str = since or get_last_epo_ingest_date(conn, corpus_id)
    try:
        date_from = datetime.strptime(last_date_str, "%Y-%m-%d").date()
    except ValueError:
        date_from = date(2020, 1, 1)
    date_to = date.today()

    token    = EpoToken(CONSUMER_ID, CONSUMER_SECRET)
    throttle = EpoThrottle()
    prefixes = [p.strip() for p in cpc_prefix.split(",")]

    grand_total = 0

    for q_start, q_end in quarterly_windows(date_from, date_to):
        q_total = 0
        for prefix in prefixes:
            q_total += _count_window_adaptive(token, throttle, prefix, q_start, q_end)
        if q_total > 0:
            print(f"    {q_start} – {q_end}: {q_total:,}")
        grand_total += q_total

    return grand_total


# ── incremental ────────────────────────────────────────────────────────────────

def run_epo_incremental(
    conn,
    corpus_id: str,
    dry_run:   bool = False,
    since:     str | None = None,
) -> int:
    """
    Fetch and insert new EPO patents for a corpus.

    Uses adaptive windowing: starts at quarterly granularity and automatically
    subdivides into monthly then weekly windows when a quarter's result count
    hits the EPO 10K cap.  This ensures complete coverage for high-volume
    classes like H01L (~7-8K patents/month across all offices).

    For each window:
      1. _fetch_window_adaptive() paginates all results, subdividing as needed.
      2. Documents are deduplicated by external_id within the quarter.
      3. Inserted into data_documents (ON CONFLICT DO NOTHING).
      4. Raw XML stored in patent_raw_api (ON CONFLICT DO NOTHING).
      5. Commit every 500 inserts.

    Abstract is pulled directly from the biblio search response for most docs
    (EP, US, CN, KR).  Run --mode backfill-abstracts to fill remaining NULLs.
    """
    cpc_prefix = CORPUS_CPC_MAP.get(corpus_id)
    if not cpc_prefix:
        print(f"  ERROR: No CPC mapping for {corpus_id}", file=sys.stderr)
        return 0

    if not CONSUMER_ID or not CONSUMER_SECRET:
        print("  ERROR: EPO_CONSUMER_ID / EPO_CONSUMER_SECRET not set", file=sys.stderr)
        return 0

    ensure_raw_table(conn)

    last_date_str = since or get_last_epo_ingest_date(conn, corpus_id)
    try:
        date_from = datetime.strptime(last_date_str, "%Y-%m-%d").date()
    except ValueError:
        date_from = date(2020, 1, 1)
    date_to = date.today()

    print(f"\n{'='*60}")
    print(f"Corpus:       {corpus_id}")
    print(f"CPC prefix:   {cpc_prefix}")
    print(f"Date range:   {date_from} → {date_to}")
    print(f"Windowing:    adaptive (quarterly → monthly → weekly on cap)")
    print(f"Mode:         {'dry-run (no writes)' if dry_run else 'insert'}")
    print(f"{'='*60}")

    token    = EpoToken(CONSUMER_ID, CONSUMER_SECRET)
    throttle = EpoThrottle()
    prefixes = [p.strip() for p in cpc_prefix.split(",")]
    quarters = list(quarterly_windows(date_from, date_to))

    inserted = 0
    t0       = time.time()

    for qi, (q_start, q_end) in enumerate(quarters, 1):
        # Collect all docs for this quarter across all prefixes, deduped by external_id
        quarter_docs: dict[str, dict] = {}

        for prefix in prefixes:
            label = f"Q{qi}/{len(quarters)} ({q_start}–{q_end}) [{prefix}]"
            docs  = _fetch_window_adaptive(token, throttle, prefix, q_start, q_end,
                                           depth=0, label=label)
            quarter_docs.update(docs)

        if not quarter_docs:
            continue

        if dry_run:
            sample = list(quarter_docs.values())[:3]
            for d in sample:
                print(f"  SAMPLE {d['external_id']} | {d['pub_date']} | {d['title'][:60]}")
                if d["abstract"]:
                    print(f"         abstract: {d['abstract'][:80]}...")
                print(f"         cpc: {d['cpc_codes'][:3]}")
            print(f"  [dry-run] quarter {qi}: {len(quarter_docs):,} docs "
                  f"(no writes)", flush=True)
            continue

        # ── Insert ─────────────────────────────────────────────────────────
        for doc in quarter_docs.values():
            if _insert_doc(conn, corpus_id, doc):
                inserted += 1
            _upsert_raw(conn, doc)

            if inserted % 500 == 0 and inserted > 0:
                conn.commit()
                elapsed = time.time() - t0
                rate    = inserted / elapsed if elapsed > 0 else 0
                print(f"  Progress: {inserted:,} inserted ({rate:.0f}/sec)", flush=True)

        conn.commit()
        print(f"  Quarter {qi} complete: {len(quarter_docs):,} fetched, "
              f"{inserted:,} total inserted", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Inserted:   {inserted:,} new documents")
    print(f"  Elapsed:    {elapsed:.0f}s")
    print(f"{'='*60}")
    return inserted


# ── backfill-abstracts ─────────────────────────────────────────────────────────

def run_epo_backfill_abstracts(
    conn,
    corpus_id: str | None = None,
    dry_run:   bool = False,
    limit:     int | None = None,
) -> None:
    """
    Fill NULL abstracts for EPO documents via per-doc abstract endpoint.

    Most EP, US, CN records already have abstracts from the original search.
    This mode handles edge cases: TW, JP, and other offices where the biblio
    search response does not include an English abstract.

    Uses /published-data/publication/epodoc/{external_id}/abstract.
    One API call per patent — use --limit to process in batches.
    Resume-safe: WHERE abstract IS NULL skips already-filled rows.
    """
    if not CONSUMER_ID or not CONSUMER_SECRET:
        print("  ERROR: EPO_CONSUMER_ID / EPO_CONSUMER_SECRET not set", file=sys.stderr)
        return

    where_parts = ["abstract IS NULL", "source_api = 'epo_ops'"]
    params: list = []
    if corpus_id:
        where_parts.append("corpus_id = %s")
        params.append(corpus_id)
    where_clause = " AND ".join(where_parts)
    limit_clause = "LIMIT %s" if limit else ""

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT DISTINCT ON (external_id)
                   external_id, corpus_id, publication_date
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
    print(f"EPO backfill-abstracts: {total:,} documents to process")
    if corpus_id:
        print(f"Corpus filter:          {corpus_id}")
    if limit:
        print(f"Limit:                  {limit}")
    if dry_run:
        print("[dry-run] No writes.")
    print(f"{'='*60}\n")

    if not rows:
        print("Nothing to do.")
        return

    token    = EpoToken(CONSUMER_ID, CONSUMER_SECRET)
    throttle = EpoThrottle()

    done     = 0
    skipped  = 0
    errors   = 0
    updated  = 0
    t0       = time.time()
    pending  = 0

    for i, (ext_id, sample_corpus, _) in enumerate(rows, 1):

        if i % 100 == 0:
            elapsed   = time.time() - t0
            rate      = i / elapsed if elapsed > 0 else 0
            remaining = (total - i) / rate if rate > 0 else 0
            print(f"  [{i:>6}/{total}]  done={done}  skip={skipped}  err={errors}"
                  f"  ({rate:.1f}/sec, ~{remaining/60:.1f} min remaining)")

        if dry_run:
            print(f"  [{i}/{total}] Would fetch abstract for {ext_id} ({sample_corpus})")
            done += 1
            continue

        try:
            abstract = _fetch_abstract_single(token, throttle, ext_id)
        except Exception as e:
            print(f"  [ERROR] {ext_id}: {e}", file=sys.stderr)
            errors += 1
            continue

        if not abstract:
            skipped += 1
            continue

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE data_documents SET abstract = %s "
                "WHERE external_id = %s AND abstract IS NULL",
                (abstract, ext_id),
            )
            updated += cur.rowcount

        done    += 1
        pending += 1
        if pending >= 100:
            conn.commit()
            pending = 0

    conn.commit()
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Backfill complete.")
    print(f"  Processed:             {done + skipped + errors:,} / {total:,}")
    print(f"  Abstracts written:     {done:,}  ({updated:,} rows across corpora)")
    print(f"  Skipped (not found):   {skipped:,}")
    print(f"  Errors:                {errors:,}")
    print(f"  Elapsed:               {elapsed/60:.1f} min")
    print(f"{'='*60}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="EPO OPS API client for ARC — worldwide patent ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 arc_epo_api.py --mode count-new --corpus H01L_quarterly
  python3 arc_epo_api.py --mode count-new --corpus all --since 2024-01-01
  python3 arc_epo_api.py --mode incremental --corpus H01L_quarterly --dry-run
  python3 arc_epo_api.py --mode incremental --corpus H01L_quarterly
  python3 arc_epo_api.py --mode backfill-abstracts --corpus H01L_quarterly --limit 500
        """,
    )
    ap.add_argument(
        "--mode",
        choices=["count-new", "incremental", "backfill-abstracts"],
        required=True,
        help=(
            "count-new: report patent counts per corpus (no writes) | "
            "incremental: fetch and insert new patents since last ingest | "
            "backfill-abstracts: fill NULL abstracts via per-doc EPO API call"
        ),
    )
    ap.add_argument(
        "--corpus",
        help='Corpus ID, or "all" for all corpora in CORPUS_CPC_MAP',
    )
    ap.add_argument(
        "--since",
        help="Override start date (YYYY-MM-DD). Default: auto from last ingest date in DB",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts and samples; no DB writes",
    )
    ap.add_argument(
        "--limit",
        type=int,
        help="backfill-abstracts: max patents to process in this run",
    )
    ap.add_argument(
        "--db",
        default="arc_v4",
        help="Target database (default: arc_v4)",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    conn = psycopg2.connect(
        host=os.environ.get("PGHOST", "/var/run/postgresql"),
        dbname=args.db,
        user=os.environ.get("PGUSER", "jeff"),
    )

    if args.mode == "count-new":
        corpus_ids = (
            list(CORPUS_CPC_MAP.keys())
            if args.corpus in (None, "all")
            else [args.corpus]
        )
        today = date.today().isoformat()
        print(f"\n{'='*60}")
        print(f"EPO OPS — new patents since last ingest  (as of {today})")
        print(f"{'='*60}")
        grand_total = 0
        for cid in corpus_ids:
            if cid not in CORPUS_CPC_MAP:
                print(f"  WARNING: no CPC mapping for {cid}", file=sys.stderr)
                continue
            since = args.since or get_last_epo_ingest_date(conn, cid)
            print(f"\n  Corpus: {cid:<40}  since: {since}")
            n = count_new_epo(conn, cid, since=since)
            print(f"  → {n:,} total")
            grand_total += n
        if len(corpus_ids) > 1:
            print(f"\n{'='*60}")
            print(f"  GRAND TOTAL: {grand_total:,}")
            print(f"{'='*60}")

    elif args.mode == "incremental":
        if not args.corpus:
            print("ERROR: --corpus required for --mode incremental", file=sys.stderr)
            sys.exit(1)
        corpus_ids = (
            list(CORPUS_CPC_MAP.keys())
            if args.corpus == "all"
            else [args.corpus]
        )
        t0             = time.time()
        total_inserted = 0
        for cid in corpus_ids:
            n = run_epo_incremental(
                conn, cid, dry_run=args.dry_run, since=args.since,
            )
            total_inserted += n
        elapsed = time.time() - t0
        print(f"\nTotal inserted: {total_inserted:,}  |  Elapsed: {elapsed/60:.1f} min")

    elif args.mode == "backfill-abstracts":
        run_epo_backfill_abstracts(
            conn,
            corpus_id=args.corpus if args.corpus and args.corpus != "all" else None,
            dry_run=args.dry_run,
            limit=args.limit,
        )

    conn.close()


if __name__ == "__main__":
    main()

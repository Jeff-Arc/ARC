#!/usr/bin/env python3
"""
arc_epo_ingest.py — Ingest DOCDB XML into normalized epo_* tables (arc_v5 schema).

Tables (18 data tables + 1 person lookup):
  epo_document, epo_title, epo_abstract, epo_application_ref, epo_publication_ref,
  epo_classification_ipc, epo_classification_ipcr, epo_patent_classification,
  epo_person, epo_document_applicant, epo_document_inventor,
  epo_priority_claim, epo_citation_patent, epo_citation_npl,
  epo_pub_availability, epo_classification_national, epo_designated_state,
  epo_related_document

ZIP structure:
  outer .zip -> Root/DOC/DOCDB-*.zip (per-country inner ZIPs)
             -> DOCDB-*.xml (one XML per inner ZIP)
             -> <exch:exchange-document> elements

Usage:
  python3 arc_epo_ingest.py --zip ~/data/docdb_backfile/docdb_xml_bck_202607_001_A.zip
  python3 arc_epo_ingest.py --dir ~/data/docdb_backfile
  python3 arc_epo_ingest.py --dir ~/data/docdb_backfile --delete-after --error-log /tmp/epo_errors.jsonl
"""

import argparse
import functools
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

# Force unbuffered stdout so monitoring works through pipes
print = functools.partial(print, flush=True)  # type: ignore[assignment]

import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_values

# ── XML parser ───────────────────────────────────────────────────────────────
try:
    from lxml import etree
    HAVE_LXML = True
except ImportError:
    import xml.etree.ElementTree as etree
    HAVE_LXML = False

NS = "http://www.epo.org/exchange"
NS_TAG = f"{{{NS}}}"

# ── DB ────────────────────────────────────────────────────────────────────────
DB_PARAMS = dict(
    host="/var/run/postgresql",
    dbname="arc_v4",
    user=os.environ.get("PGUSER", "jeff"),
)

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


_DOCTYPE_RE = _re.compile(rb'<!DOCTYPE\s[^[>]*(?:\[[^\]]*\])?\s*>', _re.DOTALL)
_ENTITY_RE  = _re.compile(rb'&(?!amp;|lt;|gt;|apos;|quot;)[a-zA-Z][a-zA-Z0-9]*;')


def _strip_entities(xml_bytes: bytes) -> bytes:
    """Remove DOCTYPE declaration and unknown named entities so iterparse works."""
    xml_bytes = _DOCTYPE_RE.sub(b"", xml_bytes, count=1)
    xml_bytes = _ENTITY_RE.sub(b"", xml_bytes)
    return xml_bytes


def _find(el, tag: str):
    """Find child element trying namespaced first, then bare."""
    r = el.find(f"{NS_TAG}{tag}")
    if r is None:
        r = el.find(tag)
    return r


def _findall(el, path: str):
    """findall trying namespaced path first, then bare."""
    # Build namespaced version: replace each tag segment
    parts = path.split("/")
    ns_parts = [f"{NS_TAG}{p}" if p and not p.startswith(".") and not p.startswith("[")
                and not p.startswith("@") else p for p in parts]
    ns_path = "/".join(ns_parts)
    results = el.findall(ns_path)
    if not results:
        results = el.findall(path)
    return results


def _find_text(el, tag: str) -> str | None:
    """Find child and return text or None."""
    child = _find(el, tag)
    return _text_or_none(child)


def _get_docid_fields(did_el) -> dict:
    """Extract country, doc-number, kind, date from a document-id element."""
    if did_el is None:
        return {}
    return {
        "country": _find_text(did_el, "country"),
        "doc_number": _find_text(did_el, "doc-number"),
        "kind": _find_text(did_el, "kind"),
        "date": _parse_date(_find_text(did_el, "date")),
        "doc_id_attr": did_el.get("doc-id"),
        "lang": did_el.get("lang"),
    }


# ── IPCR text parser ─────────────────────────────────────────────────────────

_IPCR_RE = _re.compile(
    r'^([A-H])(\d{2})([A-Z])\s+'          # section, class_num, subclass_letter
    r'(\d+)/(\d+)\s+'                      # main_group / subgroup
    r'(\d{8})'                             # version date (YYYYMMDD)
    r'([ACN])'                             # level: A=advanced, C=core, N=not-yet
    r'([FILS])'                            # position: F=first, L=later, I=inventive, S=additional
)


def _parse_ipcr_text(ipcr_text: str) -> dict:
    """Parse structured IPCR text into component fields for IPC-to-CPC matching.

    Example: "H01L  21/00        20060101AFI20250627BHCN"
    Returns: {ipc_section:'H', ipc_class:'H01', ipc_subclass:'H01L',
              ipc_main_group:'21', ipc_subgroup:'00',
              ipc_symbol:'H01L 21/00', ipc_version:'20060101',
              ipc_level:'A', ipc_position:'F'}
    """
    if not ipcr_text:
        return {}
    m = _IPCR_RE.match(ipcr_text.strip())
    if not m:
        return {}
    section = m.group(1)
    class_num = m.group(2)
    subclass_letter = m.group(3)
    main_group = m.group(4)
    subgroup = m.group(5)
    version = m.group(6)
    level = m.group(7)
    position = m.group(8)

    ipc_class = f"{section}{class_num}"
    ipc_subclass = f"{ipc_class}{subclass_letter}"
    ipc_symbol = f"{ipc_subclass} {main_group}/{subgroup}"

    return {
        "ipc_section": section,
        "ipc_class": ipc_class,
        "ipc_subclass": ipc_subclass,
        "ipc_main_group": main_group,
        "ipc_subgroup": subgroup,
        "ipc_symbol": ipc_symbol,
        "ipc_version": version,
        "ipc_level": level,
        "ipc_position": position,
    }


def _parse_cpc_symbol(symbol: str) -> dict:
    """Parse a CPC classification symbol into component fields.

    Example: "H01L  21/00" or "B01J  19/0053"
    Returns: {cpc_section:'H', cpc_class:'H01', cpc_subclass:'H01L',
              cpc_main_group:'21', cpc_subgroup:'00'}
    """
    if not symbol:
        return {}
    s = symbol.strip()
    # Remove extra spaces: "H01L  21/00" -> "H01L21/00" or keep structure
    # CPC format: SectionClassSubclass MainGroup/SubGroup
    m = _re.match(r'^([A-HY])(\d{2})([A-Z])\s*(\d+)/(\d+)', s)
    if not m:
        return {}
    section = m.group(1)
    class_num = m.group(2)
    subclass_letter = m.group(3)
    return {
        "cpc_section": section,
        "cpc_class": f"{section}{class_num}",
        "cpc_subclass": f"{section}{class_num}{subclass_letter}",
        "cpc_main_group": m.group(4),
        "cpc_subgroup": m.group(5),
    }


# ── XML Tag Monitor ─────────────────────────────────────────────────────────

# Known XML paths under exchange-document that we extract.
# Any path NOT in this set triggers a warning/pause.
KNOWN_XML_PATHS = {
    "exchange-document",
    "exchange-document/bibliographic-data",
    "exchange-document/bibliographic-data/invention-title",
    "exchange-document/abstract",
    "exchange-document/abstract/p",
    "exchange-document/bibliographic-data/application-reference",
    "exchange-document/bibliographic-data/application-reference/document-id",
    "exchange-document/bibliographic-data/application-reference/document-id/country",
    "exchange-document/bibliographic-data/application-reference/document-id/doc-number",
    "exchange-document/bibliographic-data/application-reference/document-id/kind",
    "exchange-document/bibliographic-data/application-reference/document-id/date",
    "exchange-document/bibliographic-data/publication-reference",
    "exchange-document/bibliographic-data/publication-reference/document-id",
    "exchange-document/bibliographic-data/publication-reference/document-id/country",
    "exchange-document/bibliographic-data/publication-reference/document-id/doc-number",
    "exchange-document/bibliographic-data/publication-reference/document-id/kind",
    "exchange-document/bibliographic-data/publication-reference/document-id/date",
    "exchange-document/bibliographic-data/classification-ipc",
    "exchange-document/bibliographic-data/classification-ipc/main-classification",
    "exchange-document/bibliographic-data/classification-ipc/further-classification",
    "exchange-document/bibliographic-data/classification-ipc/edition",
    "exchange-document/bibliographic-data/classification-ipc/additional-info",
    "exchange-document/bibliographic-data/classification-ipc/text",
    "exchange-document/bibliographic-data/classification-ipc/linked-indexing-code-group",
    "exchange-document/bibliographic-data/classification-ipc/linked-indexing-code-group/main-linked-indexing-code",
    "exchange-document/bibliographic-data/classification-ipc/linked-indexing-code-group/sub-linked-indexing-code",
    "exchange-document/bibliographic-data/classification-ipc/unlinked-indexing-code",
    "exchange-document/bibliographic-data/classifications-ipcr",
    "exchange-document/bibliographic-data/classifications-ipcr/classification-ipcr",
    "exchange-document/bibliographic-data/classifications-ipcr/classification-ipcr/text",
    "exchange-document/bibliographic-data/patent-classifications",
    "exchange-document/bibliographic-data/patent-classifications/patent-classification",
    "exchange-document/bibliographic-data/patent-classifications/patent-classification/classification-symbol",
    "exchange-document/bibliographic-data/patent-classifications/patent-classification/classification-scheme",
    "exchange-document/bibliographic-data/patent-classifications/patent-classification/classification-scheme/date",
    "exchange-document/bibliographic-data/patent-classifications/patent-classification/classification-status",
    "exchange-document/bibliographic-data/patent-classifications/patent-classification/classification-value",
    "exchange-document/bibliographic-data/patent-classifications/patent-classification/symbol-position",
    "exchange-document/bibliographic-data/patent-classifications/patent-classification/generating-office",
    "exchange-document/bibliographic-data/patent-classifications/patent-classification/classification-data-source",
    "exchange-document/bibliographic-data/patent-classifications/patent-classification/action-date",
    "exchange-document/bibliographic-data/patent-classifications/patent-classification/action-date/date",
    "exchange-document/bibliographic-data/patent-classifications/combination-set",
    "exchange-document/bibliographic-data/patent-classifications/combination-set/group-number",
    "exchange-document/bibliographic-data/patent-classifications/combination-set/combination-rank",
    "exchange-document/bibliographic-data/patent-classifications/combination-set/combination-rank/patent-classification",
    "exchange-document/bibliographic-data/patent-classifications/combination-set/combination-rank/rank-number",
    # All sub-elements of combination-rank/patent-classification are same as patent-classification above
    "exchange-document/bibliographic-data/patent-classifications/combination-set/combination-rank/patent-classification/classification-symbol",
    "exchange-document/bibliographic-data/patent-classifications/combination-set/combination-rank/patent-classification/classification-scheme",
    "exchange-document/bibliographic-data/patent-classifications/combination-set/combination-rank/patent-classification/classification-scheme/date",
    "exchange-document/bibliographic-data/patent-classifications/combination-set/combination-rank/patent-classification/classification-status",
    "exchange-document/bibliographic-data/patent-classifications/combination-set/combination-rank/patent-classification/classification-value",
    "exchange-document/bibliographic-data/patent-classifications/combination-set/combination-rank/patent-classification/symbol-position",
    "exchange-document/bibliographic-data/patent-classifications/combination-set/combination-rank/patent-classification/generating-office",
    "exchange-document/bibliographic-data/patent-classifications/combination-set/combination-rank/patent-classification/classification-data-source",
    "exchange-document/bibliographic-data/patent-classifications/combination-set/combination-rank/patent-classification/action-date",
    "exchange-document/bibliographic-data/patent-classifications/combination-set/combination-rank/patent-classification/action-date/date",
    "exchange-document/bibliographic-data/parties",
    "exchange-document/bibliographic-data/parties/applicants",
    "exchange-document/bibliographic-data/parties/applicants/applicant",
    "exchange-document/bibliographic-data/parties/applicants/applicant/applicant-name",
    "exchange-document/bibliographic-data/parties/applicants/applicant/applicant-name/name",
    "exchange-document/bibliographic-data/parties/applicants/applicant/residence",
    "exchange-document/bibliographic-data/parties/applicants/applicant/residence/country",
    "exchange-document/bibliographic-data/parties/applicants/applicant/address",
    "exchange-document/bibliographic-data/parties/applicants/applicant/address/text",
    "exchange-document/bibliographic-data/parties/inventors",
    "exchange-document/bibliographic-data/parties/inventors/inventor",
    "exchange-document/bibliographic-data/parties/inventors/inventor/inventor-name",
    "exchange-document/bibliographic-data/parties/inventors/inventor/inventor-name/name",
    "exchange-document/bibliographic-data/parties/inventors/inventor/residence",
    "exchange-document/bibliographic-data/parties/inventors/inventor/residence/country",
    "exchange-document/bibliographic-data/language-of-publication",
    "exchange-document/bibliographic-data/priority-claims",
    "exchange-document/bibliographic-data/priority-claims/priority-claim",
    "exchange-document/bibliographic-data/priority-claims/priority-claim/document-id",
    "exchange-document/bibliographic-data/priority-claims/priority-claim/document-id/country",
    "exchange-document/bibliographic-data/priority-claims/priority-claim/document-id/doc-number",
    "exchange-document/bibliographic-data/priority-claims/priority-claim/document-id/kind",
    "exchange-document/bibliographic-data/priority-claims/priority-claim/document-id/date",
    "exchange-document/bibliographic-data/priority-claims/priority-claim/priority-active-indicator",
    "exchange-document/bibliographic-data/priority-claims/priority-claim/priority-linkage-type",
    "exchange-document/bibliographic-data/dates-of-public-availability",
    "exchange-document/bibliographic-data/dates-of-public-availability/printed-with-grant",
    "exchange-document/bibliographic-data/dates-of-public-availability/printed-with-grant/document-id",
    "exchange-document/bibliographic-data/dates-of-public-availability/printed-with-grant/document-id/date",
    "exchange-document/bibliographic-data/dates-of-public-availability/unexamined-printed-without-grant",
    "exchange-document/bibliographic-data/dates-of-public-availability/unexamined-printed-without-grant/document-id",
    "exchange-document/bibliographic-data/dates-of-public-availability/unexamined-printed-without-grant/document-id/date",
    "exchange-document/bibliographic-data/dates-of-public-availability/modified-complete-spec-pub",
    "exchange-document/bibliographic-data/dates-of-public-availability/modified-complete-spec-pub/document-id",
    "exchange-document/bibliographic-data/dates-of-public-availability/modified-complete-spec-pub/document-id/date",
    "exchange-document/bibliographic-data/dates-of-public-availability/modified-first-page-pub",
    "exchange-document/bibliographic-data/dates-of-public-availability/modified-first-page-pub/document-id",
    "exchange-document/bibliographic-data/dates-of-public-availability/modified-first-page-pub/document-id/date",
    "exchange-document/bibliographic-data/dates-of-public-availability/supplemental-srep-pub",
    "exchange-document/bibliographic-data/dates-of-public-availability/supplemental-srep-pub/date",
    "exchange-document/bibliographic-data/classification-national",
    "exchange-document/bibliographic-data/classification-national/text",
    "exchange-document/bibliographic-data/designated-states",
    "exchange-document/bibliographic-data/designated-states/designated-states-national",
    "exchange-document/bibliographic-data/designated-states/designated-states-national/country",
    "exchange-document/bibliographic-data/designated-states/designated-states-regional",
    "exchange-document/bibliographic-data/designated-states/designated-states-regional/country",
    "exchange-document/bibliographic-data/related-documents",
    "exchange-document/references-cited",
    "exchange-document/references-cited/citation",
    "exchange-document/references-cited/citation/patcit",
    "exchange-document/references-cited/citation/patcit/document-id",
    "exchange-document/references-cited/citation/patcit/document-id/country",
    "exchange-document/references-cited/citation/patcit/document-id/doc-number",
    "exchange-document/references-cited/citation/patcit/document-id/kind",
    "exchange-document/references-cited/citation/patcit/document-id/date",
    "exchange-document/references-cited/citation/patcit/document-id/name",
    "exchange-document/references-cited/citation/nplcit",
    "exchange-document/references-cited/citation/nplcit/text",
    "exchange-document/references-cited/citation/nplcit/source-doc",
    "exchange-document/references-cited/citation/nplcit/source-doc/document-id",
    "exchange-document/references-cited/citation/nplcit/source-doc/document-id/country",
    "exchange-document/references-cited/citation/nplcit/source-doc/document-id/doc-number",
    "exchange-document/references-cited/citation/nplcit/source-doc/document-id/kind",
    "exchange-document/references-cited/citation/category",
    "exchange-document/references-cited/citation/rel-passage",
    "exchange-document/references-cited/citation/rel-passage/passage",
    "exchange-document/references-cited/citation/rel-passage/category",
    "exchange-document/references-cited/citation/corresponding-docs",
    "exchange-document/references-cited/citation/corresponding-docs/patcit",
    "exchange-document/references-cited/citation/corresponding-docs/patcit/document-id",
    "exchange-document/references-cited/citation/corresponding-docs/patcit/document-id/country",
    "exchange-document/references-cited/citation/corresponding-docs/patcit/document-id/doc-number",
    "exchange-document/references-cited/citation/corresponding-docs/patcit/document-id/kind",
    "exchange-document/references-cited/citation/corresponding-docs/patcit/document-id/date",
    "exchange-document/references-cited/citation/corresponding-docs/patcit/document-id/name",
    "exchange-document/references-cited/citation/corresponding-docs/rel-passage",
    "exchange-document/references-cited/citation/corresponding-docs/rel-passage/passage",
    # Also under bibliographic-data (some countries nest it there)
    "exchange-document/bibliographic-data/references-cited",
    "exchange-document/bibliographic-data/references-cited/citation",
    "exchange-document/bibliographic-data/references-cited/citation/patcit",
    "exchange-document/bibliographic-data/references-cited/citation/patcit/document-id",
    "exchange-document/bibliographic-data/references-cited/citation/patcit/document-id/country",
    "exchange-document/bibliographic-data/references-cited/citation/patcit/document-id/doc-number",
    "exchange-document/bibliographic-data/references-cited/citation/patcit/document-id/kind",
    "exchange-document/bibliographic-data/references-cited/citation/patcit/document-id/date",
    "exchange-document/bibliographic-data/references-cited/citation/patcit/document-id/name",
    "exchange-document/bibliographic-data/references-cited/citation/nplcit",
    "exchange-document/bibliographic-data/references-cited/citation/nplcit/text",
    "exchange-document/bibliographic-data/references-cited/citation/nplcit/source-doc",
    "exchange-document/bibliographic-data/references-cited/citation/nplcit/source-doc/document-id",
    "exchange-document/bibliographic-data/references-cited/citation/nplcit/source-doc/document-id/country",
    "exchange-document/bibliographic-data/references-cited/citation/nplcit/source-doc/document-id/doc-number",
    "exchange-document/bibliographic-data/references-cited/citation/nplcit/source-doc/document-id/kind",
    "exchange-document/bibliographic-data/references-cited/citation/category",
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage",
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage/passage",
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage/category",
    "exchange-document/bibliographic-data/references-cited/citation/corresponding-docs",
    "exchange-document/bibliographic-data/references-cited/citation/corresponding-docs/patcit",
    "exchange-document/bibliographic-data/references-cited/citation/corresponding-docs/patcit/document-id",
    "exchange-document/bibliographic-data/references-cited/citation/corresponding-docs/patcit/document-id/country",
    "exchange-document/bibliographic-data/references-cited/citation/corresponding-docs/patcit/document-id/doc-number",
    "exchange-document/bibliographic-data/references-cited/citation/corresponding-docs/patcit/document-id/kind",
    "exchange-document/bibliographic-data/references-cited/citation/corresponding-docs/patcit/document-id/date",
    "exchange-document/bibliographic-data/references-cited/citation/corresponding-docs/patcit/document-id/name",
    "exchange-document/bibliographic-data/references-cited/citation/corresponding-docs/rel-passage",
    "exchange-document/bibliographic-data/references-cited/citation/corresponding-docs/rel-passage/passage",
    # Pub availability: newly discovered date types
    "exchange-document/bibliographic-data/dates-of-public-availability/examined-not-printed-without-grant",
    "exchange-document/bibliographic-data/dates-of-public-availability/examined-not-printed-without-grant/document-id",
    "exchange-document/bibliographic-data/dates-of-public-availability/examined-not-printed-without-grant/document-id/date",
    "exchange-document/bibliographic-data/dates-of-public-availability/unexamined-not-printed-without-grant",
    "exchange-document/bibliographic-data/dates-of-public-availability/unexamined-not-printed-without-grant/document-id",
    "exchange-document/bibliographic-data/dates-of-public-availability/unexamined-not-printed-without-grant/document-id/date",
    "exchange-document/bibliographic-data/dates-of-public-availability/gazette-reference",
    "exchange-document/bibliographic-data/dates-of-public-availability/gazette-reference/date",
    "exchange-document/bibliographic-data/dates-of-public-availability/examined-printed-without-grant",
    "exchange-document/bibliographic-data/dates-of-public-availability/examined-printed-without-grant/document-id",
    "exchange-document/bibliographic-data/dates-of-public-availability/examined-printed-without-grant/document-id/date",
    "exchange-document/bibliographic-data/dates-of-public-availability/abstract-reference",
    "exchange-document/bibliographic-data/dates-of-public-availability/abstract-reference/document-id",
    "exchange-document/bibliographic-data/dates-of-public-availability/abstract-reference/document-id/date",
    # Preceding publication date (AT/other, rare)
    "exchange-document/bibliographic-data/preceding-publication-date",
    "exchange-document/bibliographic-data/preceding-publication-date/date",
    # Inventor address (same pattern as applicant address)
    "exchange-document/bibliographic-data/parties/inventors/inventor/address",
    "exchange-document/bibliographic-data/parties/inventors/inventor/address/text",
    # NPL structured citation data (article, online, book metadata)
    # Captured as flat npl_text; structured data available for future parsing
    "exchange-document/bibliographic-data/references-cited/citation/corresponding-docs/nplcit",
    "exchange-document/bibliographic-data/references-cited/citation/corresponding-docs/nplcit/text",
    "exchange-document/bibliographic-data/references-cited/citation/nplcit/article",
    "exchange-document/bibliographic-data/references-cited/citation/nplcit/online",
    "exchange-document/bibliographic-data/references-cited/citation/nplcit/source-doc",
    "exchange-document/bibliographic-data/references-cited/citation/nplcit/source-doc/document-id",
    "exchange-document/bibliographic-data/references-cited/citation/nplcit/source-doc/document-id/country",
    "exchange-document/bibliographic-data/references-cited/citation/nplcit/source-doc/document-id/doc-number",
    "exchange-document/bibliographic-data/references-cited/citation/nplcit/source-doc/document-id/kind",
    # Citation rel-passage sub-elements
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage/passage/bookmark",
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage/passage/claim",
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage/passage/colf",
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage/passage/coll",
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage/passage/compound",
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage/passage/example",
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage/passage/figure",
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage/passage/linef",
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage/passage/linel",
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage/passage/paraf",
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage/passage/paral",
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage/passage/ppf",
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage/passage/ppl",
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage/passage/sequence",
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage/passage/table",
    "exchange-document/bibliographic-data/references-cited/citation/rel-passage/rel-claims",
}
# Auto-accept deep NPL article/online/book sub-elements
# (50+ sub-paths for author, serial, imprint, DOI, ISBN, etc.)
# Rather than listing all individually, we accept any path starting with known NPL prefixes
_KNOWN_NPL_PREFIXES = (
    "exchange-document/bibliographic-data/references-cited/citation/nplcit/article/",
    "exchange-document/bibliographic-data/references-cited/citation/nplcit/online/",
    "exchange-document/bibliographic-data/references-cited/citation/nplcit/book/",
    "exchange-document/references-cited/citation/nplcit/article/",
    "exchange-document/references-cited/citation/nplcit/online/",
    "exchange-document/references-cited/citation/nplcit/book/",
)

# Paths seen but not in KNOWN_XML_PATHS — logged once per path
_unknown_paths_seen: set[str] = set()
_UNKNOWN_PATH_LOG = Path("/tmp/epo_unknown_xml_paths.jsonl")


def _strip_ns_tag(tag: str) -> str:
    """Remove namespace prefix from a tag."""
    if tag.startswith(NS_TAG):
        return tag[len(NS_TAG):]
    return tag


def _get_element_path(el) -> str:
    """Build the full path from exchange-document root (namespace-stripped)."""
    parts = []
    while el is not None:
        parts.append(_strip_ns_tag(el.tag))
        el = el.getparent()
    parts.reverse()
    try:
        idx = parts.index("exchange-document")
        return "/".join(parts[idx:])
    except ValueError:
        return "/".join(parts)


def check_unknown_tags(doc_el) -> list[str]:
    """Scan a document element for XML paths not in KNOWN_XML_PATHS.

    Returns list of newly discovered unknown paths (empty if all known).
    """
    new_unknowns = []
    for el in doc_el.iter():
        path = _get_element_path(el)
        if (path not in KNOWN_XML_PATHS
                and path not in _unknown_paths_seen
                and not path.startswith(_KNOWN_NPL_PREFIXES)):
            _unknown_paths_seen.add(path)
            new_unknowns.append(path)
            # Log to file
            entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "path": path,
                "tag": _strip_ns_tag(el.tag),
                "attrs": dict(el.attrib),
                "text_sample": (el.text or "").strip()[:200],
                "doc_id": doc_el.get("doc-id", "?"),
                "country": doc_el.get("country", "?"),
            }
            with open(_UNKNOWN_PATH_LOG, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
            print(f"  ⚠ UNKNOWN XML PATH: {path} (doc_id={entry['doc_id']}, "
                  f"country={entry['country']})", flush=True)
    return new_unknowns


# ── INSERT SQL templates ─────────────────────────────────────────────────────

INSERT_DOCUMENT = """
INSERT INTO arc_v5.epo_document
    (doc_id, country, doc_number, kind, family_id, date_publ,
     is_representative, date_of_last_exchange, date_added_docdb,
     originating_office, language_of_publication, zip_source)
VALUES %s
ON CONFLICT (doc_id) DO NOTHING
"""

INSERT_TITLE = """
INSERT INTO arc_v5.epo_title
    (doc_id, lang, data_format, title_text)
VALUES %s
"""

INSERT_ABSTRACT = """
INSERT INTO arc_v5.epo_abstract
    (doc_id, lang, data_format, abstract_source, abstract_text)
VALUES %s
"""

INSERT_APPLICATION_REF = """
INSERT INTO arc_v5.epo_application_ref
    (doc_id, ref_doc_id, is_representative,
     docdb_country, docdb_doc_number, docdb_kind, docdb_date,
     epodoc_doc_number, epodoc_kind, epodoc_date,
     original_doc_number, original_kind, original_date, original_lang)
VALUES %s
ON CONFLICT (doc_id) DO NOTHING
"""

INSERT_PUBLICATION_REF = """
INSERT INTO arc_v5.epo_publication_ref
    (doc_id,
     docdb_country, docdb_doc_number, docdb_kind, docdb_date,
     epodoc_doc_number, epodoc_kind, epodoc_date, epodoc_lang,
     original_doc_number, original_kind, original_date, original_lang)
VALUES %s
ON CONFLICT (doc_id) DO NOTHING
"""

INSERT_CLASSIFICATION_IPC = """
INSERT INTO arc_v5.epo_classification_ipc
    (doc_id, classification_type, code, edition, additional_info)
VALUES %s
"""

INSERT_CLASSIFICATION_IPCR = """
INSERT INTO arc_v5.epo_classification_ipcr
    (doc_id, sequence, ipcr_text,
     ipc_section, ipc_class, ipc_subclass, ipc_main_group, ipc_subgroup,
     ipc_symbol, ipc_version, ipc_level, ipc_position)
VALUES %s
"""

INSERT_PATENT_CLASSIFICATION = """
INSERT INTO arc_v5.epo_patent_classification
    (doc_id, classification_symbol, scheme_office, scheme,
     classification_status, classification_value, symbol_position,
     generating_office, classification_data_source, action_date,
     sequence, combination_group, combination_rank,
     cpc_section, cpc_class, cpc_subclass, cpc_main_group, cpc_subgroup)
VALUES %s
"""

INSERT_PERSON = """
INSERT INTO arc_v5.epo_person
    (name_docdb, name_epodoc, name_original, country)
VALUES (%s, %s, %s, %s)
RETURNING id
"""

INSERT_DOCUMENT_APPLICANT = """
INSERT INTO arc_v5.epo_document_applicant
    (doc_id, person_id, sequence, address_text)
VALUES %s
"""

INSERT_DOCUMENT_INVENTOR = """
INSERT INTO arc_v5.epo_document_inventor
    (doc_id, person_id, sequence)
VALUES %s
"""

INSERT_PRIORITY_CLAIM = """
INSERT INTO arc_v5.epo_priority_claim
    (doc_id, sequence,
     docdb_country, docdb_doc_number, docdb_kind, docdb_date, docdb_ref_id,
     epodoc_doc_number, epodoc_kind, epodoc_date,
     priority_active_indicator, priority_linkage_type)
VALUES %s
"""

INSERT_CITATION_PATENT = """
INSERT INTO arc_v5.epo_citation_patent
    (doc_id, cited_country, cited_doc_number, cited_kind, cited_doc_id,
     category, data_format, sequence)
VALUES %s
"""

INSERT_CITATION_NPL = """
INSERT INTO arc_v5.epo_citation_npl
    (doc_id, npl_text, category, sequence)
VALUES %s
"""

INSERT_PUB_AVAILABILITY = """
INSERT INTO arc_v5.epo_pub_availability
    (doc_id, printed_with_grant_date, unexamined_printed_wo_grant_date,
     modified_complete_spec_pub_date, modified_first_page_pub_date,
     supplemental_srep_pub_date,
     examined_not_printed_wo_grant_date, unexamined_not_printed_wo_grant_date,
     gazette_reference_date,
     examined_printed_wo_grant_date, abstract_reference_date)
VALUES %s
ON CONFLICT (doc_id) DO NOTHING
"""

INSERT_CLASSIFICATION_NATIONAL = """
INSERT INTO arc_v5.epo_classification_national
    (doc_id, nat_text)
VALUES %s
"""

INSERT_DESIGNATED_STATE = """
INSERT INTO arc_v5.epo_designated_state
    (doc_id, state_type, country)
VALUES %s
"""

INSERT_RELATED_DOCUMENT = """
INSERT INTO arc_v5.epo_related_document
    (doc_id, relation_type, parent_child,
     related_country, related_doc_number, related_kind, related_date)
VALUES %s
"""


# ── Person Cache ──────────────────────────────────────────────────────────────

class PersonCache:
    """In-memory cache mapping (name_docdb_lower, country) -> person_id.

    Person rows are inserted individually (RETURNING id) since we need the
    id immediately for junction table inserts within the same transaction.

    When cache exceeds MAX_SIZE, oldest entries are evicted. Evicted persons
    will be re-looked-up on next encounter (INSERT ... ON CONFLICT returns
    existing id via a fallback SELECT).
    """

    MAX_SIZE = 20_000_000  # ~4GB RAM at 200 bytes/entry

    def __init__(self):
        self.cache: dict[tuple[str, str], int] = {}
        self.stats = {"hits": 0, "inserts": 0, "evictions": 0, "db_lookups": 0}
        # Track person_ids inserted in current batch for rollback cleanup
        self._batch_keys: list[tuple[str, str]] = []

    def _evict_if_needed(self):
        """Evict oldest 20% of cache when it exceeds MAX_SIZE."""
        if len(self.cache) > self.MAX_SIZE:
            n_evict = len(self.cache) // 5
            keys_to_evict = list(self.cache.keys())[:n_evict]
            for k in keys_to_evict:
                del self.cache[k]
            self.stats["evictions"] += n_evict

    def get_or_create(self, cur, name_docdb: str | None,
                      name_epodoc: str | None, name_original: str | None,
                      country: str | None) -> int:
        """Look up or create a person. Uses the cursor from the batch transaction."""
        key = ((name_docdb or "").lower().strip(), (country or "").strip())
        if key in self.cache:
            self.stats["hits"] += 1
            return self.cache[key]

        # Try INSERT first; if person already exists (from evicted cache entry),
        # the INSERT will succeed (no unique constraint during bulk load) creating
        # a duplicate. Post-load dedup handles this. For safety, try a SELECT first
        # if we've had evictions.
        if self.stats["evictions"] > 0:
            cur.execute(
                "SELECT id FROM arc_v5.epo_person "
                "WHERE lower(name_docdb) = %s AND coalesce(country, '') = %s LIMIT 1",
                (key[0], key[1])
            )
            row = cur.fetchone()
            if row:
                self.cache[key] = row[0]
                self.stats["db_lookups"] += 1
                self.stats["hits"] += 1
                return row[0]

        cur.execute(
            INSERT_PERSON,
            (name_docdb, name_epodoc, name_original, country)
        )
        pid = cur.fetchone()[0]
        self.cache[key] = pid
        self._batch_keys.append(key)
        self.stats["inserts"] += 1
        self._evict_if_needed()
        return pid

    def begin_batch(self):
        """Mark the start of a new batch for rollback tracking."""
        self._batch_keys = []

    def rollback_batch(self):
        """Remove cache entries added during the current (failed) batch."""
        for key in self._batch_keys:
            self.cache.pop(key, None)
        self._batch_keys = []

    def commit_batch(self):
        """Clear batch tracking after successful commit."""
        self._batch_keys = []


# ── Document Extractor ────────────────────────────────────────────────────────

def extract_doc_normalized(doc_el) -> dict:
    """Extract one exchange-document element into a normalized dict.

    Returns a dict with top-level fields for epo_document plus lists of
    child dicts for each related table.
    """
    doc_id = doc_el.get("doc-id")
    country = doc_el.get("country")
    doc_number = doc_el.get("doc-number")
    kind = doc_el.get("kind")
    family_id = doc_el.get("family-id")
    date_publ = _parse_date(doc_el.get("date-publ"))

    is_rep_str = doc_el.get("is-representative")
    is_representative = True if is_rep_str == "YES" else (False if is_rep_str == "NO" else None)

    date_of_last_exchange = _parse_date(doc_el.get("date-of-last-exchange"))
    date_added_docdb = _parse_date(doc_el.get("date-added-docdb"))
    originating_office = doc_el.get("originating-office")

    bib = _find(doc_el, "bibliographic-data")

    # ── Language ──
    language = None
    if bib is not None:
        language = _find_text(bib, "language-of-publication")

    # ── Titles — ALL languages ──
    titles = []
    if bib is not None:
        for t in _findall(bib, "invention-title"):
            txt = _text_or_none(t)
            if txt:
                titles.append({
                    "lang": t.get("lang"),
                    "data_format": t.get("data-format"),
                    "title_text": txt,
                })

    # ── Abstracts — ALL languages ──
    abstracts = []
    for ab in doc_el.iter(f"{NS_TAG}abstract"):
        lang = ab.get("lang")
        data_format = ab.get("data-format")
        abstract_source = ab.get("abstract-source")
        # Concatenate all <p> children
        paras = _findall(ab, "p")
        if paras:
            txt = " ".join(_text_or_none(p) or "" for p in paras).strip()
        else:
            txt = _text_or_none(ab)
        if txt:
            abstracts.append({
                "lang": lang,
                "data_format": data_format,
                "abstract_source": abstract_source,
                "abstract_text": txt,
            })
    # Also check bare namespace
    for ab in doc_el.iter("abstract"):
        if ab.tag == f"{NS_TAG}abstract":
            continue  # already handled
        lang = ab.get("lang")
        data_format = ab.get("data-format")
        abstract_source = ab.get("abstract-source")
        paras = _findall(ab, "p")
        if paras:
            txt = " ".join(_text_or_none(p) or "" for p in paras).strip()
        else:
            txt = _text_or_none(ab)
        if txt:
            abstracts.append({
                "lang": lang,
                "data_format": data_format,
                "abstract_source": abstract_source,
                "abstract_text": txt,
            })

    # ── Application Reference — merge 3 data-format variants into one row ──
    app_ref = None
    if bib is not None:
        app_ref_data = {
            "docdb_country": None, "docdb_doc_number": None,
            "docdb_kind": None, "docdb_date": None,
            "docdb_is_representative": None, "docdb_doc_id_attr": None,
            "epodoc_doc_number": None, "epodoc_kind": None, "epodoc_date": None,
            "original_doc_number": None, "original_kind": None,
            "original_date": None, "original_lang": None,
        }
        has_any = False
        for ar in _findall(bib, "application-reference"):
            fmt = ar.get("data-format", "")
            did = _find(ar, "document-id")
            fields = _get_docid_fields(did)
            if fmt == "docdb":
                app_ref_data["docdb_country"] = fields.get("country")
                app_ref_data["docdb_doc_number"] = fields.get("doc_number")
                app_ref_data["docdb_kind"] = fields.get("kind")
                app_ref_data["docdb_date"] = fields.get("date")
                app_ref_data["docdb_is_representative"] = (
                    True if ar.get("is-representative") == "YES"
                    else (False if ar.get("is-representative") == "NO" else None)
                )
                app_ref_data["docdb_doc_id_attr"] = ar.get("doc-id")
                has_any = True
            elif fmt == "epodoc":
                app_ref_data["epodoc_doc_number"] = fields.get("doc_number")
                app_ref_data["epodoc_kind"] = fields.get("kind")
                app_ref_data["epodoc_date"] = fields.get("date")
                has_any = True
            elif fmt in ("original", "docdba"):
                app_ref_data["original_doc_number"] = fields.get("doc_number")
                app_ref_data["original_kind"] = fields.get("kind")
                app_ref_data["original_date"] = fields.get("date")
                app_ref_data["original_lang"] = fields.get("lang")
                has_any = True
        if has_any:
            app_ref = app_ref_data

    # ── Publication Reference — merge 3 data-format variants into one row ──
    pub_ref = None
    if bib is not None:
        pub_ref_data = {
            "docdb_country": None, "docdb_doc_number": None,
            "docdb_kind": None, "docdb_date": None,
            "epodoc_doc_number": None, "epodoc_kind": None,
            "epodoc_date": None, "epodoc_lang": None,
            "original_doc_number": None, "original_kind": None,
            "original_date": None, "original_lang": None,
        }
        has_any = False
        for pr in _findall(bib, "publication-reference"):
            fmt = pr.get("data-format", "")
            did = _find(pr, "document-id")
            fields = _get_docid_fields(did)
            if fmt == "docdb":
                pub_ref_data["docdb_country"] = fields.get("country")
                pub_ref_data["docdb_doc_number"] = fields.get("doc_number")
                pub_ref_data["docdb_kind"] = fields.get("kind")
                pub_ref_data["docdb_date"] = fields.get("date")
                has_any = True
            elif fmt == "epodoc":
                pub_ref_data["epodoc_doc_number"] = fields.get("doc_number")
                pub_ref_data["epodoc_kind"] = fields.get("kind")
                pub_ref_data["epodoc_date"] = fields.get("date")
                pub_ref_data["epodoc_lang"] = fields.get("lang")
                has_any = True
            elif fmt in ("original", "docdba"):
                pub_ref_data["original_doc_number"] = fields.get("doc_number")
                pub_ref_data["original_kind"] = fields.get("kind")
                pub_ref_data["original_date"] = fields.get("date")
                pub_ref_data["original_lang"] = fields.get("lang")
                has_any = True
        if has_any:
            pub_ref = pub_ref_data

    # ── IPC (old format) ──
    ipc_codes = []
    if bib is not None:
        ipc_el = _find(bib, "classification-ipc")
        if ipc_el is not None:
            edition = _find_text(ipc_el, "edition")
            additional_info = _find_text(ipc_el, "additional-info")
            # main-classification
            for mc in _findall(ipc_el, "main-classification"):
                txt = _text_or_none(mc)
                if txt:
                    ipc_codes.append({
                        "code_type": "main",
                        "code_text": txt,
                        "edition": edition,
                        "additional_info": additional_info,
                    })
            # further-classification
            for fc in _findall(ipc_el, "further-classification"):
                txt = _text_or_none(fc)
                if txt:
                    ipc_codes.append({
                        "code_type": "further",
                        "code_text": txt,
                        "edition": edition,
                        "additional_info": additional_info,
                    })
            # linked indexing codes
            for lig in _findall(ipc_el, "linked-indexing-code-group"):
                for mlic in _findall(lig, "main-linked-indexing-code"):
                    txt = _text_or_none(mlic)
                    if txt:
                        ipc_codes.append({
                            "code_type": "main_linked",
                            "code_text": txt,
                            "edition": edition,
                            "additional_info": additional_info,
                        })
                for slic in _findall(lig, "sub-linked-indexing-code"):
                    txt = _text_or_none(slic)
                    if txt:
                        ipc_codes.append({
                            "code_type": "sub_linked",
                            "code_text": txt,
                            "edition": edition,
                            "additional_info": additional_info,
                        })
            # unlinked indexing codes
            for uic in _findall(ipc_el, "unlinked-indexing-code"):
                txt = _text_or_none(uic)
                if txt:
                    ipc_codes.append({
                        "code_type": "unlinked",
                        "code_text": txt,
                        "edition": edition,
                        "additional_info": additional_info,
                    })
            # text child (full IPC text)
            ipc_text_el = _find(ipc_el, "text")
            if ipc_text_el is not None:
                txt = _text_or_none(ipc_text_el)
                if txt:
                    ipc_codes.append({
                        "code_type": "text",
                        "code_text": txt,
                        "edition": edition,
                        "additional_info": additional_info,
                    })

    # ── IPCR ──
    ipcr_codes = []
    if bib is not None:
        ipcr_parent = _find(bib, "classifications-ipcr")
        if ipcr_parent is not None:
            for cl in _findall(ipcr_parent, "classification-ipcr"):
                seq = cl.get("sequence")
                txt_el = _find(cl, "text")
                txt = _text_or_none(txt_el)
                if txt:
                    ipcr_codes.append({
                        "sequence": _safe_int(seq),
                        "ipcr_text": txt,
                    })

    # ── Patent Classifications (CPC etc) ──
    patent_classifications = []
    if bib is not None:
        pc_parent = _find(bib, "patent-classifications")
        if pc_parent is not None:
            # Direct patent-classification children
            for pc in _findall(pc_parent, "patent-classification"):
                patent_classifications.append(_extract_patent_classification(pc))
            # combination-set > combination-rank > patent-classification
            for cset in _findall(pc_parent, "combination-set"):
                group_number = cset.get("group-number")
                for crank in _findall(cset, "combination-rank"):
                    rank_number = crank.get("rank-number")
                    for pc in _findall(crank, "patent-classification"):
                        row = _extract_patent_classification(pc)
                        row["combination_group"] = _safe_int(group_number)
                        row["combination_rank"] = _safe_int(rank_number)
                        patent_classifications.append(row)

    # ── Applicants — group by sequence across data-formats ──
    applicants = []
    if bib is not None:
        applicants = _extract_parties(bib, "applicants", "applicant", "applicant-name")

    # ── Inventors — group by sequence across data-formats ──
    inventors = []
    if bib is not None:
        inventors = _extract_parties(bib, "inventors", "inventor", "inventor-name")

    # ── Priority Claims — group by sequence across data-formats ──
    priority_claims = []
    if bib is not None:
        pc_el = _find(bib, "priority-claims")
        if pc_el is not None:
            # Group by sequence
            by_seq: dict[str, dict] = {}
            for claim in _findall(pc_el, "priority-claim"):
                seq = claim.get("sequence", "0")
                fmt = claim.get("data-format", "")
                did = _find(claim, "document-id")
                fields = _get_docid_fields(did)

                if seq not in by_seq:
                    by_seq[seq] = {
                        "sequence": _safe_int(seq),
                        "docdb_country": None, "docdb_doc_number": None,
                        "docdb_kind": None, "docdb_date": None,
                        "docdb_doc_id_attr": None,
                        "epodoc_doc_number": None, "epodoc_kind": None,
                        "epodoc_date": None,
                        "priority_active_indicator": None,
                        "priority_linkage_type": None,
                    }

                rec = by_seq[seq]
                if fmt == "docdb":
                    rec["docdb_country"] = fields.get("country")
                    rec["docdb_doc_number"] = fields.get("doc_number")
                    rec["docdb_kind"] = fields.get("kind")
                    rec["docdb_date"] = fields.get("date")
                    rec["docdb_doc_id_attr"] = fields.get("doc_id_attr")
                elif fmt == "epodoc":
                    rec["epodoc_doc_number"] = fields.get("doc_number")
                    rec["epodoc_kind"] = fields.get("kind")
                    rec["epodoc_date"] = fields.get("date")

                # These can appear on any format variant
                pai = _find_text(claim, "priority-active-indicator")
                if pai:
                    rec["priority_active_indicator"] = pai
                plt = _find_text(claim, "priority-linkage-type")
                if plt:
                    rec["priority_linkage_type"] = plt

            priority_claims = list(by_seq.values())

    # ── Citations ──
    citation_patents = []
    citation_npls = []
    for cite in doc_el.iter(f"{NS_TAG}citation"):
        _extract_citation(cite, citation_patents, citation_npls)
    # Also bare namespace
    for cite in doc_el.iter("citation"):
        if cite.tag == f"{NS_TAG}citation":
            continue
        _extract_citation(cite, citation_patents, citation_npls)

    # ── Dates of Public Availability ──
    pub_availability = None
    if bib is not None:
        dpa = _find(bib, "dates-of-public-availability")
        if dpa is not None:
            pub_availability = {
                "printed_with_grant_date": _extract_availability_date(dpa, "printed-with-grant"),
                "unexamined_printed_date": _extract_availability_date(dpa, "unexamined-printed-without-grant"),
                "modified_complete_spec_date": _extract_availability_date(dpa, "modified-complete-spec-pub"),
                "modified_first_page_date": _extract_availability_date(dpa, "modified-first-page-pub"),
                "supplemental_srep_date": _extract_availability_date(dpa, "supplemental-srep-pub"),
                "examined_not_printed_date": _extract_availability_date(dpa, "examined-not-printed-without-grant"),
                "unexamined_not_printed_date": _extract_availability_date(dpa, "unexamined-not-printed-without-grant"),
                "gazette_reference_date": _extract_availability_date(dpa, "gazette-reference"),
                "examined_printed_wo_grant_date": _extract_availability_date(dpa, "examined-printed-without-grant"),
                "abstract_reference_date": _extract_availability_date(dpa, "abstract-reference"),
            }
            # Only keep if at least one date is present
            if not any(pub_availability.values()):
                pub_availability = None

    # ── Classification National ──
    national_codes = []
    if bib is not None:
        cn_el = _find(bib, "classification-national")
        if cn_el is not None:
            for txt_el in _findall(cn_el, "text"):
                txt = _text_or_none(txt_el)
                if txt:
                    national_codes.append(txt)

    # ── Designated States ──
    designated_states = []
    if bib is not None:
        ds_el = _find(bib, "designated-states")
        if ds_el is not None:
            # national > country
            nat = _find(ds_el, "national")
            if nat is not None:
                for c_el in _findall(nat, "country"):
                    txt = _text_or_none(c_el)
                    if txt:
                        designated_states.append({
                            "state_type": "national",
                            "country_code": txt,
                        })
            # regional > country
            reg = _find(ds_el, "regional")
            if reg is not None:
                for c_el in _findall(reg, "country"):
                    txt = _text_or_none(c_el)
                    if txt:
                        designated_states.append({
                            "state_type": "regional",
                            "country_code": txt,
                        })

    # ── Related Documents ──
    related_documents = []
    if bib is not None:
        rd_el = _find(bib, "related-documents")
        if rd_el is not None:
            # Known relation types
            for rel_type in ("continuation", "continuation-in-part", "continuing-reissue",
                             "division", "addition", "substitute", "utility-model-basis",
                             "reissue", "reexamination", "merged-analysis",
                             "continuation-of", "division-of", "previously-published"):
                for rel_container in _findall(rd_el, rel_type):
                    relation = _find(rel_container, "relation")
                    if relation is None:
                        relation = rel_container  # some formats nest directly
                    for child_or_parent in ("child-doc", "parent-doc"):
                        doc = _find(relation, child_or_parent)
                        if doc is None:
                            continue
                        did = _find(doc, "document-id")
                        fields = _get_docid_fields(did)
                        if fields.get("doc_number"):
                            related_documents.append({
                                "relation_type": rel_type,
                                "child_or_parent": child_or_parent.replace("-doc", ""),
                                "related_country": fields.get("country"),
                                "related_doc_number": fields.get("doc_number"),
                                "related_kind": fields.get("kind"),
                                "related_date": fields.get("date"),
                            })

    return {
        "doc_id": doc_id,
        "country": country,
        "doc_number": doc_number,
        "kind": kind,
        "family_id": family_id,
        "date_publ": date_publ,
        "is_representative": is_representative,
        "date_of_last_exchange": date_of_last_exchange,
        "date_added_docdb": date_added_docdb,
        "originating_office": originating_office,
        "language": language,
        "titles": titles,
        "abstracts": abstracts,
        "app_ref": app_ref,
        "pub_ref": pub_ref,
        "ipc_codes": ipc_codes,
        "ipcr_codes": ipcr_codes,
        "patent_classifications": patent_classifications,
        "applicants": applicants,
        "inventors": inventors,
        "priority_claims": priority_claims,
        "citation_patents": citation_patents,
        "citation_npls": citation_npls,
        "pub_availability": pub_availability,
        "national_codes": national_codes,
        "designated_states": designated_states,
        "related_documents": related_documents,
    }


# ── Extraction helpers ────────────────────────────────────────────────────────

def _safe_int(s: str | None) -> int | None:
    """Convert string to int or None."""
    if s is None:
        return None
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def _extract_patent_classification(pc_el) -> dict:
    """Extract one patent-classification element."""
    symbol = _find_text(pc_el, "classification-symbol")
    scheme_el = _find(pc_el, "classification-scheme")
    scheme_office = scheme_el.get("office") if scheme_el is not None else None
    scheme_name = scheme_el.get("scheme") if scheme_el is not None else None
    status = _find_text(pc_el, "classification-status")
    value = _find_text(pc_el, "classification-value")
    position = _find_text(pc_el, "symbol-position")
    gen_office = _find_text(pc_el, "generating-office")
    data_source = _find_text(pc_el, "classification-data-source")
    action_date_el = _find(pc_el, "action-date")
    action_date = None
    if action_date_el is not None:
        action_date = _parse_date(_find_text(action_date_el, "date"))
    seq = pc_el.get("sequence")

    return {
        "classification_symbol": symbol,
        "scheme_office": scheme_office,
        "scheme_name": scheme_name,
        "classification_status": status,
        "classification_value": value,
        "symbol_position": position,
        "generating_office": gen_office,
        "classification_data_source": data_source,
        "action_date": action_date,
        "sequence": _safe_int(seq),
        "combination_group": None,
        "combination_rank": None,
    }


def _extract_parties(bib, container_tag: str, item_tag: str, name_tag: str) -> list[dict]:
    """Extract applicants or inventors, grouped by sequence across data-formats.

    Returns list of dicts with keys: sequence, name_docdb, name_epodoc,
    name_original, country, address_text, data_format_source.
    """
    # Find the container: e.g. parties/applicants
    parties = _find(bib, "parties")
    if parties is None:
        return []
    container = _find(parties, container_tag)
    if container is None:
        return []

    by_seq: dict[str, dict] = {}
    for item in _findall(container, item_tag):
        seq = item.get("sequence", "0")
        fmt = item.get("data-format", "")

        if seq not in by_seq:
            by_seq[seq] = {
                "sequence": _safe_int(seq),
                "name_docdb": None,
                "name_epodoc": None,
                "name_original": None,
                "country": None,
                "address_text": None,
                "data_format_source": fmt,
            }

        rec = by_seq[seq]
        # Name extraction
        name_el = _find(item, name_tag)
        if name_el is not None:
            name_sub = _find(name_el, "name")
            name_text = _text_or_none(name_sub) if name_sub is not None else _text_or_none(name_el)
        else:
            name_text = None

        if fmt == "docdb":
            rec["name_docdb"] = name_text
            rec["data_format_source"] = "docdb"
        elif fmt == "epodoc":
            rec["name_epodoc"] = name_text
        elif fmt in ("original", "docdba"):
            rec["name_original"] = name_text

        # Country from residence (any format, docdb preferred)
        res = _find(item, "residence")
        if res is not None:
            c = _find_text(res, "country")
            if c and (rec["country"] is None or fmt == "docdb"):
                rec["country"] = c

        # Address
        addr = _find(item, "address")
        if addr is not None:
            addr_text = _find_text(addr, "text")
            if addr_text and (rec["address_text"] is None or fmt == "docdb"):
                rec["address_text"] = addr_text

    return list(by_seq.values())


def _extract_citation(cite_el, patent_list: list, npl_list: list):
    """Extract patent and NPL citations from a citation element."""
    data_format = cite_el.get("data-format")
    cat_el = _find(cite_el, "category")
    category = _text_or_none(cat_el)

    # Patent citation
    patcit = _find(cite_el, "patcit")
    if patcit is not None:
        did = _find(patcit, "document-id")
        if did is not None:
            fields = _get_docid_fields(did)
            if fields.get("country") or fields.get("doc_number"):
                patent_list.append({
                    "cited_country": fields.get("country"),
                    "cited_doc_number": fields.get("doc_number"),
                    "cited_kind": fields.get("kind"),
                    "cited_doc_id": fields.get("doc_id_attr"),
                    "category": category,
                    "data_format": data_format,
                })

    # NPL citation
    nplcit = _find(cite_el, "nplcit")
    if nplcit is not None:
        npl_text = _text_or_none(nplcit)
        if npl_text:
            npl_list.append({
                "npl_text": npl_text[:4000],
                "category": category,
            })


def _extract_availability_date(dpa_el, tag: str) -> date | None:
    """Extract a date from a dates-of-public-availability sub-element."""
    sub = _find(dpa_el, tag)
    if sub is None:
        return None
    # Try document-id/date first
    did = _find(sub, "document-id")
    if did is not None:
        return _parse_date(_find_text(did, "date"))
    # Try direct date child
    return _parse_date(_find_text(sub, "date"))


# ── XML Stream Parser ─────────────────────────────────────────────────────────

def parse_xml_stream_normalized(xml_bytes: bytes) -> Iterator[dict]:
    """Stream-parse exchange-document elements into normalized dicts."""
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
    _doc_count = 0
    for _event, doc in context:
        _doc_count += 1
        # Tag monitor: first 100 docs per XML file, then every 10,000th
        if _doc_count <= 100 or _doc_count % 10000 == 0:
            check_unknown_tags(doc)
        yield extract_doc_normalized(doc)
        doc.clear()
        while doc.getprevious() is not None:
            del doc.getparent()[0]


def iter_epo_zip(outer_zip_path: str,
                 error_sink: list | None = None) -> Iterator[tuple[dict, str]]:
    """Yield (normalized_doc_dict, inner_zip_name) from a DOCDB outer ZIP."""
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
                    for doc in parse_xml_stream_normalized(xml_bytes):
                        yield doc, inner_path
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


# ── COPY helper for fast bulk insert ─────────────────────────────────────────

def _copy_rows(cur, table: str, columns: list[str], rows: list[tuple]) -> int:
    """Bulk-insert rows using COPY FROM (2-5x faster than execute_values).

    Only for identity-PK tables where ON CONFLICT is not needed.
    Handles None values as \\N (NULL).
    """
    if not rows:
        return 0
    buf = io.StringIO()
    for row in rows:
        fields = []
        for val in row:
            if val is None:
                fields.append("\\N")
            elif isinstance(val, bool):
                fields.append("t" if val else "f")
            elif isinstance(val, date):
                fields.append(val.isoformat())
            else:
                # Escape tabs, newlines, backslashes for COPY format
                s = str(val).replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
                fields.append(s)
        buf.write("\t".join(fields) + "\n")
    buf.seek(0)
    cols = ", ".join(columns)
    cur.copy_expert(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT text)", buf)
    return len(rows)


# ── Batch Flush ───────────────────────────────────────────────────────────────

def flush_batch(conn, batch: list[dict], person_cache: PersonCache) -> dict:
    """Insert a batch of normalized docs into all epo_* tables.

    Uses COPY for identity-PK child tables (fast), execute_values for
    doc_id-PK tables that need ON CONFLICT (safe).

    Returns dict of per-table row counts.
    """
    counts: dict[str, int] = {}
    person_cache.begin_batch()

    try:
        with conn.cursor() as cur:
            # 1. Documents
            doc_rows = []
            for d in batch:
                if not d.get("doc_id"):
                    continue
                doc_rows.append((
                    d["doc_id"], d["country"], d["doc_number"], d["kind"],
                    d["family_id"], d["date_publ"], d["is_representative"],
                    d["date_of_last_exchange"], d["date_added_docdb"],
                    d["originating_office"], d["language"],
                    d.get("zip_source"),
                ))
            if doc_rows:
                execute_values(cur, INSERT_DOCUMENT, doc_rows, page_size=5000)
                counts["document"] = cur.rowcount

            # 2. Titles (COPY — identity PK)
            title_rows = []
            for d in batch:
                did = d.get("doc_id")
                if not did:
                    continue
                for t in d.get("titles", []):
                    title_rows.append((
                        did, t["lang"], t["data_format"], t["title_text"],
                    ))
            counts["title"] = _copy_rows(
                cur, "arc_v5.epo_title",
                ["doc_id", "lang", "data_format", "title_text"],
                title_rows)

            # 3. Abstracts (COPY — identity PK)
            abstract_rows = []
            for d in batch:
                did = d.get("doc_id")
                if not did:
                    continue
                for a in d.get("abstracts", []):
                    abstract_rows.append((
                        did, a["lang"], a["data_format"],
                        a["abstract_source"], a["abstract_text"],
                    ))
            counts["abstract"] = _copy_rows(
                cur, "arc_v5.epo_abstract",
                ["doc_id", "lang", "data_format", "abstract_source", "abstract_text"],
                abstract_rows)

            # 4. Application Reference
            appref_rows = []
            for d in batch:
                did = d.get("doc_id")
                if not did or d.get("app_ref") is None:
                    continue
                ar = d["app_ref"]
                appref_rows.append((
                    did,
                    ar.get("docdb_doc_id_attr"), ar.get("docdb_is_representative"),
                    ar["docdb_country"], ar["docdb_doc_number"],
                    ar["docdb_kind"], ar["docdb_date"],
                    ar["epodoc_doc_number"], ar.get("epodoc_kind"), ar["epodoc_date"],
                    ar["original_doc_number"], ar.get("original_kind"),
                    ar["original_date"], ar["original_lang"],
                ))
            if appref_rows:
                execute_values(cur, INSERT_APPLICATION_REF, appref_rows, page_size=5000)
                counts["application_ref"] = cur.rowcount

            # 5. Publication Reference
            pubref_rows = []
            for d in batch:
                did = d.get("doc_id")
                if not did or d.get("pub_ref") is None:
                    continue
                pr = d["pub_ref"]
                pubref_rows.append((
                    did,
                    pr["docdb_country"], pr["docdb_doc_number"],
                    pr["docdb_kind"], pr["docdb_date"],
                    pr["epodoc_doc_number"], pr["epodoc_kind"],
                    pr["epodoc_date"], pr["epodoc_lang"],
                    pr["original_doc_number"], pr["original_kind"],
                    pr["original_date"], pr["original_lang"],
                ))
            if pubref_rows:
                execute_values(cur, INSERT_PUBLICATION_REF, pubref_rows, page_size=5000)
                counts["publication_ref"] = cur.rowcount

            # 6. Classification IPC (old) — COPY
            ipc_rows = []
            for d in batch:
                did = d.get("doc_id")
                if not did:
                    continue
                for c in d.get("ipc_codes", []):
                    ipc_rows.append((
                        did, c["code_type"], c["code_text"],
                        c["edition"], c["additional_info"],
                    ))
            counts["classification_ipc"] = _copy_rows(
                cur, "arc_v5.epo_classification_ipc",
                ["doc_id", "classification_type", "code", "edition", "additional_info"],
                ipc_rows)

            # 7. Classification IPCR — COPY
            ipcr_rows = []
            for d in batch:
                did = d.get("doc_id")
                if not did:
                    continue
                for c in d.get("ipcr_codes", []):
                    parsed = _parse_ipcr_text(c["ipcr_text"])
                    ipcr_rows.append((
                        did, c["sequence"], c["ipcr_text"],
                        parsed.get("ipc_section"), parsed.get("ipc_class"),
                        parsed.get("ipc_subclass"), parsed.get("ipc_main_group"),
                        parsed.get("ipc_subgroup"), parsed.get("ipc_symbol"),
                        parsed.get("ipc_version"), parsed.get("ipc_level"),
                        parsed.get("ipc_position"),
                    ))
            counts["classification_ipcr"] = _copy_rows(
                cur, "arc_v5.epo_classification_ipcr",
                ["doc_id", "sequence", "ipcr_text",
                 "ipc_section", "ipc_class", "ipc_subclass", "ipc_main_group",
                 "ipc_subgroup", "ipc_symbol", "ipc_version", "ipc_level", "ipc_position"],
                ipcr_rows)

            # 8. Patent Classifications (CPC etc) — COPY
            patcls_rows = []
            for d in batch:
                did = d.get("doc_id")
                if not did:
                    continue
                for c in d.get("patent_classifications", []):
                    parsed_cpc = _parse_cpc_symbol(c["classification_symbol"])
                    patcls_rows.append((
                        did, c["classification_symbol"],
                        c["scheme_office"], c["scheme_name"],
                        c["classification_status"], c["classification_value"],
                        c["symbol_position"], c["generating_office"],
                        c["classification_data_source"], c["action_date"],
                        c["sequence"], c["combination_group"], c["combination_rank"],
                        parsed_cpc.get("cpc_section"), parsed_cpc.get("cpc_class"),
                        parsed_cpc.get("cpc_subclass"), parsed_cpc.get("cpc_main_group"),
                        parsed_cpc.get("cpc_subgroup"),
                    ))
            counts["patent_classification"] = _copy_rows(
                cur, "arc_v5.epo_patent_classification",
                ["doc_id", "classification_symbol", "scheme_office", "scheme",
                 "classification_status", "classification_value", "symbol_position",
                 "generating_office", "classification_data_source", "action_date",
                 "sequence", "combination_group", "combination_rank",
                 "cpc_section", "cpc_class", "cpc_subclass", "cpc_main_group", "cpc_subgroup"],
                patcls_rows)

            # 9. Applicants — resolve person, COPY junction
            applicant_rows = []
            for d in batch:
                did = d.get("doc_id")
                if not did:
                    continue
                for app in d.get("applicants", []):
                    pid = person_cache.get_or_create(
                        cur,
                        app.get("name_docdb"),
                        app.get("name_epodoc"),
                        app.get("name_original"),
                        app.get("country"),
                    )
                    applicant_rows.append((
                        did, pid, app.get("sequence"),
                        app.get("address_text"),
                    ))
            counts["document_applicant"] = _copy_rows(
                cur, "arc_v5.epo_document_applicant",
                ["doc_id", "person_id", "sequence", "address_text"],
                applicant_rows)

            # 10. Inventors — resolve person, COPY junction
            inventor_rows = []
            for d in batch:
                did = d.get("doc_id")
                if not did:
                    continue
                for inv in d.get("inventors", []):
                    pid = person_cache.get_or_create(
                        cur,
                        inv.get("name_docdb"),
                        inv.get("name_epodoc"),
                        inv.get("name_original"),
                        inv.get("country"),
                    )
                    inventor_rows.append((
                        did, pid, inv.get("sequence"),
                    ))
            counts["document_inventor"] = _copy_rows(
                cur, "arc_v5.epo_document_inventor",
                ["doc_id", "person_id", "sequence"],
                inventor_rows)

            # 11. Priority Claims — COPY
            prio_rows = []
            for d in batch:
                did = d.get("doc_id")
                if not did:
                    continue
                for p in d.get("priority_claims", []):
                    prio_rows.append((
                        did, p["sequence"],
                        p["docdb_country"], p["docdb_doc_number"],
                        p["docdb_kind"], p["docdb_date"], p["docdb_doc_id_attr"],
                        p["epodoc_doc_number"], p["epodoc_kind"], p["epodoc_date"],
                        p["priority_active_indicator"], p["priority_linkage_type"],
                    ))
            counts["priority_claim"] = _copy_rows(
                cur, "arc_v5.epo_priority_claim",
                ["doc_id", "sequence",
                 "docdb_country", "docdb_doc_number", "docdb_kind", "docdb_date", "docdb_ref_id",
                 "epodoc_doc_number", "epodoc_kind", "epodoc_date",
                 "priority_active_indicator", "priority_linkage_type"],
                prio_rows)

            # 12. Patent Citations — COPY
            cite_pat_rows = []
            for d in batch:
                did = d.get("doc_id")
                if not did:
                    continue
                for c in d.get("citation_patents", []):
                    cite_pat_rows.append((
                        did, c["cited_country"], c["cited_doc_number"],
                        c["cited_kind"], c.get("cited_doc_id"),
                        c["category"], c.get("data_format"),
                        c.get("sequence"),
                    ))
            counts["citation_patent"] = _copy_rows(
                cur, "arc_v5.epo_citation_patent",
                ["doc_id", "cited_country", "cited_doc_number", "cited_kind", "cited_doc_id",
                 "category", "data_format", "sequence"],
                cite_pat_rows)

            # 13. NPL Citations — COPY
            cite_npl_rows = []
            for d in batch:
                did = d.get("doc_id")
                if not did:
                    continue
                for c in d.get("citation_npls", []):
                    cite_npl_rows.append((
                        did, c["npl_text"], c.get("category"),
                        c.get("sequence"),
                    ))
            counts["citation_npl"] = _copy_rows(
                cur, "arc_v5.epo_citation_npl",
                ["doc_id", "npl_text", "category", "sequence"],
                cite_npl_rows)

            # 14. Public Availability
            pubavail_rows = []
            for d in batch:
                did = d.get("doc_id")
                if not did or d.get("pub_availability") is None:
                    continue
                pa = d["pub_availability"]
                pubavail_rows.append((
                    did,
                    pa["printed_with_grant_date"],
                    pa["unexamined_printed_date"],
                    pa["modified_complete_spec_date"],
                    pa["modified_first_page_date"],
                    pa["supplemental_srep_date"],
                    pa.get("examined_not_printed_date"),
                    pa.get("unexamined_not_printed_date"),
                    pa.get("gazette_reference_date"),
                    pa.get("examined_printed_wo_grant_date"),
                    pa.get("abstract_reference_date"),
                ))
            if pubavail_rows:
                execute_values(cur, INSERT_PUB_AVAILABILITY, pubavail_rows, page_size=5000)
                counts["pub_availability"] = cur.rowcount

            # 15. Classification National — COPY
            natcls_rows = []
            for d in batch:
                did = d.get("doc_id")
                if not did:
                    continue
                for txt in d.get("national_codes", []):
                    natcls_rows.append((did, txt))
            counts["classification_national"] = _copy_rows(
                cur, "arc_v5.epo_classification_national",
                ["doc_id", "nat_text"],
                natcls_rows)

            # 16. Designated States — COPY
            ds_rows = []
            for d in batch:
                did = d.get("doc_id")
                if not did:
                    continue
                for ds in d.get("designated_states", []):
                    ds_rows.append((
                        did, ds["state_type"], ds["country_code"],
                    ))
            counts["designated_state"] = _copy_rows(
                cur, "arc_v5.epo_designated_state",
                ["doc_id", "state_type", "country"],
                ds_rows)

            # 17. Related Documents — COPY
            reldoc_rows = []
            for d in batch:
                did = d.get("doc_id")
                if not did:
                    continue
                for rd in d.get("related_documents", []):
                    reldoc_rows.append((
                        did, rd["relation_type"], rd["child_or_parent"],
                        rd["related_country"], rd["related_doc_number"],
                        rd["related_kind"], rd["related_date"],
                    ))
            counts["related_document"] = _copy_rows(
                cur, "arc_v5.epo_related_document",
                ["doc_id", "relation_type", "parent_child",
                 "related_country", "related_doc_number", "related_kind", "related_date"],
                reldoc_rows)

        conn.commit()
        person_cache.commit_batch()

    except Exception:
        conn.rollback()
        person_cache.rollback_batch()
        raise

    return counts


# ── ZIP Ingestion ─────────────────────────────────────────────────────────────

def ingest_zip(conn, zip_path: str, person_cache: PersonCache,
               error_log: Path | None = None,
               delete_after: bool = False,
               progress_file: Path | None = None) -> dict:
    """Ingest one outer ZIP into normalized epo_* tables. Returns stats dict."""
    t0 = time.time()
    total_seen = 0
    batch_count = 0
    errors = 0
    error_samples = []
    batch: list[dict] = []
    cumulative_counts: dict[str, int] = {}
    parse_errors: list = []

    zip_name = Path(zip_path).name
    log_fh = open(error_log, "a") if error_log else None
    print(f"Processing {zip_name} ...")

    def _record_error(err_dict: dict):
        nonlocal errors
        errors += 1
        if len(error_samples) < 20:
            error_samples.append(err_dict)
        if log_fh:
            log_fh.write(json.dumps(err_dict, default=str) + "\n")
            log_fh.flush()

    def _log_progress():
        if progress_file:
            elapsed = time.time() - t0
            rate = total_seen / elapsed if elapsed > 0 else 0
            doc_count = cumulative_counts.get("document", 0)
            with open(progress_file, "a") as pf:
                pf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                         f"{zip_name}: {total_seen:,} seen, {doc_count:,} docs inserted, "
                         f"{errors:,} errors, {rate:,.0f} docs/s\n")

    for doc, inner_path in iter_epo_zip(zip_path, error_sink=parse_errors):
        # Drain parse errors
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
                "inner_zip": inner_path,
            })
            continue

        doc["zip_source"] = zip_name
        batch.append(doc)

        if len(batch) >= BATCH_SIZE:
            try:
                batch_counts = flush_batch(conn, batch, person_cache)
                for k, v in batch_counts.items():
                    cumulative_counts[k] = cumulative_counts.get(k, 0) + v
                batch_count += 1
            except Exception as exc:
                _record_error({
                    "type": "batch_insert_error",
                    "batch_number": batch_count,
                    "batch_size": len(batch),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
            batch = []

            # Progress: log to file every batch (5K docs), print every 50K
            _log_progress()
            if total_seen % 50000 == 0:
                elapsed = time.time() - t0
                rate = total_seen / elapsed
                doc_ins = cumulative_counts.get("document", 0)
                print(f"  {total_seen:>9,} seen  {doc_ins:>9,} docs  "
                      f"{errors:>5,} errors  {rate:,.0f} docs/s  "
                      f"persons: {person_cache.stats['inserts']:,} new / "
                      f"{person_cache.stats['hits']:,} cached")

    # Drain remaining parse errors
    for pe in parse_errors:
        _record_error(pe)

    # Flush final batch
    if batch:
        try:
            batch_counts = flush_batch(conn, batch, person_cache)
            for k, v in batch_counts.items():
                cumulative_counts[k] = cumulative_counts.get(k, 0) + v
            batch_count += 1
        except Exception as exc:
            _record_error({
                "type": "batch_insert_error",
                "batch_number": batch_count,
                "batch_size": len(batch),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

    if log_fh:
        log_fh.close()

    elapsed = time.time() - t0
    doc_ins = cumulative_counts.get("document", 0)
    print(f"  Done: {total_seen:,} seen  {doc_ins:,} docs inserted  "
          f"{errors:,} errors  {elapsed:.1f}s")
    print(f"  Persons: {person_cache.stats['inserts']:,} new / "
          f"{person_cache.stats['hits']:,} cached")
    if cumulative_counts:
        print(f"  Table counts: {json.dumps(cumulative_counts, indent=None)}")
    if error_log and errors:
        print(f"  Error log: {error_log}")

    if progress_file:
        with open(progress_file, "a") as pf:
            pf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                     f"COMPLETED {zip_name}: {total_seen:,} seen, "
                     f"{doc_ins:,} docs, {errors:,} errors, {elapsed:.1f}s\n")

    if delete_after and total_seen > 0:
        try:
            Path(zip_path).unlink()
            print(f"  Deleted {zip_name}")
        except OSError as e:
            print(f"  WARNING: could not delete {zip_path}: {e}")

    return {
        "total": total_seen,
        "counts": cumulative_counts,
        "errors": errors,
        "error_samples": error_samples,
        "elapsed": elapsed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Ingest DOCDB XML into normalized epo_* tables (arc_v5)")
    ap.add_argument("--zip", help="Single outer ZIP file to ingest")
    ap.add_argument("--dir", help="Directory of outer ZIP files")
    ap.add_argument("--delete-after", action="store_true",
                    help="Delete each ZIP after successful ingest to free disk space")
    ap.add_argument("--error-log", default=None,
                    help="JSONL file to append error records during ingest")
    ap.add_argument("--progress-file",
                    default="/tmp/epo_ingest_progress.txt",
                    help="File to log progress updates "
                         "(default: /tmp/epo_ingest_progress.txt)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse only, no DB writes. Reports counts per ZIP.")
    args = ap.parse_args()

    if not HAVE_LXML:
        print("ERROR: lxml is required. Install with: pip install lxml")
        sys.exit(1)

    # Gather ZIP files
    zips: list[str] = []
    if args.zip:
        zips = [args.zip]
    elif args.dir:
        zips = sorted(str(p) for p in Path(args.dir).glob("docdb_xml_bck_*.zip"))
    else:
        ap.print_help()
        sys.exit(0)

    if not zips:
        print("No ZIP files found.")
        sys.exit(1)

    print(f"Files to process: {len(zips)}")

    if args.dry_run:
        _run_dry(zips)
        return

    conn = psycopg2.connect(**DB_PARAMS)
    person_cache = PersonCache()

    error_log_path = Path(args.error_log) if args.error_log else None
    progress_file = Path(args.progress_file) if args.progress_file else None

    grand_total = 0
    grand_counts: dict[str, int] = {}
    grand_errors = 0
    t_all = time.time()

    if progress_file:
        with open(progress_file, "a") as pf:
            pf.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                     f"=== EPO INGEST SESSION START === Files: {len(zips)}\n")

    for zip_path in zips:
        stats = ingest_zip(
            conn, zip_path, person_cache,
            error_log=error_log_path,
            delete_after=args.delete_after,
            progress_file=progress_file,
        )
        grand_total += stats["total"]
        grand_errors += stats["errors"]
        for k, v in stats["counts"].items():
            grand_counts[k] = grand_counts.get(k, 0) + v

    elapsed_all = time.time() - t_all
    doc_ins = grand_counts.get("document", 0)

    summary = (f"\nAll done: {grand_total:,} total seen  {doc_ins:,} docs inserted  "
               f"{grand_errors:,} errors  {elapsed_all:.1f}s")
    print(summary)
    print(f"Persons: {person_cache.stats['inserts']:,} new / "
          f"{person_cache.stats['hits']:,} cached")
    print(f"Per-table totals: {json.dumps(grand_counts, indent=2)}")

    if progress_file:
        with open(progress_file, "a") as pf:
            pf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                     f"=== SESSION COMPLETE === {grand_total:,} seen, "
                     f"{doc_ins:,} docs, {grand_errors:,} errors, "
                     f"{elapsed_all:.1f}s\n")

    conn.close()


def _run_dry(zips: list[str]):
    """Dry-run mode: parse ZIPs and report statistics without DB writes."""
    grand_total = 0
    t_all = time.time()

    for zip_path in zips:
        t0 = time.time()
        n = 0
        table_counts: dict[str, int] = {}
        countries: dict[str, int] = {}
        parse_errors: list = []

        for doc, _inner in iter_epo_zip(zip_path, error_sink=parse_errors):
            for pe in parse_errors:
                pass  # count only
            parse_errors.clear()

            n += 1
            c = doc.get("country", "??")
            countries[c] = countries.get(c, 0) + 1

            # Count child rows
            for key in ("titles", "abstracts", "ipc_codes", "ipcr_codes",
                        "patent_classifications", "applicants", "inventors",
                        "priority_claims", "citation_patents", "citation_npls",
                        "national_codes", "designated_states", "related_documents"):
                items = doc.get(key, [])
                if items:
                    table_counts[key] = table_counts.get(key, 0) + len(items)
            if doc.get("app_ref"):
                table_counts["app_ref"] = table_counts.get("app_ref", 0) + 1
            if doc.get("pub_ref"):
                table_counts["pub_ref"] = table_counts.get("pub_ref", 0) + 1
            if doc.get("pub_availability"):
                table_counts["pub_availability"] = table_counts.get("pub_availability", 0) + 1

        elapsed = time.time() - t0
        print(f"\n{Path(zip_path).name}")
        print(f"  Documents: {n:,}  elapsed: {elapsed:.1f}s")
        print(f"  Child row counts: {json.dumps(table_counts, indent=None)}")
        top_countries = dict(sorted(countries.items(), key=lambda x: -x[1])[:10])
        print(f"  Top countries: {top_countries}")
        grand_total += n

    elapsed_all = time.time() - t_all
    print(f"\nTotal: {grand_total:,} documents across {len(zips)} ZIP(s)  "
          f"({elapsed_all:.1f}s)")


if __name__ == "__main__":
    main()

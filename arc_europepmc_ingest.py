#!/usr/bin/env python3
"""
arc_europepmc_ingest.py — Ingest Europe PMC OA full text into europepmc.* star schema.

Files: /data/downloads/europepmc/PMC*_PMC*.xml.gz (1,309 files, 164 GB)
Format: gzip XML, <articles> → <article> elements (JATS 1.4)
~16.9M articles with abstract + full text + references

Usage:
  python3 arc_europepmc_ingest.py --dir /data/downloads/europepmc
  python3 arc_europepmc_ingest.py --dir /data/downloads/europepmc --file PMC10000003
  python3 arc_europepmc_ingest.py --set-logged
  python3 arc_europepmc_ingest.py --create-indexes

Run as background:
  nohup python3 arc_europepmc_ingest.py --dir /data/downloads/europepmc >> /tmp/europepmc_ingest.log 2>&1 &
"""

import argparse
import gzip
import io
import json
import os
import re
import sys
import tarfile
import time
import traceback
from datetime import date, datetime
from pathlib import Path

import psycopg2
import psycopg2.extras
from lxml import etree

# ── Config ─────────────────────────────────────────────────────────────────────

DB_PARAMS = dict(
    host="/var/run/postgresql",
    dbname="arc_v5",
    user=os.environ.get("PGUSER", "arc"),
)

SCHEMA = "europepmc"
BATCH_SIZE = 5000
PROGRESS_INTERVAL = 10000
DOWNLOAD_DIR = Path("/data/downloads/europepmc")
ABORT_FLAG = Path("/tmp/ARC_ABORT_FLAG")

# ── Entity stripping ─────────────────────────────────────────────────────────

_DOCTYPE_RE = re.compile(rb'<!DOCTYPE\s[^[>]*(?:\[[^\]]*\])?\s*>', re.DOTALL)
_ENTITY_RE  = re.compile(rb'&(?!amp;|lt;|gt;|apos;|quot;|#)[a-zA-Z][a-zA-Z0-9]*;')

def _strip_entities(xml_bytes: bytes) -> bytes:
    xml_bytes = _DOCTYPE_RE.sub(b"", xml_bytes, count=1)
    xml_bytes = _ENTITY_RE.sub(b"", xml_bytes)
    return xml_bytes


# ── XML helpers ──────────────────────────────────────────────────────────────

def _text(el):
    if el is None:
        return None
    t = "".join(el.itertext()).strip()
    return t if t else None

def _attr(el, name, default=None):
    if el is None:
        return default
    return el.get(name, default)

def _strip_ns(tag):
    """Remove namespace prefix from a tag."""
    if not isinstance(tag, str):
        return None
    return tag.split("}", 1)[1] if "}" in tag else tag

def _find(el, path):
    """Find element, trying with and without namespace."""
    if el is None:
        return None
    r = el.find(path)
    if r is not None:
        return r
    # Try with wildcard namespace
    parts = path.split("/")
    ns_path = "/".join(f"{{{el.nsmap.get(None, '')}}}{{}}".format(p) if el.nsmap else p for p in parts)
    return el.find(ns_path)


# ── Article extraction ────────────────────────────────────────────────────────

def extract_article(article_el, source="oa_fulltext", source_file=None):
    """Extract one JATS <article> into a dict of table rows.

    source_file: name of the tar.gz / xml.gz file this article came from (for audit).
    """

    meta = article_el.find(".//article-meta")
    jmeta = article_el.find(".//journal-meta")
    if meta is None:
        return None

    # ── IDs ──
    pmcid = None
    pmid = None
    doi = None
    article_ids = []

    for aid in meta.findall("article-id"):
        id_type = aid.get("pub-id-type", "")
        val = (aid.text or "").strip()
        if not val:
            continue
        if id_type == "pmc":
            pmcid = val if val.startswith("PMC") else f"PMC{val}"
        elif id_type == "pmid":
            pmid = val
        elif id_type == "doi":
            doi = val
        article_ids.append({"id_type": id_type, "id_value": val})

    if not pmcid:
        # Try from article-id without type
        for aid in meta.findall("article-id"):
            val = (aid.text or "").strip()
            if val and val.startswith("PMC"):
                pmcid = val
                break

    if not pmcid:
        return None

    doc_id = f"PMC:{pmcid}"

    # Add doc_id + source_file to article_ids
    for aid in article_ids:
        aid["doc_id"] = doc_id
        aid["source_file"] = source_file

    # ── Journal info ──
    journal_title = None
    journal_iso = None
    journal_pmc_id = None
    journal_id_iso_abbrev = None
    journal_id_publisher = None
    publisher_name = None
    publisher_loc = None
    issn_ppub = None
    issn_epub = None

    if jmeta is not None:
        journal_title = _text(jmeta.find("journal-title-group/journal-title"))
        if journal_title is None:
            journal_title = _text(jmeta.find("journal-title"))
        for jid in jmeta.findall("journal-id"):
            jtype = jid.get("journal-id-type", "")
            jval = (jid.text or "").strip()
            if jtype == "nlm-ta":
                journal_iso = jval
            elif jtype == "iso-abbrev":
                journal_id_iso_abbrev = jval
            elif jtype == "publisher-id":
                journal_id_publisher = jval
            elif jtype in ("pmc", "pmc-domain-id"):
                journal_pmc_id = jval
        for issn_el in jmeta.findall("issn"):
            ival = (issn_el.text or "").strip()
            pt = issn_el.get("pub-type", "")
            if pt == "ppub":
                issn_ppub = ival
            elif pt == "epub":
                issn_epub = ival
            elif issn_ppub is None and pt == "":
                issn_ppub = ival
        pub = jmeta.find("publisher/publisher-name")
        publisher_name = _text(pub)
        ploc = jmeta.find("publisher/publisher-loc")
        publisher_loc = _text(ploc)

    # ── Restricted-by (from processing-meta) ──
    restricted_by = _text(article_el.find(".//processing-meta/restricted-by"))

    # ── Title ──
    title_group = meta.find("title-group")
    article_title = _text(title_group.find("article-title")) if title_group is not None else None

    # ── Pub dates (capture all types: epub, ppub, collection; first also fills display fields) ──
    pub_year = None
    pub_month = None
    pub_day = None
    pub_type = None
    pub_date_epub = None
    pub_date_ppub = None
    pub_date_collection = None
    for pd in meta.findall("pub-date"):
        pd_type = pd.get("pub-type") or pd.get("date-type")
        y = _text(pd.find("year"))
        if not y:
            continue
        m = _text(pd.find("month"))
        d = _text(pd.find("day"))
        try:
            yi = int(y)
            mi = int(m) if m and m.isdigit() else 1
            di = int(d) if d and d.isdigit() else 1
            dt = date(yi, mi, di)
        except (ValueError, TypeError):
            yi = None
            dt = None
        # First non-empty date fills display fields
        if pub_year is None and yi is not None:
            pub_year = yi
            pub_month = m
            pub_day = d
            pub_type = pd_type
        if pd_type == "epub":
            pub_date_epub = dt
        elif pd_type == "ppub":
            pub_date_ppub = dt
        elif pd_type == "collection":
            pub_date_collection = dt

    # ── License ──
    license_el = meta.find(".//license")
    license_type = None
    license_url = None
    if license_el is not None:
        license_type = license_el.get("license-type")
        link = license_el.find(".//ext-link")
        if link is not None:
            license_url = link.get("{http://www.w3.org/1999/xlink}href") or _text(link)

    # ── Copyright ──
    perms = meta.find("permissions")
    copyright_statement = _text(perms.find("copyright-statement")) if perms is not None else None
    copyright_holder = _text(perms.find("copyright-holder")) if perms is not None else None
    copyright_year = _text(perms.find("copyright-year")) if perms is not None else None

    # ── Elocation ID ──
    elocation_id = _text(meta.find("elocation-id"))

    # ── Counts block (word/fig/table/ref/page/equation) ──
    word_count = None
    fig_count = None
    table_count = None
    ref_count = None
    page_count = None
    equation_count = None
    counts_el = meta.find("counts")
    if counts_el is not None:
        def _count(name):
            el = counts_el.find(name)
            if el is not None:
                v = el.get("count")
                if v and v.isdigit():
                    return int(v)
            return None
        word_count = _count("word-count")
        fig_count = _count("fig-count")
        table_count = _count("table-count")
        ref_count = _count("ref-count")
        page_count = _count("page-count")
        equation_count = _count("equation-count")

    doc_row = {
        "doc_id": doc_id,
        "pmcid": pmcid,
        "pmid": pmid,
        "doi": doi,
        "article_title": article_title,
        "article_type": article_el.get("article-type"),
        "dtd_version": article_el.get("dtd-version"),
        "language": article_el.get("{http://www.w3.org/XML/1998/namespace}lang"),
        "journal_title": journal_title,
        "journal_iso": journal_iso,
        "journal_pmc_id": journal_pmc_id,
        "journal_id_iso_abbrev": journal_id_iso_abbrev,
        "journal_id_publisher": journal_id_publisher,
        "issn_ppub": issn_ppub,
        "issn_epub": issn_epub,
        "publisher_name": publisher_name,
        "publisher_loc": publisher_loc,
        "pub_year": pub_year,
        "pub_month": pub_month,
        "pub_day": pub_day,
        "pub_type": pub_type,
        "pub_date_epub": pub_date_epub,
        "pub_date_ppub": pub_date_ppub,
        "pub_date_collection": pub_date_collection,
        "volume": _text(meta.find("volume")),
        "issue": _text(meta.find("issue")),
        "fpage": _text(meta.find("fpage")),
        "lpage": _text(meta.find("lpage")),
        "elocation_id": elocation_id,
        "article_version": _text(meta.find("article-version")),
        "license_type": license_type,
        "license_url": license_url,
        "copyright_statement": copyright_statement,
        "copyright_holder": copyright_holder,
        "copyright_year": copyright_year,
        "restricted_by": restricted_by,
        "word_count": word_count,
        "fig_count": fig_count,
        "table_count": table_count,
        "ref_count": ref_count,
        "page_count": page_count,
        "equation_count": equation_count,
        "source": source,
        "source_file": source_file,
    }

    # ── Abstract ──
    abstracts = []
    abstract_sections = []
    for abs_el in meta.findall("abstract"):
        full_text = _text(abs_el)
        if full_text:
            abstracts.append({
                "doc_id": doc_id,
                "abstract_type": abs_el.get("abstract-type"),
                "abstract_text": full_text,
                "source_file": source_file,
            })
        # Structured sections
        for seq, sec in enumerate(abs_el.findall("sec"), 1):
            sec_text = _text(sec)
            if sec_text:
                title_el = sec.find("title")
                abstract_sections.append({
                    "doc_id": doc_id,
                    "section_title": _text(title_el),
                    "section_text": sec_text,
                    "sequence": seq,
                    "source_file": source_file,
                })

    # ── Translated titles ──
    trans_titles = []
    for tt_group in meta.findall("trans-title-group"):
        lang = tt_group.get("{http://www.w3.org/XML/1998/namespace}lang")
        for tt in tt_group.findall("trans-title"):
            ttext = _text(tt)
            if ttext:
                trans_titles.append({
                    "doc_id": doc_id,
                    "language": lang,
                    "trans_title": ttext,
                    "source_file": source_file,
                })

    # ── Translated abstracts ──
    trans_abstracts = []
    for ta in meta.findall("trans-abstract"):
        lang = ta.get("{http://www.w3.org/XML/1998/namespace}lang")
        ttext = _text(ta)
        if ttext:
            trans_abstracts.append({
                "doc_id": doc_id,
                "language": lang,
                "abstract_text": ttext,
                "source_file": source_file,
            })

    # ── Authors ──
    authors = []
    author_affiliations = []
    # Build aff lookup from article-meta/aff elements: id -> (text, country)
    aff_map = {}
    for aff in meta.findall("aff"):
        aff_id = aff.get("id")
        aff_text = _text(aff)
        aff_country = _text(aff.find("country"))
        if aff_id and aff_text:
            aff_map[aff_id] = (aff_text, aff_country)

    for contrib_group in meta.findall("contrib-group"):
        # Also check affs inside contrib-group
        for aff in contrib_group.findall("aff"):
            aff_id = aff.get("id")
            aff_text = _text(aff)
            aff_country = _text(aff.find("country"))
            if aff_id and aff_text:
                aff_map[aff_id] = (aff_text, aff_country)

        for seq, contrib in enumerate(contrib_group.findall("contrib"), 1):
            ctype = contrib.get("contrib-type")
            name_el = contrib.find("name")
            collab_el = contrib.find("collab")

            if collab_el is not None:
                # Collaboration/consortium entry — no individual name
                authors.append({
                    "doc_id": doc_id,
                    "surname": None,
                    "given_names": None,
                    "initials": None,
                    "prefix": None,
                    "suffix": None,
                    "contrib_type": ctype,
                    "orcid": None,
                    "email": None,
                    "role": _text(contrib.find("role")),
                    "bio": _text(contrib.find("bio")),
                    "is_collab": True,
                    "collab_name": _text(collab_el),
                    "sequence": seq,
                    "source_file": source_file,
                })
            else:
                surname = _text(name_el.find("surname")) if name_el is not None else None
                given = _text(name_el.find("given-names")) if name_el is not None else None
                initials = _text(name_el.find("initials")) if name_el is not None else None
                prefix = _text(name_el.find("prefix")) if name_el is not None else None
                suffix = _text(name_el.find("suffix")) if name_el is not None else None

                # ORCID
                orcid = None
                for cid in contrib.findall("contrib-id"):
                    if cid.get("contrib-id-type") == "orcid":
                        orcid = _text(cid)

                # Email
                email = _text(contrib.find("email"))
                role = _text(contrib.find("role"))
                bio = _text(contrib.find("bio"))

                authors.append({
                    "doc_id": doc_id,
                    "surname": surname,
                    "given_names": given,
                    "initials": initials,
                    "prefix": prefix,
                    "suffix": suffix,
                    "contrib_type": ctype,
                    "orcid": orcid,
                    "email": email,
                    "role": role,
                    "bio": bio,
                    "is_collab": False,
                    "collab_name": None,
                    "sequence": seq,
                    "source_file": source_file,
                })

            # Affiliations via xref[@ref-type='aff']
            for xref in contrib.findall("xref"):
                if xref.get("ref-type") == "aff":
                    rid = xref.get("rid")
                    if rid and rid in aff_map:
                        aff_text, aff_country = aff_map[rid]
                        author_affiliations.append({
                            "doc_id": doc_id,
                            "author_sequence": seq,
                            "affiliation_text": aff_text,
                            "affiliation_id": rid,
                            "country": aff_country,
                            "source_file": source_file,
                        })

            # Inline aff
            for aff in contrib.findall("aff"):
                aff_text = _text(aff)
                aff_country = _text(aff.find("country"))
                if aff_text:
                    author_affiliations.append({
                        "doc_id": doc_id,
                        "author_sequence": seq,
                        "affiliation_text": aff_text,
                        "affiliation_id": aff.get("id"),
                        "country": aff_country,
                        "source_file": source_file,
                    })

    # ── Corresponding authors ──
    corresponding_authors = []
    author_notes = meta.find("author-notes")
    if author_notes is not None:
        for corresp in author_notes.findall("corresp"):
            full_text = _text(corresp)
            email_el = corresp.find("email")
            email = _text(email_el)
            corresponding_authors.append({
                "doc_id": doc_id,
                "corresp_id": corresp.get("id"),
                "label": _text(corresp.find("label")),
                "full_text": full_text,
                "email": email,
                "source_file": source_file,
            })

    # ── Footnotes (article-level fn-group / fn) ──
    footnotes = []
    for fn_group in meta.findall(".//fn-group"):
        for fn_seq, fn in enumerate(fn_group.findall("fn"), 1):
            fn_text = _text(fn)
            if fn_text:
                footnotes.append({
                    "doc_id": doc_id,
                    "fn_id": fn.get("id"),
                    "fn_type": fn.get("fn-type"),
                    "fn_text": fn_text[:10000],
                    "sequence": fn_seq,
                    "source_file": source_file,
                })

    # ── Keywords ──
    keywords = []
    for kg in meta.findall("kwd-group"):
        kg_type = kg.get("kwd-group-type")
        for kw in kg.findall("kwd"):
            t = _text(kw)
            if t:
                keywords.append({
                    "doc_id": doc_id,
                    "keyword": t,
                    "kwd_group_type": kg_type,
                    "source_file": source_file,
                })

    # ── References ──
    references = []
    back = article_el.find(".//back")
    if back is not None:
        for seq, ref in enumerate(back.findall(".//ref"), 1):
            # Try mixed-citation first, then element-citation
            cite = ref.find("mixed-citation")
            if cite is None:
                cite = ref.find("element-citation")

            citation_text = _text(cite) if cite is not None else _text(ref)
            cited_pmid = None
            cited_pmcid = None
            cited_doi = None
            cited_year = None
            cited_source = None
            cited_title = None

            if cite is not None:
                for pid in cite.findall("pub-id"):
                    ptype = pid.get("pub-id-type", "")
                    pval = (pid.text or "").strip()
                    if ptype == "pmid":
                        cited_pmid = pval
                    elif ptype == "pmcid":
                        cited_pmcid = pval
                    elif ptype == "doi":
                        cited_doi = pval
                cited_year = _text(cite.find("year"))
                cited_source = _text(cite.find("source"))
                cited_title = _text(cite.find("article-title"))

            references.append({
                "doc_id": doc_id,
                "ref_id": ref.get("id"),
                "label": _text(ref.find("label")),
                "citation_text": citation_text[:5000] if citation_text else None,
                "cited_pmid": cited_pmid,
                "cited_pmcid": cited_pmcid,
                "cited_doi": cited_doi,
                "cited_year": cited_year,
                "cited_source": cited_source,
                "cited_article_title": cited_title,
                "sequence": seq,
                "source_file": source_file,
            })

    # ── Body sections ──
    # NOTE: body_fulltext table was renamed to x_delete_body_fulltext on 2026-04-08;
    # body_section is now the canonical store. No aggregate blob extracted.
    body_sections = []
    body_el = article_el.find(".//body")
    if body_el is not None:
        # Extract sections
        seq = [0]
        def _walk_sections(parent, level):
            for child in parent:
                tag = _strip_ns(child.tag)
                if tag == "sec":
                    seq[0] += 1
                    title_el = child.find("title")
                    sec_text = _text(child)
                    if sec_text:
                        body_sections.append({
                            "doc_id": doc_id,
                            "section_title": _text(title_el),
                            "section_text": sec_text[:50000],
                            "section_level": level,
                            "section_type": child.get("sec-type"),
                            "sequence": seq[0],
                            "source_file": source_file,
                        })
                    _walk_sections(child, level + 1)
        _walk_sections(body_el, 1)

    # ── Article categories ──
    categories = []
    for sg in meta.findall("article-categories/subj-group"):
        sg_type = sg.get("subj-group-type")
        for subj in sg.findall("subject"):
            t = _text(subj)
            if t:
                categories.append({
                    "doc_id": doc_id,
                    "subject_group_type": sg_type,
                    "subject": t,
                    "source_file": source_file,
                })

    # ── Pub history ──
    pub_history = []
    hist = meta.find("history")
    if hist is not None:
        for d in hist.findall("date"):
            y = _text(d.find("year"))
            m = _text(d.find("month"))
            day = _text(d.find("day"))
            if y:
                try:
                    dt = date(int(y), int(m) if m and m.isdigit() else 1,
                              int(day) if day and day.isdigit() else 1)
                    pub_history.append({
                        "doc_id": doc_id,
                        "event_type": d.get("date-type"),
                        "event_date": dt,
                        "source_file": source_file,
                    })
                except (ValueError, TypeError):
                    pass

    # ── Custom meta ──
    custom_meta = []
    for cm in meta.findall(".//custom-meta-group/custom-meta"):
        name = _text(cm.find("meta-name"))
        val = _text(cm.find("meta-value"))
        if name:
            custom_meta.append({
                "doc_id": doc_id,
                "meta_name": name,
                "meta_value": val,
                "source_file": source_file,
            })

    # ── Funding (with structured funder IDs) ──
    funding = []
    for fg in meta.findall(".//funding-group/award-group"):
        inst_wrap = fg.find(".//institution-wrap")
        if inst_wrap is not None:
            funder = _text(inst_wrap.find("institution"))
            funder_id_el = inst_wrap.find("institution-id")
            funder_id_val = _text(funder_id_el)
            funder_id_type = funder_id_el.get("institution-id-type") if funder_id_el is not None else None
        else:
            funder = _text(fg.find(".//institution"))
            funder_id_val = None
            funder_id_type = None
        award = _text(fg.find("award-id"))
        country = _text(fg.find(".//country"))
        if funder or award or funder_id_val:
            funding.append({
                "doc_id": doc_id,
                "funder_name": funder,
                "award_id": award,
                "award_group": fg.get("id"),
                "funder_id": funder_id_val,
                "funder_id_type": funder_id_type,
                "country": country,
                "source_file": source_file,
            })

    # ── Acknowledgments ──
    ack_rows = []
    ack = article_el.find(".//back/ack")
    if ack is not None:
        ack_text = _text(ack)
        if ack_text:
            ack_rows.append({
                "doc_id": doc_id,
                "ack_text": ack_text[:10000],
                "source_file": source_file,
            })

    # ── Supplementary material ──
    supp_materials = []
    for sm in article_el.findall(".//supplementary-material"):
        supp_materials.append({
            "doc_id": doc_id,
            "supp_id": sm.get("id"),
            "label": _text(sm.find("label")),
            "caption": _text(sm.find("caption")),
            "mimetype": sm.get("mimetype"),
            "media_type": sm.get("mime-subtype"),
            "href": sm.get("{http://www.w3.org/1999/xlink}href"),
            "source_file": source_file,
        })

    return {
        "document": doc_row,
        "abstract": abstracts,
        "abstract_section": abstract_sections,
        "author": authors,
        "author_affiliation": author_affiliations,
        "keyword": keywords,
        "reference": references,
        "body_section": body_sections,
        "article_category": categories,
        "article_id": article_ids,
        "pub_history": pub_history,
        "custom_meta": custom_meta,
        "funding": funding,
        "acknowledgment": ack_rows,
        "corresponding_author": corresponding_authors,
        "footnote": footnotes,
        "trans_title": trans_titles,
        "trans_abstract": trans_abstracts,
        "supplementary_material": supp_materials,
    }


# ── SQL ───────────────────────────────────────────────────────────────────────

DOC_COLUMNS = [
    "doc_id", "pmcid", "pmid", "doi", "article_title", "article_type",
    "dtd_version", "language",
    "journal_title", "journal_iso", "journal_pmc_id",
    "journal_id_iso_abbrev", "journal_id_publisher",
    "issn_ppub", "issn_epub",
    "publisher_name", "publisher_loc",
    "pub_year", "pub_month", "pub_day", "pub_type",
    "pub_date_epub", "pub_date_ppub", "pub_date_collection",
    "volume", "issue", "fpage", "lpage", "elocation_id", "article_version",
    "license_type", "license_url",
    "copyright_statement", "copyright_holder", "copyright_year",
    "restricted_by",
    "word_count", "fig_count", "table_count", "ref_count", "page_count", "equation_count",
    "source", "source_file",
]

DOC_INSERT = (
    f"INSERT INTO {SCHEMA}.document ({', '.join(DOC_COLUMNS)}) "
    f"VALUES %s ON CONFLICT (doc_id) DO NOTHING"
)

DIM_TABLES = {
    "abstract": ["doc_id", "abstract_type", "abstract_text", "source_file"],
    "abstract_section": ["doc_id", "section_title", "section_text", "sequence", "source_file"],
    "author": ["doc_id", "surname", "given_names", "initials", "prefix", "suffix",
               "contrib_type", "orcid", "email", "role", "bio",
               "is_collab", "collab_name", "sequence", "source_file"],
    "author_affiliation": ["doc_id", "author_sequence", "affiliation_text",
                           "affiliation_id", "country", "source_file"],
    "keyword": ["doc_id", "keyword", "kwd_group_type", "source_file"],
    "reference": ["doc_id", "ref_id", "label", "citation_text", "cited_pmid",
                   "cited_pmcid", "cited_doi", "cited_year", "cited_source",
                   "cited_article_title", "sequence", "source_file"],
    "body_section": ["doc_id", "section_title", "section_text", "section_level",
                     "section_type", "sequence", "source_file"],
    "article_category": ["doc_id", "subject_group_type", "subject", "source_file"],
    "article_id": ["doc_id", "id_type", "id_value", "source_file"],
    "pub_history": ["doc_id", "event_type", "event_date", "source_file"],
    "custom_meta": ["doc_id", "meta_name", "meta_value", "source_file"],
    "funding": ["doc_id", "funder_name", "award_id", "award_group",
                "funder_id", "funder_id_type", "country", "source_file"],
    "acknowledgment": ["doc_id", "ack_text", "source_file"],
    # NEW tables added 2026-04-08:
    "corresponding_author": ["doc_id", "corresp_id", "label", "full_text",
                             "email", "source_file"],
    "footnote": ["doc_id", "fn_id", "fn_type", "fn_text", "sequence", "source_file"],
    "trans_title": ["doc_id", "language", "trans_title", "source_file"],
    "trans_abstract": ["doc_id", "language", "abstract_text", "source_file"],
    "supplementary_material": ["doc_id", "supp_id", "label", "caption",
                               "mimetype", "media_type", "href", "source_file"],
}


def flush_batches(conn, batches):
    with conn.cursor() as cur:
        doc_rows = batches.get("document", [])
        if doc_rows:
            rows = [tuple(d[c] for c in DOC_COLUMNS) for d in doc_rows]
            psycopg2.extras.execute_values(cur, DOC_INSERT, rows, page_size=BATCH_SIZE)

        for table_name, columns in DIM_TABLES.items():
            dim_rows = batches.get(table_name, [])
            if dim_rows:
                cols = ", ".join(columns)
                sql = f"INSERT INTO {SCHEMA}.{table_name} ({cols}) VALUES %s"
                rows = [tuple(d[c] for c in columns) for d in dim_rows]
                psycopg2.extras.execute_values(cur, sql, rows, page_size=BATCH_SIZE)
    conn.commit()


# ── File iteration ────────────────────────────────────────────────────────────

def iter_articles_from_file(gz_path: Path):
    """Parse gzipped XML, yield <article> elements."""
    with gzip.open(gz_path, "rb") as f:
        raw = f.read()
    raw = _strip_entities(raw)
    parser = etree.XMLParser(huge_tree=True, recover=True)
    tree = etree.fromstring(raw, parser=parser)
    if tree.tag == "articles":
        yield from tree.findall("article")
    elif tree.tag == "article":
        yield tree
    else:
        for a in tree.findall(".//article"):
            yield a


# ── Progress ──────────────────────────────────────────────────────────────────

def check_done(conn, file_name):
    with conn.cursor() as cur:
        cur.execute(f"SELECT status FROM {SCHEMA}.ingest_progress WHERE file_name = %s", (file_name,))
        row = cur.fetchone()
        return row and row[0] == "done"

def mark_status(conn, file_name, status, **kwargs):
    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {SCHEMA}.ingest_progress (file_name, status, started_at)
            VALUES (%s, %s, now())
            ON CONFLICT (file_name) DO UPDATE SET
                status = EXCLUDED.status,
                articles_seen = COALESCE(%s, {SCHEMA}.ingest_progress.articles_seen),
                articles_inserted = COALESCE(%s, {SCHEMA}.ingest_progress.articles_inserted),
                errors = COALESCE(%s, {SCHEMA}.ingest_progress.errors),
                completed_at = CASE WHEN EXCLUDED.status IN ('done','error') THEN now() ELSE NULL END,
                error_detail = %s
        """, (file_name, status,
              kwargs.get("articles_seen"), kwargs.get("articles_inserted"),
              kwargs.get("errors"), kwargs.get("error_detail")))
    conn.commit()


# ── Ingest one file ──────────────────────────────────────────────────────────

def ingest_file(conn, gz_path: Path, source="oa_fulltext",
                error_log=None, progress_file=None):
    file_name = gz_path.name
    t0 = time.time()
    total = 0
    inserted = 0
    errors = 0
    error_fh = open(error_log, "a") if error_log else None

    all_tables = ["document"] + list(DIM_TABLES.keys())
    batches = {name: [] for name in all_tables}

    mark_status(conn, file_name, "processing")
    print(f"Processing {file_name} ...", end=" ", flush=True)

    try:
        articles = list(iter_articles_from_file(gz_path))
    except Exception as e:
        print(f"PARSE ERROR: {e}")
        mark_status(conn, file_name, "error", error_detail=str(e)[:500])
        return {"total": 0, "inserted": 0, "errors": 1, "elapsed": time.time() - t0}

    for article_el in articles:
        total += 1
        try:
            data = extract_article(article_el, source=source, source_file=file_name)
        except Exception as e:
            errors += 1
            if error_fh:
                error_fh.write(json.dumps({
                    "file": file_name, "n": total, "error": str(e),
                    "tb": traceback.format_exc()[-500:]
                }) + "\n")
            continue

        if data is None:
            errors += 1
            continue

        batches["document"].append(data["document"])
        for table_name in DIM_TABLES:
            rows = data.get(table_name, [])
            if rows:
                batches[table_name].extend(rows)

        if len(batches["document"]) >= BATCH_SIZE:
            flush_batches(conn, batches)
            inserted += BATCH_SIZE
            batches = {name: [] for name in all_tables}

            if ABORT_FLAG.exists():
                print(f"\n  Aborted by monitor at {total:,} articles — data safe")
                mark_status(conn, file_name, "error",
                            articles_seen=total, articles_inserted=inserted,
                            errors=errors, error_detail="Aborted by monitor — data safe")
                if error_fh:
                    error_fh.close()
                return {"total": total, "inserted": inserted, "errors": errors,
                        "elapsed": time.time() - t0}

    if batches["document"]:
        flush_batches(conn, batches)
        inserted += len(batches["document"])

    elapsed = time.time() - t0
    rate = total / elapsed if elapsed > 0 else 0
    print(f"{total:,} articles  {inserted:,} ins  {errors:,} err  {elapsed:.1f}s  {rate:,.0f}/s")

    mark_status(conn, file_name, "done",
                articles_seen=total, articles_inserted=inserted, errors=errors)

    if error_fh:
        error_fh.close()
    return {"total": total, "inserted": inserted, "errors": errors, "elapsed": elapsed}


# ── Index SQL ─────────────────────────────────────────────────────────────────

INDEX_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_epmc_doc_pmcid ON {SCHEMA}.document (pmcid);
CREATE INDEX IF NOT EXISTS idx_epmc_doc_pmid ON {SCHEMA}.document (pmid);
CREATE INDEX IF NOT EXISTS idx_epmc_doc_doi ON {SCHEMA}.document (doi);
CREATE INDEX IF NOT EXISTS idx_epmc_doc_year ON {SCHEMA}.document (pub_year);
CREATE INDEX IF NOT EXISTS idx_epmc_abstract_docid ON {SCHEMA}.abstract (doc_id);
CREATE INDEX IF NOT EXISTS idx_epmc_author_docid ON {SCHEMA}.author (doc_id);
CREATE INDEX IF NOT EXISTS idx_epmc_kw_docid ON {SCHEMA}.keyword (doc_id);
CREATE INDEX IF NOT EXISTS idx_epmc_ref_docid ON {SCHEMA}.reference (doc_id);
CREATE INDEX IF NOT EXISTS idx_epmc_ref_doi ON {SCHEMA}.reference (cited_doi);
CREATE INDEX IF NOT EXISTS idx_epmc_ref_pmid ON {SCHEMA}.reference (cited_pmid);
CREATE INDEX IF NOT EXISTS idx_epmc_body_docid ON {SCHEMA}.body_fulltext (doc_id);
CREATE INDEX IF NOT EXISTS idx_epmc_section_docid ON {SCHEMA}.body_section (doc_id);
CREATE INDEX IF NOT EXISTS idx_epmc_artid_docid ON {SCHEMA}.article_id (doc_id);
"""


# ── PMC OA tar.gz ingest ──────────────────────────────────────────────────────

def _run_pmc_tars(conn, args):
    """Ingest PMC OA tar.gz files containing individual JATS XML articles.

    Filename filter: tar files are filtered by a prefix derived from --source.
    `--source oa_comm`    → only files matching `oa_comm*.tar.gz`
    `--source oa_noncomm` → only files matching `oa_noncomm*.tar.gz`
    Other source values keep the legacy behavior of all `*.tar.gz`.
    This prevents cross-source contamination when both subset directories
    coexist in the same --pmc-dir.
    """
    pmc_dir = args.pmc_dir
    source = args.source

    # Source-based filename prefix filter (safer than blind *.tar.gz)
    if source == "oa_comm":
        tar_files = sorted(pmc_dir.glob("oa_comm*.tar.gz"))
    elif source == "oa_noncomm":
        tar_files = sorted(pmc_dir.glob("oa_noncomm*.tar.gz"))
    else:
        tar_files = sorted(pmc_dir.glob("*.tar.gz"))

    if args.file:
        tar_files = [f for f in tar_files if args.file in f.name]
    if not tar_files:
        print(f"No tar.gz files matching source '{source}' found in {pmc_dir}")
        return
    progress_file = Path(args.progress_file) if args.progress_file else None
    error_log = Path(args.error_log) if args.error_log else None

    print(f"[{datetime.utcnow().isoformat()}Z] PMC OA Fulltext Ingest (tar.gz)")
    print(f"  Dir: {pmc_dir}")
    print(f"  Files: {len(tar_files)}")
    print(f"  Source: {source}")

    if progress_file:
        with open(progress_file, "a") as pf:
            pf.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                     f"=== PMC OA INGEST: {len(tar_files)} tar.gz files ===\n")

    all_tables = ["document"] + list(DIM_TABLES.keys())
    grand_total = 0
    grand_inserted = 0
    grand_errors = 0
    skipped = 0
    t_all = time.time()

    for ti, tar_path in enumerate(tar_files, 1):
        tar_name = tar_path.name

        if check_done(conn, tar_name):
            skipped += 1
            continue

        if ABORT_FLAG.exists():
            print("Aborted by monitor — data safe")
            break

        mark_status(conn, tar_name, "processing")
        t0 = time.time()
        total = 0
        inserted = 0
        errors = 0
        batches = {name: [] for name in all_tables}
        error_fh = open(error_log, "a") if error_log else None

        print(f"  [{ti:>3}/{len(tar_files)}] {tar_name}...", end=" ", flush=True)

        try:
            with tarfile.open(str(tar_path), "r:gz") as tf:
                for member in tf:
                    if not member.isfile() or not member.name.endswith(".xml"):
                        continue
                    try:
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        xml_bytes = f.read()
                        # Parse single JATS article
                        tree = etree.fromstring(xml_bytes)
                        data = extract_article(tree, source=source, source_file=tar_name)
                    except Exception as e:
                        errors += 1
                        if error_fh:
                            error_fh.write(json.dumps({
                                "tar": tar_name, "member": member.name,
                                "error": str(e),
                                "tb": traceback.format_exc()[-500:]
                            }) + "\n")
                        continue

                    total += 1
                    if data is None:
                        errors += 1
                        continue

                    batches["document"].append(data["document"])
                    for table_name in DIM_TABLES:
                        rows = data.get(table_name, [])
                        if rows:
                            batches[table_name].extend(rows)

                    if len(batches["document"]) >= BATCH_SIZE:
                        flush_batches(conn, batches)
                        inserted += BATCH_SIZE
                        batches = {name: [] for name in all_tables}

        except Exception as e:
            errors += 1
            print(f"TAR ERROR: {e}")
            if error_fh:
                error_fh.write(json.dumps({
                    "tar": tar_name, "error": str(e),
                    "tb": traceback.format_exc()[-500:]
                }) + "\n")

        # Final flush
        if batches["document"]:
            flush_batches(conn, batches)
            inserted += len(batches["document"])

        elapsed = time.time() - t0
        rate = total / elapsed if elapsed > 0 else 0
        print(f"{total:,} articles  {inserted:,} ins  {errors:,} err  "
              f"{elapsed:.1f}s  {rate:,.0f}/s")

        mark_status(conn, tar_name, "done",
                    articles_seen=total, articles_inserted=inserted, errors=errors)

        grand_total += total
        grand_inserted += inserted
        grand_errors += errors

        if error_fh:
            error_fh.close()

        if progress_file:
            elapsed_all = time.time() - t_all
            with open(progress_file, "a") as pf:
                pf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                         f"[{ti}/{len(tar_files)}] {tar_name}: "
                         f"{total:,} articles, {inserted:,} ins, "
                         f"{errors:,} err, {elapsed:.1f}s "
                         f"(total: {grand_total:,}, {elapsed_all/3600:.1f}h)\n")

    elapsed_all = time.time() - t_all
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] PMC OA COMPLETE: "
          f"{grand_total:,} articles, {grand_inserted:,} inserted, "
          f"{grand_errors:,} errors, {skipped:,} skipped, "
          f"{elapsed_all/3600:.1f}h")

    if progress_file:
        with open(progress_file, "a") as pf:
            pf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                     f"=== PMC OA COMPLETE === {grand_total:,} articles, "
                     f"{grand_inserted:,} ins, {grand_errors:,} err, "
                     f"{skipped:,} skipped, {elapsed_all/3600:.1f}h\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Ingest Europe PMC JATS XML into europepmc.* schema")
    ap.add_argument("--dir", type=Path, default=DOWNLOAD_DIR)
    ap.add_argument("--pmc-dir", type=Path, help="Directory of PMC OA tar.gz files (individual JATS XMLs)")
    ap.add_argument("--file", help="Process only files matching this substring")
    ap.add_argument("--source", default="oa_fulltext",
                    choices=["oa_fulltext", "ncbi_pmc", "pmclite", "pmc_oa",
                             "oa_comm", "oa_noncomm"])
    ap.add_argument("--create-indexes", action="store_true")
    ap.add_argument("--set-logged", action="store_true")
    ap.add_argument("--error-log", default="/tmp/europepmc_ingest_errors.jsonl")
    ap.add_argument("--progress-file", default="/tmp/europepmc_ingest_progress.txt")
    args = ap.parse_args()

    conn = psycopg2.connect(**DB_PARAMS)

    if args.create_indexes:
        print("Creating indexes...")
        with conn.cursor() as cur:
            cur.execute(INDEX_SQL)
        conn.commit()
        print("Done.")
        return

    if args.set_logged:
        print("Converting tables to LOGGED...")
        with conn.cursor() as cur:
            cur.execute(f"SELECT tablename FROM pg_tables WHERE schemaname = '{SCHEMA}'")
            tables = [r[0] for r in cur.fetchall()]
        for t in sorted(tables):
            print(f"  {SCHEMA}.{t}...", end=" ", flush=True)
            with conn.cursor() as cur:
                cur.execute(f"ALTER TABLE {SCHEMA}.{t} SET LOGGED")
            conn.commit()
            print("OK")
        return

    # ── PMC OA tar.gz mode (explicit or auto-detected from --dir) ──
    if args.pmc_dir:
        _run_pmc_tars(conn, args)
        conn.close()
        return

    # Auto-detect: if --dir contains tar.gz files (and no PMC*_PMC*.xml.gz files),
    # dispatch to tar mode. This makes `--dir /data/downloads/pmc_fulltext/` work
    # for the oa_comm / oa_noncomm tar.gz layout.
    tar_files_check = sorted(args.dir.glob("*.tar.gz"))
    xml_files_check = sorted(args.dir.glob("PMC*_PMC*.xml.gz"))
    if tar_files_check and not xml_files_check:
        print(f"[auto-detect] --dir contains {len(tar_files_check)} tar.gz files; "
              f"dispatching to PMC tar mode")
        args.pmc_dir = args.dir
        _run_pmc_tars(conn, args)
        conn.close()
        return

    files = xml_files_check
    if args.file:
        files = [f for f in files if args.file in f.name]

    if not files:
        print(f"No files found in {args.dir}")
        sys.exit(1)

    print(f"[{datetime.utcnow().isoformat()}Z] Europe PMC Ingest")
    print(f"  Files: {len(files)}")
    print(f"  Source: {args.source}")

    progress_file = Path(args.progress_file) if args.progress_file else None
    error_log = Path(args.error_log) if args.error_log else None

    if progress_file:
        with open(progress_file, "a") as pf:
            pf.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                     f"=== EUROPEPMC INGEST: {len(files)} files ===\n")

    grand_total = 0
    grand_inserted = 0
    grand_errors = 0
    skipped = 0
    t_all = time.time()

    for gz_path in files:
        if check_done(conn, gz_path.name):
            skipped += 1
            continue

        stats = ingest_file(conn, gz_path, source=args.source,
                            error_log=error_log, progress_file=progress_file)
        grand_total += stats["total"]
        grand_inserted += stats["inserted"]
        grand_errors += stats["errors"]

        if progress_file and grand_total % 50000 < stats["total"]:
            elapsed = time.time() - t_all
            with open(progress_file, "a") as pf:
                pf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                         f"Total: {grand_total:,} articles, {grand_inserted:,} ins, "
                         f"{grand_errors:,} err, {elapsed/3600:.1f}h\n")

    elapsed = time.time() - t_all
    summary = (f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] COMPLETE: "
               f"{grand_total:,} articles, {grand_inserted:,} inserted, "
               f"{grand_errors:,} errors, {skipped} skipped, {elapsed/3600:.1f}h")
    print(summary)
    if progress_file:
        with open(progress_file, "a") as pf:
            pf.write(summary + "\n")

    conn.close()


if __name__ == "__main__":
    main()

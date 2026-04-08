# arc_v5 Database Maintenance Analysis — 2026-04-08

**Host:** arc-sx135
**Database:** `arc_v5`
**Report date:** 2026-04-08
**Status at time of audit:** S2 citations ingest 85%+ done, EPO fulltext ingest ~22% done, both running

---

## Executive Summary

Full audit of `arc_v5` database identified three classes of maintenance work:

| Finding | Count | Total size | Action |
|---|---|---|---|
| **UNLOGGED tables** (not crash-safe) | **90 tables** | **~3.5–4 TB** | Convert to LOGGED before treating as durable |
| **Tables > 1 GB with zero indexes** | **57 tables** | **~2.8 TB** | Build FK indexes for join queries |
| **Cold/unused indexes** (idx_scan = 0) | **5 indexes** | ~22 GB | Investigate before dropping |

The entire ingest pipeline (crossref, semanticscholar, europepmc, pubmed, openalex schemas) was built with UNLOGGED tables for bulk-load speed, and indexes were dropped before ingestion to minimize WAL and speed up COPY. **Neither has been converted back**, which means:

1. A crash, power loss, or OOM kill will **zero-truncate 3.5+ TB of data** on recovery
2. Every join query against these tables currently seq-scans hundreds of gigabytes
3. The tables are also excluded from `pg_basebackup` and not replicated

This report details the audit findings, proposes a 5-phase maintenance schedule (to run after current ingests complete), and codifies 4 operational rules that arose from today's debugging of a runaway COUNT(*) loop.

---

## Table of Contents

1. [Finding #1 — UNLOGGED tables audit](#finding-1--unlogged-tables-audit)
2. [Finding #2 — Missing indexes audit](#finding-2--missing-indexes-audit)
3. [Finding #3 — Index usage analysis](#finding-3--index-usage-analysis)
4. [FK column inspection — Phase 3 satellite tables](#fk-column-inspection--phase-3-satellite-tables)
5. [Proposed maintenance schedule](#proposed-maintenance-schedule)
6. [Operational rules](#operational-rules)
7. [Insights and lessons learned](#insights-and-lessons-learned)
8. [Appendix A — Full diagnostic query outputs](#appendix-a--full-diagnostic-query-outputs)

---

## Finding #1 — UNLOGGED tables audit

### Query

```sql
SELECT schemaname, relname,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||relname)) AS size
FROM pg_tables t
JOIN pg_class c ON c.relname = t.tablename
 AND c.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = t.schemaname)
WHERE c.relpersistence = 'u'
ORDER BY pg_total_relation_size(schemaname||'.'||relname) DESC;
```

### Result: 90 UNLOGGED tables

| # | Schema | Table | Size |
|---|---|---|---|
| 1 | crossref | reference | 584 GB |
| 2 | semanticscholar | paper_citation | 552 GB |
| 3 | europepmc | body_section | 442 GB |
| 4 | openalex | work_concept | 399 GB |
| 5 | europepmc | reference | 265 GB |
| 6 | openalex | work_keyword | 225 GB |
| 7 | openalex | work | 220 GB |
| 8 | europepmc | **x_delete_body_fulltext** | **207 GB** (marked for deletion) |
| 9 | semanticscholar | paper | 191 GB |
| 10 | openalex | work_related_work | 175 GB |
| 11 | openalex | work_referenced_work | 152 GB |
| 12 | openalex | work_topic | 138 GB |
| 13 | crossref | document | 120 GB |
| 14 | openalex | work_authorship | 97 GB |
| 15 | openalex | work_location | 67 GB |
| 16 | semanticscholar | paper_author | 56 GB |
| 17 | openalex | work_authorship_affiliation | 56 GB |
| 18 | crossref | author | 49 GB |
| 19 | pubmed | reference | 49 GB |
| 20 | openalex | work_mesh | 44 GB |
| 21 | pubmed | abstract_section | 39 GB |
| 22 | crossref | link | 36 GB |
| 23 | pubmed | reference_article_id | 27 GB |
| 24 | pubmed | mesh_heading | 26 GB |
| 25 | crossref | author_affiliation | 22 GB |
| 26 | europepmc | author_affiliation | 21 GB |
| 27 | pubmed | author_affiliation | 21 GB |
| 28 | openalex | work_counts_by_year | 21 GB |
| 29 | europepmc | abstract | 20 GB |
| 30 | crossref | assertion | 19 GB |
| 31 | openalex | work_authorship_country | 19 GB |
| 32 | pubmed | document | 18 GB |
| 33 | openalex | work_corresponding_author | 18 GB |
| 34 | semanticscholar | paper_field | 16 GB |
| 35 | openalex | work_indexed_in | 15 GB |
| 36 | crossref | issn | 15 GB |
| 37 | crossref | license | 15 GB |
| 38 | pubmed | author | 14 GB |
| 39 | openalex | work_sdg | 14 GB |
| 40 | pubmed | mesh_qualifier | 13 GB |
| 41 | europepmc | custom_meta | 10 GB |
| 42 | openalex | work_corresponding_institution | 10 GB |
| 43 | europepmc | abstract_section | 10186 MB |
| 44 | pubmed | history_date | 9818 MB |
| 45 | europepmc | author | 9557 MB |
| 46 | pubmed | article_id | 7156 MB |
| 47 | crossref | alternative_id | 6828 MB |
| 48 | europepmc | article_id | 5598 MB |
| 49 | openalex | author_entity | 5463 MB |
| 50 | pubmed | chemical | 5321 MB |
| 51 | crossref | funder | 5179 MB |
| 52 | pubmed | publication_type | 5106 MB |
| 53 | pubmed | keyword | 4812 MB |
| 54 | europepmc | keyword | 4055 MB |
| 55 | europepmc | acknowledgment | 3759 MB |
| 56 | europepmc | document | 3667 MB |
| 57 | semanticscholar | paper_source | 3471 MB |
| 58 | arc_v5 | epo_person_staging | 3255 MB |
| 59 | europepmc | article_category | 2014 MB |
| 60 | europepmc | pub_history | 1885 MB |
| 61 | openalex | work_funder | 1477 MB |
| 62 | pubmed | grant | 1439 MB |
| 63 | pubmed | author_identifier | 1196 MB |
| 64 | europepmc | funding | 997 MB |
| 65 | _front_staging | epo_document | 825 MB |
| 66 | pubmed | investigator | 454 MB |
| 67 | pubmed | comment_correction | 426 MB |
| 68 | pubmed | other_abstract | 415 MB |
| 69 | pubmed | databank_accession | 293 MB |
| 70 | crossref | update_to | 59 MB |
| 71 | arc_v5 | _journal_emb_staging | 30 MB |
| 72 | pubmed | general_note | 26 MB |
| 73 | pubmed | supplemental_mesh | 23 MB |
| 74 | pubmed | gene_symbol | 7488 kB |
| 75 | crossref | clinical_trial | 7320 kB |
| 76 | crossref | ingest_progress | 4720 kB |
| 77 | semanticscholar | ingest_progress | 1968 kB |
| 78 | europepmc | ingest_progress | 320 kB |
| 79 | pubmed | ingest_progress | 288 kB |
| 80 | openalex | ingest_progress | 208 kB |
| 81 | openalex | topic_entity | 48 kB |
| 82 | openalex | institution_entity | 32 kB |
| 83 | openalex | funder_entity | 32 kB |
| 84 | openalex | concept_entity | 32 kB |
| 85 | openalex | source_entity | 32 kB |
| 86 | openalex | author_entity_affiliation | 16 kB |
| 87 | crossref | subject | 16 kB |
| 88 | openalex | author_entity_topic | 16 kB |
| 89 | openalex | work_award | 16 kB |
| 90 | _front_staging | delete_ids | 16 kB |

### Implications

**UNLOGGED tables are:**
- Truncated to zero rows on any unclean shutdown (power loss, OOM killer, kernel panic) — Postgres skips their WAL on crash recovery
- Not included in `pg_basebackup` (their contents are zeroed in base backups)
- Not replicated to physical standbys
- Fine for ingest staging and crash-resumable bulk loads, but **catastrophic if treated as durable before conversion**

**Conversion cost estimate:** `ALTER TABLE … SET LOGGED` rewrites the entire table and writes full content to WAL. Rough IO budget: **~2× table size** (one read, one WAL write, plus heap write). For `crossref.reference` at 584 GB, that's ~1.2 TB of IO. At RAID6 sequential write speeds (~500 MB/s when uncontended), that's about **40 minutes per 500 GB**. Total conversion time for everything: **~24–36 hours of serial work**, or faster if parallelized across tables.

**`x_delete_body_fulltext` (207 GB) was marked for deletion earlier on 2026-04-08** (renamed from `body_fulltext` with a `COMMENT ON TABLE` explaining it's redundant with `body_section`). Do NOT convert it to LOGGED — drop it first.

---

## Finding #2 — Missing indexes audit

### Query

```sql
SELECT t.schemaname, t.tablename,
       pg_size_pretty(pg_total_relation_size(t.schemaname||'.'||t.tablename)) AS size,
       (SELECT COUNT(*) FROM pg_index
        WHERE indrelid = (t.schemaname||'.'||t.tablename)::regclass) AS direct_idx_count
FROM pg_tables t
WHERE pg_total_relation_size(t.schemaname||'.'||t.tablename) > 1073741824
  AND (SELECT COUNT(*) FROM pg_index
       WHERE indrelid = (t.schemaname||'.'||t.tablename)::regclass) = 0
ORDER BY pg_total_relation_size(t.schemaname||'.'||t.tablename) DESC;
```

**Note:** The original version of this query used `pg_indexes` (the view) in a correlated subquery, which returned inconsistent results during active autovacuum ANALYZE on `semanticscholar.paper_citation`. Using direct `pg_index` with `::regclass` casting is the authoritative source and matches actual table state.

### Result: 57 tables > 1 GB with zero indexes

| # | Schema | Table | Size | Rosetta-critical? |
|---|---|---|---|---|
| 1 | crossref | reference | 584 GB | — |
| 2 | semanticscholar | paper_citation | 556 GB | ★ YES (needs paper_id, cited_id) |
| 3 | europepmc | body_section | 442 GB | ★ YES (needs doc_id for has_fulltext) |
| 4 | openalex | work_concept | 399 GB | — |
| 5 | europepmc | reference | 265 GB | ★ YES (needs doc_id for citation graph) |
| 6 | openalex | work_keyword | 225 GB | — |
| 7 | europepmc | x_delete_body_fulltext | 207 GB | NO — DROP instead |
| 8 | openalex | work_related_work | 175 GB | — |
| 9 | openalex | work_referenced_work | 152 GB | — |
| 10 | openalex | work_topic | 138 GB | — |
| 11 | openalex | work_authorship | 97 GB | — |
| 12 | openalex | work_location | 67 GB | — |
| 13 | semanticscholar | paper_author | 56 GB | — |
| 14 | openalex | work_authorship_affiliation | 56 GB | — |
| 15 | crossref | author | 49 GB | — |
| 16 | pubmed | reference | 49 GB | — |
| 17 | openalex | work_mesh | 44 GB | — |
| 18 | pubmed | abstract_section | 39 GB | ★ YES (needs doc_id for pubmed abstracts) |
| 19 | crossref | link | 36 GB | — |
| 20 | pubmed | reference_article_id | 27 GB | — |
| 21 | pubmed | mesh_heading | 26 GB | ★ YES (needs doc_id for MeSH terms) |
| 22 | crossref | author_affiliation | 22 GB | — |
| 23 | europepmc | author_affiliation | 21 GB | — |
| 24 | pubmed | author_affiliation | 21 GB | — |
| 25 | openalex | work_counts_by_year | 21 GB | — |
| 26 | europepmc | abstract | 20 GB | ★ YES (needs doc_id for epmc abstracts) |
| 27 | crossref | assertion | 19 GB | — |
| 28 | openalex | work_authorship_country | 19 GB | — |
| 29 | openalex | work_corresponding_author | 18 GB | — |
| 30 | semanticscholar | paper_field | 16 GB | ★ YES (needs paper_id for fields_of_study) |
| 31 | openalex | work_indexed_in | 15 GB | — |
| 32 | crossref | issn | 15 GB | — |
| 33 | crossref | license | 15 GB | — |
| 34 | pubmed | author | 14 GB | — |
| 35 | openalex | work_sdg | 14 GB | — |
| 36 | pubmed | mesh_qualifier | 13 GB | — |
| 37 | europepmc | custom_meta | 10 GB | — |
| 38 | openalex | work_corresponding_institution | 10 GB | — |
| 39 | europepmc | abstract_section | 10186 MB | — |
| 40 | pubmed | history_date | 9818 MB | — |
| 41 | europepmc | author | 9557 MB | — |
| 42 | pubmed | article_id | 7156 MB | — |
| 43 | crossref | alternative_id | 6828 MB | — |
| 44 | europepmc | article_id | 5598 MB | — |
| 45 | pubmed | chemical | 5321 MB | — |
| 46 | crossref | funder | 5179 MB | — |
| 47 | pubmed | publication_type | 5106 MB | — |
| 48 | pubmed | keyword | 4812 MB | — |
| 49 | europepmc | keyword | 4055 MB | — |
| 50 | europepmc | acknowledgment | 3759 MB | — |
| 51 | semanticscholar | paper_source | 3471 MB | — |
| 52 | arc_v5 | epo_person_staging | 3255 MB | — (staging, temporary) |
| 53 | europepmc | article_category | 2014 MB | — |
| 54 | europepmc | pub_history | 1885 MB | — |
| 55 | openalex | work_funder | 1477 MB | — |
| 56 | pubmed | grant | 1439 MB | — |
| 57 | pubmed | author_identifier | 1196 MB | — |

### Main document tables (NOT in this list — they have PKs)

| Table | Size | Index | Scans |
|---|---|---|---|
| `openalex.work` | 220 GB | `work_pkey` (19 GB) | 272,933,744 — hot |
| `crossref.document` | 120 GB | `document_pkey` (11 GB) | 179,541,204 — hot |
| `semanticscholar.paper` | 191 GB | `paper_pkey` (13 GB) | 5 — cold |
| `pubmed.document` | 18 GB | `document_pkey` (2 GB) | 40,004,988 — hot |
| `europepmc.document` | 3.7 GB | (PK exists, too small for top 30) | — |

---

## Finding #3 — Index usage analysis

### Query

```sql
SELECT schemaname, relname AS tablename, indexrelname AS indexname,
       pg_size_pretty(pg_relation_size(indexrelid)) AS size,
       idx_scan, idx_tup_read
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC
LIMIT 30;
```

### Top 30 largest indexes

| # | Schema | Table | Index | Size | idx_scan | idx_tup_read |
|---|---|---|---|---|---|---|
| 1 | openalex | work | work_pkey | 19 GB | 272,933,744 | 62,401,853 |
| 2 | arc_v5 | journal_embeddings | journal_embeddings_pkey | 19 GB | 270,405,519 | 104,295,310 |
| 3 | arc_v5 | epo_patent_classification | idx_epo_patclass_docid | 18 GB | 9,258 | 3,408,530,033 |
| 4 | semanticscholar | paper | paper_pkey | 13 GB | **5** | 5 |
| 5 | crossref | document | document_pkey | 11 GB | 179,541,204 | 4,887 |
| 6 | arc_v5 | epo_classification_ipcr | idx_epo_ipcr_docid | 8672 MB | 2,480,806 | 674,787,291 |
| 7 | arc_v5 | epo_patent_classification | **idx_epo_patclass_subclass** | 7886 MB | **0** | 0 |
| 8 | arc_v5 | epo_patent_classification | **idx_epo_patclass_section** | 7881 MB | **0** | 0 |
| 9 | arc_v5 | epo_document_inventor | idx_epo_inv_docid | 6943 MB | 3,231 | 25,108 |
| 10 | arc_v5 | epo_priority_claim | idx_epo_prio_docid | 6020 MB | 3,231 | 10,882 |
| 11 | arc_v5 | epo_title | idx_epo_title_docid | 5972 MB | 1,438,876 | 359,717,262 |
| 12 | arc_v5 | epo_citation_patent | idx_epo_citpat_docid | 5804 MB | 3,231 | 2,006 |
| 13 | arc_v5 | epo_application_ref | idx_epo_appref_docid | 5295 MB | 13,228 | 16,504 |
| 14 | arc_v5 | epo_publication_ref | idx_epo_pubref_docid | 5295 MB | 590,646 | 593,920 |
| 15 | arc_v5 | epo_application_ref | idx_epo_appref_docid_uniq | 5291 MB | 9,155,559 | 2,861,675 |
| 16 | arc_v5 | epo_publication_ref | idx_epo_pubref_docid_uniq | 5291 MB | 9,155,559 | 2,861,693 |
| 17 | arc_v5 | epo_document | idx_epo_doc_id_uniq | 5275 MB | 16,457,697 | 21,408,559 |
| 18 | arc_v5 | epo_pub_availability | idx_epo_pubavail_docid_uniq | 5128 MB | 8,881,981 | 2,743,015 |
| 19 | arc_v5 | epo_pub_availability | idx_epo_pubavail_docid | 5128 MB | 1,119,938 | 1,072,715 |
| 20 | arc_v5 | epo_document_applicant | idx_epo_app_docid | 4938 MB | 3,235 | 6,923 |
| 21 | arc_v5 | epo_abstract | idx_epo_abstract_docid | 4540 MB | 9,958,097 | 37,463,583 |
| 22 | arc_v5 | epo_knn | idx_epo_knn_cpc | 3817 MB | 1,325 | 1,533,342,163 |
| 23 | arc_v5 | epo_document | idx_epo_doc_family_id | 3637 MB | 4 | 109,691,068 |
| 24 | arc_v5 | epo_embeddings | epo_embeddings_pkey | 3574 MB | 3,272,779 | 733,396,695 |
| 25 | arc_v5 | epo_document | idx_epo_doc_doc_number | 3513 MB | 423,623 | 1,788,126 |
| 26 | arc_v5 | epo_document_inventor | **idx_epo_inv_personid** | 3326 MB | **0** | 0 |
| 27 | arc_v5 | epo_document_dedup | idx_epo_dedup_doc | 2860 MB | 163,835,100 | 485,437,427 |
| 28 | arc_v5 | epo_document_dedup | **idx_epo_dedup_family** | 2783 MB | **0** | 0 |
| 29 | pubmed | document | document_pkey | 2030 MB | 40,004,988 | 0 |
| 30 | arc_v5 | epo_classification_ipc | idx_epo_ipc_docid | 1817 MB | 13,242 | 82,165,336 |

### Hot indexes (well-utilized)

Top 5 most-scanned indexes:

1. **`openalex.work_pkey`** — 272.9 M scans, the core work lookup
2. **`arc_v5.journal_embeddings_pkey`** — 270.4 M scans, hot for embedding retrieval
3. **`crossref.document_pkey`** — 179.5 M scans, crossref document lookups
4. **`arc_v5.epo_document_dedup idx_epo_dedup_doc`** — 163.8 M scans, dedup lookups
5. **`pubmed.document_pkey`** — 40.0 M scans, pubmed document lookups

These are doing their jobs — leave them alone.

### Cold/unused indexes (candidates for eventual drop, investigate first)

| Index | Table | Size | Scans | Notes |
|---|---|---|---|---|
| `idx_epo_patclass_subclass` | arc_v5.epo_patent_classification | **7.9 GB** | **0** | CPC subclass lookups — might be needed for future queries |
| `idx_epo_patclass_section` | arc_v5.epo_patent_classification | **7.9 GB** | **0** | CPC section lookups — same concern |
| `idx_epo_inv_personid` | arc_v5.epo_document_inventor | 3.3 GB | **0** | Person→inventor lookup (inventor search feature) |
| `idx_epo_dedup_family` | arc_v5.epo_document_dedup | 2.8 GB | **0** | Family→docs lookup (family-level dedup queries) |
| `semanticscholar.paper_pkey` | semanticscholar.paper | 13 GB | **5** | Essentially unused — S2 paper table isn't being queried yet |

**Total unused index space:** ~22 GB on EPO classification tables + 3.3 GB on inventor + 2.8 GB on family dedup = **~28 GB**.

**Important caveat:** `idx_scan = 0` does NOT mean "useless" — it means "hasn't been scanned since the stats counter was last reset" (server start or manual reset). The server has been up 7d 21h at the time of this audit, which validates "hasn't been used today" but NOT "never useful". Patent search workloads are bursty — an index for CPC classification or inventor search could be critical during a search session and idle for weeks otherwise.

**Do NOT drop these without:**
1. Confirming the expected query patterns with the application layer (patent search, inventor search, family dedup)
2. Waiting at least 2 weeks to see if they remain at 0 scans
3. Confirming with Jeff that the functionality they support is either unused or will be replaced

---

## FK column inspection — Phase 3 satellite tables

### Rosetta-critical tables (7 confirmed earlier)

| Table | Size | FK column(s) | Type |
|---|---|---|---|
| `semanticscholar.paper_citation` | 556 GB | `paper_id`, `cited_id` | text, text |
| `europepmc.body_section` | 442 GB | `doc_id` | text |
| `europepmc.reference` | 265 GB | `doc_id` | text |
| `pubmed.abstract_section` | 39 GB | `doc_id` | text |
| `pubmed.mesh_heading` | 26 GB | `doc_id` | text |
| `europepmc.abstract` | 20 GB | `doc_id` | text |
| `semanticscholar.paper_field` | 16 GB | `paper_id` | text |

### Phase 3 satellite tables (46 inspected, all confirmed)

**All satellite tables use consistent FK naming:**
- `openalex.work_*` tables → `doc_id` text (links to `openalex.work.doc_id`)
- `pubmed.*` child tables → `doc_id` text (stored as `'PMID:' || pmid` namespaced form)
- `europepmc.*` child tables → `doc_id` text (stored as `'PMC:' || pmcid` namespaced form)
- `crossref.*` child tables → `doc_id` text
- `semanticscholar.paper_*` tables → `paper_id` text (matches `semanticscholar.paper.s2_id`)

### OpenAlex work_* family (16 tables, all FK=`doc_id` text)

| Table | Size | Additional indexable columns |
|---|---|---|
| work_concept | 399 GB | `concept_id`, `wikidata` |
| work_keyword | 225 GB | `keyword_id` |
| work_related_work | 175 GB | `related_work_id` |
| work_referenced_work | 152 GB | `referenced_work_id` (citation graph!) |
| work_topic | 138 GB | `topic_id` |
| work_authorship | 97 GB | `author_id` |
| work_location | 67 GB | `source_id` (venue) |
| work_authorship_affiliation | 56 GB | `authorship_sequence`, `institution_id` |
| work_mesh | 44 GB | `descriptor_ui`, `descriptor_name` |
| work_counts_by_year | 21 GB | `year` |
| work_authorship_country | 19 GB | `authorship_sequence`, `country_code` |
| work_corresponding_author | 18 GB | `author_id` |
| work_indexed_in | 15 GB | `index_name` |
| work_sdg | 14 GB | `sdg_id` |
| work_corresponding_institution | 10 GB | `institution_id` |
| work_funder | 1.5 GB | `funder_id` |

### Crossref child tables (9 tables, all FK=`doc_id` text)

reference, author, link, author_affiliation, assertion, issn, license, funder, alternative_id

### PubMed child tables (12 tables, all FK=`doc_id` text, namespaced `PMID:<pmid>`)

reference, reference_article_id, author_affiliation, author, mesh_qualifier, history_date, article_id, chemical, publication_type, keyword, grant, author_identifier

### EuropePMC child tables (8 tables, all FK=`doc_id` text, namespaced `PMC:<pmcid>`)

author_affiliation, custom_meta, abstract_section, author, article_id, keyword, acknowledgment, article_category, pub_history

### Semantic Scholar child tables (2 additional, FK=`paper_id` text)

paper_author, paper_source

### Why this matters

**The schema designer did the right thing** by standardizing on two FK column names (`doc_id` for main document-based tables, `paper_id` for semanticscholar, which has its own `s2_id`/`paper_id` naming history). This means the Phase 3 index-build script can be templated:

```sql
-- OpenAlex work_* family
CREATE INDEX CONCURRENTLY idx_<table>_doc_id ON openalex.<table>(doc_id);

-- Crossref child tables
CREATE INDEX CONCURRENTLY idx_<table>_doc_id ON crossref.<table>(doc_id);

-- PubMed child tables
CREATE INDEX CONCURRENTLY idx_<table>_doc_id ON pubmed.<table>(doc_id);

-- EuropePMC child tables
CREATE INDEX CONCURRENTLY idx_<table>_doc_id ON europepmc.<table>(doc_id);

-- SemanticScholar child tables
CREATE INDEX CONCURRENTLY idx_<table>_paper_id ON semanticscholar.<table>(paper_id);
```

No special cases, no per-table column-name investigations. The entire Phase 3 can be a for-loop over a table list.

---

## Proposed maintenance schedule

> **Run only after both S2 citations AND EPO fulltext ingests complete.**
> Expected start: ~14:00 local for S2 finish, ~01:00–13:00 next day for EPO finish. Both must be done.

### Phase 0 — Immediate cleanup (runs first, after both ingests)

| Task | Command | Est. time |
|---|---|---|
| Drop the marked-for-deletion table (frees 207 GB) | `DROP TABLE europepmc.x_delete_body_fulltext;` | seconds |
| Checkpoint before any heavy writes | `CHECKPOINT;` | ~30 s |

**Note:** per standing rule, only DROP after Jeff confirms the rename-and-comment marker is still valid and the table hasn't been needed since the rename.

### Phase 1 — Rosetta-critical indexes (required before running arc_rosetta_build.py)

These 7 indexes unblock the metadata enrichment passes in `arc_rosetta_build.py`:

```sql
-- Run with generous maintenance_work_mem
SET maintenance_work_mem = '4GB';

-- Smallest first (quick wins, quick validation)
CREATE INDEX CONCURRENTLY idx_paper_field_paper_id
  ON semanticscholar.paper_field(paper_id);   -- 16 GB table

CREATE INDEX CONCURRENTLY idx_epmc_abstract_doc_id
  ON europepmc.abstract(doc_id);              -- 20 GB table

CREATE INDEX CONCURRENTLY idx_pubmed_mesh_heading_doc_id
  ON pubmed.mesh_heading(doc_id);             -- 26 GB table

CREATE INDEX CONCURRENTLY idx_pubmed_abstract_section_doc_id
  ON pubmed.abstract_section(doc_id);         -- 39 GB table

CREATE INDEX CONCURRENTLY idx_epmc_reference_doc_id
  ON europepmc.reference(doc_id);             -- 265 GB table

CREATE INDEX CONCURRENTLY idx_epmc_body_section_doc_id
  ON europepmc.body_section(doc_id);          -- 442 GB table

-- Largest, run last (two indexes needed — citation graph both directions)
CREATE INDEX CONCURRENTLY idx_paper_citation_paper_id
  ON semanticscholar.paper_citation(paper_id);   -- 556 GB table
CREATE INDEX CONCURRENTLY idx_paper_citation_cited_id
  ON semanticscholar.paper_citation(cited_id);   -- 556 GB table
```

**Estimated time:** 2–6 hours total at `maintenance_work_mem = 4GB`. `paper_citation` indexes dominate — ~2 hours each for the two on the 556 GB table.

**Important:** `CONCURRENTLY` allows builds to proceed without blocking other queries, but each concurrent build takes ~2× longer than a blocking build. If no other work is happening (which should be true post-ingest), consider dropping `CONCURRENTLY` for speed.

### Phase 2 — Convert core tables UNLOGGED → LOGGED

Priority order (most-queried first):

```sql
-- Core document tables
ALTER TABLE openalex.work SET LOGGED;         -- 220 GB, ~30 min
ALTER TABLE crossref.document SET LOGGED;     -- 120 GB, ~15 min
ALTER TABLE semanticscholar.paper SET LOGGED; -- 191 GB, ~25 min
ALTER TABLE pubmed.document SET LOGGED;       -- 18 GB, ~3 min
ALTER TABLE europepmc.document SET LOGGED;    -- 3.7 GB, ~1 min
```

**Estimated time:** 1.5–2 hours serial, less if parallelized.

### Phase 3 — Build remaining satellite indexes

Template-driven from the FK inspection above. Approximately 50 CREATE INDEX statements. Can be parallelized across different tables:

```bash
# Pseudocode — run each on its own connection
for schema in openalex crossref pubmed europepmc; do
  # All FKs are doc_id text
  ...
done
# semanticscholar uses paper_id
```

**Estimated time:** 12–24 hours serial, 4–8 hours with 4-way parallelization. Largest index jobs: `openalex.work_concept(doc_id)` (~20 GB index), `openalex.work_keyword(doc_id)` (~15 GB), `europepmc.reference(doc_id)` (already in Phase 1).

### Phase 4 — Convert remaining satellites UNLOGGED → LOGGED

The other ~80 UNLOGGED tables, in size order. Largest wins:
- `crossref.reference` (584 GB, ~75 min)
- `openalex.work_concept` (399 GB, ~50 min)
- `openalex.work_keyword` (225 GB, ~30 min)
- `openalex.work_related_work` (175 GB, ~25 min)
- `openalex.work_referenced_work` (152 GB, ~20 min)
- ...

**Estimated time:** 24–36 hours serial. Can overlap with Phase 3 since they touch different table sets.

### Phase 5 — ANALYZE all touched tables

After every SET LOGGED or CREATE INDEX, planner statistics may be stale:

```sql
-- Full ANALYZE of all ingest schemas
ANALYZE openalex.work;
ANALYZE openalex.work_concept;
ANALYZE openalex.work_keyword;
ANALYZE openalex.work_related_work;
ANALYZE openalex.work_referenced_work;
-- ... etc for every table touched
ANALYZE crossref.document;
ANALYZE crossref.reference;
ANALYZE semanticscholar.paper;
ANALYZE semanticscholar.paper_citation;
-- etc
```

Or equivalently `VACUUM ANALYZE` (includes a full scan + stats refresh). Runs in seconds to minutes per table (not full table rewrite — just stats sampling).

**Estimated time:** 30–60 minutes total.

### Total time envelope

**Best case (parallelized):** ~30–40 hours, done within 2 days.
**Worst case (serial):** ~60–80 hours, done within 4 days.

---

## Operational rules

These four rules arose from debugging incidents on 2026-04-08 and have been saved to persistent memory:

### Rule 1 — Never run COUNT(*) on arc_v5 production tables

**Use `SELECT reltuples::bigint FROM pg_class WHERE relname = 'tablename'` instead.**

*Why:* On 2026-04-08, a `/loop` cron in an orphaned "Sentinel" claude session was firing `SELECT COUNT(*) FROM semanticscholar.paper_citation` every 10 minutes during active ingest. Each COUNT seq-scanned the 200+ GB table for ~29 minutes, evicting hot index pages and stalling the COPY writer. Load average went from 7 to 18, S2 ingest delayed by ~30 minutes.

*How to apply:* For any table > 1 GB, default to `reltuples`. Only use true `COUNT(*)` when exactness matters AND no concurrent write workload exists. Same rule applies to any full-table aggregation (SUM, AVG, MAX/MIN without an indexable filter).

### Rule 2 — Exclude `pg_backend_pid()` from pg_terminate_backend filter queries

**Always add `AND pid != pg_backend_pid()` when filtering by query text or state.**

*Why:* On 2026-04-08, running
```sql
SELECT pg_terminate_backend(pid) FROM pg_stat_activity
WHERE query LIKE '%COUNT%paper_citation%' AND state='active';
```
resulted in the terminate query's own text matching the LIKE pattern, killing its own backend mid-execution.

*How to apply:* Safe template for any pattern-matched kill:
```sql
SELECT pg_terminate_backend(pid) FROM pg_stat_activity
WHERE <filter> AND pid != pg_backend_pid();
```

### Rule 3 — Check for remote clients before killing queries

**Before killing "stray" queries, verify via `pg_stat_activity.client_addr` and `ss -tnp` that no remote client (laptop SSH, TCP pg client) is the source.**

*Why:* arc-sx135 occasionally receives SSH sessions from a laptop that runs interactive queries. Reflexively killing "runaway" queries could terminate legitimate work.

*How to apply:* Two checks:
1. `client_addr IS NULL` + `client_port = -1` in `pg_stat_activity` means Unix socket (local)
2. `ss -tnp` shows TCP connections and their peer IPs; `who` / `w` shows active SSH sessions

Unless both confirm local-only, prefer surgical per-PID kills with user confirmation.

### Rule 4 — Convert UNLOGGED to LOGGED before treating as durable

**Every arc_v5 ingest schema table is UNLOGGED. They will be zero-truncated on any unclean shutdown. Convert with `ALTER TABLE ... SET LOGGED` before relying on them.**

*Why:* 90 UNLOGGED tables totaling ~3.5–4 TB identified today. UNLOGGED tables are not crash-safe, not backed up by pg_basebackup, and not replicated.

*How to apply:*
- Check persistence: `SELECT relpersistence FROM pg_class WHERE relname = 'x'` — `'u'` = unlogged, `'p'` = permanent
- Conversion cost ≈ 2× table size in IO
- Schedule outside active ingest windows

---

## Insights and lessons learned

### 1. The unlogged-then-relog pattern has unbounded duration risk

UNLOGGED + bulk load is a recognized PostgreSQL technique, but has a sharp edge: the unlogged phase is unbounded in time. The pattern fails when teams go unlogged for a load, finish the load, and move on without relogging because "it's working fine." Months later, a crash loses everything. The 90 UNLOGGED tables in this database are the result of that deferral pattern extended across multi-week ingests.

**Useful discipline:** Any UNLOGGED table populated for > 7 days should have a dated `COMMENT ON TABLE` explaining *why* it's still unlogged and when it will be converted. The COMMENT pattern used on `x_delete_body_fulltext` is a perfect vehicle for this.

### 2. `idx_scan = 0` is a partial truth, not a definitive one

`pg_stat_user_indexes.idx_scan` counts since the last stats reset. If the server was restarted recently, counters may be misleadingly low. The real discriminator for "truly unused" is long uptime + diverse workload + zero scans. This server has been up 7 days 21 hours at audit time — enough to validate "not used today" but NOT "never useful".

The 22+ GB in unused EPO classification indexes is suspicious enough to flag but not enough to drop yet — patent search workloads tend to be bursty, not continuous. An index for CPC classification search could be critical during a search session and idle for weeks otherwise.

### 3. Why "biggest first" index rebuilds are the right ordering (for Phase 3)

The final table state is commutative — you'll end up with the same indexes regardless of order. But the practical consideration is **work completed before potential interruption**. If Phase 3 is interrupted by an unrelated crash after 6 hours, you want the already-built indexes to be the ones providing the most query value. Since query cost scales with table size, building the largest-table indexes first maximizes the value of partial completion.

(This is the opposite of the ordering for Phase 1, where smallest-first gives quick validation wins. Different goals.)

### 4. The consistency of FK column naming is a gift

All 46 Phase 3 satellite tables + 7 Rosetta-critical tables use just two FK column names: `doc_id` and `paper_id`, both `text` type. This wasn't accidental — the original schema designer standardized on this convention. As a result, the Phase 3 index build can be fully templated and run as a for-loop over a table list, without per-table column-name investigations. This alone saves days of work and eliminates a whole class of human-error bugs.

### 5. `ClientRead` wait state = Postgres isn't the bottleneck

When a writer backend is in `wait_event = ClientRead`, Postgres is idle waiting for the application to send the next chunk. This was observed during S2 ingest cleanup: after killing the COUNT(*) backends, the COPY writer spent a meaningful fraction of time in ClientRead. The implication is that the ingest is **Python-bound** (gzip+JSON parsing), not DB-bound — so Postgres-side optimization (better indexes, WAL tuning, vacuum settings) can't speed it up. The fix would have to be in the producer.

### 6. "Why is this doing work nobody asked for?" is a crucial observability question

The Sentinel `/loop` cron ran undisturbed for 5 days because nothing tracked "who is driving this process?". It was a live, non-zombied, non-stuck process that happened to be running work nobody had explicitly requested. Monitoring tools that track "last user interaction" as a first-class metric on long-running daemons would have caught this immediately.

**Practical check:** Periodically run `SELECT pid, now()-query_start AS runtime, query FROM pg_stat_activity WHERE state='active'` and ask: is this query being driven by a live session, or is it a zombie's autopilot?

### 7. Orphaned session files after SIGKILL are a recoverable hygiene issue

When `claude` is SIGKILL'd, the cleanup handler doesn't run, so `/home/arc/.claude/sessions/<pid>.json` files get orphaned. Individually harmless (~200 bytes each), but they accumulate. Cleanup one-liner:

```bash
for f in ~/.claude/sessions/*.json; do
  pid=$(basename $f .json)
  kill -0 $pid 2>/dev/null || rm -v $f
done
```

Safe to run at any time — only deletes files whose PIDs are already dead.

---

## Appendix A — Full diagnostic query outputs

All raw query outputs are preserved in the main conversation log and in `/tmp/` files generated during the 2026-04-08 debugging session. The key data has been transcribed into the tables above; if exact reproduction is needed, re-run the queries from the `Query` sections of each finding.

**Queries used:**
1. UNLOGGED tables: `SELECT ... FROM pg_tables WHERE c.relpersistence = 'u' ...`
2. Missing indexes: `SELECT ... FROM pg_tables WHERE pg_index count = 0 ...` (use direct `pg_index` lookup, not `pg_indexes` view, for authoritative results)
3. Index usage: `SELECT ... FROM pg_stat_user_indexes ORDER BY pg_relation_size(indexrelid) DESC LIMIT 30`
4. FK columns: `SELECT ... FROM information_schema.columns WHERE (table_schema, table_name) IN (...)`

---

*Generated by Claude Code on 2026-04-08 during post-Sentinel-incident cleanup. Incident: orphaned `/loop` cron in zombie claude session firing COUNT(*) every 10 minutes on growing paper_citation table, causing load spike to 18 and ~30 min ingest delay. Resolution: identified and killed zombie session (PID 52076, session 942fdf0a), cleaned up suspended session 471648, consolidated tmux, audited schema, produced this report.*

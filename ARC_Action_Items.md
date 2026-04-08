# ARC Action Items

Living checklist of pending operational and development tasks for the ARC project.

## DB Maintenance Queue (post-ingest, do not run until both EPO fulltext AND S2 citations complete)

**Full details:** [`docs/schema/db_maintenance_analysis_20260408.md`](docs/schema/db_maintenance_analysis_20260408.md) — audit findings, full table lists, time estimates, raw query outputs.

**STANDING RULE:** Never DROP or DELETE any table or data without explicit confirmation from Jeff that it is redundant. Rename with `x_delete_` prefix as soft-delete only.

### Phase 1 — Rosetta-critical indexes (run before arc_rosetta_build.py)

- Inspect FK column names on all 7 critical tables first — **DONE 2026-04-08**, all confirmed: `paper_id`/`cited_id` for `paper_citation`, `doc_id` for body_section/abstract/reference/abstract_section/mesh_heading, `paper_id` for paper_field
- CREATE INDEX on:
  - `semanticscholar.paper_citation(paper_id)` — 556 GB table, ~2 hr
  - `semanticscholar.paper_citation(cited_id)` — 556 GB table, ~2 hr (second index for reverse citation direction)
  - `europepmc.body_section(doc_id)` — 442 GB table, ~90 min
  - `europepmc.abstract(doc_id)` — 20 GB
  - `europepmc.reference(doc_id)` — 265 GB, ~75 min
  - `pubmed.abstract_section(doc_id)` — 39 GB
  - `pubmed.mesh_heading(doc_id)` — 26 GB
  - `semanticscholar.paper_field(paper_id)` — 16 GB
- Set `maintenance_work_mem = '4GB'` for the session
- Run smallest-first for quick validation (`paper_field` → `abstract` → `mesh_heading` → `abstract_section` → `reference` → `body_section` → `paper_citation` × 2)

### Phase 2 — Convert core tables UNLOGGED → LOGGED

Priority order (most-queried first):
- `openalex.work` (220 GB) — ~30 min
- `crossref.document` (120 GB) — ~15 min
- `semanticscholar.paper` (191 GB) — ~25 min
- `pubmed.document` (18 GB) — ~3 min
- `europepmc.document` (3.7 GB) — ~1 min
- Then remaining large tables in size order

### Phase 3 — Remaining satellite indexes

All openalex.work_* child tables, crossref/pubmed/europepmc/s2 child tables — **FK columns inspected 2026-04-08**, all confirmed to use `doc_id text` (openalex/pubmed/europepmc/crossref) or `paper_id text` (semanticscholar). Templated for-loop over table list is feasible, no per-table investigation needed.

- Biggest wins first: `openalex.work_concept` (399 GB), `openalex.work_keyword` (225 GB), `openalex.work_related_work` (175 GB), `openalex.work_referenced_work` (152 GB), `openalex.work_topic` (138 GB)
- Then crossref.reference (584 GB — biggest index job of the whole phase)
- Then pubmed and europepmc child tables in size order

### Phase 4 — Convert remaining satellites UNLOGGED → LOGGED

Largest first:
- `crossref.reference` (584 GB)
- `openalex.work_concept` (399 GB)
- `openalex.work_keyword` (225 GB)
- All remaining tables from the 90-table UNLOGGED list in `docs/schema/db_maintenance_analysis_20260408.md`

### Phase 5 — ANALYZE all touched tables

After every SET LOGGED or CREATE INDEX, planner stats may be stale:
- Run `ANALYZE <schema>.<table>` on every table touched in Phases 1–4
- Or equivalently `VACUUM ANALYZE` for a full stats refresh
- Runs in seconds to minutes per table

### Phase 0 — Immediate pre-maintenance cleanup

- **DROP `europepmc.x_delete_body_fulltext`** (207 GB, marked for deletion 2026-04-08) — pending Jeff confirmation per standing rule
- `CHECKPOINT;` before starting Phase 1

### Future ingests pending

- **PMC commercial subset:** Status as of 2026-04-08 — **already ingested** (81 oa_comm tars, 81 oa_noncomm tars, all marked done in `europepmc.ingest_progress`, processed Apr 3–4). **However**, those rows were extracted with the OLD schema (before today's 18-column document additions + 5 new tables for corresponding_author, footnote, trans_title, trans_abstract, supplementary_material). To fully populate the new fields you'd need to re-ingest with the truncate-and-reload pattern OR write a backfill script that re-parses the source XML and UPDATEs existing rows. Decide later.

- **arc_europepmc_ingest.py field-coverage audit (2026-04-08):** All known JATS fields now extracted. See `docs/schema/db_maintenance_analysis_20260408.md` and the in-script DIM_TABLES for the full list. Schema added: 18 doc columns, 5 child tables, source_file on 14 child tables. Filename filtering by source added (`oa_comm` only matches `oa_comm*.tar.gz`).

### Audit other ingest scripts (post-Rosetta task)

After `arc_rosetta_build.py` Phase 1 is complete, audit each of the other source ingest scripts using the same approach used for `arc_europepmc_ingest.py` on 2026-04-08:

1. **For each script in this list:**
   - `arc_openalex_ingest.py`
   - `arc_crossref_ingest.py`
   - `arc_pubmed_ingest.py`
   - `arc_semanticscholar_ingest.py`

2. **Methodology:**
   - Extract one or two sample files from the raw downloads dir
   - Enumerate all unique element/field names in the sample
   - Compare against the corresponding `<schema>.<table>` columns in `arc_v5`
   - Identify gaps: fields present in raw data but not extracted, columns in DB but not populated, source-format quirks the script doesn't handle (e.g., UNLOGGED-vs-LOGGED, body_fulltext-style renames, missing source_file column for audit)
   - Apply patches per the same model: `ALTER TABLE ADD COLUMN IF NOT EXISTS`, code patches, run smoke test against samples before committing
   - Document the audit in `docs/schema/<source>_field_audit_YYYYMMDD.md`

3. **Priority order** (by stakes / coverage):
   - `arc_openalex_ingest.py` first — largest corpus (~250M works), most rosetta-critical
   - `arc_crossref_ingest.py` second — second largest, citation graph source
   - `arc_pubmed_ingest.py` third — biomed metadata, MeSH terms
   - `arc_semanticscholar_ingest.py` last — already integrated via paper_citation linkage

4. **Schedule:** After Rosetta Stone build completes successfully. Each audit ~2-4 hours of investigation + patches + smoke tests.

- **PMC commercial subset launch (alt path if you want to re-ingest):**
  ```bash
  # Truncate the existing ingest_progress for oa_comm to force re-process:
  psql -d arc_v5 -c "DELETE FROM europepmc.ingest_progress WHERE file_name LIKE 'oa_comm%';"
  # Then launch with the fixed script:
  nohup python3 /home/arc/ARC/arc_europepmc_ingest.py --source oa_comm --pmc-dir /data/downloads/pmc_fulltext/ >> /tmp/pmc_comm_ingest.log 2>&1 &
  ```
  WARNING: Re-ingest would create duplicate rows in europepmc.document (ON CONFLICT DO NOTHING handles document_pkey collisions, but child tables have no such guard and would double up). Better approach: write a targeted backfill that reads the raw XMLs and runs UPDATE statements only for the new columns.
- **JPO Japanese patents:** register at PA0630@jpo.go.jp, download bulk XML, ingest Japanese abstracts for 8.6M missing JP patents (see `memory/project_epo_embeddings_coverage.md` for the 66% JP gap context)
- **ArXiv re-extraction:** reprocess 407K truncated papers (`char_count = 50000`) from `/data/downloads/arxiv/src/` using GROBID or LaTeX parser

### Rosetta Stone

- **Dry-run first:**
  ```bash
  python /home/arc/ARC/arc_rosetta_build.py --dry-run
  ```
- **Full run** (only after Phase 1 indexes are built):
  ```bash
  nohup python /home/arc/ARC/arc_rosetta_build.py >> /tmp/rosetta_build.log 2>&1 &
  ```
- After completion: migrate `journal_embeddings.doc_id` → `canonical_id`
- After completion: add `canonical_id` FK to `europepmc.body_section`, `arc_v5.arxiv_fulltext`, and citation tables

### Git

- Fix SSH remote: `git remote set-url origin git@github.com:Jeff-Arc/ARC.git` — **DONE 2026-04-08**
- Commit pending:
  - `arc_rosetta_build.py` — **DONE 2026-04-08**
  - `docs/schema/db_maintenance_analysis_20260408.md` — **DONE 2026-04-08**
  - `arc_semanticscholar_ingest.py` — still pending
  - `arc_docdb_download.py`, `arc_docdb_ingest.py` — modified, pending
  - Various other untracked `arc_*_ingest.py` files

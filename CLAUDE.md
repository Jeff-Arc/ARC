# ARC Project — Claude Code Context

## PROJECT

ARC (Adaptive Research Cartography) maps the semantic topology of patent and journal corpora over time. It clusters document embeddings into persistent research topics, tracks their birth/death/drift across periods, runs ML models to predict cluster lifecycle events, and surfaces structural laws (phase transitions, void geometry, cross-resolution cascades).

## DB CONNECTION

```bash
PGHOST=/var/run/postgresql   # Unix socket — peer auth for jeff (superuser)
PGDATABASE=arc_v4
PGUSER=jeff
# Always pass PGUSER=jeff explicitly — shell may have arc_developer set
```

```python
import psycopg2
conn = psycopg2.connect(host="/var/run/postgresql", dbname="arc_v4", user="jeff")
```

Credentials in `/home/jeff/arc/.env`.

## PYTHON ENVS

- `/home/jeff/miniconda3/bin/python3` — main env: pipeline, ML, API (psycopg2, numpy, sklearn, xgboost, sentence-transformers)
- `/home/jeff/miniconda3/envs/arc311/bin/python` — embed + kNN only (faiss-gpu)

## CANONICAL SCRIPTS

```
arc_ingest.py              Corpus ingest → data_* tables
arc_embed_knn.py           Embed chunks + build kNN  [arc311 env]
ml/arc_ml.py               ML pipeline orchestrator
ml/arc_ml_train.py         Train XGBoost death/birth models
ml/arc_ml_score.py         Score clusters → death_probability
ml/arc_backtest.py         Walk-forward backtest
ml/arc_survival.py         Kaplan-Meier survival curves
ml/arc_discovery.py        Lasso signal discovery + archetype clustering
ml/arc_gnn.py              GraphSAGE GNN experiments
```

## DB NAMING CONVENTIONS

| Prefix | Contains |
|---|---|
| `pipe_*` | Pipeline data (clusters, period stats, voids, edges) |
| `data_*` | Raw input (documents, chunks, embeddings) |
| `sys_*` | Config, task queue, run tracking |
| `sci_*` | Findings, laws, hypotheses |
| `log_*` | Logs, query patterns |
| `ml_*` | ML outputs (models, results, SHAP) |
| `v_*` | Complex views — treat as tables |
| `embeddings.*` | Vector storage (chunk_periods, knn_edges — partitioned) |

## TASK QUEUE

`sys_task_queue` is the single source of truth for all scheduled work.

**Chain:** `embed_knn → pipeline → label → ml → cartographer`

**Task types:**
- `embed_knn` — GPU embed + kNN (pipeline_worker container)
- `pipeline` — `SELECT arc_run_pipeline(corpus_id, NULL)`
- `label` — Claude labels unlabeled clusters (see Section 13 below)
- `ml` — train + score + backtest + survival
- `cartographer` — Claude cartographer session
- `report` — sys_reports generation

**Key functions:**
```sql
SELECT enqueue_corpus_chain('[corpus_id]');   -- full chain (new corpus)
SELECT enqueue_pipeline_only('[corpus_id]');  -- skip embed, rerun pipeline
SELECT arc_run_all_periods('[corpus_id]');    -- run all periods + cluster_matching in one call
-- NOTE: arc_run_all_periods now calls run_cluster_matching automatically after all periods.
-- If using p_start_from to resume, cluster_matching still runs over the full corpus.
SELECT * FROM sys_task_queue_ready LIMIT 1;  -- check for work
SELECT task_start([id], 'agent_YYYYMMDD_HHMM');
SELECT task_complete([id], '[summary]', [rows], ARRAY[[finding_ids]]);
SELECT task_fail([id], '[error]');            -- auto-retries 3x (5m→15m→45m)
SELECT * FROM sys_task_chain_status ORDER BY chain_started DESC LIMIT 5;
```

## SESSION START

1. Run: `arc-status`
   (alias: `PGHOST=/var/run/postgresql psql -U jeff -d arc_v4 -Atq \
   -c "SELECT get_session_context();" > ~/arc/docs/arc_session_context.md`)
2. Read `~/arc/docs/arc_session_context.md` into context
3. You now have full system state — proceed with the task

Generate a session_id: `[session_type]_[YYYYMMDD]_[HHMM]`
Example: `cartographer_20260317_0900`

## POST-PIPELINE CARTOGRAPHER SESSION

After run_corpus.sh completes, launch a cartographer session:
```bash
CORPUS_ID=H01L_quarterly
PERIOD=$(PGHOST=/var/run/postgresql psql -U jeff -d arc_v4 -Atc \
  "SELECT MAX(period_start) FROM pipe_clusters WHERE corpus_id='$CORPUS_ID'")

claude --dangerously-skip-permissions -p "
You are the ARC Cartographer. Corpus: $CORPUS_ID, Period: $PERIOD.
DB: PGHOST=/var/run/postgresql psql -U jeff -d arc_v4

INSTRUCTIONS:
Use SQLCoder (ollama sqlcoder:7b) to generate all SQL queries. Do not
write SQL directly. Generate query → execute via psql → interpret result.

Before spawning any tasks: run the schema context query from Section 10
and store it. All subagents must use this schema block when calling SQLCoder.

Make a plan first. Then spawn parallel subagents using the Task tool for:

Task 1 — Death Risk Review
  Find the 10 highest death_probability clusters this period.
  Pull their score history from ml_score_history (last 4 periods).
  Flag any cluster with 3+ consecutive rising scores.
  Use pipe_cluster_shap_values to identify top risk drivers per cluster.

Task 2 — Law Validation
  Pull all sci_findings WHERE type='law' AND outcome='pending'
  AND '$CORPUS_ID' = ANY(corpus_ids).
  For each law: query the metrics its condition_expression references.
  Assess: confirmed / refuted / ambiguous. Update outcome in DB.

Task 3 — Void Intelligence
  Pull all active pipe_voids for $CORPUS_ID.
  For each void: assess narrowing vs stable using centroid_drift.
  Flag voids where period_count crossed 4, 8, or 12.
  For the largest void by void_size: describe what bridging invention
  would collapse it. File as sci_findings type='finding' subtype='void_bridge'.

Task 4 — Period Anomaly Detection
  Pull pipe_period_stats for $CORPUS_ID last 10 periods.
  Compute mean and stdev for: entropy, algebraic_connectivity,
  mean_drift, n_deaths, n_births.
  Flag any metric more than 2 stdev from mean this period.
  Check if any anomaly matches condition_expression of a known law.

Task 5 — New Hypothesis Generation
  Based on all findings from Tasks 1-4, propose 1-3 new hypotheses
  not already in sci_findings for this corpus.
  Each must be falsifiable with measurable geometric conditions.
  INSERT into sci_findings (type='hypothesis', outcome='pending',
  confidence=0.X, created_by='cartographer_session', corpus_ids=ARRAY['$CORPUS_ID']).

CONFIDENCE SCORES REQUIRED: Every finding and hypothesis filed must include a
confidence score between 0.5 and 1.0 in the INSERT statement. Use these guidelines:
  0.95+ : directly observed, unambiguous metric evidence
  0.85–0.94 : strong evidence, minor interpretive uncertainty
  0.70–0.84 : moderate evidence, some confounders present
  0.50–0.69 : speculative or limited data; flag reasoning
Never file a finding without a confidence column value.

After all tasks complete: synthesize a 1-paragraph period summary.
INSERT into sci_findings (type='finding', subtype='period_summary',
confidence=0.X, corpus_ids=ARRAY['$CORPUS_ID'], period='$PERIOD',
title='Period summary: $CORPUS_ID $PERIOD', created_by='cartographer_session').
Print: laws confirmed/refuted, hypotheses filed, anomalies found, elapsed time.
"
```

Add to run_corpus.sh as final step so it runs automatically after every pipeline.

## LABEL TASK — Cluster Labeling Methodology

Prompt source: `SELECT value FROM sci_cartographer_config WHERE key = 'label_cluster_instructions'`

**v2 procedure uses 12 parallel subagents** (see full instructions in sci_cartographer_config):

1. Call `get_unlabeled_clusters('[corpus_id]')` — returns one row per unlabeled non-junk persistent cluster at peak period
2. Divide into 12 equal batches
3. Spawn 12 parallel subagents via Task tool, each handling one batch
4. Each subagent fetches top 10 chunks by `distance_to_centroid`, generates `cluster_label` (3–5 word noun phrase) and `cluster_summary` (1–2 sentences), writes with:
   ```sql
   UPDATE pipe_clusters
   SET cluster_label = '[label]', cluster_summary = '[summary]'
   WHERE persistent_cluster_id = '[pid]' AND corpus_id = '[corpus_id]'
   ```
5. Collect counts from all subagents; report total labeled

**Rules:**
- Never re-label — `get_unlabeled_clusters()` enforces this, trust its output
- Each subagent writes directly; no coordination needed between batches
- corpus_id filter on UPDATE prevents cross-corpus contamination

## GPU ENVIRONMENT SPLIT

Two Python environments — never cross them:

| Phase | Env | Command |
|---|---|---|
| embed + kNN + Leiden (full run) | MAIN `/home/jeff/miniconda3/bin/python3` | `python3 arc_embed_knn.py --corpus-id X` |
| kNN-only re-run (skip embed+Leiden) | arc311 `/home/jeff/miniconda3/envs/arc311/bin/python` | `python arc_embed_knn.py --corpus-id X --skip-embed --skip-leiden` |

- Never run embed in arc311 (no torch installed)
- Never run two full embed tasks simultaneously — GPU OOM risk
- kNN-only and pipeline SQL tasks can run in parallel safely

## QUEUE WORKER MONITORING

When running embed tasks from the queue:
- Check `nvidia-smi` every 10 min during embed phases — GPU memory should stabilize within 2 min
- Check `SELECT pid, query, now()-query_start as elapsed FROM pg_stat_activity WHERE state='active' ORDER BY elapsed DESC LIMIT 10;` for processes running >2 hours on the same period
- If stuck process found: `SELECT pg_terminate_backend(pid)` (for SQL-only phases)
- SVD hangs (arc_run_pipeline compute_svd_geometry): require `sudo kill -9 <pid>` — uninterruptible
- Run max 1 embed task at a time if GPU memory < 8GB free; 2 max otherwise
- SQL pipeline tasks (`pipeline`, `label`, `ml`) can run 4–6 in parallel safely
- If RAM free < 4GB: pause new pipeline tasks until memory recovers

## Session Loop

On every session start or context refresh:

STEP 1 — Read context:
Run arc-status alias and read output.

STEP 2 — Check queue:
```sql
SELECT corpus_id, task_type, priority, chain_position
FROM sys_task_queue
WHERE status='pending'
ORDER BY priority, corpus_id, chain_position
LIMIT 20
```

IF pending tasks exist:
- Execute highest priority task
- Spawn parallel subagents for independent tasks
  (max 2 embed tasks simultaneously — GPU OOM risk)
  (max 6 SQL pipeline tasks simultaneously — CPU bound)
- Monitor: nvidia-smi every 10 min during embed
- Monitor:
  ```sql
  SELECT pid, now()-query_start as duration,
    LEFT(query,80) FROM pg_stat_activity
    WHERE state='active' AND query LIKE '%arc_run_pipeline%'
  ```
  Kill any process running >2 hours on same period:
  ```sql
  SELECT pg_terminate_backend(pid)
  ```
- After each task: `SELECT task_complete(...)`

STEP 3 — IF queue empty: EXPLORATION MODE
See Section: Exploration Mode below.

## Exploration Mode

When queue is empty, Claude explores freely and files findings.
Spin up subagents for parallel exploration threads.

Available exploration types (pick based on what was least
recently explored — check sci_findings.created_at):

TYPE 1 — Cross-domain void analysis:
Find voids that span multiple corpora domains.
```sql
SELECT v1.corpus_id, v2.corpus_id,
  v1.persistent_cluster_a, v2.persistent_cluster_a,
  v1.void_size, v2.void_size
FROM pipe_voids v1
JOIN pipe_voids v2 ON
  v1.persistent_cluster_a != v2.persistent_cluster_a
WHERE v1.status='active' AND v2.status='active'
AND v1.corpus_id != v2.corpus_id
```
Look for clusters in different domains that are
semantically adjacent — potential cross-domain voids.

TYPE 2 — Crystallization sequence monitoring:
For each active corpus check if any cluster is at
step 3-4 of the 5-step crystallization sequence.
These are imminent paradigm shifts.
Check: density increasing, entropy dropping,
boundary_sharpness rising, convergence_score > 0.7

TYPE 3 — Law validation across new corpora:
Take each law in sci_findings WHERE type='law'
and check if it holds for corpora that were added
since the law was discovered.
File confirmation or refutation finding.

TYPE 4 — Assignee convergence patterns:
For each active void, check if any assignee
is filing in BOTH adjacent clusters simultaneously.
```sql
SELECT a.assignee_name, COUNT(DISTINCT cluster_id)
FROM data_assignee_normalized a
JOIN data_documents d ON d.document_id=a.document_id
JOIN embeddings.chunk_periods cp ON cp.chunk_id IN (
  SELECT chunk_id FROM data_chunks
  WHERE document_id=d.document_id
)
WHERE cp.corpus_id=[corpus]
AND cp.persistent_cluster_id IN
  (void.persistent_cluster_a, void.persistent_cluster_b)
GROUP BY a.assignee_name
HAVING COUNT(DISTINCT cluster_id) > 1
```

TYPE 5 — Identify new corpus opportunities:
Look at active voids and findings.
If a void spans a domain with no corresponding corpus,
or if a finding references a technology area not yet ingested:
- Determine the relevant CPC code or OpenAlex concept
- Check if already in sys_run_config
- If not: INSERT into sys_run_config with appropriate settings
- Queue an ingest task
- File a finding: 'Queued [corpus] — identified as adjacent
  to void [id] in [domain]'

TYPE 6 — PLS predictions:
Find journal clusters (openalex corpora) that are
splitting or showing high entropy —
these precede patent filings by 6-9 months.
Check openalex corpora for clusters with
high phase_transition_score in latest 2 periods.
Cross-reference with patent voids in same domain.

Store exploration instructions in sci_cartographer_config:
```sql
INSERT INTO sci_cartographer_config (key, value)
VALUES ('exploration_types', '[JSON of above types]')
ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value;
```

## GPU Environment Split

MAIN env (`/home/jeff/miniconda3/bin/python3`):
  Full run: `python3 arc_embed_knn.py --corpus-id X`
  (embed + kNN + Leiden — requires torch + faiss + leidenalg)

arc311 env (`/home/jeff/miniconda3/envs/arc311/bin/python`):
  kNN-only: `python arc_embed_knn.py --corpus-id X --skip-embed --skip-leiden`
  (FAISS only — no torch required)

NEVER run embed tasks in parallel on same GPU.
Run embed tasks sequentially: one at a time.
kNN-only and SQL pipeline tasks can run in parallel.

## Identify Task

When exploration finds a domain with no corpus:
1. Determine CPC code (patents) or concept_id (OpenAlex)
2. Check sys_run_config for existing registration
3. If not registered:
   ```sql
   INSERT INTO sys_run_config (legacy_name, source_type,
   source_filter, resolution, year_from, status, domain)
   VALUES (...)
   ```
4. Queue ingest:
   ```sql
   INSERT INTO sys_task_queue (task_type='ingest',
   corpus_id=legacy_name, status='pending', priority=3)
   ```
5. File sci_finding:
   ```sql
   -- type='hypothesis', title='New corpus queued: [name]',
   -- description='Queued because [reason from exploration]'
   ```

Do NOT ingest large corpora (>500K docs estimated) without
checking with user first. Flag and wait.

## Corpus Discovery and Registration

### Before registering a new corpus, run discovery queries:

#### Patents — estimate count for a CPC prefix (no ingest needed):
```bash
# Fast count from USPTO bulk data (seconds, no DB required):
grep -c $'\tH01S' /home/jeff/data/g_cpc_current.tsv

# More precise: unique patents matching a CPC prefix:
awk -F'\t' '$4 ~ /^H01S/' /home/jeff/data/g_cpc_current.tsv | cut -f1 | sort -u | wc -l
```

```sql
-- CPC codes inside already-ingested corpora:
SELECT UNNEST(cpc_codes) as cpc_code, COUNT(*) as patents
FROM data_documents
WHERE source_type = 'patents' AND corpus_id = 'H01L_quarterly'
GROUP BY cpc_code ORDER BY patents DESC LIMIT 20;

-- Find adjacent CPC codes near a void cluster:
SELECT UNNEST(d.cpc_codes) as cpc, COUNT(*) as patents
FROM data_documents d
JOIN data_chunks dc ON dc.document_id = d.document_id
JOIN embeddings.chunk_periods cp ON cp.chunk_id = dc.chunk_id
WHERE cp.persistent_cluster_id = 'H01L_quarterly_0110'
GROUP BY cpc ORDER BY patents DESC LIMIT 10;
```

#### OpenAlex — check fields already ingested and available sizes:
```sql
-- Fields already ingested:
SELECT source_api, COUNT(*) as papers
FROM data_documents WHERE source_type = 'papers'
GROUP BY source_api ORDER BY papers DESC;

-- Topics from already-ingested OpenAlex data:
SELECT cpc_codes[1] as topic, COUNT(*) as papers
FROM data_documents WHERE source_type = 'papers'
GROUP BY cpc_codes[1] ORDER BY papers DESC LIMIT 20;
```

OpenAlex field IDs (use as `source_filter` for papers corpora):
- Engineering: `https://openalex.org/fields/22`
- Computer Science: `https://openalex.org/fields/41`
- Physics: `https://openalex.org/fields/51`
- Materials Science: `https://openalex.org/fields/88`
- Chemistry: `https://openalex.org/fields/15`
- Medicine: `https://openalex.org/fields/27`
- Biology: `https://openalex.org/fields/11`

### Register a new corpus:

Use `register_corpus()` — validates inputs, guards against duplicates, sets defaults.

```sql
-- Patents:
SELECT register_corpus(
  'H01S_quarterly',        -- corpus_id (becomes legacy_name)
  'patents',               -- source_type: 'patents' | 'papers' | 'arc'
  'H01S',                  -- source_filter: CPC prefix
  'quarterly',             -- resolution: 'quarterly' | 'annual' | 'biannual'
  1990,                    -- year_from
  'semiconductors',        -- domain: 'ai_ml'|'semiconductors'|'longevity'|'narrative'|'other'|NULL
  'Lasers and Photonics'   -- label (defaults to corpus_id if NULL)
);

-- OpenAlex papers (domain 'physics' is NOT valid — use 'other' or NULL):
SELECT register_corpus(
  'openalex_physics_full',
  'papers',
  'https://openalex.org/fields/51',
  'quarterly',
  2000,
  'other',
  'Physics Journals',
  ARRAY['photonics','optics','laser','semiconductor']  -- concept keywords filter
);
```

**IMPORTANT:** `domain` CHECK constraint only accepts: `ai_ml`, `semiconductors`, `longevity`,
`narrative`, `other` (or NULL). The function returns an `ERROR:` string (not exception) if
invalid — always check the return value.

Returns:
- `REGISTERED: corpus_id (...)` — success
- `EXISTS: corpus_id already registered` — idempotent, no change
- `ERROR: ...` — validation failed, nothing written

### After registration, ingest:
```bash
python3 arc_ingest.py --corpus-id [corpus_name]
# Then queue the full chain:
```
```sql
SELECT enqueue_corpus_chain('[corpus_name]');
```

### Check what's registered:
```sql
SELECT legacy_name as corpus_id, source_type, source_filter,
       resolution, year_from, domain, status
FROM sys_run_config ORDER BY status, legacy_name;
```

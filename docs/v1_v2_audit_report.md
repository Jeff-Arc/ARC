# ARC V1 vs V2 Pipeline Audit Report

**Generated:** 2026-03-20
**Corpora examined:** G06N_quarterly (910 cluster-period rows, 64 periods), C23C_quarterly (2020 rows, 143 periods)
**V1 tables:** `pipe_clusters`, `pipe_period_stats`, `pipe_voids`, `pipe_cluster_edges`
**V2 tables:** `f_cluster_period`, `f_period`, `f_void`, `f_edge`

---

## Executive Summary

The f_ tables represent a **planned V2 pipeline schema** currently populated for exactly 2 corpora: G06N_quarterly and C23C_quarterly. The table comment on `f_cluster_period` states: *"V2 pipeline: cluster-level measures per period. Replaces pipe_clusters + ml_score_history."*

**V2 is not yet ready to replace V1.** The schema design is sound and conceptually cleaner, but several critical metrics diverge substantially from V1, the death probability scoring methodology has changed fundamentally (from continuous XGBoost scores to near-binary values), and the system that populates f_ tables is not connected to the standard pipeline: no `arc_run_pipeline`, `arc_compute_period`, or any other standard procedure writes to any f_ table. The population mechanism is unknown or not yet implemented in DB-resident code.

**Key findings:**

1. **Row counts are identical** for cluster and period tables. Void and edge tables diverge significantly (V2 voids: 2836 vs V1 2669 for G06N; V2 edges: 5948 vs V1 14879 for G06N).
2. **Cohesion, drift_magnitude, persistence_score, convergence_score, age_periods:** correlation ≥ 0.985 — effectively identical.
3. **boundary_pressure_rate:** correlation 0.2956, mean delta -0.789. V1 range [0.017, 1.000], V2 range [-0.554, 0.653]. **Fundamentally different formula.**
4. **death_probability:** V1 produces continuous XGBoost probabilities (full distribution); V2 produces near-binary values (789/910 rows = 0.0, 117 rows = 1.0, 4 rows ~0.9). **Scoring approach completely changed.**
5. **system_entropy:** correlation 0.155, V1 mean 3.50, V2 mean 2.59. **Different formula or base.**
6. **algebraic_connectivity:** correlation -0.231. V1 uses count-based edge weights (max 908), V2 uses normalized weights (max 0.549). **Fundamentally different computation.**
7. **V2 adds 7 new cluster-level columns** not in V1: `compactness`, `compactness_percentile`, `novelty_percentile`, `drift_vector`, `corpus_centroid_drift`, `is_precursor`, `mean_novelty_score`.
8. **V2 voids gain per-period snapshot rows** (period_start present); V1 voids are lifetime records without period_start. V2 has 100% void_size fill vs V1's 41.8%.
9. **V2 edges use normalized fractional weights** (proportion of kNN connections) vs V1's integer counts.

---

## Section 1: Column Coverage

### 1.1 pipe_clusters vs f_cluster_period

**Columns present in BOTH tables (shared):**

| Column | V1 Type | V2 Type | Notes |
|--------|---------|---------|-------|
| corpus_id | text | text | |
| period_start | date | date | |
| cluster_id | integer | integer | |
| persistent_cluster_id | text | text | |
| cluster_label | text | text | |
| cluster_summary | text | text | |
| is_junk | boolean | boolean | |
| centroid | USER-DEFINED | USER-DEFINED | |
| spread | real | real | |
| cohesion | real | real | Same formula |
| elongation_ratio | real | real | |
| dist_cv | real | real | |
| distance_entropy | real | real | |
| size | integer | integer | |
| mean_betweenness | real | real | |
| avg_path_length | real | real | |
| cluster_reachability | integer | integer | |
| propagation_speed | real | real | |
| semantic_distance_to_nearest | real | real | |
| size_percentile | real | real | |
| drift_percentile | real | real | |
| boundary_percentile | real | real | |
| mean_prosecution_duration | real | real | |
| prosecution_heat_index | real | real | |
| drift_magnitude | real | real | |
| velocity | real | real | |
| acceleration | real | real | |
| jerk | real | real | |
| boundary_pressure_rate | real | real | **Different formula — see Section 7** |
| velocity_direction_stability | real | real | |
| convergence_score | real | real | |
| velocity_vector | USER-DEFINED | USER-DEFINED | |
| velocity_alignment | real | real | |
| inheritance_score | real | real | |
| field_surprise_index | real | real | |
| age_periods | integer | integer | |
| persistence_score | real | real | |
| membership_churn | real | real | |
| is_new | boolean | boolean | |
| is_dead | boolean | boolean | |
| is_split_child | boolean | boolean | |
| marginal_entropy_impact | real | real | |
| death_probability | real | real | **Different scoring — see Section 7** |
| birth_probability | real | real | |
| mean_boundary_score | real | real | |
| cohesion_percentile | real | real | |

**V1-only columns (35 columns in pipe_clusters not in f_cluster_period):**

| V1 Column | Data Type | Notes |
|-----------|-----------|-------|
| corpus_type | text | V2 infers from corpus_id |
| mean_l2_spread | real | L2 spread metric |
| mean_intra_distance | real | Intra-cluster distance |
| volume_estimate | real | Cluster volume |
| skewness_pc1 | real | SVD geometry |
| skewness_pc2 | real | SVD geometry |
| kurtosis_pc1 | real | SVD geometry |
| kurtosis_pc2 | real | SVD geometry |
| mean_boundary_proximity | real | |
| max_boundary_proximity | real | |
| mean_density | real | |
| outlier_fraction | real | |
| mean_uncertainty | real | |
| mean_energy | real | |
| n_attractors | integer | |
| n_saddle_points | integer | |
| growth_rate | real | |
| max_betweenness | real | Max (vs mean) betweenness |
| is_merger_target | boolean | |
| period_end | date | V2 omits (computable) |
| period | text | V2 omits (computable) |
| mean_intrinsic_dim | real | |
| prosecution_duration_stddev | real | |
| prosecution_duration_trend | real | |
| labelled_at | timestamp with time zone | |
| match_similarity | double precision | Cluster matching score |
| match_type | text | Matching type |
| event_model_version | text | |
| boundary_sharpness | real | |
| citations_per_chunk | real | |
| citation_rate_trend | real | |
| dist_skewness | real | |
| dist_kurtosis | real | |
| shape_type | text | |
| mean_perturbation_score | real | |

**V2-only columns (7 new columns in f_cluster_period):**

| V2 Column | Data Type | Notes |
|-----------|-----------|-------|
| compactness | real | Replaces cohesion conceptually (different scale) |
| compactness_percentile | real | Percentile rank of compactness |
| novelty_percentile | real | Percentile rank of novelty |
| drift_vector | USER-DEFINED | Full drift direction vector |
| corpus_centroid_drift | real | Drift relative to corpus centroid |
| is_precursor | boolean | Flags clusters that precede a birth event |
| mean_novelty_score | real | Mean novelty from f_chunk_period |

### 1.2 pipe_period_stats vs f_period

**pipe_period_stats:** 49 columns
**f_period:** 28 columns

**Shared columns (28):**

| Column | Notes |
|--------|-------|
| corpus_id | |
| period_start | |
| period_end | |
| n_clusters | |
| n_junk | |
| n_births | Different counts — see Section 7 |
| n_deaths | Different counts — see Section 7 |
| n_splits | |
| n_mergers | |
| n_chunks_total | |
| n_documents_total | |
| n_dark_matter_chunks | |
| n_converging_pairs | |
| void_count | Correlation 0.999 — essentially identical |
| system_entropy | **Correlation 0.155 — different formula** |
| phase_transition_score | **Correlation 0.421 — different formula** |
| anomaly_score | |
| global_avg_path_length | |
| graph_diameter | V1: integer; V2: real |
| max_void_persistence | |
| field_surprise_index | |
| velocity_shock_ratio | |
| corpus_centroid_drift | |
| topology_reorganization | |
| entropy_trend | |
| algebraic_connectivity | **Correlation -0.231 — fundamentally different** |
| spectral_gap | |
| n_communities_spectral | |

**V1-only columns in pipe_period_stats (21):**

| Column | Notes |
|--------|-------|
| corpus_type | |
| n_chunks_new | |
| n_documents_new | |
| cluster_size_entropy | |
| kl_divergence_from_uniform | |
| mean_density | |
| global_clustering_coefficient | |
| rewiring_rate | |
| n_new_connections | |
| n_lost_connections | |
| rupture_rate | |
| n_reassignments | |
| is_refresh_period | |
| leiden_modularity | |
| n_shadow_clusters | |
| n_precursor_clusters | |
| n_cluster_changes | |
| period | text label |
| mean_prosecution_duration | |
| anomaly_model_version | |
| void_fill_probability | |

**f_period has NO additional columns beyond those in pipe_period_stats.** It is a strict subset.

### 1.3 pipe_voids vs f_void

**V1-only columns in pipe_voids:**

| Column | Notes |
|--------|-------|
| id | Surrogate key (V1 only) |
| last_seen | Lifetime record field (V2 is per-period snapshot) |
| centroid_drift | 0% fill in V1 — effectively dead |

**V2-only columns in f_void:**

| Column | Notes |
|--------|-------|
| period_start | V2 is a per-period snapshot (key architectural difference) |
| void_density | 0% fill in V2 currently |
| is_new_void | Boolean: first appearance in this period |
| is_closing | Boolean: last appearance in this period |

### 1.4 pipe_cluster_edges vs f_edge

**V1-only columns:**

| Column | Notes |
|--------|-------|
| mean_boundary_score | |
| n_bridge_chunks | |
| period_end | |
| period | text label |
| graph_distance | 41.5% fill in V1 |
| void_dark_matter_count | |
| void_density | |
| semantic_overlap_max | |

**Shared columns:** corpus_id, period_start, cluster_a, cluster_b, connection_weight, convergence_score, semantic_overlap_a_to_b, semantic_overlap_b_to_a, is_new, is_lost, weight_change

---

## Section 2: Row Count Comparison

### 2.1 Cluster Tables

| Source | Corpus | Rows |
|--------|--------|------|
| pipe_clusters | G06N_quarterly | 910 |
| f_cluster_period | G06N_quarterly | 910 |
| pipe_clusters | C23C_quarterly | 2020 |
| f_cluster_period | C23C_quarterly | 2020 |

**Finding:** Cluster-period row counts are identical. Both tables contain the same (corpus_id, period_start, cluster_id) tuples.

### 2.2 Period Tables

| Source | Corpus | Rows |
|--------|--------|------|
| pipe_period_stats | G06N_quarterly | 64 |
| f_period | G06N_quarterly | 64 |
| pipe_period_stats | C23C_quarterly | 143 |
| f_period | C23C_quarterly | 143 |

**Finding:** Period row counts are identical.

### 2.3 Void Tables

| Source | Corpus | Rows |
|--------|--------|------|
| pipe_voids | G06N_quarterly | 2,669 |
| f_void | G06N_quarterly | 2,836 |
| pipe_voids | C23C_quarterly | 1,413 |
| f_void | C23C_quarterly | 4,040 |

**Finding:** Void counts diverge significantly. V1 stores one row per unique void pair (lifetime record). V2 stores one row per (void pair, period_start) — a per-period snapshot. This explains the higher V2 counts. C23C shows a 2.86× ratio.

### 2.4 Edge Tables

| Source | Corpus | Rows |
|--------|--------|------|
| pipe_cluster_edges | G06N_quarterly | 14,879 |
| f_edge | G06N_quarterly | 5,948 |
| pipe_cluster_edges | C23C_quarterly | 17,996 |
| f_edge | C23C_quarterly | 13,459 |

**Finding:** V1 maintains fixed k=15 edges per cluster (all 225 maximum edges per period for G06N in later periods). V2 appears to filter by a weight threshold, keeping only edges with meaningful semantic overlap. V1 average weight per period: 232 edges/period; V2: 93 edges/period. V2 connection_weight is normalized [0.001, 0.549]; V1 is a raw count [0, 908].

---

## Section 3: Numerical Comparison — Cluster Level (G06N_quarterly)

All correlations computed on matched rows (INNER JOIN on corpus_id, period_start, cluster_id). Non-null pairs only.

| Measure | n | V1 Mean | V2 Mean | Mean Delta | Correlation | Assessment |
|---------|---|---------|---------|------------|-------------|------------|
| cohesion | 910 | 0.901570 | 0.901570 | 0.000000 | **1.0000** | Identical |
| drift_magnitude | 775 | 0.107230 | 0.107230 | 0.000000 | **1.0000** | Identical |
| age_periods | 910 | 10.2637 | 10.2637 | 0.000000 | **0.9856** | Near-identical |
| persistence_score | 902 | 0.161655 | 0.161655 | 0.000000 | **0.9855** | Near-identical |
| convergence_score | 861 | 0.034180 | 0.034180 | 0.000000 | **1.0000** | Identical |
| mean_betweenness | 910 | 0.004479 | 0.004470 | -0.000009 | **0.9938** | Essentially identical |
| death_probability | 910 | 0.143645 | 0.132437 | -0.011208 | **0.5363** | **DIVERGED — different methodology** |
| boundary_pressure_rate | 775 | 0.789648 | 0.000369 | -0.789279 | **0.2956** | **FUNDAMENTALLY DIFFERENT** |
| compactness (V2) vs cohesion (V1) | 905 | 0.906551 | 0.308320 | -0.598231 | 0.1756 | Different metric, different scale |

**Critical observations:**

- **cohesion, drift_magnitude, convergence_score:** Perfectly identical (correlation = 1.000, delta = 0). These are computed by the same stored procedures writing to the same underlying columns — V2 copies from V1.
- **boundary_pressure_rate:** V1 range [0.017, 1.000] with mean 0.790. V2 range [-0.554, 0.653] with mean 0.000. Completely different formula. V1 appears to store a rate bounded [0,1]; V2 stores a signed delta (change in pressure). This is a breaking semantic change.
- **death_probability:** V1 = continuous XGBoost probability (full distribution from ~0 to 1). V2 = near-binary (86.7% are exactly 0.0, 12.9% are exactly 1.0). V2 is using is_dead as a proxy rather than an ML model score.
- **compactness (V2):** Distinct from V1 cohesion — different formula producing range [0.15, 0.55] vs cohesion [0.83, 0.99].

---

## Section 4: Period-Level Comparison (G06N_quarterly)

| Measure | n | V1 Mean | V2 Mean | Mean Delta | Correlation | Assessment |
|---------|---|---------|---------|------------|-------------|------------|
| void_count | 64 | 44.19 | 44.31 | +0.125 | **0.9989** | Essentially identical |
| n_births | 64 | 2.109 | 1.922 | -0.188 | **0.9856** | Near-identical |
| n_deaths | 64 | 2.109 | 1.922 | -0.188 | 0.3358 | **DIVERGED — counting difference** |
| phase_transition_score | 64 | 0.185 | 0.106 | -0.079 | 0.4207 | **Partially correlated, different scale** |
| system_entropy | 64 | 3.504 | 2.587 | -0.917 | 0.1554 | **FUNDAMENTALLY DIFFERENT** |
| algebraic_connectivity | 64 | 1.879 | 0.445 | -1.434 | **-0.2314** | **FUNDAMENTALLY DIFFERENT (negative corr!)** |

**Critical observations:**

- **void_count:** Near-perfect correlation (0.999). Both versions count the same voids.
- **n_births / n_deaths:** Both have mean delta -0.188. n_births has high correlation (0.986), but n_deaths has low correlation (0.336). This suggests V2 classifies deaths differently — possibly misidentifying some deaths as something else. The means are identical (both 2.109 V1, 1.922 V2), which means the same average but different per-period assignment.
- **system_entropy:** V1 mean 3.50, V2 mean 2.59. The ratio (~0.74) is consistent with a log-base change or inclusion of junk clusters in V1. Low correlation (0.155) means V2 entropy is not tracking the same temporal patterns.
- **algebraic_connectivity:** Negative correlation (-0.231) confirms these are computed from fundamentally different edge representations. V1 uses kNN count-based Laplacian; V2 uses f_edge normalized weights via `f_compute_spectral_geometry`. V1 range: [0.000, 120.25] (unscaled). V2 range: [0.145, 0.740] (normalized). The 2010 periods show V1 = 0 (disconnected under count metric) but V2 = 0.58-0.74 (connected under normalized metric).

---

## Section 5: Columns in V1 with No V2 Equivalent

### High-value V1-only cluster columns (fill rates from G06N_quarterly):

| Column | Fill Rate (G06N) | Notes |
|--------|-----------------|-------|
| avg_path_length | 905/910 = 99.5% | Graph diameter metric — high value |
| propagation_speed | 905/910 = 99.5% | Information propagation — high value |
| cluster_reachability | 910/910 = 100% | Full fill — high value |
| boundary_sharpness | 902/910 = 99.1% | Boundary definition quality |
| dist_skewness / dist_kurtosis | 902/910 = 99.1% | Distribution shape |
| shape_type | 902/910 = 99.1% | Cluster geometry classification |
| is_split_child | 104/910 = 11.4% | Low fill but meaningful for splits |
| mean_perturbation_score | 352/910 = 38.7% | Partial fill |
| growth_rate | Computed | Growth trajectory |
| volume_estimate | Computed | Cluster spatial volume |
| n_attractors / n_saddle_points | Computed | Topological features |
| mean_density / outlier_fraction | Computed | Density distribution |
| max_betweenness | Computed | Peak centrality |
| boundary_sharpness | Computed | Boundary definition |
| citations_per_chunk | Full | Patent-specific metric |
| mean_intrinsic_dim | Computed | Dimensionality |
| skewness_pc1/2, kurtosis_pc1/2 | Computed | SVD higher-order moments |

### High-value V1-only period columns:

| Column | Notes |
|--------|-------|
| leiden_modularity | Community detection quality — critical for cluster validity |
| cluster_size_entropy | Distribution of cluster sizes |
| kl_divergence_from_uniform | Entropy relative to uniform distribution |
| global_clustering_coefficient | Triangle density in cluster graph |
| rewiring_rate | Edge turnover rate |
| rupture_rate | Sudden disconnections |
| n_reassignments | Chunk migration between clusters |
| n_shadow_clusters | Small temporary clusters |
| n_precursor_clusters | Count of precursor-flagged clusters |
| void_fill_probability | (0% fill even in V1) |

---

## Section 6: New V2 Measures

| V2-only Column | Table | Description | Fill Rate (G06N/C23C) |
|----------------|-------|-------------|----------------------|
| compactness | f_cluster_period | Normalized cohesion metric on different scale [0.15–0.55] | 99.5% / 97.9% |
| compactness_percentile | f_cluster_period | Percentile rank of compactness within corpus-period | 100% / 100% |
| novelty_percentile | f_cluster_period | Percentile rank of mean_novelty_score | 100% / 100% |
| drift_vector | f_cluster_period | Full vector direction of drift (not just magnitude) | 85.2% / 91.3% |
| corpus_centroid_drift | f_cluster_period | Drift relative to whole-corpus centroid | **0% (no data)** |
| is_precursor | f_cluster_period | Boolean: cluster precedes a birth event | **0% (no data)** |
| mean_novelty_score | f_cluster_period | Mean novelty from f_chunk_period.novelty_score | 99.5% / 97.9% |
| void_density | f_void | Density of chunks in void region | 0% (no data) |
| is_new_void | f_void | First appearance of this void pair | 100% |
| is_closing | f_void | Last appearance of this void pair | 100% |

**Notable gaps:** corpus_centroid_drift and is_precursor are 0% filled despite being V2-defined columns. void_density is also empty. These are schema placeholders not yet computed.

---

## Section 7: Methodology Differences

### 7.1 boundary_pressure_rate

**V1:** Bounded rate in [0, 1]. Appears to measure the proportion of clusters exerting boundary pressure, stored as a positive-only fraction.

**V2:** Signed delta metric in approximately [-0.55, 0.65]. Appears to be the *change* in boundary pressure from the previous period, not the absolute rate. V1 mean 0.790 (high, suggesting most clusters are under pressure), V2 mean 0.000 (centered on zero, a delta).

**Impact:** Direct comparison is invalid. V1 BPR feeds death model training (v_cluster_event_training). V2 BPR cannot substitute.

### 7.2 death_probability

**V1:** Continuous XGBoost probability from the arc_ml pipeline. Distribution: ~0 to 1.0 across all buckets. Mean 0.144, stddev 0.207.

**V2:** Near-binary. 86.7% of rows = 0.0 exactly, 12.9% = 1.0 exactly, 0.4% ≈ 0.9. This is effectively `is_dead::numeric` — not an ML model score. The V2 death_probability appears to be set to 1.0 for clusters where `is_dead = true` and 0.0 otherwise. This destroys predictive value.

**Impact:** V2 cannot be used for death risk ranking. The entire ML scoring pipeline would need to be rebuilt for V2.

### 7.3 system_entropy

**V1:** Mean 3.50 nats (base e, includes all clusters). V2: Mean 2.59 nats. Consistent offset of ~0.9 nats. Low correlation (0.155) means divergence is not a simple scale factor. Possible explanations: V2 excludes junk clusters from entropy computation; V2 uses a different cluster size weight; V2 uses log2 vs loge; or V2 samples differently. Without inspecting the V2 population logic, the exact cause cannot be determined.

### 7.4 algebraic_connectivity

**V1:** Computed from `pipe_cluster_edges` using raw kNN connection counts (integers, range 0–908). Laplacian built from count-weighted adjacency. Result: large eigenvalues when clusters are densely connected by raw count. Many periods show 0.0 (disconnected graph after threshold), one outlier period shows 120.25.

**V2:** Computed by `f_compute_spectral_geometry` from `f_edge` using normalized fractional weights (range 0.001–0.549). Produces small, stable eigenvalues in [0.14, 0.74]. V2 approach is more mathematically principled but not comparable to V1 values.

**Impact:** The negative correlation means the two versions rank periods in opposite order by connectivity. Any law or hypothesis derived from V1 algebraic_connectivity cannot be applied to V2 without re-derivation.

### 7.5 n_deaths counting

**V1 and V2 share the same n_births average** (2.109 vs 1.922, corr 0.986 for births) but **n_deaths shows low correlation** (0.336) despite identical averages. Inspection reveals specific periods where V1 counts 13 deaths and V2 counts 2 (2010-01-01), or V1 counts 1 death and V2 counts 16 (2025-07-01). V2 appears to classify some terminal-period deaths differently.

### 7.6 Void architecture

**V1:** Lifetime records in `pipe_voids`. One row per unique (persistent_cluster_a, persistent_cluster_b) pair. first_seen and last_seen dates. 41.8% void_size fill (1116/2669 rows), 0.5% void_type fill (13/2669). centroid_drift: 0% fill system-wide.

**V2:** Per-period snapshots in `f_void`. One row per (persistent_cluster_a, persistent_cluster_b, period_start). 100% void_size fill. is_new_void and is_closing boolean flags allow tracking void lifecycle. void_density: 0% fill.

**Impact:** V2 void architecture is significantly better for temporal analysis. The per-period snapshot allows tracking void evolution. V1 void geometry is effectively non-functional (0% centroid_drift, 0.5% void_type).

### 7.7 Edge weight normalization

**V1:** `connection_weight` = raw count of kNN edges between clusters (integer). V1 maintains exactly `k` edges per cluster regardless of semantic proximity.

**V2:** `connection_weight` = fraction of kNN neighbors that point to each cluster (normalized [0,1]). V2 filters by weight threshold, keeping only meaningful connections. Average edges per period: V1=232, V2=93.

---

## Section 8: Data Quality Comparison

### 8.1 Fill Rate Comparison — Non-junk Clusters

| Corpus | Version | Total | cohesion | drift_mag | betweenness | convergence | bpr_fill | death_prob |
|--------|---------|-------|----------|-----------|-------------|-------------|----------|------------|
| C23C | V1 | 1989 | 0.990 | 0.913 | 0.437 | 0.905 | 0.984 | 0.988 |
| C23C | V2 | 1989 | 0.990 | 0.914 | **1.000** | **0.998** | 0.906 | **1.000** |
| G06N | V1 | 898 | 1.000 | 0.863 | 1.000 | 0.952 | 1.000 | 1.000 |
| G06N | V2 | 898 | 1.000 | 0.863 | 1.000 | 0.952 | 0.863 | 1.000 |

**Observations:**
- V2 achieves 100% betweenness fill for C23C vs V1's 43.7%. This is a meaningful improvement.
- V2 convergence fill is slightly higher for C23C (0.998 vs 0.905).
- V2 BPR fill is lower for G06N (0.863 vs 1.000) — consistent with the different formula producing more NULLs.

### 8.2 V1-only Columns — Fill Rates (G06N_quarterly, all rows)

| Column | Non-null Count | Total | Fill Rate |
|--------|----------------|-------|-----------|
| cluster_reachability | 910 | 910 | 100.0% |
| avg_path_length | 905 | 910 | 99.5% |
| propagation_speed | 905 | 910 | 99.5% |
| boundary_sharpness | 902 | 910 | 99.1% |
| dist_skewness | 902 | 910 | 99.1% |
| dist_kurtosis | 902 | 910 | 99.1% |
| shape_type | 902 | 910 | 99.1% |
| mean_perturbation_score | 352 | 910 | 38.7% |
| is_split_child | 104 | 910 | 11.4% |

### 8.3 V2-only Columns — Fill Rates

| Column | G06N Fill | C23C Fill |
|--------|-----------|-----------|
| compactness | 99.5% | 97.9% |
| compactness_percentile | 100% | 100% |
| novelty_percentile | 100% | 100% |
| drift_vector | 85.2% | 91.3% |
| corpus_centroid_drift | **0%** | **0%** |
| is_precursor | **0%** | **0%** |
| mean_novelty_score | 99.5% | 97.9% |

### 8.4 Void Fill Rates (G06N_quarterly)

| Metric | V1 (pipe_voids) | V2 (f_void) |
|--------|-----------------|-------------|
| void_size | 41.8% (1116/2669) | **100%** (2836/2836) |
| void_type | 0.5% (13/2669) | **0%** (0/2836) |
| void_centroid | 41.8% | 100% |
| void_density | n/a | 0% |
| is_new_void | n/a | 100% |
| is_closing | n/a | 100% |

---

## Section 9: Summary Table

One row per measure, all key metadata:

| Measure | V1 Table | V2 Table | Correlation | V1→V2 Status | ML Risk |
|---------|----------|----------|-------------|--------------|---------|
| cohesion | pipe_clusters | f_cluster_period | 1.0000 | Identical | None |
| drift_magnitude | pipe_clusters | f_cluster_period | 1.0000 | Identical | None |
| convergence_score | pipe_clusters | f_cluster_period | 1.0000 | Identical | None |
| mean_betweenness | pipe_clusters | f_cluster_period | 0.9938 | Near-identical | Negligible |
| persistence_score | pipe_clusters | f_cluster_period | 0.9855 | Near-identical | Negligible |
| age_periods | pipe_clusters | f_cluster_period | 0.9856 | Near-identical | Negligible |
| void_count (period) | pipe_period_stats | f_period | 0.9989 | Near-identical | None |
| n_births | pipe_period_stats | f_period | 0.9856 | Near-identical | Low |
| n_deaths | pipe_period_stats | f_period | 0.3358 | **Diverged** | High |
| phase_transition_score | pipe_period_stats | f_period | 0.4207 | **Diverged** | High |
| death_probability | pipe_clusters | f_cluster_period | 0.5363 | **Methodology changed** | Critical |
| boundary_pressure_rate | pipe_clusters | f_cluster_period | 0.2956 | **Fundamentally different** | Critical |
| system_entropy | pipe_period_stats | f_period | 0.1554 | **Fundamentally different** | High |
| algebraic_connectivity | pipe_period_stats | f_period | -0.2314 | **Fundamentally different** | High |
| compactness | n/a | f_cluster_period | n/a | V2-new | n/a |
| mean_novelty_score | n/a | f_cluster_period | n/a | V2-new | n/a |
| corpus_centroid_drift | n/a | f_cluster_period | n/a | V2-new (0% fill) | n/a |
| is_precursor | n/a | f_cluster_period | n/a | V2-new (0% fill) | n/a |
| void (per-period) | n/a | f_void | n/a | V2-new architecture | n/a |
| edge weights (normalized) | n/a | f_edge | n/a | V2-new methodology | n/a |

---

## Section 10: C23C_quarterly Specific Checks

### Period and Cluster Coverage

**f_cluster_period (V2):**
- First period: 1980-01-01
- Last period: 2025-04-01
- Distinct periods: 143
- Total rows: 2,020
- Distinct clusters: 21

**pipe_clusters (V1):**
- First period: 1980-01-01
- Last period: 2025-04-01
- Distinct periods: 143
- Total rows: 2,020
- Distinct clusters: 21

Both tables are structurally identical in coverage for C23C_quarterly.

---

## Section 11: Pipeline Architecture Findings

### What populates f_ tables?

The standard ARC pipeline (`arc_run_pipeline`, `arc_compute_period`, `populate_period_stats`) does **not** write to any f_ table. Investigation of all plpgsql procedures shows no INSERT or UPDATE targeting `f_cluster_period`, `f_period`, `f_void`, `f_edge`, or `f_centroids`.

The two V2-specific functions found (`f_compute_betweenness`, `f_compute_spectral_geometry`) read FROM f_edge but are plpython3u functions that compute values (returning jsonb), not procedures that populate the tables.

The f_ tables were populated by an unknown external mechanism — most likely a Python script or manual SQL session that ran for G06N_quarterly and C23C_quarterly on or around 2026-03-20 (based on autovacuum timestamps). No production pipeline function exists to maintain them.

**Table comment confirms intent:** `f_cluster_period` comment reads: *"V2 pipeline: cluster-level measures per period. Populated by incremental INSERT per new period. Historical rows never modified."*

### V2 Schema Design Intent

- `f_centroids`: Pre-computed per-period centroids. Written once, immutable.
- `f_chunk_period`: Per-chunk metrics (novelty, betweenness, perturbation, etc.). Source for f_cluster_period aggregations.
- `f_cluster_period`: Cluster-level aggregations from f_chunk_period + V1 pipe_clusters fields. Intended to merge pipe_clusters + ml_score_history.
- `f_period`: Period-level aggregations (strict subset of pipe_period_stats columns).
- `f_void`: Per-period void snapshots (improved over V1's lifetime records).
- `f_edge`: Filtered, normalized edges (improved over V1's count-weighted edges).

The V2 schema covers exactly 2 corpora (G06N_quarterly, C23C_quarterly) out of 36 active corpora. It is a pilot implementation.

---

## Conclusions

### Is V2 ready to replace V1?

**No.** V2 is not ready to replace V1 for these reasons:

1. **Coverage gap:** V2 covers 2/36 active corpora. 34 corpora have no f_ table data.

2. **No production pipeline:** No stored procedure or canonical script populates the f_ tables. The population mechanism is not in production.

3. **Critical metric divergences:**
   - `boundary_pressure_rate`: Completely different formula (signed delta vs absolute rate). Death model training data would be invalidated.
   - `death_probability`: Near-binary (0/1) rather than continuous ML scores. Predictive capability is lost.
   - `system_entropy`: Low correlation (0.155); temporal trend tracking would change.
   - `algebraic_connectivity`: Negative correlation; all graph-connectivity laws would need re-derivation.

4. **Missing V1 features:** V2 omits 35 V1 cluster columns and 21 V1 period columns. Many are actively used (leiden_modularity, boundary_sharpness, mean_density, growth_rate, n_attractors, etc.).

5. **Unfilled V2 columns:** corpus_centroid_drift, is_precursor, and void_density are 0% filled — V2 is not internally complete.

### What V2 does better:

1. **Void architecture:** Per-period snapshots with is_new_void/is_closing flags are substantially better than V1's lifetime records. Void_size fill is 100% vs V1's 41.8%.
2. **Betweenness fill:** 100% for C23C vs V1's 43.7%.
3. **Normalized edge weights:** More semantically meaningful than raw count weights.
4. **New measures:** compactness, mean_novelty_score, novelty_percentile, drift_vector add genuine new signal.
5. **Immutable historical rows:** The append-only design prevents historical data corruption.

### Recommendation:

V2 should be developed as a **parallel schema** (not a replacement) until:
1. A complete production pipeline procedure is implemented
2. All 36 active corpora are covered
3. `boundary_pressure_rate` and `death_probability` formulas are reconciled with the ML training pipeline
4. `system_entropy` and `algebraic_connectivity` divergences are investigated and documented
5. `corpus_centroid_drift` and `is_precursor` are computed

The V2 void and edge tables (`f_void`, `f_edge`) are closer to production-ready and could potentially replace their V1 equivalents sooner.

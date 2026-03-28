# ARC Session Context
*Generated: 2026-03-25 07:01:47.158012 (UTC)*

---

## SECTION 1 — System State

| Key | Value |
|-----|-------|
| Last migration | 166 (166_arc_f_pipeline_enrich_single_cluster_period_guard) |
| Active corpora | 36 |
| Total clusters | 56534 |
| Total findings | 1313 |
| Active voids | 923 |
| Queue pending | 0 |
| Queue running | 0 |
| Queue failed | 0 |

---

## SECTION 2 — Schema Anchor

### Base Tables

| Table | Est. Rows | Convention |
|-------|-----------|------------|
| data_assignee_normalized | 370819 | raw input |
| data_chunks | 5923087 | raw input |
| data_documents | 3228317 | raw input |
| data_patent_citations | 5703052 | raw input |
| f_centroids | 56534 |  |
| f_chunk_period | 6639743 |  |
| f_cluster_id_sequences | 32 |  |
| f_cluster_match_staging | 49359 |  |
| f_cluster_period | 65396 |  |
| f_cluster_shap_values | 519038 |  |
| f_edge | 340621 |  |
| f_period | 4143 |  |
| f_void | 172058 |  |
| log_backup_runs | 0 | logs |
| log_bugs | 101 | logs |
| log_knn_runs | 1264 | logs |
| log_pipeline_config_history | 70 | logs |
| log_pipeline_phase_runs | 19360 | logs |
| log_pipeline_run_steps | 238371 | logs |
| log_pipeline_runs | 6511 | logs |
| ml_config | 0 | ML outputs |
| ml_feature_registry | 2664 | ML outputs |
| ml_models | 0 | ML outputs |
| ml_prediction_cone | 13044 | ML outputs |
| ml_results | 5463 | ML outputs |
| ml_score_history | 57651 | ML outputs |
| narr_profiles | 0 |  |
| narr_sessions | 1212 |  |
| narr_shared_events | 0 |  |
| pipe_cluster_edges | 783259 | pipeline data |
| pipe_cluster_shap_values | 3324189 | pipeline data |
| pipe_cluster_survival | 10713 | pipeline data |
| pipe_clusters | 118014 | pipeline data |
| pipe_period_stats | 4143 | pipeline data |
| pipe_periods | 1180 | pipeline data |
| pipe_shadow_clusters | 3885 | pipeline data |
| pipe_voids | 104735 | pipeline data |
| sci_cartographer_config | 0 | findings/science |
| sci_findings | 1267 | findings/science |
| sys_available_models | 8 | system/config |
| sys_corpus_classes | 0 | system/config |
| sys_corpus_pipeline_steps | 468 | system/config |
| sys_migrations | 0 | system/config |
| sys_pipeline_config | 64 | system/config |
| sys_principles | 0 | system/config |
| sys_query_consolidations | 0 | system/config |
| sys_query_log | 671 | system/config |
| sys_reports | 83 | system/config |
| sys_role_prompts | 0 | system/config |
| sys_run_config | 26 | system/config |
| sys_task_queue | 183 | system/config |

### Key Procedures & Functions

| Name | Returns | Notes |
|------|---------|-------|
| arc_compute_period | void | Master SQL procedure orchestrating all geometry, graph, stats, and sig |
| arc_f_match_clusters | void |  |
| arc_f_pipeline_core | void |  |
| arc_f_pipeline_enrich | void |  |
| arc_run_all_periods | void |  |
| arc_run_pipeline | void |  |
| arc_run_pipeline_all | void | Convenience wrapper that calls arc_run_pipeline for every period_start |
| compute_attractors | void | State primitive (text-overload). Energy-gradient attractor and saddle  |
| compute_avg_path_length | jsonb | Runs intra-cluster BFS on the _age_edges session temp table to compute |
| compute_belief_persistence | jsonb | Computes embeddings.chunk_periods.belief_persistence_score = fraction  |
| compute_betweenness_age | jsonb | Computes normalised betweenness centrality for all chunks via weighted |
| compute_boundary_percentile | jsonb | Computes per-period fractional rank of pipe_clusters.mean_boundary_sco |
| compute_boundary_pressure_rate | void | Change primitive. Computes the period-over-period change in mean_bound |
| compute_boundary_scores | void | Instability primitive. Computes boundary_score (fraction of foreign KN |
| compute_boundary_sharpness | jsonb | Computes pipe_clusters.boundary_sharpness = P90 - P10 of chunk boundar |
| compute_centroids | void | Computes the mean embedding vector (centroid) for each cluster by aver |
| compute_citation_rate | jsonb | Populates pipe_clusters.citations_per_chunk and citation_rate_trend fr |
| compute_cluster_age | void | Stability primitive. Counts consecutive periods each cluster has been  |
| compute_cluster_edges | void | Date-parameter overload of compute_cluster_edges. Inserts or updates p |
| compute_cluster_reachability | jsonb | Computes the number of other clusters reachable within one hop via pip |
| compute_cohesion_percentile | jsonb | Computes per-period fractional rank of pipe_clusters.cohesion. Writes  |
| compute_convergence_score | jsonb | Estimates convergence probability for each cluster pair by combining c |
| compute_converging_pairs | void | Identifies cluster pairs whose convergence_score exceeds the merger th |
| compute_corpus_centroid_drift | void | Computes the cosine distance between the corpus mean embedding in the  |
| compute_curvature_edges | jsonb | Computes boundary_proximity for each chunk as the STDDEV of cosine dis |
| compute_dark_matter | void | Identifies chunks that appear in fewer than p_min_periods as non-outli |
| compute_distance_entropy | void | State primitive. Computes Shannon entropy of the 20-bin distribution o |
| compute_distance_to_centroid | void | State primitive. Computes cosine distance from each chunk embedding to |
| compute_distribution_moments | void | Computes distance-to-centroid shape measures for all clusters: dist_sk |
| compute_drift | void |  |
| compute_drift_all_periods | void |  |
| compute_drift_percentile | jsonb | Computes per-period fractional rank of pipe_clusters.drift_magnitude.  |
| compute_energy | void | Date-parameter overload of compute_energy. Writes energy = 1/point_den |
| compute_entropy_trend | void | Computes the linear slope of system_entropy over the last p_window per |
| compute_field_surprise_index | jsonb | Computes the corpus-level average of cluster centroid deviations from  |
| compute_first_appearance | jsonb | Sets embeddings.chunk_periods.first_appearance_period = the earliest p |
| compute_graph_diameter | void | BFS-based computation of graph diameter and average path length on the |
| compute_graph_distance | jsonb | Computes shortest-path distances between cluster centroids in the kNN  |
| compute_idea_lineage_depth | jsonb | Computes embeddings.chunk_periods.idea_lineage_depth = number of perio |
| compute_inheritance_score | jsonb | Computes pipe_clusters.inheritance_score = weighted Jaccard overlap be |
| compute_intrinsic_dim | jsonb | Estimates local intrinsic dimensionality for each chunk via the TwoNN  |
| compute_jerk | void | Change primitive. Computes jerk = acceleration(T) - acceleration(T-1)  |
| compute_l2_spread | void | Date-parameter overload of compute_l2_spread. Fixes period column mism |
| compute_lifecycle | void | Stability / Instability primitive (corpus-scoped overload). Same as 2- |
| compute_marginal_entropy_impact | jsonb | For each cluster, computes the change in corpus system_entropy if that |
| compute_mean_intra_distance | void | Date-parameter overload of compute_mean_intra_distance. Reads chunk_pe |
| compute_mean_perturbation_score | jsonb |  |
| compute_membership_churn | void | Change primitive. Computes membership_churn = fraction of cluster memb |
| compute_membership_volatility | jsonb | Computes the number of clusters a chunk has belonged to over its obser |
| compute_merger_targets | void | Merger detection via pgvector LATERAL distance operator. Flags is_merg |
| compute_nearest_clusters | void | For each chunk, identifies the second-nearest cluster (excluding its o |
| compute_novelty_percentile | jsonb | Computes per-period fractional rank of mean novelty_score per cluster. |
| compute_novelty_scores | void | Latent Structure / perturbation primitive. Computes z-score of distanc |
| compute_pattern_significance | double precision | STUB. Returns NULL until significance scoring is implemented (v4 miles |
| compute_perturbation_score | jsonb | Computes embeddings.chunk_periods.perturbation_score = change in dista |
| compute_phase_transition_score | void | Recomputes pipe_period_stats.phase_transition_score = weighted combina |
| compute_point_density | void | Date-parameter overload of compute_point_density. Computes point_densi |
| compute_precursor_flags | void | Identifies clusters with combined spectral and convergence signals ind |
| compute_prosecution_aggregates | jsonb | Aggregates prosecution_duration_days statistics from data_documents th |
| compute_prosecution_heat | jsonb | Computes pipe_clusters.prosecution_heat_index = composite score combin |
| compute_prosecution_percentile | jsonb | Computes per-period fractional rank of pipe_clusters.mean_prosecution_ |
| compute_semantic_distance_to_nearest | void | Computes cosine distance from each cluster's centroid to the nearest o |
| compute_semantic_overlap | jsonb | Computes pipe_cluster_edges.semantic_overlap_a_to_b/b_to_a/max from em |
| compute_semantic_velocity_vector | jsonb | Computes pipe_clusters.velocity_vector = current_centroid - prev_centr |
| compute_shadow_clusters | jsonb | Identifies sub-threshold Leiden communities (below junk_threshold) and |
| compute_size_percentile | jsonb | Computes per-period fractional rank of pipe_clusters.size within each  |
| compute_spectral_geometry | void | Spectral analysis on the cluster graph (pipe_cluster_edges): computes  |
| compute_split_children | jsonb | Detects cluster fission events by finding pairs of new clusters whose  |
| compute_svd_geometry | void | Computes SVD-based shape descriptors for each cluster: spread, cohesio |
| compute_system_entropy | jsonb | Computes Shannon entropy of the cluster-size distribution and KL diver |

### v_* Views

| View | Description |
|------|-------------|
| v_centroid_stability |  |
| v_chunk_trajectory | Per-chunk longitudinal view showing cluster membership changes across periods. Used to stu |
| v_cluster_assignee_features | Per-cluster assignee diversity features: count of distinct assignees, HHI concentration in |
| v_cluster_birth_training | Training view for the arc_cluster_birth_prob model. Target is will_be_born derived from is |
| v_cluster_centroid_history | Centroid position history per persistent cluster. Provides a time-ordered sequence of cent |
| v_cluster_event_training | Primary XGBoost training view for the arc_cluster_death_prob model. Produces one row per n |
| v_cluster_pair_distance | Pairwise cosine distance between all connected cluster centroids for each corpus×period. D |
| v_cluster_trajectory | Longitudinal view of cluster state across periods for each persistent_cluster_id. Ordered  |
| v_cross_resolution_period_stats | Cross-resolution GNN training features joining child corpus periods (monthly/weekly) to th |
| v_drift_direction | Cosine alignment of each cluster's velocity_vector with the corpus mean velocity for that  |
| v_f_cluster_assignee_features | V2-native assignee features per cluster per period.
Joins raw USPTO data (data_assignee_no |
| v_f_cluster_death_train | pgml training feed for arc_cluster_death_v2.
Numeric features only — no identifier columns |
| v_f_cluster_death_train_patent_cross | pgml training feed for arc_cluster_death_patent_cross_corpus.
13 features: cohesion, size, |
| v_f_cluster_event_training | V2 equivalent of v_cluster_event_training. Same column names and COALESCE
defaults — drop- |
| v_f_cluster_event_training_patent_cross | Combined training view for cross-corpus patent death model.
Returns G06N + C23C + H01L wit |
| v_period_stats_ml_features | Feature view for the arc_period_anomaly XGBoost model. Selects the 11 period-level feature |
| v_prediction_cone_age | Age in periods of each open ml_prediction_cone entry, computed as the count of completed p |
| v_recovery_rate | Post-shock recovery analysis. For each cluster, counts and averages the number of periods  |
| v_run_lookup | Joins sys_run_config to pipe_period_stats to provide a quick lookup of available corpus+pe |
| v_surprise_hierarchy | Hierarchical ranking of clusters by field_surprise_index within each corpus×period. Groups |
| v_temporal_period_stats | Temporal rolling statistics for pipe_period_stats. Adds lag-1 and lag-2 values for key met |
| v_void_count_per_period | [DEPRECATED — use pipe_period_stats.void_count instead. This view uses different counting  |

**Naming:** pipe_*=pipeline, data_*=raw input, sys_*=config, sci_*=science, log_*=logs, ml_*=ML, v_*=views, embeddings.*=vectors

---

## SECTION 3 — Corpus Pipeline Status

| corpus_id | domain | done | pend | ml | label% | death_prob% | flags |
|-----------|--------|------|------|----|--------|-------------|-------|
| C23C_quarterly | patents | 0 | 13 | yes | 99% | 97% |  |
| C30B_quarterly | patents | 0 | 13 | yes | 82% | 81% |  |
| G01B_quarterly | patents | 0 | 13 | yes | 96% | 95% |  |
| G01N_quarterly | patents | 0 | 13 | yes | 99% | 98% |  |
| G02B_quarterly | patents | 0 | 13 | yes | 100% | 99% |  |
| G06F_quarterly | patents | 0 | 13 | yes | 99% | 99% |  |
| G06N_10_quarterly | patents | 9 | 4 | yes | 36% | 26% |  |
| G06N_20_quarterly | patents | 8 | 5 | yes | 67% | 62% |  |
| G06N_3_quarterly | patents | 9 | 4 | yes | 59% | 52% |  |
| G06N_5_quarterly | patents | 8 | 5 | yes | 64% | 56% |  |
| G06N_7_quarterly | patents | 8 | 5 | yes | 59% | 51% |  |
| G06N_annual | patents | 9 | 4 | yes | 94% | 96% |  |
| G06N_monthly | patents | 9 | 4 | yes | 88% | 89% |  |
| G06N_quarterly | patents | 11 | 2 | yes | 99% | 100% |  |
| G06N_weekly | patents | 9 | 4 | yes | 73% | 67% |  |
| H01L_21_quarterly | patents | 10 | 3 | yes | 100% | 99% |  |
| H01L_22_quarterly | patents | 10 | 3 | yes | 100% | 100% |  |
| H01L_23_quarterly | patents | 10 | 3 | yes | 100% | 100% |  |
| H01L_24_quarterly | patents | 10 | 3 | yes | 100% | 99% |  |
| H01L_25_quarterly | patents | 10 | 3 | yes | 97% | 90% |  |
| H01L_quarterly | patents | 11 | 2 | yes | 100% | 100% |  |
| H01S_quarterly | patents | 0 | 13 | no | \u2014% | \u2014% |  |
| longevity_cardio_quarterly | patents | 11 | 2 | yes | 99% | 99% |  |
| longevity_cellular_quarterly | patents | 10 | 3 | yes | 100% | 100% |  |
| longevity_genetic_quarterly | patents | 11 | 2 | yes | 100% | 100% |  |
| longevity_neuro_quarterly | patents | 11 | 2 | yes | 99% | 99% |  |
| longevity_patents_quarterly | patents | 11 | 2 | yes | 100% | 100% |  |
| longevity_quarterly | papers | 10 | 3 | yes | 100% | 100% |  |
| openalex_chemeng_quarterly | papers | 0 | 13 | no | \u2014% | \u2014% |  |
| openalex_chemistry_quarterly | papers | 0 | 13 | no | \u2014% | \u2014% |  |
| openalex_cs_full_quarterly | papers | 0 | 13 | no | \u2014% | \u2014% |  |
| openalex_cs_sample | papers | 10 | 3 | yes | 100% | 100% |  |
| openalex_ee_sample | papers | 10 | 3 | yes | 100% | 100% |  |
| openalex_longevity_sample | papers | 10 | 3 | yes | 100% | 100% |  |
| openalex_materials_quarterly | papers | 8 | 5 | yes | 100% | 100% |  |
| openalex_physics_quarterly | papers | 2 | 11 | yes | 96% | 96% |  |

---

## SECTION 4 — Queue State

Queue is empty (no pending/running/failed tasks).

---

## SECTION 5 — Science State

### 5a. Findings Summary

| type | subtype | count | avg_confidence |
|------|---------|-------|----------------|
| finding | \u2014 | 292 | 0.618 |
| finding | anomaly | 7 | 0.861 |
| finding | anomaly_detection | 1 | 0.920 |
| finding | assignee_convergence | 5 | 0.852 |
| finding | assignee_pattern | 1 | 0.880 |
| finding | collective_summary | 1 | 0.870 |
| finding | competitive_intelligence | 7 | 0.887 |
| finding | corpus_data_quality | 6 | 0.908 |
| finding | corpus_opportunity | 5 | 0.824 |
| finding | cross_corpus_void | 3 | 0.783 |
| finding | cross_domain_void | 5 | 0.820 |
| finding | crystallization | 6 | 0.845 |
| finding | data_quality | 2 | 0.935 |
| finding | death_landscape | 1 | 0.850 |
| finding | death_risk | 3 | 0.887 |
| finding | death_risk_alert | 1 | 0.870 |
| finding | death_risk_review | 11 | 0.895 |
| finding | field_collapse | 2 | 0.925 |
| finding | institutional_gap | 1 | 0.880 |
| finding | law_exception | 1 | 0.850 |
| finding | law_validation | 24 | 0.807 |
| finding | methodology | 3 | 0.907 |
| finding | ml_prediction | 1 | 0.890 |
| finding | period_anomaly | 12 | 0.918 |
| finding | period_summary | 35 | 0.889 |
| finding | persistent_void | 2 | 0.865 |
| finding | pls_analysis | 1 | 0.750 |
| finding | pls_prediction | 7 | 0.766 |
| finding | prediction | 6 | 0.752 |
| finding | query_optimization | 6 | 0.950 |
| finding | structural | 2 | 0.875 |
| finding | technology_emergence | 1 | 0.760 |
| finding | trade_secret | 21 | 0.861 |
| finding | void_bridge | 35 | 0.759 |
| finding | void_geometry | 1 | 0.890 |
| finding | void_intelligence | 3 | 0.757 |
| hypothesis | \u2014 | 455 | 0.650 |
| hypothesis | cluster_lifecycle | 2 | 0.765 |
| hypothesis | competitive_dynamics | 1 | 0.820 |
| hypothesis | corpus_data_quality | 1 | 0.910 |
| hypothesis | corpus_dynamics | 2 | 0.880 |
| hypothesis | corpus_gap | 1 | 0.700 |
| hypothesis | data_artifact | 1 | 0.920 |
| hypothesis | death_risk | 2 | 0.700 |
| hypothesis | field_lifecycle | 1 | 0.780 |
| hypothesis | frontier_artifact | 1 | 0.720 |
| hypothesis | ml_prediction | 8 | 0.768 |
| hypothesis | phase_transition | 2 | 0.685 |
| hypothesis | pls_prediction | 52 | 0.858 |
| hypothesis | prediction | 5 | 0.694 |
| hypothesis | strategic | 3 | 0.727 |
| hypothesis | structural | 1 | 0.720 |
| hypothesis | structural_law | 2 | 0.780 |
| hypothesis | trade_secret | 1 | 0.750 |
| hypothesis | uninvented_combination | 5 | 0.820 |
| hypothesis | void_dynamics | 1 | 0.760 |
| law | \u2014 | 137 | 0.725 |
| law | predictive_model | 2 | 0.720 |
| law | structural_law | 1 | 0.880 |
| principle | \u2014 | 105 | 0.708 |

### 5b. Top 10 Laws by Confidence

| id | title | conf | corpus | outcome |
|----|-------|------|--------|--------|
| 206 | Persistence Score Dominant Survival Predictor (Cox PH) | 0.91 | G06N_quarterly,H01L_quart | ambiguous |
| 644 | Semiconductor Persistence Invariance Law | 0.9 | H01L_quarterly | confirmed |
| 100 | Large Cluster Cohesion Collapse Fragmentation Law | 0.89 | G06N_quarterly,H01L_quart | confirmed |
| 645 | Semiconductor GNN Citation Dominance Law | 0.88 | H01L_quarterly | confirmed |
| 1369 | Quantum Computing Paradigm Isolation Law: Semantic Isol | 0.88 | G06N_quarterly | \u2014 |
| 201 | Size-Persistence Universal Death Predictor | 0.88 | G06N_quarterly,G06N_annua | confirmed |
| 99 | Boundary Pressure Dissolution Law | 0.88 | G06N_quarterly,H01L_quart | ambiguous |
| 98 | Field Maturation Drift Deceleration Law | 0.87 | G06N_quarterly,H01L_quart | confirmed |
| 478 | Quantum Computing: Patent Commercialization Precedes Jo | 0.87 | G06N_quarterly,openalex_c | confirmed |
| 101 | Algebraic Connectivity Drop Fragmentation Law | 0.87 | G06N_quarterly,H01L_quart | ambiguous |

### 5c. Open Hypotheses

| id | title | conf | corpus | created |
|----|-------|------|--------|--------|
| 662 | PLS: Network Resource Optimization likely to patent by  | 0.9802 | openalex_cs_sample | 2026-03-15 |
| 663 | PLS: ML Model Evaluation likely to patent by 2026_Q3 | 0.98 | openalex_cs_sample | 2026-03-15 |
| 664 | PLS: AI-Powered Interactive Tools likely to patent by 2 | 0.9798 | openalex_cs_sample | 2026-03-15 |
| 665 | PLS: Time-Series Anomaly Detection likely to patent by  | 0.9747 | openalex_cs_sample | 2026-03-15 |
| 666 | PLS: Text-guided diffusion generation likely to patent  | 0.9711 | openalex_cs_sample | 2026-03-15 |
| 667 | PLS: Robust Group Learning likely to patent by 2026_Q3 | 0.9691 | openalex_cs_sample | 2026-03-15 |
| 668 | PLS: Neural Speech Enhancement likely to patent by 2026 | 0.9618 | openalex_cs_sample | 2026-03-15 |
| 669 | PLS: Explainable AI Methods likely to patent by 2026_Q3 | 0.9609 | openalex_cs_sample | 2026-03-15 |
| 670 | PLS: Multilingual NLP Processing likely to patent by 20 | 0.9602 | openalex_cs_sample | 2026-03-15 |
| 671 | PLS: Neuromorphic Photonic Computing likely to patent b | 0.9602 | openalex_cs_sample | 2026-03-15 |

### 5d. Significant Voids (period_count >= 3)

| id | corpus | cluster_a | cluster_b | periods | size | tier | centroid | label |
|----|--------|-----------|-----------|---------|------|------|----------|-------|
| 161531 | G06F_quarterly | G06F_quarterly_003 | G06F_quarterly_003 | 128 | 0.538 | strategic | ok | NULL |
| 174801 | G06F_quarterly | G06F_quarterly_003 | G06F_quarterly_020 | 44 | 0.522 | strategic | ok | NULL |
| 103957 | G06N_quarterly | G06N_quarterly_006 | G06N_quarterly_008 | 38 | 0.351 | strategic | ok | NULL |
| 177301 | G06F_quarterly | G06F_quarterly_003 | G06F_quarterly_024 | 33 | 0.559 | strategic | ok | NULL |
| 89661 | G06N_quarterly | G06N_quarterly_008 | G06N_quarterly_009 | 32 | 0.475 | strategic | ok | NULL |
| 101634 | longevity_quarterly | longevity_quarterl | longevity_quarterl | 30 | 0.390 | strategic | ok | NULL |
| 89902 | G06N_quarterly | G06N_quarterly_008 | G06N_quarterly_009 | 29 | 0.509 | strategic | ok | NULL |
| 111864 | longevity_patents_quarterly | longevity_patents_ | longevity_patents_ | 27 | 0.309 | strategic | ok | NULL |
| 111474 | longevity_patents_quarterly | longevity_patents_ | longevity_patents_ | 24 | 0.269 | persistent | ok | NULL |
| 110698 | longevity_patents_quarterly | longevity_patents_ | longevity_patents_ | 22 | 0.191 | persistent | ok | NULL |
| 106618 | longevity_quarterly | longevity_quarterl | longevity_quarterl | 22 | 0.322 | strategic | ok | NULL |
| 92590 | G06N_quarterly | G06N_quarterly_008 | G06N_quarterly_010 | 21 | 0.462 | strategic | ok | NULL |
| 155980 | C23C_quarterly | C23C_quarterly_014 | C23C_quarterly_014 | 19 | 0.401 | strategic | ok | NULL |
| 90677 | G06N_quarterly | G06N_quarterly_008 | G06N_quarterly_011 | 17 | 0.501 | strategic | ok | NULL |
| 120355 | longevity_genetic_quarterly | longevity_genetic_ | longevity_genetic_ | 16 | 0.475 | strategic | ok | NULL |

---

## SECTION 6 — ML Model Status

| corpus_id | auc_global | auc_within_period | f1 | dp_coverage% |
|-----------|-----------|-------------------|----|--------------|
| C23C_quarterly | \u2014 | 0.745 | 0.421 | 97% |
| C30B_quarterly | \u2014 | 0.837 | 0.549 | 81% |
| G01B_quarterly | \u2014 | 0.832 | 0.362 | 95% |
| G01N_quarterly | \u2014 | 0.823 | 0.479 | 98% |
| G02B_quarterly | \u2014 | 0.801 | 0.410 | 99% |
| G06F_quarterly | \u2014 | 0.864 | 0.453 | 99% |
| G06N_10_quarterly | \u2014 | 0.700 | 0.353 | 26% |
| G06N_20_quarterly | \u2014 | 0.863 | 0.705 | 62% |
| G06N_3_quarterly | \u2014 | 0.812 | 0.682 | 52% |
| G06N_5_quarterly | \u2014 | 0.909 | 0.679 | 56% |
| G06N_7_quarterly | \u2014 | 0.785 | 0.692 | 51% |
| G06N_annual | \u2014 | 0.937 | 0.752 | 96% |
| G06N_monthly | \u2014 | 0.865 | 0.854 | 89% |
| G06N_quarterly | \u2014 | 0.931 | 0.582 | 100% |
| G06N_weekly | \u2014 | 0.889 | 0.893 | 67% |
| H01L_21_quarterly | \u2014 | 0.915 | 0.373 | 99% |
| H01L_22_quarterly | \u2014 | 0.926 | 0.235 | 100% |
| H01L_23_quarterly | \u2014 | 0.826 | 0.478 | 100% |
| H01L_24_quarterly | \u2014 | 0.813 | 0.286 | 99% |
| H01L_25_quarterly | \u2014 | 0.564 | 0.311 | 90% |
| H01L_quarterly | \u2014 | 0.824 | 0.465 | 100% |
| longevity_cardio_quarterly | \u2014 | 0.637 | 0.208 | 99% |
| longevity_cellular_quarterly | \u2014 | 0.860 | 0.340 | 100% |
| longevity_genetic_quarterly | \u2014 | 0.836 | 0.430 | 100% |
| longevity_neuro_quarterly | \u2014 | 0.865 | 0.410 | 99% |
| longevity_patents_quarterly | \u2014 | 0.799 | 0.392 | 100% |
| longevity_quarterly | \u2014 | 0.746 | 0.333 | 100% |
| openalex_cs_sample | \u2014 | 0.874 | 0.609 | 100% |
| openalex_ee_sample | \u2014 | 0.894 | 0.458 | 100% |
| openalex_longevity_sample | \u2014 | 0.810 | 0.600 | 100% |
| openalex_materials_quarterly | \u2014 | 0.802 | 0.471 | 100% |
| openalex_physics_quarterly | \u2014 | 0.845 | 0.618 | 96% |
| patent_cross_corpus | \u2014 | 0.872 | 0.584 | \u2014% |

**Last trained (death model) per corpus:**

| corpus_id | last_trained |
|-----------|-------------|
| C23C_quarterly | 2026-03-18 14:38 |
| C30B_quarterly | 2026-03-18 11:40 |
| G01B_quarterly | 2026-03-18 00:01 |
| G01N_quarterly | 2026-03-18 14:55 |
| G02B_quarterly | 2026-03-18 14:55 |
| G06F_quarterly | 2026-03-18 21:17 |
| G06N_10_quarterly | 2026-03-17 23:47 |
| G06N_20_quarterly | 2026-03-17 23:46 |
| G06N_3_quarterly | 2026-03-17 23:47 |
| G06N_5_quarterly | 2026-03-17 22:17 |
| G06N_7_quarterly | 2026-03-17 22:17 |
| G06N_annual | 2026-03-17 23:41 |
| G06N_monthly | 2026-03-17 22:17 |
| G06N_quarterly | 2026-03-17 23:55 |
| G06N_weekly | 2026-03-17 22:17 |
| H01L_21_quarterly | 2026-03-17 23:41 |
| H01L_22_quarterly | 2026-03-17 23:41 |
| H01L_23_quarterly | 2026-03-17 23:41 |
| H01L_24_quarterly | 2026-03-17 23:42 |
| H01L_25_quarterly | 2026-03-17 23:42 |
| H01L_quarterly | 2026-03-17 20:33 |
| longevity_cardio_quarterly | 2026-03-17 23:33 |
| longevity_cellular_quarterly | 2026-03-17 23:33 |
| longevity_genetic_quarterly | 2026-03-17 23:33 |
| longevity_neuro_quarterly | 2026-03-17 23:34 |
| longevity_patents_quarterly | 2026-03-17 23:33 |
| longevity_quarterly | 2026-03-17 23:33 |
| openalex_cs_sample | 2026-03-18 00:15 |
| openalex_ee_sample | 2026-03-18 00:16 |
| openalex_longevity_sample | 2026-03-18 00:04 |
| openalex_materials_quarterly | 2026-03-17 23:55 |
| openalex_physics_quarterly | 2026-03-19 18:20 |
| patent_cross_corpus | 2026-03-20 17:59 |

---

## SECTION 7 — Open Work / Recommended Next Actions

### HIGH PRIORITY

- **VOID BRIDGE NEEDED:** id=105992 openalex_materials_qua↔openalex_materials_qua (13 periods)
- **VOID BRIDGE NEEDED:** id=106045 openalex_materials_qua↔openalex_materials_qua (13 periods)
- **VOID BRIDGE NEEDED:** id=107745 openalex_materials_qua↔openalex_materials_qua (7 periods)
- **VOID BRIDGE NEEDED:** id=111586 openalex_materials_qua↔openalex_materials_qua (6 periods)
- **VOID BRIDGE NEEDED:** id=111535 openalex_materials_qua↔openalex_materials_qua (6 periods)

### MEDIUM PRIORITY

- 919 active voids with NULL semantic_label — run: PGHOST=/var/run/postgresql python3 api/arc_call.py run-jobs
- **LOW DP COVERAGE:** G06N_10_quarterly — 26% scored
- **LOW DP COVERAGE:** G06N_20_quarterly — 62% scored
- **LOW DP COVERAGE:** G06N_3_quarterly — 52% scored
- **LOW DP COVERAGE:** G06N_5_quarterly — 56% scored
- **LOW DP COVERAGE:** G06N_7_quarterly — 51% scored
- **LOW DP COVERAGE:** G06N_weekly — 67% scored

### Queue Chain Summary

- chain=G06N_10_quarterly_20260318_0020  corpus=G06N_10_quarterly  total=4  done=4  pend=0  run=0  fail=0
- chain=G06N_3_quarterly_20260318_0020  corpus=G06N_3_quarterly  total=4  done=4  pend=0  run=0  fail=0
- chain=G06N_20_quarterly_20260318_0020  corpus=G06N_20_quarterly  total=4  done=4  pend=0  run=0  fail=0
- chain=H01L_24_quarterly_20260318_0014  corpus=H01L_24_quarterly  total=4  done=4  pend=0  run=0  fail=0
- chain=H01L_22_quarterly_20260318_0014  corpus=H01L_22_quarterly  total=4  done=4  pend=0  run=0  fail=0
- chain=H01L_23_quarterly_20260318_0014  corpus=H01L_23_quarterly  total=4  done=4  pend=0  run=0  fail=0
- chain=G06N_annual_20260318_0014  corpus=G06N_annual  total=4  done=4  pend=0  run=0  fail=0
- chain=H01L_25_quarterly_20260318_0014  corpus=H01L_25_quarterly  total=4  done=4  pend=0  run=0  fail=0
- chain=H01L_21_quarterly_20260318_0014  corpus=H01L_21_quarterly  total=4  done=4  pend=0  run=0  fail=0
- chain=G06N_monthly_20260317_2259  corpus=G06N_monthly  total=2  done=2  pend=0  run=0  fail=0

---

## SECTION 8 — Data Quality Flags

| corpus_id | min_death_prob% | min_label% | avg_cohesion% |
|-----------|----------------|------------|---------------|
| G06N_20_quarterly | 0% | 0% | 75% |
| G01B_quarterly | 0% | 0% | 96% |
| openalex_ee_sample | 0% | 100% | 100% |
| G06N_weekly | 0% | 0% | 100% |
| H01L_23_quarterly | 0% | 75% | 100% |
| G06N_3_quarterly | 0% | 0% | 75% |
| openalex_materials_quarterly | 0% | 0% | 97% |
| G06N_monthly | 0% | 0% | 100% |
| openalex_cs_sample | 0% | 0% | 100% |
| H01L_24_quarterly | 0% | 0% | 100% |


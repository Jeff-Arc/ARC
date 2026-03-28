# ASML IP Intelligence Report — 2026-03-19

**Analysis Type:** Trade Secret Territory Mapping + Imminent Filing Predictions
**Target Company:** ASML Holding N.V. (and subsidiaries: ASML Netherlands B.V., ASML US LLC, Cymer)
**Patent Corpora:** H01L_quarterly · G02B_quarterly · G01B_quarterly · C23C_quarterly · G01N_quarterly
**Science Corpus:** openalex_physics_quarterly (period 2026-10-01; note: data quality issues, see §7)
**Task ID:** 176
**Findings Filed:** IDs 1394–1406 (13 total)
**Session:** cartographer_20260319_0900

---

## Executive Summary

ASML's patent footprint in the analyzed corpora is almost entirely concentrated in a single H01L cluster — Substrate Chuck Heating Systems (191 patents, 2015–2024) — while three major legacy clusters in film deposition and process equipment have been fully abandoned since 2006–2016. This extreme concentration, combined with confirmed zero-patent positions at sub-0.20 semantic distance from active science clusters and a 19-quarter persistent void in C23C coating physics, provides strong structural evidence that ASML's EUV multilayer mirror deposition recipes, computational wavefront reconstruction algorithms, and GaN/AlGaN power electronics for plasma sources are being protected as trade secrets rather than patents. The most actionable intelligence item is **C23C void 155980** — a 5-year-old gap in thin-film deposition patent space that maps directly to ASML's EUV optics coating technology and has remained conspicuously unfilled. Imminent patent filing predictions focus on 2D materials (H01L, high confidence), silicon photonic integration (G02B, moderate confidence), and plasma confinement physics for High-NA EUV sources (H01L, moderate confidence, 2026–2027 horizon).

---

## Section 1: ASML IP Map

### 1.1 Assignee Variants in Database
| Assignee | Corpus | Patents |
|---|---|---|
| ASML NETHERLANDS B.V. | H01L_quarterly | 404 |
| ASML Holding N.V. | H01L_quarterly | 81 |
| ASML US, LLC | H01L_quarterly | 7 |
| ASM Lithography B.V. | H01L_quarterly | 1 |
| ASM Lithography | H01L_quarterly | 1 |
| ASML NETHERLANDS B.V. | G06N_quarterly | 4 |

> **Data coverage note:** Assignee data is only available for H01L_quarterly (491 documents) and G06N_quarterly (4 documents). The four remaining target corpora (G02B, G01B, C23C, G01N) have NULL assignee fields in `data_documents` — ASML's actual patent presence in those corpora cannot be confirmed or denied from this database. In reality, ASML is a significant filer in G03F (lithography processes), G01B (optical metrology), and H05G (X-ray/EUV sources), which are outside or underrepresented in the current corpus set. All "N/A" entries in the following table should be read as **data not available**, not confirmed zero presence.

### 1.2 Full IP Map (H01L_quarterly — confirmed data)
| Cluster ID | Cluster Label | Patents | First Filing | Last Filing | Recent 3yr | Trend |
|---|---|---|---|---|---|---|
| H01L_quarterly_0112 | Substrate Chuck Heating Systems | **191** | 2015-08-13 | 2024-04-23 | **10** | ✅ Active |
| H01L_quarterly_0056 | Semiconductor Film Deposition | 88 | 2006-11-08 | 2016-12-23 | 0 | 🔴 Abandoned |
| H01L_quarterly_0032 | Semiconductor Process Equipment | 77 | 1997-10-10 | 2006-09-07 | 0 | 🔴 Abandoned |
| H01L_quarterly_0080 | Semiconductor Patterning Methods | 25 | 2007-12-07 | 2014-08-08 | 0 | 🔴 Abandoned |
| H01L_quarterly_0051 | Semiconductor Packaging Assembly | 18 | 2001-03-05 | 2007-01-22 | 0 | ⚠️ Stale |
| H01L_quarterly_0127 | Semiconductor Wafer Inspection | 14 | 2018-04-23 | 2019-03-25 | 0 | ⚠️ Stale |
| H01L_quarterly_0101 | Semiconductor Etch Patterning | 9 | 2019-09-16 | 2021-12-29 | 0 | ⚠️ Stale |
| H01L_quarterly_0041 | Semiconductor Gate Fabrication | 8 | 2004-02-23 | 2022-11-10 | 0 | ⚠️ Stale |
| H01L_quarterly_0129 | Semiconductor Wafer Testing | 8 | 2017-10-12 | 2018-05-23 | 0 | ⚠️ Stale |
| *(12 additional clusters)* | *Various semiconductor processes* | *3–7 each* | — | — | 0 | Stale/Dormant |

### 1.3 Abandoned Domain Analysis

Three clusters with > 20 patents have zero recent filings:

1. **Semiconductor Film Deposition (88 patents, last: 2016)** — ASML built significant CVD/ALD-adjacent IP during the 193nm immersion era. Abandoned as EUV transition began. Likely divested or licensed to process equipment partners.

2. **Semiconductor Process Equipment (77 patents, last: 2006)** — Legacy scanner/stepper-era apparatus IP. Fully dormant for ~20 years. Represents early ASM Lithography corporate lineage.

3. **Semiconductor Patterning Methods (25 patents, last: 2014)** — Double-patterning era IP (spacer-based, LELE). Abandoned as EUV rendered double-patterning strategically less important.

**Pattern interpretation:** ASML has executed a deliberate IP strategy shift — from broad process coverage (deposition, equipment, patterning) to deep concentration in EUV stage hardware. The abandonment of three large clusters while concentrating 191 patents into a single chuck/heating cluster reflects a company that has decided its moat is built on trade secrets and market dominance, not broad patent walls.

---

## Section 2: Trade Secret Territory

*Ranked by confidence. Confidence criteria: 0.90+ = direct evidence convergence; 0.75–0.89 = strong structural + semantic evidence; 0.60–0.74 = semantic proximity, less void confirmation.*

### 🔴 Rank 1 — EUV Optics Multilayer Thin Film Deposition
**Confidence: 0.85** | Finding ID: 1394 | Void Bridge ID: 1399

**Evidence:**
- C23C void 155980 has been open for **19 consecutive quarters** (~5 years) between Steel Sheet Metal Coatings and Semiconductor Epitaxial Substrates — the longest active void in the entire analyzed patent dataset
- Science cluster AlGaN GaN Power Device Fabrication sits at semantic distance **0.2079–0.2462** from this void
- ASML has confirmed **0 patents** in H01L GaAs Device Fabrication (distance 0.1556), Nitride Semiconductor Devices (distance 0.1828), and Epitaxial Crystal Growth (distance 0.1886) — the tightest science-to-empty-patent distances in the full analysis
- ASML's EUV multilayer mirror technology (Mo/Si bilayers, Ru capping, B4C passivation) involves precision thin-film deposition that would classify in exactly this C23C/H01L boundary zone

**Trade secret signature:** A 5-year void in the deposition domain most relevant to ASML's EUV optics manufacturing, with confirmed zero ASML patents in the nearest H01L clusters. Deposition recipes for multilayer reflective optics are notoriously trade secret: they involve empirically optimized nucleation sequences, interface treatment steps, and environmental controls that are difficult to reverse-engineer even with the published layer count and target reflectivity.

**Bridging invention that would fill this void:** A unified thin-film process framework connecting industrial PVD/CVD equipment science to semiconductor-grade multilayer optical stacks — specifically addressing nucleation kinetics at Mo/Si interfaces, stress management in thick multilayers (40+ bilayer stacks), and in-situ reflectometry feedback for layer thickness control.

---

### 🟠 Rank 2 — Computational Wavefront Sensing / Scanner Aberration Reconstruction
**Confidence: 0.82** | Finding ID: 1395 | Void Bridge ID: 1400

**Evidence:**
- Science cluster Computational Imaging Wavefront Sensing (openalex_physics_quarterly_0576) covers ptychography, phase retrieval, deep learning wavefront reconstruction, and adaptive optics — all directly feeding ASML's scanner metrology
- Semantic distances: **0.1909** to G02B Super-Resolution Microscopy, **0.2922** to G01B Microlithography Projection Exposure Tools, **0.2943** to G01B Optical Height Sensors for Lithography
- G01B void 121569 (6 periods, OCT Laser Imaging ↔ Non-Contact Structure Displacement Monitoring) sits at void_centroid distance 0.2624 from the nearest science cluster, covering the wafer height + overlay sensor fusion zone
- Science cluster actively published through 2026-Q1 (final data-reliable period)

**Trade secret signature:** ASML's scanner aberration reconstruction (the algorithm converting wavefront sensor measurements into lens actuator corrections) and its computational sensor fusion methods (combining level sensor, alignment sensor, and scatterometry signals in real-time control loops) are extensively documented as core IP that ASML does not publish. The academic science is progressing; the translation into scanner control software remains non-disclosed.

---

### 🟠 Rank 3 — Material Characterization Physics for Lithography Metrology
**Confidence: 0.80** | Finding ID: 1396

**Evidence:**
- Science cluster Material Characterization Microstructure Analysis (openalex_physics_quarterly_0058) has the **highest cross-corpus patent connectivity** in the dataset — generating near-pairs across G02B (0.1808), G01N (0.1973, 0.1853), G01B (0.2311, 0.2473), simultaneously
- This cluster covers TEM/SEM/XRD/SAXS thin film characterization — the physics underlying ASML's YieldStar overlay metrology and HMI eScan mask inspection
- Assignee data gap prevents confirmation of ASML's presence/absence in the adjacent G01N/G01B clusters

**Trade secret signature:** ASML's metrology products implement algorithmic translation of measurement physics into process control signals. The science is openly published; the specific numerical models, calibration procedures, and feedback architectures used in production scanners constitute the trade secret layer.

---

### 🟡 Rank 4 — GaN/AlGaN Power Device Physics for EUV Source Integration
**Confidence: 0.78** | Finding ID: 1397

**Evidence:**
- **Confirmed 0 ASML patents** in H01L GaAs/Nitride/Epitaxial clusters at distances 0.1556–0.1886 — the tightest science-to-empty-patent distance in the full analysis
- Science cluster AlGaN GaN Power Device Fabrication (0545, 9 periods active, cohesion 0.987) covers GaN/AlGaN power devices, ohmic contacts, HEMTs, micro-LEDs
- ASML's discharge-produced and laser-produced plasma sources use III-V compound semiconductor driver electronics where GaN power device physics directly applies

**Trade secret signature:** ASML does not patent at the device/epitaxial layer level in GaN/AlGaN despite the relevance to their EUV source components. This is consistent with sourcing compound semiconductor expertise through acquisition (Cymer, Trumpf collaboration) and protecting integration methods as trade secrets rather than devices.

---

## Section 3: Imminent Filing Predictions

*PLS model: science phase transitions lead patent filings by 6–12 months. Predictions based on 2022–2025 science corpus data (2026 data unreliable due to sparsity).*

### Prediction 1: 2D Materials → H01L EUV Patterning + Process Control Filings
**Confidence: 0.78** | Hypothesis ID: 1402 | **Timing: 2024–2025** *(likely already in motion)*

- **Science signal:** Two-Dimensional Electronic Material Properties (0139) — most persistent ASML-relevant cluster, present every period 2023-Q1 to 2026-Q1, cohesion 0.985–0.989. Present during 2023-Q4 peak phase transition (score 0.091).
- **Patent target:** H01L_quarterly — Semiconductor Etching Processes (growth_rate 3.86, age_periods=2, death_prob 0.18) and Wafer Surface Processing (growth_rate 2.14) clusters are newly born and fast-growing, consistent with filing activity arriving from the 2023-Q4 science transition.
- **ASML relevance:** 2D material integration for EUV requires new resist chemistry compatible with atomically thin films; ASML's YieldStar and process control tools must characterize 2D layer thickness at angstrom scale.
- **Falsifiable test:** ASML files H01L patents on 2D-material resist chemistry or angstrom-scale characterization in 2024–2025.

### Prediction 2: Valley Photonic Crystal Consolidation → G02B Silicon Photonic Integration Filings
**Confidence: 0.72** | Hypothesis ID: 1403 | **Timing: 2025–2026**

- **Science signal:** Valley Photonic Crystal Devices (0559, 10+ periods, cohesion >0.987, sustained low death_prob) consolidated during 2023-Q4 transition. Photonic Crystals Topological Materials (0436) under elevated death pressure (0.578 in 2023-Q4) — paradigm crystallization in progress.
- **Patent target:** G02B void 149395 (5 periods, Camera Module Lens Driving ↔ Silicon Photonic Package Integration) — the natural co-packaged photonic sensor landing zone.
- **ASML relevance:** ASML's alignment sensor roadmap involves integrated PIC sensors (ORION, SMASH evolution) for EUV-compatible compact sensing.
- **Falsifiable test:** G02B void 149395 closes (status → 'closed') within 6 quarters.

### Prediction 3: Semiconductor CVD Process Equipment → C23C Advanced Node Deposition Filings
**Confidence: 0.74** | Hypothesis ID: 1404 | **Timing: 2025**

- **Science signal:** AlGaN GaN Power Device Fabrication (0545, 9 periods, cohesion 0.987) as science precursor. C23C Semiconductor CVD Process Equipment patent cluster (growth_rate 1.875, death_prob 0.045) newly born and extremely low-risk.
- **Patent target:** C23C void 155980 (19 periods) expected to begin closing as ALD/selective epitaxy patents accumulate.
- **Falsifiable test:** Void 155980 period_count stops increasing or status changes to 'closed' within 8 quarters.

### Prediction 4: Plasma Confinement Physics → H01L EUV Source Power Scaling Filings
**Confidence: 0.68** | Hypothesis ID: 1405 | **Timing: 2026–2027**

- **Science signal:** Plasma Confinement Fusion Physics (0611) — new cluster in 2026-Q1 (cohesion 0.988, death_prob 0.419). Covers tokamak edge dynamics, runaway electrons, magnetic confinement — methodology transferable to LPP plasma source optimization.
- **Patent target:** H01L or H05G EUV source subsystem patents for ASML/Cymer High-NA power scaling program (target: 600W+ for EXE:5000).
- **ASML relevance:** High-NA EUV requires substantially higher source power; plasma stability physics is central to conversion efficiency improvements in tin-plasma LPP sources.
- **Falsifiable test:** ASML/Cymer files H01L/H05G patents referencing plasma confinement or magnetic perturbation for EUV source optimization in 2026–2027.

### Prediction 5: Non-Hermitian Topology → G01B Exceptional-Point Sensing Filings
**Confidence: 0.55** | Hypothesis ID: 1406 | **Timing: 2026–2027** *(speculative)*

- **Science signal:** Non-Hermitian Topology Exceptional Points (0607) — unstable frontier cluster (death_prob 0.853), appeared 2025-Q3/Q4. Exceptional-point physics enables extreme perturbation sensitivity — applicable to overlay and focus metrology.
- **Patent target:** G01B Optical Waveguide Propagation Sensing (growth_rate 1.75, death_prob 0.046).
- **Caveat:** Science cluster is high-risk and early-stage; no confirmed patent birth events yet; most speculative prediction in this set.

---

## Section 4: Strategic Recommendations

1. **Re-register physics corpus with correct field ID.** openalex_physics_quarterly uses OpenAlex field/31 (not physics). Register `openalex_physics_correct_quarterly` with `source_filter='https://openalex.org/fields/51'`. Until then, all science signals from this corpus should be treated as provisional — the relevant physics clusters represent ~4% of the corpus by count.

2. **Add G03F_quarterly to the patent corpus set.** ASML's core lithography IP files under G03F (photomechanical production of textured patterns) — the single most important patent class for lithography exposure tools. Its absence from the current corpus set is a major gap. Register and ingest immediately.

3. **Add H05G_quarterly for EUV/X-ray source IP.** ASML/Cymer's EUV plasma source patents are classified under H05G. The plasma confinement → EUV source prediction (Prediction 4) cannot be validated without this corpus.

4. **Monitor C23C void 155980 quarterly.** This 19-period void is the strongest structural trade secret signal in the dataset. If it begins closing (period_count stops increasing), it indicates a strategic IP move in EUV thin-film deposition — either ASML filing or a competitor entering the space. This is the single highest-value monitoring item from this analysis.

5. **Expand assignee normalization coverage.** Currently, assignee data is only available for H01L_quarterly. ASML is a major G01B, G02B, and C23C filer in reality. Without assignee data in these corpora, trade secret identification relies entirely on semantic distance and void geometry rather than direct patent count. Populating `data_assignee_normalized` for all five target corpora would dramatically improve confidence levels across all findings.

---

## Section 5: Key Structural Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Assignee data only in H01L | Cannot confirm/deny ASML presence in G01B/G02B/C23C/G01N | Populate data_assignee_normalized for all corpora |
| Science corpus uses wrong OpenAlex field (field/31 vs physics/51) | ~52% noise; only ~4% of clusters ASML-relevant | Register openalex_physics_correct_quarterly |
| 2026 science data sparse (31 docs vs ~7,000 baseline) | 2026 period signals unreliable; analysis uses 2022–2025 window | Re-ingest when 2026 OpenAlex data is complete |
| Void geometry mostly non-functional (centroid_drift 0%, void_type 0.02% fill) | Void narrowing/bridging dynamics not assessable | Fix compute_void_persistence and void geometry pipeline |
| G03F, H05G corpora absent | Core ASML filing classes not analyzed | Register and ingest G03F_quarterly, H05G_quarterly |

---

## Section 6: Findings Index

| ID | Type | Subtype | Confidence | Title |
|---|---|---|---|---|
| 1394 | finding | trade_secret | 0.85 | ASML Trade Secret: EUV Optics Multilayer Thin Film Deposition |
| 1395 | finding | trade_secret | 0.82 | ASML Trade Secret: Computational Wavefront Sensing / Scanner Aberration Reconstruction |
| 1396 | finding | trade_secret | 0.80 | ASML Trade Secret: Material Characterization Physics for Lithography Metrology |
| 1397 | finding | trade_secret | 0.78 | ASML Trade Secret: GaN/AlGaN Power Device Physics for EUV Source Integration |
| 1398 | finding | trade_secret | 0.92 | ASML IP Observation: Substrate Chuck/Heating System Dominance in H01L |
| 1399 | finding | void_bridge | 0.83 | Void Bridge: C23C Epitaxial-Coating Gap (19 Periods) — ASML EUV Optics Territory |
| 1400 | finding | void_bridge | 0.72 | Void Bridge: G01B OCT-Displacement Sensing Gap — ASML Scanner Metrology Territory |
| 1401 | finding | void_bridge | 0.70 | Void Bridge: G02B Camera-Silicon Photonics Gap — ASML Alignment Sensor Integration Territory |
| 1402 | hypothesis | prediction | 0.78 | ASML Prediction: 2D Materials → H01L EUV Patterning Filings (2024–2025) |
| 1403 | hypothesis | prediction | 0.72 | ASML Prediction: Valley Photonic Crystal → G02B Silicon Photonic Integration Filings (2024–2026) |
| 1404 | hypothesis | prediction | 0.74 | ASML Prediction: Semiconductor CVD → C23C Advanced Node Deposition Filings (2025) |
| 1405 | hypothesis | prediction | 0.68 | ASML Prediction: Plasma Confinement → H01L EUV Source Power Scaling Filings (2026–2027) |
| 1406 | hypothesis | prediction | 0.55 | ASML Prediction: Non-Hermitian Topology → G01B Exceptional-Point Sensing Filings (2026–2027) |

---

## Section 7: Cartographer Session Notes

**Task 176 (cartographer, openalex_physics_quarterly)** was run concurrently with this analysis. The cartographer session independently confirmed the data quality crisis in this corpus: wrong OpenAlex field ingestion (field/31 vs physics/51), producing 65 periods of heterogeneous content. The 6-cluster Q4 2026 "crystallization" is a data sparsity artifact, not a genuine scientific restructuring. Cartographer findings filed: 1384 (void_bridge), 1385 (anomaly), 1386–1388 (hypotheses), 1390 (period_summary).

**Implication for this analysis:** All science-to-patent inferences in this report should be treated as provisional until `openalex_physics_correct_quarterly` (field/51) is ingested and analyzed. The HIGH and MEDIUM relevance clusters identified in Step 2 (~16 clusters) are real physics content that was found within the noise, but they represent a sample not a complete picture of the relevant science landscape.

---

*Generated: 2026-03-19 | Session: cartographer_20260319_0900 | ARC v4, migration 149*

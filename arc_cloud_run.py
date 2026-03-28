#!/usr/bin/env python3
import os
if os.environ.get('CONDA_PREFIX') is None:
    os.environ['CONDA_PREFIX'] = '/home/jeff/miniconda3/envs/arc_embed'
"""
arc_cloud_run.py — Cloud pipeline: embed → kNN → Leiden → graph analytics

Runs on a cloud GPU machine. No DB connection at any point.
Downloads chunk TSVs from Hetzner Object Storage (S3-compatible), runs the
full pipeline, and uploads output TSVs back to S3.

Environment:  /home/jeff/miniconda3/envs/arc_embed/bin/python3
  Packages:   sentence-transformers, faiss-gpu-cu12, leidenalg, cugraph, cudf

Usage:
    python3 arc_cloud_run.py --corpus-id G06N_quarterly

    # Override config values:
    python3 arc_cloud_run.py --corpus-id G06N_quarterly --k 16 --batch-size 16

    # Local mode (skip S3):
    python3 arc_cloud_run.py --corpus-id G06N_quarterly \
        --local-input  /home/jeff/arc/data/cloud_in \
        --local-output /home/jeff/arc/data/cloud_out
"""

import argparse
import collections
import datetime
import gzip
import json
import logging
import os
import random
import shutil
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging — PhaseAdapter injects mutable phase name into every log record
# ---------------------------------------------------------------------------

class _PhaseFallbackFormatter(logging.Formatter):
    """
    Formatter that renders %(phase)s when present and falls back to a plain
    format when the LogRecord lacks it (e.g. records from third-party libraries
    that log without going through PhaseAdapter).
    """
    _PHASE_FMT = "%(asctime)s %(levelname)s [%(phase)s] %(message)s"
    _PLAIN_FMT = "%(asctime)s %(levelname)s %(message)s"

    _phase_formatter = logging.Formatter(_PHASE_FMT)
    _plain_formatter = logging.Formatter(_PLAIN_FMT)

    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "phase"):
            return self._phase_formatter.format(record)
        return self._plain_formatter.format(record)


class PhaseAdapter(logging.LoggerAdapter):
    """LoggerAdapter that injects a mutable 'phase' key into every LogRecord."""

    def set_phase(self, phase: str) -> None:
        self.extra["phase"] = phase

    def process(self, msg, kwargs):
        kwargs.setdefault("extra", {})["phase"] = self.extra["phase"]
        return msg, kwargs


def _setup_logging() -> PhaseAdapter:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_PhaseFallbackFormatter())
        root.addHandler(handler)
    else:
        # basicConfig may have already attached a handler — replace its formatter
        for handler in root.handlers:
            handler.setFormatter(_PhaseFallbackFormatter())
    return PhaseAdapter(logging.getLogger(__name__), {"phase": "init"})


log = _setup_logging()

# ---------------------------------------------------------------------------
# PeriodOutputs dataclass — groups write_period_outputs parameters
# ---------------------------------------------------------------------------

@dataclass
class PeriodOutputs:
    corpus_id: str
    period_start: str
    period_indices: List[int]
    chunk_ids: List[str]
    knn_edges: List
    cluster_map: Dict[str, int]
    chunk_graph_df: Any          # cuDF DataFrame or None
    f_edge_df: Any               # cuDF DataFrame or None
    centroids: Dict[int, "np.ndarray"]
    chunk_measures: Dict[str, Dict]
    f_void_rows: List
    f_gap_rows: List[Dict]
    spectral: Tuple              # (alg_conn, spectral_gap) or (None, None)
    cluster_geometry: Dict[int, Dict]
    cluster_stats: Dict[int, Dict]
    topology_stats: Dict[int, Dict]
    field_surprise: Dict[int, Optional[float]]
    phase_scores: Dict           # keys: phase_transition_score, n_dark_matter_chunks,
                                 #       is_dark_matter, membership_volatility,
                                 #       belief_persistence_score
    leiden_modularity: float


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cloud embed → kNN → Leiden → graph pipeline"
    )
    p.add_argument("--corpus-id",    required=True, help="Corpus identifier string")
    p.add_argument("--k",            type=int, default=None, help="Override kNN k from config")
    p.add_argument("--batch-size",   type=int, default=16,   help="Embedding batch size")
    p.add_argument("--local-input",  default=None,
                   help="Local input root (skip S3 download). "
                        "Expects {dir}/{corpus_id}/chunks_{corpus_id}.tsv and config.json")
    p.add_argument("--local-output", default=None,
                   help="Local output root (skip S3 upload). "
                        "Writes cold/ and import/ subdirs here.")
    return p.parse_args()

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _get_s3_client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
    )


def _s3_download(s3, bucket: str, key: str, local_path: str, retries: int = 3) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            s3.download_file(bucket, key, local_path)
            size = os.path.getsize(local_path)
            log.info("Downloaded s3://%s/%s → %s (%d bytes)", bucket, key, local_path, size)
            return
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"S3 download failed after {retries} attempts: {key}"
                ) from exc
            log.warning("S3 download attempt %d failed (%s) — retrying in 10s", attempt, exc)
            time.sleep(10)


def _s3_upload(s3, bucket: str, key: str, local_path: str, retries: int = 3) -> None:
    size = os.path.getsize(local_path)
    for attempt in range(1, retries + 1):
        try:
            s3.upload_file(local_path, bucket, key)
            log.info("Uploaded %s → s3://%s/%s (%d bytes)", local_path, bucket, key, size)
            return
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"S3 upload failed after {retries} attempts: {key}"
                ) from exc
            log.warning("S3 upload attempt %d failed (%s) — retrying in 10s", attempt, exc)
            time.sleep(10)


def _s3_download_optional(s3, bucket: str, key: str, local_path: str) -> bool:
    """Download from S3 if the key exists. Returns True on success, False if not found."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        s3.download_file(bucket, key, local_path)
        size = os.path.getsize(local_path)
        log.info("Downloaded optional s3://%s/%s → %s (%d bytes)", bucket, key, local_path, size)
        return True
    except Exception as exc:
        err_code = ""
        if hasattr(exc, "response") and isinstance(exc.response, dict):
            err_code = exc.response.get("Error", {}).get("Code", "")
        if err_code in ("404", "NoSuchKey"):
            log.info("Optional input not found in S3: %s", key)
        else:
            log.warning("Optional input unavailable (%s — %s): %s", type(exc).__name__, exc, key)
        return False

# ---------------------------------------------------------------------------
# Period bucketing
# ---------------------------------------------------------------------------

def _period_start(filing_date: str, resolution: str) -> Optional[str]:
    """Return YYYY-MM-DD period_start for a filing date, or None on parse failure."""
    try:
        d = datetime.date.fromisoformat(filing_date)
    except (ValueError, TypeError):
        return None
    if resolution == "quarterly":
        quarter = (d.month - 1) // 3
        return datetime.date(d.year, quarter * 3 + 1, 1).isoformat()
    elif resolution == "monthly":
        return datetime.date(d.year, d.month, 1).isoformat()
    elif resolution == "annual":
        return datetime.date(d.year, 1, 1).isoformat()
    else:
        raise ValueError(f"Unknown resolution: {resolution!r}")

# ---------------------------------------------------------------------------
# FAISS index builder — GPU with silent CPU fallback
# ---------------------------------------------------------------------------

def _build_faiss_index(dim: int):
    import faiss
    try:
        res = faiss.StandardGpuResources()
        cpu_index = faiss.IndexFlatIP(dim)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        log.info("FAISS: using GPU IndexFlatIP (dim=%d)", dim)
        return gpu_index
    except (AttributeError, RuntimeError) as exc:
        log.info("FAISS GPU unavailable (%s) — falling back to CPU IndexFlatIP", exc)
        return faiss.IndexFlatIP(dim)

# ---------------------------------------------------------------------------
# Phase 0: Download inputs
# ---------------------------------------------------------------------------

def phase_download(
    corpus_id: str,
    work_dir: str,
    local_input: Optional[str],
) -> Tuple[str, str]:
    """
    Return (chunks_tsv_path, config_json_path) on local disk.
    Downloads from S3 unless --local-input is set.
    """
    log.set_phase("download")

    if local_input:
        chunks_path = os.path.join(local_input, corpus_id, f"chunks_{corpus_id}.tsv")
        config_path = os.path.join(local_input, corpus_id, "config.json")
        for p in (chunks_path, config_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Local input file not found: {p}")
        log.info("Local mode: reading inputs from %s", local_input)
        return chunks_path, config_path

    s3 = _get_s3_client()
    bucket = os.environ["S3_BUCKET"]
    local_dir = os.path.join(work_dir, "input", corpus_id)

    chunks_path = os.path.join(local_dir, f"chunks_{corpus_id}.tsv")
    config_path = os.path.join(local_dir, "config.json")

    _s3_download(s3, bucket, f"input/{corpus_id}/chunks_{corpus_id}.tsv", chunks_path)
    _s3_download(s3, bucket, f"input/{corpus_id}/config.json", config_path)

    return chunks_path, config_path

# ---------------------------------------------------------------------------
# Phase 1: Period splitting
# ---------------------------------------------------------------------------

def phase_period_split(
    chunks_path: str,
    resolution: str,
    year_from: int,
) -> Tuple[List[str], List[str], Dict[str, List[int]]]:
    """
    Parse chunks TSV and bucket rows into periods.

    Returns:
        chunk_ids   — list[str] of all chunk_id values (index = row position)
        texts       — list[str] of idea_text values (same order as chunk_ids)
        period_map  — dict{period_start: [row_indices]}
    """
    log.set_phase("period_split")

    chunk_ids: List[str] = []
    texts:     List[str] = []
    period_map: Dict[str, List[int]] = {}
    skipped = 0
    duplicates = 0

    # Tracks (chunk_id, period) pairs already added.  A chunk_id should appear
    # in exactly one period, but upstream SQL joins sometimes emit duplicate rows
    # for the same (chunk_id, filing_date).  Without this guard, every duplicate
    # lands in the same period_map bucket and write_period_outputs writes it
    # twice to chunk_periods.tsv.  chunk_graph is immune (cuGraph deduplicates
    # nodes), which is why only chunk_periods is affected.
    seen: set = set()

    with open(chunks_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            parts = line.split("\t", 2)
            if len(parts) < 3:
                log.warning("Skipping line %d: expected 3 tab-separated fields", lineno)
                skipped += 1
                continue

            cid, text, filing_date = parts[0], parts[1], parts[2]
            period = _period_start(filing_date, resolution)
            if period is None:
                log.warning("Skipping chunk %s: unparseable date %r", cid, filing_date)
                skipped += 1
                continue

            try:
                if datetime.date.fromisoformat(period).year < year_from:
                    skipped += 1
                    continue
            except ValueError:
                skipped += 1
                continue

            key = (cid, period)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)

            idx = len(chunk_ids)
            chunk_ids.append(cid)
            texts.append(text)
            period_map.setdefault(period, []).append(idx)

    if duplicates:
        log.warning(
            "Period split: dropped %d duplicate (chunk_id, period) rows from input",
            duplicates,
        )
    log.info(
        "Period split: %d chunks across %d periods (skipped %d)",
        len(chunk_ids), len(period_map), skipped,
    )
    for ps in sorted(period_map):
        log.info("  %s: %d chunks", ps, len(period_map[ps]))

    return chunk_ids, texts, period_map

# ---------------------------------------------------------------------------
# Phase 2: Embed all chunks
# ---------------------------------------------------------------------------

def phase_embed(
    texts: List[str],
    model_name: str,
    batch_size: int,
    max_seq_length: int = 512,
) -> np.ndarray:
    """
    Encode all chunks at once with sentence-transformers.
    Returns float32 ndarray of shape (n_chunks, dim).
    """
    log.set_phase("embed")
    from sentence_transformers import SentenceTransformer

    log.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    # Cap max_seq_length for patent/journal abstract corpora.
    # Qwen3-Embedding-0.6B defaults to 32,768 tokens — this pads every batch
    # to the longest sequence, causing ~73x slowdown when any abstract exceeds
    # a few hundred tokens. Patent abstracts are typically 100-300 tokens.
    # 512 tokens captures 99%+ of abstract content with no quality loss.
    # Comment: increase this value only for full-text corpora (claims + description).
    _model_default = model.max_seq_length
    model.max_seq_length = max_seq_length
    log.info("[embed] max_seq_length set to %d (model default was %d)",
             max_seq_length, _model_default)

    log.info("Embedding %d chunks (batch_size=%d)...", len(texts), batch_size)
    t0 = time.monotonic()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we normalize per-period before FAISS
    )
    elapsed = time.monotonic() - t0
    log.info(
        "Embedding complete: %d chunks in %.1fs (%.1f chunks/sec), dim=%d",
        len(texts), elapsed, len(texts) / max(elapsed, 1e-6), emb.shape[1],
    )
    return emb.astype(np.float32)

# ---------------------------------------------------------------------------
# Phase 3a: kNN via FAISS
# ---------------------------------------------------------------------------

def run_knn(
    period_indices: List[int],
    chunk_ids: List[str],
    embeddings: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, List[Tuple[str, str, float]]]:
    """
    Build FAISS IndexFlatIP for this period's chunks, search k neighbours.

    Returns:
        period_emb  — (n, dim) L2-normalised float32 embeddings for this period
        knn_edges   — list of (chunk_id, neighbor_id, cosine_similarity) tuples
                      Contains both A→B and B→A directions (Graph() deduplicates).
    """
    import faiss

    k = int(k)  # guard: coerce in case caller passes a string from JSON
    period_emb = embeddings[period_indices].copy()
    n, dim = period_emb.shape

    # L2-normalise → inner product == cosine similarity
    norms = np.linalg.norm(period_emb, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    period_emb /= norms

    index = _build_faiss_index(dim)
    index.add(period_emb)

    actual_k = min(k + 1, n)  # +1 to exclude self; cap at corpus size
    distances, nbr_indices = index.search(period_emb, actual_k)

    knn_edges: List[Tuple[str, str, float]] = []
    for row_i, (dists, nbrs) in enumerate(zip(distances, nbr_indices)):
        src_id = chunk_ids[period_indices[row_i]]
        for dist, nbr_idx in zip(dists, nbrs):
            if nbr_idx == -1 or nbr_idx == row_i:
                continue  # skip invalid / self
            dst_id = chunk_ids[period_indices[int(nbr_idx)]]
            knn_edges.append((src_id, dst_id, float(dist)))

    log.info("kNN: %d chunks → %d edges (k=%d)", n, len(knn_edges), k)
    return period_emb, knn_edges

# ---------------------------------------------------------------------------
# Phase 3b: Leiden clustering
# ---------------------------------------------------------------------------

def run_leiden(
    period_indices: List[int],
    chunk_ids: List[str],
    knn_edges: List[Tuple[str, str, float]],
    leiden_res: float,
    leiden_seed: int,
) -> Tuple[Dict[str, int], float]:
    """
    Build undirected igraph from knn_edges, run Leiden with RBConfiguration.
    Returns {chunk_id: integer_cluster_id}.
    """
    import igraph as ig
    import leidenalg

    period_chunk_ids = [chunk_ids[i] for i in period_indices]
    chunk_to_idx     = {cid: i for i, cid in enumerate(period_chunk_ids)}

    # Deduplicate edges: take max similarity for each undirected pair
    best_weight: Dict[Tuple[int, int], float] = {}
    for src, dst, sim in knn_edges:
        si = chunk_to_idx.get(src)
        di = chunk_to_idx.get(dst)
        if si is None or di is None or si == di:
            continue
        key = (min(si, di), max(si, di))
        if sim > best_weight.get(key, -1.0):
            best_weight[key] = sim

    ig_edges = list(best_weight.keys())
    # Clip to [0, ∞) — leidenalg raises "Cannot accept negative weights" on any
    # negative value, which can appear for anti-correlated FAISS cosine scores.
    weights  = [max(0.0, best_weight[e]) for e in ig_edges]

    g = ig.Graph(n=len(period_chunk_ids), edges=ig_edges, directed=False)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=leiden_res,
        seed=leiden_seed,
        weights="weight",
    )

    cluster_map = {
        period_chunk_ids[i]: int(partition.membership[i])
        for i in range(len(period_chunk_ids))
    }
    n_clusters = len(set(cluster_map.values()))

    # modularity: how well the Leiden partition separates the graph into communities
    # vs what would be expected by chance.  Range: [-0.5, 1.0].
    # High modularity = strong community structure, distinct research areas.
    # Low modularity = clusters not well-defined in this period's kNN graph.
    # Computed directly from the leidenalg partition object — zero extra cost.
    modularity = float(partition.modularity)

    log.info(
        "Leiden: %d chunks → %d clusters (res=%.2f, seed=%d, modularity=%.4f)",
        len(period_chunk_ids), n_clusters, leiden_res, leiden_seed, modularity,
    )
    return cluster_map, modularity

# ---------------------------------------------------------------------------
# Phase 3c: cuGraph analytics
# ---------------------------------------------------------------------------

def run_cugraph(
    knn_edges: List[Tuple[str, str, float]],
    betweenness_k: int = 256,
):
    """
    Compute per-chunk graph measures via cuGraph.
    Returns cuDF DataFrame with columns:
        chunk_id, betweenness_centrality, core_number, clustering_coeff,
        triangle_count, degree, pagerank, eigenvector_centrality,
        katz_centrality, harmonic_centrality, in_degree_centrality
    Returns None on OOM or if cuGraph unavailable.
    """
    try:
        import cudf
        import cugraph
    except Exception as exc:
        log.warning("cuGraph/cuDF import failed (%s) — skipping graph analytics", exc)
        return None

    try:
        edges_df = cudf.DataFrame({
            "chunk_id":    cudf.Series([e[0] for e in knn_edges], dtype="str"),
            "neighbor_id": cudf.Series([e[1] for e in knn_edges], dtype="str"),
            "distance":    cudf.Series([e[2] for e in knn_edges], dtype="float32"),
        })

        G = cugraph.Graph()
        G.from_cudf_edgelist(
            edges_df,
            source="chunk_id",
            destination="neighbor_id",
            edge_attr="distance",
        )
        n_vertices = G.number_of_vertices()
        log.info("cuGraph: %d vertices, %d edges", n_vertices, G.number_of_edges())

        if n_vertices == 0:
            log.warning("cuGraph: graph has 0 vertices — skipping graph analytics")
            return None

        # Cap betweenness k to graph size (cuGraph raises if k > n_vertices)
        effective_k = min(betweenness_k, n_vertices)
        log.info("cuGraph: betweenness k=%d (requested %d)", effective_k, betweenness_k)

        # Betweenness centrality (approximate)
        bc = cugraph.betweenness_centrality(G, k=effective_k, normalized=True)
        bc = bc.rename(columns={"vertex": "chunk_id"})

        # Core number
        core = cugraph.core_number(G)
        core = core.rename(columns={"vertex": "chunk_id"})

        # Triangle count → clustering coefficient
        tri = cugraph.triangle_count(G)
        tri = tri.rename(columns={"vertex": "chunk_id", "counts": "triangle_count"})

        deg = G.degrees()[["vertex", "in_degree"]].rename(
            columns={"vertex": "chunk_id", "in_degree": "degree"}
        )

        chunk_graph = bc.merge(core, on="chunk_id", how="outer")
        chunk_graph = chunk_graph.merge(tri, on="chunk_id", how="left")
        chunk_graph = chunk_graph.merge(deg, on="chunk_id", how="left")

        # clustering_coeff = 2t / (d*(d-1)); 0 for degree < 2
        denom = chunk_graph["degree"] * (chunk_graph["degree"] - 1)
        chunk_graph["clustering_coeff"] = (
            (2.0 * chunk_graph["triangle_count"]) / denom
        ).fillna(0.0)
        chunk_graph["clustering_coeff"] = chunk_graph["clustering_coeff"].replace(
            [float("inf"), float("-inf")], 0.0
        )
        # triangle_count — raw closed-triangle count per chunk; kept as direct
        # structural density signal (high = tightly-knit local community)

        # degree — number of kNN edges per chunk; kept as local connectivity
        # baseline (high = hub concept, low = niche or peripheral chunk)

        # PageRank — stationary probability of a random walk landing on this
        # chunk; high = semantic landmark reached by many paths, not just hubs
        pr = cugraph.pagerank(G)
        pr = pr.rename(columns={"vertex": "chunk_id", "pagerank": "pagerank"})
        chunk_graph = chunk_graph.merge(pr, on="chunk_id", how="left")

        # Eigenvector centrality — chunk importance weighted by neighbor
        # importance; high = connected to other highly-central chunks (core
        # of an influential sub-region of the kNN graph).
        # cuGraph power iteration can fail to converge on certain topologies
        # (disconnected components, star graphs); fall back to NetworkX on CPU.
        try:
            ec = cugraph.eigenvector_centrality(G)
            ec = ec.rename(columns={"vertex": "chunk_id",
                                     "eigenvector_centrality": "eigenvector_centrality"})
            chunk_graph = chunk_graph.merge(ec, on="chunk_id", how="left")
        except Exception as _ec_err:
            log.warning("cugraph.eigenvector_centrality failed (%s) — computing via NetworkX CPU fallback", _ec_err)
            import networkx as nx
            nx_G_ec = nx.Graph()
            nx_G_ec.add_nodes_from(edges_df["chunk_id"].to_arrow().to_pylist())
            nx_G_ec.add_edges_from(
                zip(edges_df["chunk_id"].to_arrow().to_pylist(),
                    edges_df["neighbor_id"].to_arrow().to_pylist())
            )
            ec_dict = nx.eigenvector_centrality_numpy(nx_G_ec)
            max_ec = max(abs(v) for v in ec_dict.values()) if ec_dict else 1.0
            if max_ec == 0.0:
                max_ec = 1.0
            ec_norm = {k: v / max_ec for k, v in ec_dict.items()}
            ec_df = cudf.DataFrame({
                "chunk_id": list(ec_norm.keys()),
                "eigenvector_centrality": list(ec_norm.values()),
            })
            chunk_graph = chunk_graph.merge(ec_df, on="chunk_id", how="left")

        # Katz centrality — damped path-count; less hub-dominated than
        # eigenvector; high = reachable from many chunks via short paths
        kz = cugraph.katz_centrality(G)
        kz = kz.rename(columns={"vertex": "chunk_id",
                                  "katz_centrality": "katz_centrality"})
        chunk_graph = chunk_graph.merge(kz, on="chunk_id", how="left")

        # Harmonic centrality — sum of inverse shortest-path (hop count)
        # to all other vertices; well-defined on disconnected graphs
        # (unreachable pairs contribute 0). High = global integrator.
        # cuGraph 26.x has no harmonic_centrality and multi_source_bfs is
        # unimplemented. NetworkX is faster than a per-vertex BFS GPU loop
        # due to kernel launch overhead (~40ms/vertex vs ~0.1ms/vertex CPU).
        import networkx as nx
        nx_G = nx.Graph()
        nx_G.add_nodes_from(edges_df["chunk_id"].to_arrow().to_pylist())
        nx_G.add_edges_from(
            zip(edges_df["chunk_id"].to_arrow().to_pylist(),
                edges_df["neighbor_id"].to_arrow().to_pylist())
        )
        hc_dict = nx.harmonic_centrality(nx_G)
        max_hc = max(hc_dict.values()) if hc_dict else 1.0
        if max_hc == 0.0:
            max_hc = 1.0
        hc_norm = {k: v / max_hc for k, v in hc_dict.items()}
        hc_df = cudf.DataFrame({
            "chunk_id": list(hc_norm.keys()),
            "harmonic_centrality": list(hc_norm.values()),
        })
        chunk_graph = chunk_graph.merge(hc_df, on="chunk_id", how="left")

        # in_degree_centrality — degree normalised by (n-1); derived from
        # degree rather than a separate kernel; scale-free connectivity
        # fraction usable across corpora of different sizes
        chunk_graph["in_degree_centrality"] = (
            chunk_graph["degree"].astype("float32") / max(n_vertices - 1, 1)
        )

        log.info("cuGraph analytics complete: %d vertex records", len(chunk_graph))
        return chunk_graph

    except Exception as exc:
        exc_str = str(exc).lower()
        if "out of memory" in exc_str or "cudaerrormemoryal" in exc_str:
            log.warning(
                "cuGraph OOM — skipping graph analytics for this period. Error: %s", exc
            )
            return None
        raise

# ---------------------------------------------------------------------------
# Phase 3d: f_edge construction (cuDF)
# ---------------------------------------------------------------------------

def run_fedge(
    knn_edges: List[Tuple[str, str, float]],
    cluster_map: Dict[str, int],
):
    """
    Compute cluster-level connection metrics using cuDF joins.
    Returns cuDF DataFrame with columns:
        cluster_a, cluster_b, connection_weight,
        semantic_overlap_a_to_b, semantic_overlap_b_to_a, n_shared_edges
    Returns None if cuDF is unavailable (no GPU / CUDA env not configured).
    """
    try:
        import cudf
        if not hasattr(cudf, 'DataFrame'):
            raise AttributeError("module 'cudf' has no attribute 'DataFrame'")
    except Exception as exc:
        log.warning("cuDF import failed in run_fedge (%s) — skipping f_edge", exc)
        return None

    knn_df = cudf.DataFrame({
        "chunk_id":    cudf.Series([e[0] for e in knn_edges], dtype="str"),
        "neighbor_id": cudf.Series([e[1] for e in knn_edges], dtype="str"),
    })

    cp_keys = list(cluster_map.keys())
    cp_df = cudf.DataFrame({
        "chunk_id":   cudf.Series(cp_keys,                          dtype="str"),
        "cluster_id": cudf.Series([cluster_map[c] for c in cp_keys], dtype="int64"),
    })

    # Attach cluster labels to both edge endpoints
    edges = knn_df.merge(
        cp_df.rename(columns={"cluster_id": "cluster_a"}),
        on="chunk_id",
        how="inner",
    )
    edges = edges.merge(
        cp_df.rename(columns={"chunk_id": "neighbor_id", "cluster_id": "cluster_b"}),
        on="neighbor_id",
        how="inner",
    )

    # Cross-cluster edges only
    cross = edges[edges["cluster_a"] != edges["cluster_b"]].copy()
    log.info("f_edge: cross-cluster edges before dedup: %d", len(cross))

    # Deduplicate undirected pairs: keep canonical direction chunk_id < neighbor_id
    # (string/UUID lexicographic order is consistent and well-defined)
    cross = cross[cross["chunk_id"] < cross["neighbor_id"]]
    log.info("f_edge: cross-cluster edges after dedup:  %d", len(cross))

    # Cluster sizes for normalisation
    sizes = cp_df.groupby("cluster_id").size().reset_index(name="cluster_size")

    # Raw edge count per cluster pair (kept as an independent signal)
    n_edges = (
        cross.groupby(["cluster_a", "cluster_b"])
        .size()
        .reset_index(name="n_shared_edges")
    )

    # Distinct source chunks in cluster_a that touch cluster_b (A→B direction)
    n_src = (
        cross.groupby(["cluster_a", "cluster_b"])["chunk_id"]
        .nunique()
        .reset_index(name="n_source_chunks")
    )

    # Distinct target chunks in cluster_b reached from cluster_a (B side)
    n_dst = (
        cross.groupby(["cluster_a", "cluster_b"])["neighbor_id"]
        .nunique()
        .reset_index(name="n_dest_chunks")
    )

    f_edge = n_edges.merge(n_src, on=["cluster_a", "cluster_b"], how="left")
    f_edge = f_edge.merge(n_dst, on=["cluster_a", "cluster_b"], how="left")
    f_edge = f_edge.merge(
        sizes.rename(columns={"cluster_id": "cluster_a", "cluster_size": "size_a"}),
        on="cluster_a", how="left",
    )
    f_edge = f_edge.merge(
        sizes.rename(columns={"cluster_id": "cluster_b", "cluster_size": "size_b"}),
        on="cluster_b", how="left",
    )

    # connection_weight: fraction of cluster_a chunks with at least one cross-cluster edge
    # semantic_overlap_a_to_b: same fraction (A-side perspective)
    # semantic_overlap_b_to_a: fraction of cluster_b chunks touched from cluster_a (B-side)
    f_edge["connection_weight"]       = f_edge["n_source_chunks"] / f_edge["size_a"]
    f_edge["semantic_overlap_a_to_b"] = f_edge["n_source_chunks"] / f_edge["size_a"]
    f_edge["semantic_overlap_b_to_a"] = f_edge["n_dest_chunks"]   / f_edge["size_b"]
    f_edge = f_edge.drop(columns=["n_source_chunks", "n_dest_chunks", "size_a", "size_b"])

    # semantic_overlap_max: stronger directional overlap regardless of direction.
    # Captures asymmetric relationships — one cluster may pull toward the other
    # without reciprocation; max gives the strongest signal either way.
    # NOTE: cuDF .max(axis=1) triggers cuPy CUB kernel compilation which fails on some
    # CUDA builds (NVRTCError: NVRTC_ERROR_COMPILATION). Use pandas for this operation.
    if len(f_edge) > 0:
        _pdf = f_edge[["semantic_overlap_a_to_b", "semantic_overlap_b_to_a"]].to_pandas()
        import cudf as _cudf
        f_edge["semantic_overlap_max"] = _cudf.Series(
            _pdf.max(axis=1).values, dtype="float32"
        )
    else:
        import cudf as _cudf
        f_edge["semantic_overlap_max"] = _cudf.Series([], dtype="float32")

    max_cw = float(f_edge["connection_weight"].max()) if len(f_edge) > 0 else 0.0
    if max_cw > 1.0:
        log.warning("f_edge: connection_weight max=%.4f > 1.0 — check edge dedup", max_cw)
    else:
        log.info("f_edge: %d cluster pairs, connection_weight max=%.4f (OK)", len(f_edge), max_cw)

    return f_edge

# ---------------------------------------------------------------------------
# Phase 3e: Centroids (numpy)
# ---------------------------------------------------------------------------

def run_centroids(
    period_indices: List[int],
    chunk_ids: List[str],
    cluster_map: Dict[str, int],
    embeddings: np.ndarray,
) -> Dict[int, np.ndarray]:
    """
    Compute mean embedding centroid per cluster using numpy (CPU).
    Returns {cluster_id: centroid_float32_array}.
    """
    cluster_vecs: Dict[int, List[np.ndarray]] = {}
    for idx in period_indices:
        cid = chunk_ids[idx]
        clu = cluster_map.get(cid)
        if clu is None:
            continue
        cluster_vecs.setdefault(clu, []).append(embeddings[idx])

    centroids = {
        clu: np.mean(np.stack(vecs), axis=0).astype(np.float32)
        for clu, vecs in cluster_vecs.items()
    }
    log.info("Centroids: %d clusters", len(centroids))
    return centroids

# ---------------------------------------------------------------------------
# Phase 3e-b: Cluster geometry (SVD-based shape measures, numpy, CPU)
# ---------------------------------------------------------------------------

def run_cluster_geometry(
    cluster_map: Dict[str, int],
    embeddings: np.ndarray,
    period_indices: List[int],
    chunk_ids: List[str],
) -> Dict[int, dict]:
    """
    Compute SVD-based shape measures for each cluster in the period.

    For each cluster, extracts its embedding matrix and runs SVD to compute:
        elongation_ratio  — sv[0]/sv[1]: high = needle-like, low = spherical
        volume_estimate   — product of top-3 singular values: size of occupied space
        skewness_pc1      — distribution skewness along primary axis (PC1)
        kurtosis_pc1      — distribution kurtosis along primary axis
        skewness_pc2      — distribution skewness along secondary axis (PC2)
        kurtosis_pc2      — distribution kurtosis along secondary axis

    Returns {cluster_id: {measure: value}}.
    All values are None for clusters with fewer than 2 chunks (SVD undefined);
    elongation_ratio additionally requires >= 5 chunks and sv[1] > 1e-6 to guard
    against degenerate near-collinear clusters that produce overflow values.
    """
    from scipy import stats

    # Build per-cluster lists of local row indices into period_embeddings
    period_chunk_ids  = [chunk_ids[i] for i in period_indices]
    period_embeddings = embeddings[period_indices]          # (n_period, dim)

    cluster_indices: Dict[int, List[int]] = {}
    for local_i, cid in enumerate(period_chunk_ids):
        clu = cluster_map.get(cid)
        if clu is not None:
            cluster_indices.setdefault(clu, []).append(local_i)

    geometry: Dict[int, dict] = {}
    n_skipped = 0
    elongation_values: List[float] = []

    for cluster_id, local_mask in cluster_indices.items():
        if len(local_mask) < 2:
            # Cannot compute SVD for single-chunk clusters — set NULL
            geometry[cluster_id] = {
                "elongation_ratio": None,
                "volume_estimate":  None,
                "skewness_pc1":     None,
                "kurtosis_pc1":     None,
                "skewness_pc2":     None,
                "kurtosis_pc2":     None,
            }
            n_skipped += 1
            continue

        cluster_emb = period_embeddings[local_mask]         # shape [n, dim]

        # SVD — singular values only (fast, no U/V matrices needed)
        # sv[0] = extent along primary axis (dominant direction of variation)
        # sv[1] = extent along secondary axis
        # sv[0]/sv[1] = elongation — high means cluster is needle-like, low means spherical
        singular_values = np.linalg.svd(cluster_emb, compute_uv=False)

        # elongation_ratio: sv[0]/sv[1] — undefined when cluster has fewer than 5
        # chunks or when sv[1] ≈ 0 (all chunks nearly collinear in embedding space).
        # Clamp to [1.0, 100.0] as a safety net — genuine elongation above 100
        # is not meaningful and indicates numerical instability.
        if len(local_mask) < 5:
            elongation_ratio = None
        elif len(singular_values) < 2 or singular_values[1] < 1e-6:
            elongation_ratio = None  # degenerate — second axis effectively zero
        else:
            elongation_ratio = float(np.clip(singular_values[0] / singular_values[1], 1.0, 100.0))

        # Volume: product of top-3 singular values (embedding space volume occupied).
        # Large = spread across many directions, small = tight/focused.
        # Guard: if any of the top-3 singular values is near zero the product is
        # degenerate — set None rather than a misleadingly tiny non-zero number.
        volume_estimate = (
            float(np.prod(singular_values[:3]))
            if len(singular_values) >= 3 and singular_values[2] >= 1e-6 else None
        )

        # Project onto PC1 and PC2 for distribution shape
        # skewness > 0 means tail toward high-value end of axis
        # kurtosis > 0 means heavier tails than normal (leptokurtic)
        # These capture whether the cluster has outlier sub-topics pulling it asymmetrically
        U, s, Vt = np.linalg.svd(
            cluster_emb - cluster_emb.mean(axis=0), full_matrices=False
        )
        projections_pc1 = U[:, 0] * s[0]
        projections_pc2 = (U[:, 1] * s[1]) if s.shape[0] > 1 else None

        skewness_pc1 = (
            float(stats.skew(projections_pc1))
            if len(projections_pc1) >= 3 else None
        )
        kurtosis_pc1 = (
            float(stats.kurtosis(projections_pc1))
            if len(projections_pc1) >= 3 else None
        )
        skewness_pc2 = (
            float(stats.skew(projections_pc2))
            if projections_pc2 is not None and len(projections_pc2) >= 3 else None
        )
        kurtosis_pc2 = (
            float(stats.kurtosis(projections_pc2))
            if projections_pc2 is not None and len(projections_pc2) >= 3 else None
        )

        geometry[cluster_id] = {
            "elongation_ratio": elongation_ratio,
            "volume_estimate":  volume_estimate,
            "skewness_pc1":     skewness_pc1,
            "kurtosis_pc1":     kurtosis_pc1,
            "skewness_pc2":     skewness_pc2,
            "kurtosis_pc2":     kurtosis_pc2,
        }
        if elongation_ratio is not None:
            elongation_values.append(elongation_ratio)

    mean_elongation = float(np.mean(elongation_values)) if elongation_values else float("nan")
    log.debug(
        "cluster_geometry: %d clusters computed, %d skipped (<2 chunks), mean_elongation=%.4f",
        len(geometry) - n_skipped, n_skipped, mean_elongation,
    )
    return geometry


# ---------------------------------------------------------------------------
# Phase 3f: Per-chunk derived measures (numpy, CPU)
# ---------------------------------------------------------------------------

def run_chunk_measures(
    period_indices: List[int],
    chunk_ids: List[str],
    period_emb: np.ndarray,           # L2-normalised, shape (n_period, dim)
    centroids: Dict[int, np.ndarray],
    cluster_map: Dict[str, int],
    knn_edges: List[Tuple[str, str, float]],
) -> Dict[str, dict]:
    """
    Compute point_density, distance_to_centroid, energy, boundary_score,
    and intrinsic_dim for every chunk in the period.

    Returns {chunk_id: {measure: value}}.
    intrinsic_dim is None for chunks with fewer than 3 available neighbours.
    """
    n = len(period_indices)

    # Local index lookup: chunk_id → row position in period_emb
    chunk_id_to_local: Dict[str, int] = {
        chunk_ids[period_indices[li]]: li for li in range(n)
    }

    # Pre-normalise centroids for dot-product cosine distance
    cent_norm: Dict[int, np.ndarray] = {}
    for clu, vec in centroids.items():
        norm = np.linalg.norm(vec)
        cent_norm[clu] = vec / norm if norm > 0 else vec

    # Neighbour maps built from knn_edges (only period chunks)
    nbr_sims: Dict[str, List[float]] = {}      # chunk_id → [cosine_sim values]
    nbr_ids:  Dict[str, List[str]]   = {}      # chunk_id → [neighbor chunk_ids]
    for src, dst, sim in knn_edges:
        if src in chunk_id_to_local:
            nbr_sims.setdefault(src, []).append(sim)
            nbr_ids.setdefault(src, []).append(dst)

    measures: Dict[str, dict] = {}

    for local_i, global_i in enumerate(period_indices):
        cid = chunk_ids[global_i]
        clu = cluster_map.get(cid)

        # ---- point_density: mean(1 - distance) across kNN neighbours ----
        # knn distance stored as cosine similarity; spec treats similarity = 1 - d
        sims = nbr_sims.get(cid, [])
        point_density = float(np.mean([1.0 - d for d in sims])) if sims else 0.0

        # ---- energy: inverse of point_density ----
        energy = 1.0 / max(point_density, 1e-9)

        # ---- distance_to_centroid: cosine distance to cluster centroid ----
        if clu is not None and clu in cent_norm:
            dtc = float(1.0 - np.dot(period_emb[local_i], cent_norm[clu]))
        else:
            dtc = None

        # ---- boundary_score: fraction of neighbours in a different cluster ----
        neighbours = nbr_ids.get(cid, [])
        if neighbours:
            n_cross = sum(1 for nid in neighbours if cluster_map.get(nid) != clu)
            boundary_score = float(n_cross / len(neighbours))
        else:
            boundary_score = 0.0

        # ---- intrinsic_dim: PCA on k-neighbourhood matrix ----
        # Collect normalised embeddings of neighbours that are in this period
        nbr_embs = [
            period_emb[chunk_id_to_local[nid]]
            for nid in neighbours
            if nid in chunk_id_to_local
        ]
        if len(nbr_embs) >= 3:
            mat = np.stack(nbr_embs)          # (k_avail, dim)
            mat = mat - mat.mean(axis=0)      # center before SVD
            _, S, _ = np.linalg.svd(mat, full_matrices=False)
            total_var = float(np.sum(S ** 2))
            if total_var > 0:
                cum_var = np.cumsum(S ** 2) / total_var
                # +1 because searchsorted returns index before threshold
                intrinsic_dim: Optional[int] = int(np.searchsorted(cum_var, 0.90) + 1)
            else:
                intrinsic_dim = 1
        else:
            intrinsic_dim = None

        # ---- uncertainty: normalised stddev of kNN cosine distances ----
        # High uncertainty = chunk sits in an ambiguous region between clusters;
        # its neighbours span a wide range of distances rather than clustering tightly.
        # Measures how ambiguous cluster membership is at this chunk's location.
        if len(sims) >= 2:
            dists_for_unc = [1.0 - s for s in sims]   # cosine distances
            uncertainty = float(np.std(dists_for_unc) / (np.mean(dists_for_unc) + 1e-9))

            # ---- boundary_proximity: raw STDDEV of kNN cosine distances ----
            # Renamed from 'curvature' in earlier pipeline versions.
            # Low STDDEV = all kNN neighbours equidistant = chunk in smooth cluster interior
            # High STDDEV = neighbours at varied distances = chunk near a semantic boundary
            #   where embedding space curves between two research directions.
            # Chunk-level instability signal.  When many chunks in a cluster have high
            # boundary_proximity, the cluster is under boundary pressure
            # (cf. mean_boundary_score at cluster level).
            # Uses the same dists_for_unc array as uncertainty — no extra computation.
            boundary_proximity = float(np.std(dists_for_unc))
        else:
            uncertainty = 0.0
            boundary_proximity = 0.0

        measures[cid] = {
            "point_density":        point_density,
            "distance_to_centroid": dtc,
            "energy":               energy,
            "boundary_score":       boundary_score,
            "intrinsic_dim":        intrinsic_dim,
            "uncertainty":          uncertainty,
            "boundary_proximity":   boundary_proximity,
        }

    # ── Cluster-level aggregates ───────────────────────────────────────────────
    # Group chunks by cluster (only chunks in this period), then compute
    # four aggregate shape statistics that characterise the cluster as a whole.
    cluster_to_chunks: Dict[int, List[str]] = {}
    for cid, m in measures.items():
        clu = cluster_map.get(cid)
        if clu is not None:
            cluster_to_chunks.setdefault(clu, []).append(cid)

    cluster_stats: Dict[int, dict] = {}
    for clu, chunk_list in cluster_to_chunks.items():
        # mean_density: cluster-level mean of per-chunk point_density.
        # High = tightly packed in embedding space; low = diffuse.
        densities = [measures[c]["point_density"] for c in chunk_list]
        mean_density = float(np.mean(densities))

        # outlier_fraction: fraction of chunks whose distance_to_centroid exceeds
        # mean + 2*stddev.  High = cluster has peripheral chunks being pulled away;
        # possible precursor to split or dissolution.
        dtcs = [
            measures[c]["distance_to_centroid"] for c in chunk_list
            if measures[c]["distance_to_centroid"] is not None
        ]
        if len(dtcs) >= 2:
            mean_d = float(np.mean(dtcs))
            std_d  = float(np.std(dtcs))
            threshold = mean_d + 2.0 * std_d
            outlier_fraction: Optional[float] = float(
                sum(1 for d in dtcs if d > threshold) / len(dtcs)
            )
        else:
            outlier_fraction = None

        # mean_uncertainty: cluster-level mean of per-chunk uncertainty.
        # Measures how ambiguous cluster membership is across all cluster chunks.
        uncertainties = [measures[c]["uncertainty"] for c in chunk_list]
        mean_uncertainty = float(np.mean(uncertainties))

        # boundary_sharpness: P90 - P10 of boundary_score within the cluster.
        # High = clear edge (chunks clearly inside or outside);
        # low = fuzzy edge (chunks continuously grade from core to boundary).
        bscores = [measures[c]["boundary_score"] for c in chunk_list]
        if len(bscores) >= 5:
            boundary_sharpness: Optional[float] = float(
                np.percentile(bscores, 90) - np.percentile(bscores, 10)
            )
        else:
            boundary_sharpness = None

        cluster_stats[clu] = {
            "mean_density":       mean_density,
            "outlier_fraction":   outlier_fraction,
            "mean_uncertainty":   mean_uncertainty,
            "boundary_sharpness": boundary_sharpness,
        }

    log.info(
        "chunk_measures: %d chunks — boundary>0.5: %d, intrinsic_dim nulls: %d, "
        "uncertainty>0.5: %d  |  cluster_stats: %d clusters",
        len(measures),
        sum(1 for m in measures.values() if m["boundary_score"] > 0.5),
        sum(1 for m in measures.values() if m["intrinsic_dim"] is None),
        sum(1 for m in measures.values() if m["uncertainty"] > 0.5),
        len(cluster_stats),
    )
    return measures, cluster_stats


# ---------------------------------------------------------------------------
# Phase 3t: Topology measures (CPU, post-parallel block)
# ---------------------------------------------------------------------------

def run_topology_measures(
    cluster_map: Dict[str, int],
    knn_edges: List[Tuple[str, str, float]],
    period_indices: List[int],
    chunk_ids: List[str],
    period_emb: np.ndarray,             # L2-normalised, shape (n_period, dim)
    measures: Dict[str, dict],          # from run_chunk_measures; needs point_density
) -> Dict[int, dict]:
    """
    Compute 4 topology measures per cluster using the kNN subgraph.

    Returns {cluster_id: {n_attractors, n_saddle_points, avg_path_length,
                           propagation_speed}} for every cluster in cluster_map.
    Values are None when size guards prevent computation.
    """
    # Build chunk_id → local row index in period_emb
    chunk_id_to_local: Dict[str, int] = {
        chunk_ids[period_indices[li]]: li for li in range(len(period_indices))
    }

    # Build cluster → list of chunk_ids
    cluster_to_chunks: Dict[int, List[str]] = {}
    for cid, clu in cluster_map.items():
        cluster_to_chunks.setdefault(clu, []).append(cid)

    # Build full neighbor list (all knn_edges, both directions) — used for
    # attractor/saddle classification which considers the full local topology,
    # not just within-cluster neighbors.
    all_nbr_ids: Dict[str, List[str]] = {}
    for src, dst, _sim in knn_edges:
        all_nbr_ids.setdefault(src, []).append(dst)
        all_nbr_ids.setdefault(dst, []).append(src)

    # Build in-cluster adjacency per cluster (for avg_path_length + propagation_speed).
    # Only retain edges where BOTH endpoints belong to the same cluster.
    adj: Dict[str, Set[str]] = {}
    for src, dst, _sim in knn_edges:
        if cluster_map.get(src) == cluster_map.get(dst) and cluster_map.get(src) is not None:
            adj.setdefault(src, set()).add(dst)
            adj.setdefault(dst, set()).add(src)

    # Build per-chunk point_density map from run_chunk_measures output
    point_density_map: Dict[str, float] = {
        cid: m["point_density"] for cid, m in measures.items()
    }

    results: Dict[int, dict] = {}

    for clu, chunk_list in cluster_to_chunks.items():
        n = len(chunk_list)
        entry: dict = {
            "n_attractors":     None,
            "n_saddle_points":  None,
            "avg_path_length":  None,
            "propagation_speed": None,
        }

        # ── Attractors and saddle points ────────────────────────────────────
        # Requires >= 5 chunks; smaller clusters lack meaningful topology.
        if n >= 5:
            n_attr, n_sad = _run_attractors_saddle(
                chunk_list, all_nbr_ids, point_density_map
            )
            entry["n_attractors"]    = n_attr
            entry["n_saddle_points"] = n_sad

        # ── Average path length ──────────────────────────────────────────────
        # Requires >= 3 chunks.
        if n >= 3:
            entry["avg_path_length"] = _run_avg_path_length(chunk_list, adj)

        # ── Propagation speed ────────────────────────────────────────────────
        # Requires >= 5 chunks.
        if n >= 5:
            entry["propagation_speed"] = _run_propagation_speed(
                chunk_list, adj, chunk_id_to_local, period_emb
            )

        results[clu] = entry

    n_attr_clusters = sum(1 for v in results.values() if v["n_attractors"] is not None)
    n_apl_clusters  = sum(1 for v in results.values() if v["avg_path_length"] is not None)
    log.info(
        "topology_measures: %d clusters — attractors computed: %d, avg_path_length: %d",
        len(results), n_attr_clusters, n_apl_clusters,
    )
    return results


def _run_attractors_saddle(
    cluster_chunk_ids: List[str],
    all_nbr_ids: Dict[str, List[str]],
    point_density_map: Dict[str, float],
) -> Tuple[int, int]:
    """
    Classify each chunk as attractor, saddle point, or valley using full kNN
    neighborhood (not restricted to within-cluster neighbors).

    Attractors: chunks that are local density maxima — every kNN neighbor has
    strictly lower point_density.  These are the semantic 'cores' of sub-topics
    within the cluster.  A cluster with 3 attractors likely contains 3 distinct
    research threads; high n_attractors relative to cluster size suggests the
    cluster should split.

    Saddle points: chunks whose neighborhood is mixed — some neighbors have
    higher density, some lower.  Saddle points sit between two density peaks
    and serve as semantic bridges.  Removing them would disconnect the cluster's
    internal topology.

    Valley points (all neighbors denser) are ignored — they are peripheral chunks.
    """
    n_attractors   = 0
    n_saddle_points = 0

    for chunk_id in cluster_chunk_ids:
        neighbors = all_nbr_ids.get(chunk_id, [])
        if len(neighbors) < 2:
            # Isolated or degree-1 nodes cannot be reliably classified
            continue

        my_density = point_density_map.get(chunk_id, 0.0)
        neighbor_densities = [
            point_density_map[n] for n in neighbors if n in point_density_map
        ]
        if not neighbor_densities:
            continue

        n_higher = sum(1 for d in neighbor_densities if d > my_density)
        n_lower  = sum(1 for d in neighbor_densities if d < my_density)

        if n_higher == 0 and n_lower > 0:
            n_attractors += 1     # local density maximum
        elif n_higher > 0 and n_lower > 0:
            n_saddle_points += 1  # mixed neighborhood — topological bridge

    return n_attractors, n_saddle_points


def _run_avg_path_length(
    cluster_chunk_ids: List[str],
    adj: Dict[str, Set[str]],
) -> Optional[float]:
    """
    Mean shortest path length within the cluster's kNN subgraph (in-cluster
    edges only).

    Low path length = highly connected, any two ideas reachable quickly (dense
    topic).  High path length = chain-like structure (niche subtopic).

    Disconnected clusters (path length = infinity for some pairs) use only
    reachable pairs in the mean.  Logs a warning if >10% of pairs are
    unreachable.

    Computational note: O(V*(V+E)) BFS.  For clusters > 200 chunks we subsample
    50 source nodes and estimate; a log.info line records when estimation fires.
    """
    n = len(cluster_chunk_ids)
    if n < 3:
        return None

    # Subsample sources for large clusters to bound runtime
    if n > 200:
        sources = random.sample(cluster_chunk_ids, 50)
        log.info(
            "avg_path_length: cluster size %d > 200 — subsampling 50 sources", n
        )
    else:
        sources = cluster_chunk_ids

    total_path  = 0
    total_pairs = 0
    unreachable = 0

    for source in sources:
        # BFS to find shortest distances from this source to all reachable nodes
        distances: Dict[str, int] = {source: 0}
        queue: deque = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in adj.get(node, set()):
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)

        for node in cluster_chunk_ids:
            if node == source:
                continue
            if node in distances:
                total_path  += distances[node]
                total_pairs += 1
            else:
                unreachable += 1

    # Warn if more than 10% of pairs are unreachable (disconnected subgraph)
    total_possible = len(sources) * (n - 1)
    if total_possible > 0 and unreachable / total_possible > 0.10:
        log.warning(
            "avg_path_length: %.1f%% of pairs unreachable (disconnected cluster, n=%d)",
            100.0 * unreachable / total_possible, n,
        )

    return total_path / total_pairs if total_pairs > 0 else None


def _run_propagation_speed(
    cluster_chunk_ids: List[str],
    adj: Dict[str, Set[str]],
    chunk_id_to_local: Dict[str, int],
    period_emb: np.ndarray,
) -> Optional[float]:
    """
    Rate of cosine-similarity decay as graph distance increases (hops 1–3).

    Measures how quickly high semantic similarity propagates through the cluster's
    kNN graph.  Fast propagation (similarity stays high at hops 2–3) = coherent
    cluster where ideas naturally chain from one to the next.  Slow propagation
    (sharp drop at hop 1) = loosely connected cluster, likely a catch-all
    category rather than a coherent research topic.

    Returns the linear slope of mean cosine-similarity vs hop distance (fitted
    over hops 1, 2, 3).  Negative slope = similarity decays with distance
    (expected).  Near-zero slope = uniform similarity regardless of graph
    distance.

    Samples up to 20 source chunks for efficiency.
    """
    if len(cluster_chunk_ids) < 5:
        return None

    sources = random.sample(cluster_chunk_ids, min(20, len(cluster_chunk_ids)))
    hop_sims: Dict[int, List[float]] = {1: [], 2: [], 3: []}

    for source in sources:
        src_local = chunk_id_to_local.get(source)
        if src_local is None:
            continue
        src_emb = period_emb[src_local]

        # BFS out to 3 hops
        visited: Dict[str, int] = {source: 0}
        queue: deque = deque([source])
        while queue:
            node = queue.popleft()
            if visited[node] >= 3:
                continue
            for neighbor in adj.get(node, set()):
                if neighbor not in visited:
                    visited[neighbor] = visited[node] + 1
                    queue.append(neighbor)

        for node, hop in visited.items():
            if hop == 0:
                continue
            node_local = chunk_id_to_local.get(node)
            if node_local is None:
                continue
            # period_emb is already L2-normalised → dot product = cosine similarity
            sim = float(np.dot(src_emb, period_emb[node_local]))
            if hop in hop_sims:
                hop_sims[hop].append(sim)

    means = [
        float(np.mean(hop_sims[h])) if hop_sims[h] else None
        for h in [1, 2, 3]
    ]
    if any(m is None for m in means):
        return None

    # Fit linear slope of similarity vs hop distance.
    # propagation_speed is the linear slope of mean cosine similarity vs hop
    # distance (hops 1, 2, 3).  Expected to be NEGATIVE for normal clusters —
    # similarity decays as you move further from any starting point.
    # Near-zero = very coherent cluster (similarity uniform across hops).
    # Positive = hub-and-spoke topology (unusual, warrants investigation).
    # Typical range for patent corpora: -0.3 to -0.01.
    # All-negative results are CORRECT — do not treat them as a sign bug.
    slope = float(np.polyfit([1.0, 2.0, 3.0], means, 1)[0])
    return slope


# ---------------------------------------------------------------------------
# Helper: field_surprise_index per cluster
# ---------------------------------------------------------------------------

def _compute_surprise_for_cluster(
    current_centroid: np.ndarray,
    prev_cents_t1: Dict[int, np.ndarray],
    prev_cents_t2: Dict[int, np.ndarray],
) -> Optional[float]:
    """
    Predict where this cluster's centroid should be based on its prior
    two-period trajectory, then measure the error (surprise).

    Algorithm:
      1. Find the T-1 cluster most similar to the current centroid (best match).
      2. Find the T-2 cluster most similar to that T-1 cluster (chaining backward).
      3. Linear extrapolation: predicted_T = c_t1 + (c_t1 - c_t2).
      4. surprise = cosine_distance(predicted_T_normalised, current_centroid).

    High surprise = cluster moved in an unexpected direction — possible external
    shock, breakthrough paper, regulatory event, or technology discontinuity.
    Low surprise = cluster is evolving predictably along its established trajectory.

    Returns None if T-1 or T-2 centroid pools are empty (first two periods).
    """
    if not prev_cents_t1 or not prev_cents_t2:
        return None

    t1_vecs  = list(prev_cents_t1.values())
    t1_ids   = list(prev_cents_t1.keys())
    t1_mat   = np.stack(t1_vecs)       # (n_t1, dim)

    # Step 1: best T-1 match for the current centroid (max cosine similarity)
    sims_t1 = t1_mat @ current_centroid  # (n_t1,)
    best_t1_idx = int(np.argmax(sims_t1))
    centroid_t1 = t1_vecs[best_t1_idx]


    t2_vecs = list(prev_cents_t2.values())
    t2_mat  = np.stack(t2_vecs)        # (n_t2, dim)

    # Step 2: best T-2 match for the T-1 cluster
    sims_t2 = t2_mat @ centroid_t1     # (n_t2,)
    best_t2_idx = int(np.argmax(sims_t2))
    centroid_t2 = t2_vecs[best_t2_idx]

    # Step 3: linear extrapolation along the prior trajectory
    predicted = centroid_t1 + (centroid_t1 - centroid_t2)
    pred_norm = np.linalg.norm(predicted)
    if pred_norm > 0:
        predicted = predicted / pred_norm

    # Step 4: surprise = cosine distance between prediction and reality
    surprise = float(1.0 - np.dot(predicted, current_centroid))
    # Clamp to [0, 2] (theoretical range of cosine distance) for safety
    return max(0.0, min(2.0, surprise))


# ---------------------------------------------------------------------------
# Helper: system entropy + birth/death estimation + phase transition score
# ---------------------------------------------------------------------------

def _compute_system_entropy(cluster_map: Dict[str, int]) -> float:
    """
    Shannon entropy of the cluster size distribution for this period.

    H = -sum(p_i * log2(p_i)) where p_i = fraction of chunks in cluster i.

    High entropy = many similarly-sized clusters (diverse, balanced field).
    Low entropy  = one or two dominant clusters (concentrated, mature field).
    Used as a component of phase_transition_score — a sudden entropy jump
    signals rapid structural reorganization.
    """
    if not cluster_map:
        return 0.0
    counts = collections.Counter(cluster_map.values())
    total  = len(cluster_map)
    probs  = np.array(list(counts.values()), dtype=float) / total
    # Add tiny epsilon to avoid log(0); negligible for any realistic distribution
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    return entropy


def _estimate_births_deaths(
    centroids_curr: Dict[int, np.ndarray],
    centroids_prev: Dict[int, np.ndarray],
    threshold: float = 0.3,
) -> Tuple[int, int]:
    """
    Estimate n_births and n_deaths for the phase_transition_score birth/death
    component without running the full match_clusters algorithm.

    A cluster in T is a 'birth'  if its best cosine distance to any T-1 cluster > threshold.
    A cluster in T-1 is a 'death' if its best cosine distance to any T cluster > threshold.

    This is an approximation; the definitive birth/death counts come from
    match_clusters() in Phase 3j.  The approximation is sufficient for the
    phase_transition_score composite signal.

    Runs in O(|T| * |T-1|) — fast for typical cluster counts (< 500 clusters).
    Returns (0, 0) when T-1 data is unavailable (first period).
    """
    if not centroids_prev or not centroids_curr:
        return 0, 0

    curr_vecs = list(centroids_curr.values())
    prev_vecs = list(centroids_prev.values())

    curr_mat = np.stack(curr_vecs)   # (n_curr, dim)
    prev_mat = np.stack(prev_vecs)   # (n_prev, dim)

    # Cosine similarity matrix (both sets already L2-normalised as centroids
    # are computed as mean of L2-normalised chunk embeddings then re-normalised
    # in cent_norm above; but compute defensively)
    sim_matrix = curr_mat @ prev_mat.T  # (n_curr, n_prev)

    # Birth: current cluster with no sufficiently close T-1 cluster
    best_sim_to_prev = sim_matrix.max(axis=1)   # (n_curr,)
    n_births = int(np.sum(best_sim_to_prev < (1.0 - threshold)))

    # Death: T-1 cluster with no sufficiently close current cluster
    best_sim_to_curr = sim_matrix.max(axis=0)   # (n_prev,)
    n_deaths = int(np.sum(best_sim_to_curr < (1.0 - threshold)))

    return n_births, n_deaths


def _compute_phase_transition_score(
    n_births: int,
    n_deaths: int,
    n_clusters: int,
    alg_conn: Optional[float],
    prev_alg_conn: Optional[float],
    system_entropy: float,
    prev_system_entropy: Optional[float],
    prev_n_clusters: Optional[int],
) -> Optional[float]:
    """
    Composite signal [0, 1] indicating whether the field is undergoing a rapid
    structural phase transition this period.

    Components and weights:
      1. Turnover rate (w=0.35): (n_births + n_deaths) / n_clusters
         High cluster turnover = structural instability.
      2. Connectivity change (w=0.30): |Δ algebraic_connectivity| / prev
         Large shift in Fiedler value = the kNN graph topology reorganised.
      3. Entropy change (w=0.20): |Δ system_entropy| / prev_system_entropy
         Sudden entropy jump = cluster size distribution shifted dramatically.
      4. Cluster count change (w=0.15): |Δ n_clusters| / prev_n_clusters
         Sudden count change = fragmentation or consolidation event.

    All components are individually capped at 1.0 before weighting so that
    extreme events in one component don't dominate the composite.

    Scores > 0.7 correspond to paradigm-shift-level events (e.g. transformer
    breakthrough 2017, AlphaFold 2021) based on prior ARC analysis (Law #617,
    crystallization sequence).

    Returns None for the first period (no T-1 data available).
    """
    if prev_alg_conn is None or prev_system_entropy is None or prev_n_clusters is None:
        return None

    # Component 1: cluster turnover rate
    turnover     = (n_births + n_deaths) / max(n_clusters, 1)
    turnover_norm = min(turnover, 1.0)

    # Component 2: algebraic connectivity change (Fiedler value stability)
    conn_change  = abs((alg_conn or 0.0) - prev_alg_conn) / max(prev_alg_conn, 1e-6)
    conn_norm    = min(conn_change, 1.0)

    # Component 3: entropy change (cluster-size distribution shift)
    entropy_change = abs(system_entropy - prev_system_entropy)
    entropy_norm   = min(entropy_change / max(prev_system_entropy, 1e-6), 1.0)

    # Component 4: cluster count change (fragmentation / consolidation)
    count_change = abs(n_clusters - prev_n_clusters) / max(prev_n_clusters, 1)
    count_norm   = min(count_change, 1.0)

    score = (
        0.35 * turnover_norm +
        0.30 * conn_norm     +
        0.20 * entropy_norm  +
        0.15 * count_norm
    )
    return float(score)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Phase 3f-b: Bridge chunks (CPU, post-f_edge)
# ---------------------------------------------------------------------------

def run_bridge_chunks(
    knn_edges: List[Tuple[str, str, float]],
    cluster_map: Dict[str, int],
    f_edge_df,     # cuDF DataFrame (always present, may be empty)
) -> object:       # returns the same cuDF DataFrame with n_bridge_chunks added
    """
    Add n_bridge_chunks column to f_edge_df.

    n_bridge_chunks = (chunks in cluster_a with ≥1 neighbour in cluster_b)
                    + (chunks in cluster_b with ≥1 neighbour in cluster_a)

    Bridge chunks are the semantic connectors between two clusters — they have
    one foot in each topic.  High n_bridge_chunks = porous boundary;
    low = hard semantic separation.  Critical for void analysis and split detection.

    Returns f_edge_df with the new column appended.
    If cuDF is unavailable, returns f_edge_df unchanged (n_bridge_chunks absent).
    """
    if f_edge_df is None or len(f_edge_df) == 0:
        return f_edge_df

    try:
        import cudf
    except Exception as exc:
        log.warning("cuDF import failed in run_bridge_chunks (%s) — skipping n_bridge_chunks", exc)
        return f_edge_df

    # Build chunk → set(neighbour chunk_ids) lookup on CPU
    knn_neighbors: Dict[str, set] = {}
    for src, dst, _ in knn_edges:
        knn_neighbors.setdefault(src, set()).add(dst)

    # Build cluster → set(chunk_ids) lookup on CPU
    cluster_chunks_map: Dict[int, set] = {}
    for cid, cl in cluster_map.items():
        cluster_chunks_map.setdefault(cl, set()).add(cid)

    # Iterate over cluster pairs and compute bridge counts
    pairs = f_edge_df[["cluster_a", "cluster_b"]].to_pandas()
    bridge_counts: List[int] = []
    for _, row in pairs.iterrows():
        ca, cb = int(row["cluster_a"]), int(row["cluster_b"])
        chunks_a = cluster_chunks_map.get(ca, set())
        chunks_b = cluster_chunks_map.get(cb, set())
        bridge_a = sum(
            1 for cid in chunks_a if knn_neighbors.get(cid, set()) & chunks_b
        )
        bridge_b = sum(
            1 for cid in chunks_b if knn_neighbors.get(cid, set()) & chunks_a
        )
        bridge_counts.append(bridge_a + bridge_b)

    f_edge_df["n_bridge_chunks"] = cudf.Series(bridge_counts, dtype="int64")
    log.debug(
        "bridge_chunks: %d cluster pairs, mean=%.1f max=%d",
        len(bridge_counts),
        float(np.mean(bridge_counts)) if bridge_counts else 0.0,
        max(bridge_counts) if bridge_counts else 0,
    )
    return f_edge_df


# Phase 3g: f_void — close cluster pairs with no direct kNN edge
# ---------------------------------------------------------------------------

def run_fvoid(
    centroids: Dict[int, np.ndarray],
    f_edge_df,            # cuDF DataFrame (may be None or empty)
    void_threshold: float,
) -> List[Tuple[int, int, float]]:
    """
    Identify cluster pairs that are semantically close (centroid cosine distance
    <= void_threshold) but share no cross-cluster kNN edges.

    Returns list of (cluster_a, cluster_b, centroid_distance) with cluster_a < cluster_b.
    """
    cluster_ids = sorted(centroids.keys())
    n_clusters  = len(cluster_ids)
    if n_clusters < 2:
        return []

    # Normalised centroid matrix for batch cosine computation
    cent_matrix = np.stack([centroids[c] for c in cluster_ids]).astype(np.float32)
    norms = np.linalg.norm(cent_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    cent_matrix /= norms

    # Pairwise cosine distance: (n, n)
    sim_matrix  = cent_matrix @ cent_matrix.T
    dist_matrix = 1.0 - sim_matrix

    # Build set of cluster pairs that already have a kNN edge (either direction)
    existing: set = set()
    if f_edge_df is not None and len(f_edge_df) > 0:
        pairs = f_edge_df[["cluster_a", "cluster_b"]].to_pandas()
        for row in pairs.itertuples(index=False):
            ca, cb = int(row.cluster_a), int(row.cluster_b)
            existing.add((ca, cb))
            existing.add((cb, ca))

    voids: List[Tuple[int, int, float]] = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            ca, cb = cluster_ids[i], cluster_ids[j]
            if (ca, cb) in existing or (cb, ca) in existing:
                continue
            d = float(dist_matrix[i, j])
            if d <= void_threshold:
                voids.append((ca, cb, d))

    log.debug("f_void: %d void pairs (threshold=%.3f)", len(voids), void_threshold)
    return voids


# ---------------------------------------------------------------------------
# Phase 3g2: f_gap — geometric description of inter-cluster gap space
# ---------------------------------------------------------------------------

def run_fgap(
    centroids: Dict[int, np.ndarray],
    cluster_map: Dict[str, int],
    period_embeddings: np.ndarray,
    period_chunk_ids: List[str],
    knn_edges: List[Tuple[str, str, float]],
    measures: Dict[str, dict],
    corpus_id: str,
    period_start: str,
    config: dict,
    void_radius: float = 0.15,
    fringe_radius: float = 0.20,
    f_edge_df=None,
) -> List[dict]:
    """
    Compute continuous geometric properties of the space between every cluster pair.

    Covers all pairs (a < b) regardless of kNN connectivity — bridges, fringes,
    and genuine voids alike.  Intended for exploratory analysis; gap_type is a
    soft classification only.
    """
    junk_threshold = int(config.get("junk_threshold", 2))

    # Build chunk_id → index in period_embeddings
    chunk_idx: Dict[str, int] = {cid: i for i, cid in enumerate(period_chunk_ids)}

    # Per-cluster chunk lists
    cluster_chunks: Dict[int, List[str]] = defaultdict(list)
    for chunk_id, cluster_id in cluster_map.items():
        if chunk_id in chunk_idx:   # only chunks in this period
            cluster_chunks[cluster_id].append(chunk_id)

    # kNN neighbour lookup (undirected)
    knn_neighbors: Dict[str, Set[str]] = defaultdict(set)
    for src, dst, _dist in knn_edges:
        knn_neighbors[src].add(dst)
        knn_neighbors[dst].add(src)

    # Dark matter: chunks not in any real cluster (None, -1, or junk-sized)
    cluster_sizes: Dict[int, int] = {cid: len(chunks) for cid, chunks in cluster_chunks.items()}

    dark_matter_ids: List[str] = []
    for chunk_id in period_chunk_ids:
        cl = cluster_map.get(chunk_id)
        if cl is None or cl == -1 or cluster_sizes.get(cl, 0) <= junk_threshold:
            dark_matter_ids.append(chunk_id)

    if dark_matter_ids:
        dm_embs = period_embeddings[[chunk_idx[c] for c in dark_matter_ids]]  # (n_dm, dim)
    else:
        dm_embs = None

    cluster_ids = sorted(centroids.keys())
    n_pairs = len(cluster_ids) * (len(cluster_ids) - 1) // 2

    MAX_FGAP_PAIRS = 10_000  # above this, skip fgap for this period
    if n_pairs > MAX_FGAP_PAIRS:
        log.warning(
            "[fgap] Period %s: %d cluster pairs exceeds cap %d — skipping fgap. "
            "Consider increasing MAX_FGAP_PAIRS or filtering corpora.",
            period_start, n_pairs, MAX_FGAP_PAIRS
        )
        return []
    if n_pairs > 5000:
        log.warning(
            "f_gap: %d cluster pairs — computation may be slow (n_clusters=%d)",
            n_pairs, len(cluster_ids),
        )

    # Build connection_weight lookup from already-computed f_edge.
    # Avoids O(n_clusters² × cluster_size) recomputation of connection measures.
    # Comment: f_edge_df already has connection_weight, semantic_overlap_a_to_b,
    # semantic_overlap_b_to_a for all connected cluster pairs.
    edge_lookup: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    if f_edge_df is not None and len(f_edge_df) > 0:
        _fe = f_edge_df[["cluster_a", "cluster_b", "connection_weight",
                          "semantic_overlap_a_to_b", "semantic_overlap_b_to_a"]].to_pandas()
        for _, _r in _fe.iterrows():
            _ca, _cb = int(_r["cluster_a"]), int(_r["cluster_b"])
            edge_lookup[(_ca, _cb)] = (
                float(_r["connection_weight"]),
                float(_r["semantic_overlap_a_to_b"]),
                float(_r["semantic_overlap_b_to_a"]),
            )
            edge_lookup[(_cb, _ca)] = (
                float(_r["connection_weight"]),
                float(_r["semantic_overlap_b_to_a"]),   # reversed
                float(_r["semantic_overlap_a_to_b"]),
            )

    rows: List[dict] = []

    for i, ca in enumerate(cluster_ids):
        for cb in cluster_ids[i + 1:]:

            centroid_a = centroids[ca]
            centroid_b = centroids[cb]
            chunks_a   = cluster_chunks[ca]
            chunks_b   = cluster_chunks[cb]

            # ── Distance measures ────────────────────────────────────────

            # centroid_distance: cosine distance between cluster centers.
            # The most basic gap measure — does not account for cluster shape
            # or what exists in between.
            centroid_distance = float(1.0 - np.dot(centroid_a, centroid_b))

            # midpoint: normalized mean of two centroids.
            # Represents the hypothetical semantic midpoint of the gap.
            midpoint = centroid_a + centroid_b
            mid_norm = np.linalg.norm(midpoint)
            if mid_norm > 1e-9:
                midpoint = midpoint / mid_norm

            # boundary_distance: distance between nearest boundary chunks of A and B.
            # More accurate than centroid_distance — measures from where clusters
            # actually end, not from their centers.
            boundary_chunks_a = [
                c for c in chunks_a
                if measures.get(c, {}).get("boundary_score", 0) > 0.3
            ] or chunks_a

            boundary_chunks_b = [
                c for c in chunks_b
                if measures.get(c, {}).get("boundary_score", 0) > 0.3
            ] or chunks_b

            if boundary_chunks_a and boundary_chunks_b:
                ba_idx = [chunk_idx[c] for c in boundary_chunks_a if c in chunk_idx]
                bb_idx = [chunk_idx[c] for c in boundary_chunks_b if c in chunk_idx]
                if ba_idx and bb_idx:
                    embs_ba = period_embeddings[ba_idx]
                    embs_bb = period_embeddings[bb_idx]
                    sims = embs_ba @ embs_bb.T
                    boundary_distance = float(1.0 - sims.max())
                else:
                    boundary_distance = centroid_distance
            else:
                boundary_distance = centroid_distance

            # void_depth: cosine distance from midpoint to nearest chunk of any kind.
            # High = gap is genuinely empty (deep void).
            # Low = something exists near midpoint (shallow or occupied gap).
            sims_to_mid = period_embeddings @ midpoint   # (n_period, ) cosine sims
            void_depth  = float(1.0 - sims_to_mid.max())
            nearest_mid_idx     = int(sims_to_mid.argmax())
            nearest_mid_chunk   = period_chunk_ids[nearest_mid_idx]
            nearest_mid_cluster = cluster_map.get(nearest_mid_chunk)

            # ── Content measures ─────────────────────────────────────────

            # n_dark_matter_near_midpoint: dark matter chunks within void_radius
            # of gap midpoint.  > 0 = proto-cluster signal.
            if dm_embs is not None and len(dm_embs) > 0:
                dm_sims_mid  = dm_embs @ midpoint
                dm_dists_mid = 1.0 - dm_sims_mid
                n_dm_near    = int((dm_dists_mid < void_radius).sum())
                if n_dm_near > 0:
                    dm_density_near = float(
                        n_dm_near / (void_radius ** 2 * len(dark_matter_ids) + 1e-9)
                    )
                else:
                    dm_density_near = 0.0
            else:
                n_dm_near       = 0
                dm_density_near = 0.0

            # fringe_density_a/b: dark matter near each cluster's centroid,
            # normalized by cluster size.  High = fuzzy cluster boundary.
            if dm_embs is not None and len(dm_embs) > 0:
                dm_sims_a    = dm_embs @ centroid_a
                fringe_density_a = float(
                    ((1.0 - dm_sims_a) < fringe_radius).sum() / max(len(chunks_a), 1)
                )
                dm_sims_b    = dm_embs @ centroid_b
                fringe_density_b = float(
                    ((1.0 - dm_sims_b) < fringe_radius).sum() / max(len(chunks_b), 1)
                )
            else:
                fringe_density_a = 0.0
                fringe_density_b = 0.0

            # ── Connection measures ──────────────────────────────────────
            # Look up from f_edge_df if available; else compute from knn_neighbors.
            # Comment: avoids O(n_clusters² × cluster_size) recomputation.
            if (ca, cb) in edge_lookup:
                connection_weight, connection_weight_ab, connection_weight_ba = edge_lookup[(ca, cb)]
            else:
                chunks_a_set = set(chunks_a)
                chunks_b_set = set(chunks_b)
                chunks_a_touching_b = sum(
                    1 for c in chunks_a
                    if any(n in chunks_b_set for n in knn_neighbors.get(c, set()))
                )
                chunks_b_touching_a = sum(
                    1 for c in chunks_b
                    if any(n in chunks_a_set for n in knn_neighbors.get(c, set()))
                )
                connection_weight_ab = chunks_a_touching_b / max(len(chunks_a), 1)
                connection_weight_ba = chunks_b_touching_a / max(len(chunks_b), 1)
                connection_weight    = max(connection_weight_ab, connection_weight_ba)

            # ── Soft classification ──────────────────────────────────────
            # Exploratory only — thresholds should be calibrated from data.
            if connection_weight > 0.15:
                gap_type = "bridge"
            elif n_dm_near >= 3:
                gap_type = "proto_cluster"
            elif n_dm_near > 0 or fringe_density_a > 0.1 or fringe_density_b > 0.1:
                gap_type = "fringe"
            else:
                gap_type = "void"

            # nearest_chunk_is_dark_matter: is the chunk closest to the midpoint
            # unclassified or in a junk cluster?
            nearest_is_dm = (
                nearest_mid_cluster is None
                or nearest_mid_cluster == -1
                or cluster_sizes.get(nearest_mid_cluster, 0) <= junk_threshold
            )

            rows.append({
                "cluster_a":                    ca,
                "cluster_b":                    cb,
                "period_start":                 period_start,
                "corpus_id":                    corpus_id,
                "centroid_distance":            centroid_distance,
                "boundary_distance":            boundary_distance,
                "void_depth":                   void_depth,
                "n_dark_matter_near_midpoint":  n_dm_near,
                "dark_matter_density_near_void": dm_density_near,
                "fringe_density_a":             fringe_density_a,
                "fringe_density_b":             fringe_density_b,
                "connection_weight":            connection_weight,
                "connection_weight_ab":         connection_weight_ab,
                "connection_weight_ba":         connection_weight_ba,
                "size_a":                       len(chunks_a),
                "size_b":                       len(chunks_b),
                "nearest_chunk_is_dark_matter": nearest_is_dm,
                "gap_type":                     gap_type,
            })

    log.debug(
        "f_gap: %d pairs (%d clusters)  types=%s",
        len(rows), len(cluster_ids),
        {t: sum(1 for r in rows if r["gap_type"] == t) for t in ("void","fringe","bridge","proto_cluster")},
    )
    return rows


# ---------------------------------------------------------------------------
# Phase 3h: Spectral measures — algebraic connectivity and spectral gap
# ---------------------------------------------------------------------------

def run_spectral(
    period_indices: List[int],
    chunk_ids: List[str],
    knn_edges: List[Tuple[str, str, float]],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Build the graph Laplacian from kNN edges and compute:
        algebraic_connectivity — second smallest eigenvalue (λ₂)
        spectral_gap           — λ₃ - λ₂ (gap between 2nd and 3rd eigenvalues)

    Uses cupy.linalg.eigh for n_chunks > 100, numpy otherwise.
    Returns (None, None) for periods with fewer than 3 chunks.
    """
    n = len(period_indices)
    if n < 3:
        return None, None

    local_of: Dict[str, int] = {
        chunk_ids[period_indices[li]]: li for li in range(n)
    }

    # Symmetric adjacency: take max weight per undirected pair
    best: Dict[Tuple[int, int], float] = {}
    for src, dst, sim in knn_edges:
        si = local_of.get(src)
        di = local_of.get(dst)
        if si is None or di is None or si == di:
            continue
        key = (min(si, di), max(si, di))
        if sim > best.get(key, -1.0):
            best[key] = sim

    # Build dense adjacency in numpy (single bulk GPU transfer if needed)
    A_np = np.zeros((n, n), dtype=np.float32)
    for (si, di), w in best.items():
        w_clip = max(0.0, w)
        A_np[si, di] = w_clip
        A_np[di, si] = w_clip

    try:
        if n > 100:
            import cupy as cp
            A  = cp.asarray(A_np)
            D  = cp.diag(A.sum(axis=1))
            L  = D - A
            lam = cp.asnumpy(cp.linalg.eigh(L)[0])
        else:
            D   = np.diag(A_np.sum(axis=1))
            L   = D - A_np
            lam = np.linalg.eigh(L)[0]
    except Exception as exc:
        # Fall back to numpy for any cupy failure: OOM, CUDA errors, or import
        # failures (e.g. TypeError when CONDA_PREFIX is unset).  Only return
        # None for genuine numpy-level failures that numpy can't handle either.
        log.warning("Spectral cupy failed (n=%d, %s) — falling back to numpy", n, exc)
        try:
            D   = np.diag(A_np.sum(axis=1))
            L   = D - A_np
            lam = np.linalg.eigh(L)[0]
        except Exception as np_exc:
            log.warning("Spectral numpy fallback also failed: %s — returning None", np_exc)
            return None, None

    lam_sorted = np.sort(np.real(lam))

    # algebraic_connectivity = λ₂ (second smallest Laplacian eigenvalue).
    # λ₁ is always 0 for any connected graph.  λ₂ measures overall cohesion —
    # how well the full cluster graph is connected.
    algebraic_connectivity = float(lam_sorted[1]) if len(lam_sorted) > 1 else None

    # spectral_gap = λ₃ - λ₂ (gap between second and third eigenvalues).
    # Measures how clearly two dominant communities are separated within the graph.
    # Large gap = two well-separated communities; small gap = communities blend.
    # NOTE: the previous code computed λ₂ - λ₁ = λ₂ - 0 = λ₂, which is identical
    # to algebraic_connectivity.  That was wrong — both columns carried the same value.
    spectral_gap = float(lam_sorted[2] - lam_sorted[1]) if len(lam_sorted) > 2 else None

    log.debug(
        "Spectral: algebraic_connectivity=%.6f  spectral_gap=%.6f",
        algebraic_connectivity or 0.0, spectral_gap or 0.0,
    )
    return algebraic_connectivity, spectral_gap


# ---------------------------------------------------------------------------
# Phase 3i: Write period outputs
# ---------------------------------------------------------------------------

def write_period_outputs(
    work_dir: str,
    outputs: "PeriodOutputs",
) -> Dict[str, int]:
    """
    Write 1 cold TSV (per-period) + append rows to 7 shared import TSVs.
    Cold storage: cold/{corpus_id}/             — one file per period, flat layout.
    Import files: import/{corpus_id}/*.tsv      — all periods concatenated, header once.
    Returns {relative_s3_key: row_count_this_period} for accumulation in main().
    """
    # Unpack dataclass fields to local variables — preserves all internal logic below.
    corpus_id              = outputs.corpus_id
    period_start           = outputs.period_start
    period_indices         = outputs.period_indices
    chunk_ids              = outputs.chunk_ids
    knn_edges              = outputs.knn_edges
    cluster_map            = outputs.cluster_map
    chunk_graph_df         = outputs.chunk_graph_df
    f_edge_df              = outputs.f_edge_df
    centroids              = outputs.centroids
    chunk_measures         = outputs.chunk_measures
    f_void_rows            = outputs.f_void_rows
    f_gap_rows             = outputs.f_gap_rows
    spectral               = outputs.spectral
    cluster_geometry       = outputs.cluster_geometry
    cluster_stats          = outputs.cluster_stats
    topology_stats         = outputs.topology_stats
    field_surprise         = outputs.field_surprise
    leiden_modularity      = outputs.leiden_modularity
    phase_transition_score = outputs.phase_scores["phase_transition_score"]
    n_dark_matter_chunks   = outputs.phase_scores["n_dark_matter_chunks"]
    is_dark_matter         = outputs.phase_scores["is_dark_matter"]
    membership_volatility  = outputs.phase_scores["membership_volatility"]
    belief_persistence_score = outputs.phase_scores["belief_persistence_score"]
    ps  = period_start
    cid = corpus_id
    cold_dir   = os.path.join(work_dir, "output", "cold",   cid)
    import_dir = os.path.join(work_dir, "output", "import", cid)
    os.makedirs(cold_dir,   exist_ok=True)
    os.makedirs(import_dir, exist_ok=True)

    row_counts: Dict[str, int] = {}

    # ---- Cold: knn_edges (per-period, gzip-compressed) ----
    knn_file = f"knn_edges_{cid}_{ps}.tsv.gz"
    knn_path = os.path.join(cold_dir, knn_file)
    with gzip.open(knn_path, "wt", encoding="utf-8") as fh:
        fh.write("chunk_id\tneighbor_id\tdistance\n")
        for src, dst, dist in knn_edges:
            fh.write(f"{src}\t{dst}\t{dist:.6f}\n")
    row_counts[f"cold/{cid}/{knn_file}"] = len(knn_edges)
    log.info("  knn_edges: %d rows → %s", len(knn_edges), knn_file)

    # Helper: returns ('w', True) for a new file, ('a', False) for an existing one.
    def _open_mode(path: str):
        exists = os.path.exists(path)
        return ("a", False) if exists else ("w", True)

    # ---- Import: cloud_chunk_measures (merged chunk_periods + chunk_graph) ----
    # Write one row per chunk with all embedding-derived and graph-derived measures.
    # Replaces the separate chunk_periods.tsv + chunk_graph.tsv writes.
    ccm_path = os.path.join(import_dir, "cloud_chunk_measures.tsv")
    mode, write_header = _open_mode(ccm_path)

    # Build chunk_id → graph measures lookup from cuDF if available
    cg_lookup: Dict[str, dict] = {}
    if chunk_graph_df is not None:
        _cg_pd = chunk_graph_df[[
            "chunk_id", "betweenness_centrality", "core_number", "clustering_coeff",
            "triangle_count", "degree", "pagerank", "eigenvector_centrality",
            "katz_centrality", "harmonic_centrality", "in_degree_centrality",
        ]].to_pandas()
        for _, _r in _cg_pd.iterrows():
            cg_lookup[str(_r["chunk_id"])] = {
                "betweenness_centrality":  _r.get("betweenness_centrality"),
                "core_number":             _r.get("core_number"),
                "clustering_coeff":        _r.get("clustering_coeff"),
                "triangle_count":          _r.get("triangle_count"),
                "degree":                  _r.get("degree"),
                "pagerank":               _r.get("pagerank"),
                "eigenvector_centrality":  _r.get("eigenvector_centrality"),
                "katz_centrality":         _r.get("katz_centrality"),
                "harmonic_centrality":     _r.get("harmonic_centrality"),
                "in_degree_centrality":    _r.get("in_degree_centrality"),
            }

    with open(ccm_path, mode, encoding="utf-8") as fh:
        if write_header:
            fh.write(
                "chunk_id\tcluster_id\tperiod_start\tcorpus_id\t"
                "point_density\tdistance_to_centroid\tenergy\tboundary_score\tintrinsic_dim\t"
                "uncertainty\tboundary_proximity\tis_dark_matter\tmembership_volatility\t"
                "belief_persistence_score\t"
                "betweenness_centrality\tcore_number\tclustering_coeff\t"
                "triangle_count\tdegree\tpagerank\teigenvector_centrality\t"
                "katz_centrality\tharmonic_centrality\tin_degree_centrality\n"
            )
        for global_i in period_indices:
            chunk   = chunk_ids[global_i]
            cluster = cluster_map.get(chunk, -1)
            m       = chunk_measures.get(chunk, {})
            pd_val  = m.get("point_density")
            dtc_val = m.get("distance_to_centroid")
            en_val  = m.get("energy")
            bs_val  = m.get("boundary_score")
            id_val  = m.get("intrinsic_dim")
            unc_val = m.get("uncertainty")
            bp_val  = m.get("boundary_proximity")
            dm_val  = is_dark_matter.get(chunk, False)
            mv_val  = membership_volatility.get(chunk, 0)
            bps_val = belief_persistence_score.get(chunk)
            cg = cg_lookup.get(chunk, {})
            bc_val  = cg.get("betweenness_centrality")
            cn_val  = cg.get("core_number")
            cc_val  = cg.get("clustering_coeff")
            tc_val  = cg.get("triangle_count")
            dg_val  = cg.get("degree")
            pr_val  = cg.get("pagerank")
            ec_val  = cg.get("eigenvector_centrality")
            kz_val  = cg.get("katz_centrality")
            hc_val  = cg.get("harmonic_centrality")
            ic_val  = cg.get("in_degree_centrality")
            fh.write(
                f"{chunk}\t{cluster}\t{ps}\t{cid}\t"
                f"{'NULL' if pd_val  is None else f'{pd_val:.6f}'}\t"
                f"{'NULL' if dtc_val is None else f'{dtc_val:.6f}'}\t"
                f"{'NULL' if en_val  is None else f'{en_val:.6f}'}\t"
                f"{'NULL' if bs_val  is None else f'{bs_val:.6f}'}\t"
                f"{'NULL' if id_val  is None else str(id_val)}\t"
                f"{'NULL' if unc_val is None else f'{unc_val:.6f}'}\t"
                f"{'NULL' if bp_val  is None else f'{bp_val:.6f}'}\t"
                f"{'true' if dm_val else 'false'}\t"
                f"{mv_val}\t"
                f"{'NULL' if bps_val is None else f'{bps_val:.6f}'}\t"
                f"{'NULL' if bc_val  is None else f'{bc_val:.6f}'}\t"
                f"{'NULL' if cn_val  is None else str(int(cn_val))}\t"
                f"{'NULL' if cc_val  is None else f'{cc_val:.6f}'}\t"
                f"{'NULL' if tc_val  is None else str(int(tc_val))}\t"
                f"{'NULL' if dg_val  is None else str(int(dg_val))}\t"
                f"{'NULL' if pr_val  is None else f'{pr_val:.6f}'}\t"
                f"{'NULL' if ec_val  is None else f'{ec_val:.6f}'}\t"
                f"{'NULL' if kz_val  is None else f'{kz_val:.6f}'}\t"
                f"{'NULL' if hc_val  is None else f'{hc_val:.6f}'}\t"
                f"{'NULL' if ic_val  is None else f'{ic_val:.6f}'}\n"
            )
    n_ccm = len(period_indices)
    row_counts["import/{cid}/cloud_chunk_measures.tsv".format(cid=cid)] = n_ccm
    log.info("  cloud_chunk_measures: %d rows (chunk_periods+graph merged)", n_ccm)

    # ---- Import: f_edge ----
    fe_path  = os.path.join(import_dir, "f_edge.tsv")
    fe_mode, fe_header = _open_mode(fe_path)
    n_fe = len(f_edge_df) if f_edge_df is not None else 0
    if n_fe > 0:
        fe_out = f_edge_df.copy()
        fe_out["period_start"] = ps
        fe_out["corpus_id"]    = cid
        _fe_cols = [
            "cluster_a", "cluster_b", "period_start", "corpus_id",
            "connection_weight", "semantic_overlap_a_to_b",
            "semantic_overlap_b_to_a", "n_shared_edges",
            "semantic_overlap_max", "n_bridge_chunks",
        ]
        fe_out[_fe_cols].to_pandas().to_csv(
            fe_path, sep="\t", index=False, mode=fe_mode, header=fe_header
        )
    else:
        if fe_header:
            with open(fe_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "cluster_a\tcluster_b\tperiod_start\tcorpus_id\t"
                    "connection_weight\tsemantic_overlap_a_to_b\t"
                    "semantic_overlap_b_to_a\tn_shared_edges\t"
                    "semantic_overlap_max\tn_bridge_chunks\n"
                )
    row_counts[f"import/{cid}/f_edge.tsv"] = n_fe
    log.info("  f_edge: %d rows", n_fe)

    # ---- Import: centroids ----
    cen_path = os.path.join(import_dir, "centroids.tsv")
    mode, write_header = _open_mode(cen_path)
    with open(cen_path, mode, encoding="utf-8") as fh:
        if write_header:
            fh.write(
                "cluster_id\tperiod_start\tcorpus_id\tcentroid\t"
                "elongation_ratio\tvolume_estimate\t"
                "skewness_pc1\tkurtosis_pc1\tskewness_pc2\tkurtosis_pc2\t"
                "mean_density\toutlier_fraction\tmean_uncertainty\tboundary_sharpness\t"
                "n_attractors\tn_saddle_points\tavg_path_length\tpropagation_speed\t"
                "field_surprise_index\n"
            )
        for cluster_id, centroid in sorted(centroids.items()):
            vec = " ".join(f"{v:.6f}" for v in centroid)
            g   = cluster_geometry.get(cluster_id, {})
            elo = g.get("elongation_ratio")
            vol = g.get("volume_estimate")
            sk1 = g.get("skewness_pc1")
            ku1 = g.get("kurtosis_pc1")
            sk2 = g.get("skewness_pc2")
            ku2 = g.get("kurtosis_pc2")
            cs  = cluster_stats.get(cluster_id, {})
            mdn = cs.get("mean_density")
            olf = cs.get("outlier_fraction")
            mun = cs.get("mean_uncertainty")
            bsh = cs.get("boundary_sharpness")
            ts_stat = topology_stats.get(cluster_id, {})
            nat = ts_stat.get("n_attractors")
            nsp = ts_stat.get("n_saddle_points")
            apl = ts_stat.get("avg_path_length")
            psp = ts_stat.get("propagation_speed")
            fsi = field_surprise.get(cluster_id)
            fh.write(
                f"{cluster_id}\t{ps}\t{cid}\t{vec}\t"
                f"{'NULL' if elo is None else f'{elo:.6f}'}\t"
                f"{'NULL' if vol is None else f'{vol:.6f}'}\t"
                f"{'NULL' if sk1 is None else f'{sk1:.6f}'}\t"
                f"{'NULL' if ku1 is None else f'{ku1:.6f}'}\t"
                f"{'NULL' if sk2 is None else f'{sk2:.6f}'}\t"
                f"{'NULL' if ku2 is None else f'{ku2:.6f}'}\t"
                f"{'NULL' if mdn is None else f'{mdn:.6f}'}\t"
                f"{'NULL' if olf is None else f'{olf:.6f}'}\t"
                f"{'NULL' if mun is None else f'{mun:.6f}'}\t"
                f"{'NULL' if bsh is None else f'{bsh:.6f}'}\t"
                f"{'NULL' if nat is None else str(nat)}\t"
                f"{'NULL' if nsp is None else str(nsp)}\t"
                f"{'NULL' if apl is None else f'{apl:.6f}'}\t"
                f"{'NULL' if psp is None else f'{psp:.6f}'}\t"
                f"{'NULL' if fsi is None else f'{fsi:.6f}'}\n"
            )
    n_cen = len(centroids)
    row_counts[f"import/{cid}/centroids.tsv"] = n_cen
    log.info("  centroids: %d rows", n_cen)

    # ---- Import: f_void ----
    fv_path = os.path.join(import_dir, "f_void.tsv")
    mode, write_header = _open_mode(fv_path)
    with open(fv_path, mode, encoding="utf-8") as fh:
        if write_header:
            fh.write("cluster_a\tcluster_b\tcentroid_distance\tperiod_start\tcorpus_id\n")
        for ca, cb, dist in f_void_rows:
            fh.write(f"{ca}\t{cb}\t{dist:.6f}\t{ps}\t{cid}\n")
    n_fv = len(f_void_rows)
    row_counts[f"import/{cid}/f_void.tsv"] = n_fv
    log.info("  f_void: %d rows", n_fv)

    # ---- Import: f_gap ----
    fg_path = os.path.join(import_dir, "f_gap.tsv")
    mode, write_header = _open_mode(fg_path)
    with open(fg_path, mode, encoding="utf-8") as fh:
        if write_header:
            fh.write(
                "cluster_a\tcluster_b\tperiod_start\tcorpus_id\t"
                "centroid_distance\tboundary_distance\tvoid_depth\t"
                "n_dark_matter_near_midpoint\tdark_matter_density_near_void\t"
                "fringe_density_a\tfringe_density_b\t"
                "connection_weight\tconnection_weight_ab\tconnection_weight_ba\t"
                "size_a\tsize_b\tnearest_chunk_is_dark_matter\tgap_type\n"
            )
        for r in f_gap_rows:
            fh.write(
                f"{r['cluster_a']}\t{r['cluster_b']}\t{r['period_start']}\t{r['corpus_id']}\t"
                f"{r['centroid_distance']:.6f}\t{r['boundary_distance']:.6f}\t{r['void_depth']:.6f}\t"
                f"{r['n_dark_matter_near_midpoint']}\t{r['dark_matter_density_near_void']:.6f}\t"
                f"{r['fringe_density_a']:.6f}\t{r['fringe_density_b']:.6f}\t"
                f"{r['connection_weight']:.6f}\t{r['connection_weight_ab']:.6f}\t{r['connection_weight_ba']:.6f}\t"
                f"{r['size_a']}\t{r['size_b']}\t"
                f"{'true' if r['nearest_chunk_is_dark_matter'] else 'false'}\t"
                f"{r['gap_type']}\n"
            )
    n_fg = len(f_gap_rows)
    row_counts[f"import/{cid}/f_gap.tsv"] = n_fg
    log.info("  f_gap: %d rows", n_fg)

    # ---- Import: f_period (one row per period) ----
    fp_path = os.path.join(import_dir, "f_period.tsv")
    mode, write_header = _open_mode(fp_path)
    alg_conn, spec_gap    = spectral
    n_clusters_period     = len(centroids)
    with open(fp_path, mode, encoding="utf-8") as fh:
        if write_header:
            fh.write(
                "period_start\tcorpus_id\talgebraic_connectivity\t"
                "spectral_gap\tn_clusters\tn_chunks\tphase_transition_score\t"
                "leiden_modularity\tn_dark_matter_chunks\t"
                "n_births\tn_deaths\tn_reborn\tn_split_child\n"
            )
        fh.write(
            f"{ps}\t{cid}\t"
            f"{'NULL' if alg_conn is None else f'{alg_conn:.6f}'}\t"
            f"{'NULL' if spec_gap is None else f'{spec_gap:.6f}'}\t"
            f"{n_clusters_period}\t{len(period_indices)}\t"
            f"{'NULL' if phase_transition_score is None else f'{phase_transition_score:.6f}'}\t"
            f"{leiden_modularity:.6f}\t"
            f"{n_dark_matter_chunks}\t"
            f"NULL\tNULL\tNULL\tNULL\n"
        )
    row_counts[f"import/{cid}/f_period.tsv"] = 1
    log.info(
        "  f_period: algebraic_connectivity=%s  spectral_gap=%s  n_clusters=%d  "
        "phase_transition_score=%s  leiden_modularity=%.4f  n_dark_matter=%d",
        f"{alg_conn:.6f}" if alg_conn is not None else "NULL",
        f"{spec_gap:.6f}" if spec_gap is not None else "NULL",
        n_clusters_period,
        f"{phase_transition_score:.4f}" if phase_transition_score is not None else "NULL",
        leiden_modularity,
        n_dark_matter_chunks,
    )

    return row_counts

# ---------------------------------------------------------------------------
# Phase 3j helpers
# ---------------------------------------------------------------------------

def patch_f_period_lifecycle(
    work_dir: str,
    corpus_id: str,
    lifecycle: Dict[str, Tuple[int, int, int, int]],
) -> None:
    """
    Patch f_period.tsv in-place with lifecycle counts from ClusterMatcher.finalize().

    f_period.tsv is written per-period (with NULL for lifecycle columns) because
    ClusterMatcher.process_period() accumulates counts across all periods and
    the final values are only available after the loop via finalize().
    This function reads the file back and fills in the four NULL columns.

    lifecycle: {period_start_str → (n_births, n_deaths, n_reborn, n_split_child)}
    """
    import_dir = os.path.join(work_dir, "output", "import", corpus_id)
    fp_path    = os.path.join(import_dir, "f_period.tsv")
    if not os.path.exists(fp_path):
        log.warning("patch_f_period_lifecycle: %s not found — skipping", fp_path)
        return

    with open(fp_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    if len(lines) < 2:
        return  # header-only or empty file

    cols = lines[0].rstrip("\n").split("\t")
    try:
        ps_idx     = cols.index("period_start")
        nb_idx     = cols.index("n_births")
        nd_idx     = cols.index("n_deaths")
        nr_idx     = cols.index("n_reborn")
        ns_idx     = cols.index("n_split_child")
    except ValueError as exc:
        log.warning("patch_f_period_lifecycle: column not found (%s) — skipping", exc)
        return

    patched = [lines[0]]
    for line in lines[1:]:
        parts = line.rstrip("\n").split("\t")
        ps    = parts[ps_idx] if ps_idx < len(parts) else None
        if ps and ps in lifecycle:
            nb, nd, nr, ns = lifecycle[ps]
            parts[nb_idx] = str(nb)
            parts[nd_idx] = str(nd)
            parts[nr_idx] = str(nr)
            parts[ns_idx] = str(ns)
        patched.append("\t".join(parts) + "\n")

    with open(fp_path, "w", encoding="utf-8") as fh:
        fh.writelines(patched)

    log.info(
        "patch_f_period_lifecycle: patched %d periods in %s",
        len(lifecycle), fp_path,
    )


def load_context_file(
    context_path: str,
    emb_dict: Dict[str, np.ndarray],
) -> Tuple[str, Dict[str, int], Dict[str, int], Dict[int, np.ndarray]]:
    """
    Load T-1 context file for incremental matching.

    File format (TSV with header): chunk_id, cluster_id, core_number, period_start
    One row per chunk from the most recently completed period.

    Returns (period_start, chunk_cluster, core_numbers, centroids).
    Centroids are computed from emb_dict — chunks not present in emb_dict are skipped
    for centroid computation but still contribute to the chunk→cluster mapping.
    """
    chunk_cluster: Dict[str, int] = {}
    core_numbers: Dict[str, int] = {}
    period_start_found: Optional[str] = None

    with open(context_path, encoding="utf-8") as fh:
        fh.readline()  # skip header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            cid, clu_str, core_str, ps = parts[0], parts[1], parts[2], parts[3]
            try:
                clu = int(clu_str)
            except ValueError:
                continue
            core = int(core_str) if core_str not in ("", "NULL", "None") else 0
            chunk_cluster[cid] = clu
            core_numbers[cid] = core
            if period_start_found is None:
                period_start_found = ps

    if period_start_found is None:
        raise ValueError(f"Context file {context_path} is empty or has no valid rows")

    # Compute centroids from embeddings for chunks that are in emb_dict
    cluster_vecs: Dict[int, List[np.ndarray]] = {}
    for cid, clu in chunk_cluster.items():
        emb = emb_dict.get(cid)
        if emb is not None:
            cluster_vecs.setdefault(clu, []).append(emb)

    centroids: Dict[int, np.ndarray] = {
        clu: np.mean(np.stack(vecs), axis=0).astype(np.float32)
        for clu, vecs in cluster_vecs.items()
        if vecs
    }

    log.info(
        "[match] context file loaded: period=%s  chunks=%d  clusters=%d  centroids=%d",
        period_start_found, len(chunk_cluster),
        len(set(chunk_cluster.values())), len(centroids),
    )
    return period_start_found, chunk_cluster, core_numbers, centroids


# ---------------------------------------------------------------------------
# Phase 3j: Incremental cluster matching
# ---------------------------------------------------------------------------

class ClusterMatcher:
    """
    Incremental cluster matcher: call process_period() at the end of each
    period iteration (after saving to the rolling buffer, before pruning),
    then finalize() once after the loop to flush matching.tsv and return
    lifecycle counts.

    Why incremental instead of post-loop batch:
        The rolling buffer is pruned to MAX_BUFFER_PERIODS=3 inside the
        period loop for memory efficiency.  A post-loop batch call therefore
        only has buffer data for the last 3 periods; all earlier periods are
        skipped with "no buffer data — skipping".  By running per-period,
        T-1 is always present in the buffer when process_period() is called,
        so every period receives correct matching.

    State (id_counter, period_pid_map, period_drift_vectors) accumulates
    across every process_period() call so persistent IDs are globally unique
    and velocity_alignment can look back one extra period.
    """

    def __init__(
        self,
        corpus_id: str,
        work_dir: str,
        config: dict,
        context_period_start: Optional[str] = None,
        context_chunk_cluster: Optional[Dict[str, Dict[str, int]]] = None,
        context_centroids:     Optional[Dict[str, Dict[int, np.ndarray]]] = None,
    ) -> None:
        self.corpus_id = corpus_id

        # ── Config thresholds ─────────────────────────────────────────────────
        # Mirrors arc_f_match_clusters() in the DB so cloud-computed matching
        # is directly comparable to the SQL-side staging table.
        self.continuation_threshold   = float(config.get("continuation_threshold",   0.90))
        self.weak_threshold           = float(config.get("weak_threshold",            0.70))
        # SPLIT_COVERAGE_THRESHOLD: minimum combined chunk coverage of the T-1 cluster
        # that the split-child candidates must collectively achieve to confirm a split.
        # 0.60 = children together account for at least 60% of the parent's content.
        self.split_coverage_threshold = float(config.get("split_coverage_threshold", 0.60))
        # REBORN_LOOKBACK_PERIODS: how many periods before T-1 to search for a matching
        # lineage when a cluster is initially classified as 'new'.
        # Searches T-2, T-3, ... T-(N+1) where N = reborn_lookback_periods.
        # Comment: 8 periods (~2 years quarterly) captures genuine hibernation vs death.
        self.reborn_lookback_periods  = int(config.get("reborn_lookback_periods", 8))

        # ── Persistent state across all process_period() calls ────────────────
        self._id_counter = 0
        # {period_start: {cluster_id: persistent_cluster_id}}
        # Needed so T can look up T-1's inherited IDs.  Never pruned — all
        # periods are kept so reborn detection can find any dormant lineage.
        self._period_pid_map: Dict[str, Dict[int, str]] = {}
        # {period_start: {persistent_cluster_id: drift_vector}}
        # Used for velocity_alignment (cosine sim between consecutive drifts).
        self._period_drift_vectors: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        # Chronological list of all periods processed (including context seed).
        # Index arithmetic here mirrors sorted_periods[period_idx] in the old code.
        self._processed_periods: List[str] = []

        # ── Cumulative counters ───────────────────────────────────────────────
        self._totals: Dict[str, int] = {
            "continuation": 0, "weak": 0, "new": 0, "split_child": 0, "reborn": 0,
        }
        self._reborn_gaps_all: List[int] = []
        self._period_lifecycle: Dict[str, Tuple[int, int, int, int]] = {}
        self._total_rows      = 0
        self._total_demotions = 0

        # ── Output file ───────────────────────────────────────────────────────
        import_dir = os.path.join(work_dir, "output", "import", corpus_id)
        os.makedirs(import_dir, exist_ok=True)
        self._out_path = os.path.join(import_dir, "matching.tsv")
        self._out_fh   = open(self._out_path, "w", encoding="utf-8")
        self._out_fh.write(
            "corpus_id\tperiod_start\tcluster_id\tpersistent_cluster_id\t"
            "match_type\tcomposite_score\tcentroid_sim\tchunk_overlap\tcore_continuity\t"
            "reborn_after_periods\treborn_from_period\t"
            "membership_churn\tdrift_vector\tvelocity_alignment\n"
        )

        # ── Context period pre-seeding (incremental mode) ─────────────────────
        # Assign fresh persistent IDs to the context period's clusters so the
        # first real period can inherit them via the normal matching path.
        if (context_period_start
                and context_chunk_cluster
                and context_period_start in context_chunk_cluster):
            context_clusters = sorted(
                clu for clu in set(context_chunk_cluster[context_period_start].values())
                if clu in (context_centroids or {}).get(context_period_start, {})
            )
            self._period_pid_map[context_period_start] = {
                clu: self._next_pid() for clu in context_clusters
            }
            self._processed_periods.append(context_period_start)
            log.info(
                "[match] context file found — incremental mode: T-1 from context "
                "(period=%s  clusters=%d)",
                context_period_start, len(context_clusters),
            )
        else:
            log.info("[match] backfill mode — matching runs inside period loop")

        log.info(
            "ClusterMatcher: corpus=%s  cont_thresh=%.2f  weak_thresh=%.2f  "
            "split_cov_thresh=%.2f  reborn_lookback=%d",
            corpus_id,
            self.continuation_threshold, self.weak_threshold,
            self.split_coverage_threshold, self.reborn_lookback_periods,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _next_pid(self) -> str:
        self._id_counter += 1
        return f"{self.corpus_id}_{self._id_counter:04d}"

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 0.0 else v

    # ── Per-period matching ────────────────────────────────────────────────────

    def process_period(
        self,
        period_start:       str,
        prev_chunk_cluster: Dict[str, Dict[str, int]],
        prev_centroids:     Dict[str, Dict[int, np.ndarray]],
        prev_core_numbers:  Dict[str, Dict[str, int]],
    ) -> None:
        """
        Match period_start clusters against T-1 using the current rolling buffer.

        Call after period_start has been saved to the rolling buffer
        (prev_chunk_cluster[period_start] and prev_centroids[period_start] exist)
        but before the buffer is pruned — T-1 is still present.

        self._processed_periods[-1] is T-1 at the time of the call.
        """
        out_fh    = self._out_fh
        corpus_id = self.corpus_id

        if period_start not in prev_chunk_cluster:
            log.warning("[match] no buffer data for %s — skipping", period_start)
            return

        cluster_map_T = prev_chunk_cluster[period_start]
        centroids_T   = prev_centroids[period_start]

        # Inverted index: cluster_id → set of chunk_ids for period T.
        cluster_chunks_T: Dict[int, set] = {}
        for chunk_id, clu in cluster_map_T.items():
            cluster_chunks_T.setdefault(clu, set()).add(chunk_id)

        # Only include clusters that have centroids (guards against any gap)
        t_cluster_ids = sorted(
            clu for clu in cluster_chunks_T if clu in centroids_T
        )

        # ── First period: no T-1 available — assign all as new ────────────────
        # self._processed_periods is empty iff no context was seeded in __init__
        # and no prior period has been processed.
        if not self._processed_periods:
            pid_map: Dict[int, str] = {}
            for clu in t_cluster_ids:
                pid = self._next_pid()
                pid_map[clu] = pid
                out_fh.write(
                    f"{corpus_id}\t{period_start}\t{clu}\t{pid}\t"
                    f"new\tNULL\tNULL\tNULL\tNULL\tNULL\tNULL\t"
                    f"1.000000\tNULL\tNULL\n"
                )
                self._total_rows += 1
            self._totals["new"] += len(t_cluster_ids)
            self._period_pid_map[period_start] = pid_map
            self._processed_periods.append(period_start)
            log.info(
                "[match] %s  n_new=%d (first period, all fresh IDs)",
                period_start, len(t_cluster_ids),
            )
            return

        # T-1 is always the last successfully processed period.
        prev_period = self._processed_periods[-1]

        # ── T-1 data unavailable (pruned or failed earlier) ───────────────────
        if prev_period not in prev_chunk_cluster:
            # Previous period was pruned from the rolling buffer.  The buffer
            # only holds MAX_BUFFER_PERIODS entries; this happens when T-1
            # failed to complete and was never saved to the buffer.
            log.warning(
                "[match] prev period %s missing from buffer — treating %s as first",
                prev_period, period_start,
            )
            pid_map = {}
            for clu in t_cluster_ids:
                pid = self._next_pid()
                pid_map[clu] = pid
                out_fh.write(
                    f"{corpus_id}\t{period_start}\t{clu}\t{pid}\t"
                    f"new\tNULL\tNULL\tNULL\tNULL\tNULL\tNULL\t"
                    f"1.000000\tNULL\tNULL\n"
                )
                self._total_rows += 1
            self._totals["new"] += len(t_cluster_ids)
            self._period_pid_map[period_start] = pid_map
            self._processed_periods.append(period_start)
            return

        cluster_map_prev  = prev_chunk_cluster[prev_period]
        centroids_prev    = prev_centroids[prev_period]
        core_numbers_prev = prev_core_numbers.get(prev_period, {})
        pid_map_prev      = self._period_pid_map.get(prev_period, {})

        # Inverted index for T-1
        cluster_chunks_prev: Dict[int, set] = {}
        for chunk_id, clu in cluster_map_prev.items():
            cluster_chunks_prev.setdefault(clu, set()).add(chunk_id)

        prev_cluster_ids = sorted(
            clu for clu in cluster_chunks_prev if clu in centroids_prev
        )

        log.info(
            "[match] %s  t_clusters=%d  prev_clusters=%d",
            period_start, len(t_cluster_ids), len(prev_cluster_ids),
        )

        if not t_cluster_ids or not prev_cluster_ids:
            # Degenerate period — assign all as new
            log.warning(
                "[match] %s  degenerate — t_clusters=%d prev_clusters=%d, "
                "skipping scoring (all new)",
                period_start, len(t_cluster_ids), len(prev_cluster_ids),
            )
            pid_map = {}
            for clu in t_cluster_ids:
                pid = self._next_pid()
                pid_map[clu] = pid
                out_fh.write(
                    f"{corpus_id}\t{period_start}\t{clu}\t{pid}\t"
                    f"new\tNULL\tNULL\tNULL\tNULL\tNULL\tNULL\t"
                    f"1.000000\tNULL\tNULL\n"
                )
                self._total_rows += 1
            self._totals["new"] += len(t_cluster_ids)
            self._period_pid_map[period_start] = pid_map
            self._period_lifecycle[period_start] = (
                len(t_cluster_ids),
                len(prev_cluster_ids) if not t_cluster_ids else 0,
                0,  # n_reborn
                0,  # n_split_child
            )
            self._processed_periods.append(period_start)
            return

        # ── Core chunk sets for T-1 ───────────────────────────────────────────
        # Core chunks are those whose core_number >= median for the period.
        # k-core decomposition assigns higher numbers to chunks in denser,
        # more central subgraphs — these form the stable semantic heart of
        # a cluster.  If cuGraph was unavailable, core_numbers_prev is empty
        # and core_continuity defaults to 0.0 for all pairs, gracefully
        # degrading to a centroid + chunk_overlap composite (weights 0.5 + 0.3).
        if core_numbers_prev:
            median_core = float(np.median(list(core_numbers_prev.values())))
        else:
            median_core = 0.0  # no core data; all core_continuity signals = 0

        core_chunks_prev: Dict[int, set] = {
            cp: {
                c for c in cluster_chunks_prev[cp]
                if core_numbers_prev.get(c, 0) >= median_core
            }
            for cp in prev_cluster_ids
        }

        # ── Vectorised centroid similarity matrix ─────────────────────────────
        # Stack normalised centroids into (n_T, dim) and (n_prev, dim) matrices
        # and compute all pairwise cosine similarities in one matmul.
        T_mat    = np.stack([self._norm(centroids_T[clu])    for clu in t_cluster_ids]).astype(np.float32)
        prev_mat = np.stack([self._norm(centroids_prev[clu]) for clu in prev_cluster_ids]).astype(np.float32)
        # sim_matrix[i, j] = cosine_similarity(t_cluster_ids[i], prev_cluster_ids[j])
        sim_matrix = np.clip(T_mat @ prev_mat.T, 0.0, 1.0)  # (n_T, n_prev)

        # ── Candidate pair pre-filter ─────────────────────────────────────────
        # The tightest safe pre-filter: centroid_sim >= weak_threshold.
        cand_is, cand_js = np.where(sim_matrix >= self.weak_threshold)

        # ── Compute composite scores for candidate pairs ───────────────────────
        # scores[(t_cluster, prev_cluster)] = (composite, centroid_sim, chunk_overlap, core_continuity)
        scores: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
        aw_values: List[float] = []

        # best_for_t[t_cluster] = best score tuple (even if below weak_threshold).
        # Written to 'new' rows so the output shows WHY the cluster missed threshold.
        best_for_t: Dict[int, Tuple[float, float, float, float]] = {}

        for i, j in zip(cand_is.tolist(), cand_js.tolist()):
            ct = t_cluster_ids[i]
            cp = prev_cluster_ids[j]
            centroid_sim = float(sim_matrix[i, j])

            chunks_T_str    = {str(c) for c in cluster_chunks_T[ct]}
            chunks_prev_str = {str(c) for c in cluster_chunks_prev[cp]}

            # SIGNAL 2: Chunk membership overlap (weight 0.3)
            # Fraction of T-1 cluster's chunks that reappear in T cluster.
            # chunk_ids are UUIDs stored as strings throughout the pipeline.
            # Explicit str() coercion guards against type mismatches when the rolling
            # buffer accumulates chunk_ids from code paths that may produce int indices.
            chunk_overlap = (
                len(chunks_T_str & chunks_prev_str) / len(chunks_prev_str)
                if chunks_prev_str else 0.0
            )

            # SIGNAL 3: Core chunk continuity (weight 0.2)
            # Fraction of T-1 cluster's high-core chunks that reappear in T cluster.
            cc_prev = core_chunks_prev[cp]
            core_continuity = (
                len({str(c) for c in cc_prev} & chunks_T_str) / len(cc_prev)
                if cc_prev else 0.0
            )

            # COMPOSITE SCORE — weight rationale:
            # 0.5 centroid_sim: most noise-resistant signal.
            # 0.3 chunk_overlap: direct document continuity.
            # 0.2 core_continuity: quality-weighted overlap.
            # Normalise by available_weight so composite equals centroid_sim when
            # chunk_overlap and core_continuity are both zero (chunk-disjoint corpora).
            available_weight = (
                0.5
                + (0.3 if chunk_overlap   > 0 else 0.0)
                + (0.2 if core_continuity > 0 else 0.0)
            )
            aw_values.append(available_weight)
            composite = (
                0.5 * centroid_sim +
                0.3 * chunk_overlap +
                0.2 * core_continuity
            ) / available_weight

            existing_best = best_for_t.get(ct)
            if existing_best is None or composite > existing_best[0]:
                best_for_t[ct] = (composite, centroid_sim, chunk_overlap, core_continuity)

            if composite >= self.weak_threshold:
                scores[(ct, cp)] = (composite, centroid_sim, chunk_overlap, core_continuity)

        # Capture best raw centroid similarity for T clusters with no candidate pairs.
        for t_i, ct in enumerate(t_cluster_ids):
            if ct not in best_for_t:
                best_cs = float(sim_matrix[t_i].max()) if sim_matrix.shape[1] > 0 else 0.0
                if best_cs > 0.0:
                    best_for_t[ct] = (0.5 * best_cs, best_cs, 0.0, 0.0)

        log.info(
            "[match] %s  candidate_pairs=%d  qualifying_pairs=%d  "
            "t_with_any_score=%d",
            period_start, len(cand_is), len(scores), len(best_for_t),
        )

        # ── Split detection — runs BEFORE 1-to-1 assignment ──────────────────
        # A split occurs when one research topic genuinely fragments into multiple
        # subtopics.  Detection criteria (both must hold):
        #   1. Two or more T clusters score > weak_threshold against same T-1 cluster.
        #   2. Their combined chunk coverage of T-1 >= split_coverage_threshold.

        # Reverse map: prev_cluster → list of qualifying T clusters
        prev_to_cands: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
        for (ct, cp), (comp, cs, co, cc) in scores.items():
            prev_to_cands.setdefault(cp, []).append((ct, comp, cs, co, cc))

        split_assignments: Dict[int, Tuple[str, float, float, float, float]] = {}
        split_source_prev: set = set()  # T-1 clusters confirmed as split sources
        split_ct_to_cp: Dict[int, int] = {}

        for cp, candidates in prev_to_cands.items():
            if len(candidates) < 2:
                continue  # only one T cluster qualifies — not a split

            chunks_prev_set = cluster_chunks_prev.get(cp, set())
            if not chunks_prev_set:
                continue

            combined: set = set()
            for ct, comp, cs, co, cc in candidates:
                combined |= (cluster_chunks_T[ct] & chunks_prev_set)
            combined_coverage = len(combined) / len(chunks_prev_set)

            if combined_coverage >= self.split_coverage_threshold:
                split_source_prev.add(cp)
                prev_pid = pid_map_prev.get(cp, self._next_pid())
                for ct, comp, cs, co, cc in candidates:
                    split_assignments[ct] = (prev_pid, comp, cs, co, cc)
                    split_ct_to_cp[ct] = cp

        # ── 1-to-1 greedy assignment (non-split clusters) ────────────────────
        # Mirrors arc_f_match_clusters() two-pass greedy bipartite matching.
        # Pass A: each T cluster picks its best T-1 partner.
        pass_a: Dict[int, Tuple[int, float, float, float, float]] = {}
        for (ct, cp), (comp, cs, co, cc) in scores.items():
            if ct in split_assignments:
                continue  # already resolved as split_child
            if cp in split_source_prev:
                continue  # this T-1's lineage was split; skip for 1-to-1
            existing = pass_a.get(ct)
            if existing is None or comp > existing[1]:
                pass_a[ct] = (cp, comp, cs, co, cc)

        # Pass B: each T-1 cluster keeps only its best T cluster.
        prev_to_winner: Dict[int, Tuple[int, float, float, float, float]] = {}
        for ct, (cp, comp, cs, co, cc) in pass_a.items():
            existing = prev_to_winner.get(cp)
            if existing is None or comp > existing[1]:
                prev_to_winner[cp] = (ct, comp, cs, co, cc)

        # Invert to a t_cluster-keyed lookup for the write loop.
        winners: Dict[int, Tuple[int, float, float, float, float]] = {
            ct: (cp, comp, cs, co, cc)
            for cp, (ct, comp, cs, co, cc) in prev_to_winner.items()
        }

        # ── Reborn detection ──────────────────────────────────────────────────
        # A cluster classified as 'new' may be a reborn cluster — one that existed
        # previously, disappeared for 1-N periods, and has now returned.
        #
        # Search window: reborn_lookback_periods (default 8).
        # Searches T-2, T-3, ... using self._processed_periods index arithmetic
        # (equivalent to sorted_periods[period_idx-1-lb] in the old batch code).
        # NOTE: reborn lookback is bounded by both reborn_lookback_periods config
        # AND the rolling buffer size (prev_centroids only holds the last
        # MAX_BUFFER_PERIODS entries).  self._period_pid_map holds ALL periods
        # (never pruned) so ID inheritance is always available; centroid lookup
        # degrades gracefully when a lookback period has been pruned from buffer.

        # active_pcids_at_T1: persistent IDs still alive going into T.
        active_pcids_at_T1 = set(pid_map_prev.values())

        reborn_assignments: Dict[int, Tuple[str, float, str, int]] = {}

        for ct in t_cluster_ids:
            if ct in split_assignments or ct in winners:
                continue  # already resolved

            new_centroid = self._norm(centroids_T[ct])
            best_reborn_sim    = 0.0
            best_reborn_pcid:   Optional[str] = None
            best_reborn_period: Optional[str] = None
            best_reborn_gap:    Optional[int] = None

            # self._processed_periods[-1] is T-1.
            # lb=1 → T-2 at index [-2], lb=2 → T-3 at index [-3], etc.
            for lb in range(1, self.reborn_lookback_periods + 1):
                lb_idx = len(self._processed_periods) - 1 - lb
                if lb_idx < 0:
                    break
                lookback_period = self._processed_periods[lb_idx]

                lookback_centroids = prev_centroids.get(lookback_period, {})
                lookback_pid_map   = self._period_pid_map.get(lookback_period, {})

                for prev_clu_id, prev_centroid in lookback_centroids.items():
                    pcid = lookback_pid_map.get(prev_clu_id)
                    if pcid is None:
                        continue
                    if pcid in active_pcids_at_T1:
                        continue  # still active — would have been matched already
                    sim = float(np.clip(
                        np.dot(new_centroid, self._norm(prev_centroid)), 0.0, 1.0
                    ))
                    if sim >= self.weak_threshold and sim > best_reborn_sim:
                        best_reborn_sim    = sim
                        best_reborn_pcid   = pcid
                        best_reborn_period = lookback_period
                        best_reborn_gap    = lb  # lb=1 → absent 1 period (T-1 only)

            if best_reborn_pcid is not None:
                reborn_assignments[ct] = (
                    best_reborn_pcid, best_reborn_sim,
                    best_reborn_period, best_reborn_gap,  # type: ignore[arg-type]
                )

        if reborn_assignments:
            gaps = [g for _, _, _, g in reborn_assignments.values()]
            log.info(
                "[match] %s  %d reborn clusters detected (gaps: %s)",
                period_start, len(reborn_assignments),
                ", ".join(f"{g} period{'s' if g != 1 else ''}" for g in sorted(gaps)),
            )
            self._reborn_gaps_all.extend(gaps)

        # ── Post-matching deduplication ────────────────────────────────────────
        # Enforce 1-to-1 (persistent_cluster_id → cluster_id) across winners
        # and reborn_assignments combined.  Split children are exempt by design.
        pcid_to_winner_candidates: Dict[str, List[Tuple]] = {}
        for ct_w, (cp_w, comp_w, cs_w, co_w, cc_w) in winners.items():
            pcid_w = pid_map_prev.get(cp_w)
            if pcid_w is not None:
                pcid_to_winner_candidates.setdefault(pcid_w, []).append(
                    (ct_w, comp_w, 'winner')
                )
        for ct_r, (pcid_r, sim_r, _from_p, _gap) in reborn_assignments.items():
            pcid_to_winner_candidates.setdefault(pcid_r, []).append(
                (ct_r, sim_r, 'reborn')
            )

        n_demotions = 0
        demoted_winners: set = set()
        demoted_reborns: set = set()

        for pcid_dup, candidates in pcid_to_winner_candidates.items():
            if len(candidates) <= 1:
                continue
            candidates.sort(key=lambda x: x[1], reverse=True)
            for ct_d, _score_d, kind_d in candidates[1:]:
                new_pcid = self._next_pid()
                log.warning(
                    "[match] Demoted duplicate pcid %s (%s): cluster %d -> new ID %s",
                    pcid_dup, kind_d, ct_d, new_pcid,
                )
                if kind_d == 'winner':
                    demoted_winners.add(ct_d)
                else:
                    demoted_reborns.add(ct_d)
                n_demotions += 1

        if demoted_winners:
            winners = {ct: v for ct, v in winners.items() if ct not in demoted_winners}
        if demoted_reborns:
            reborn_assignments = {ct: v for ct, v in reborn_assignments.items()
                                  if ct not in demoted_reborns}

        if n_demotions > 0:
            log.info(
                "[match] Deduplication: %d duplicate pcid assignment(s) demoted to 'new'",
                n_demotions,
            )
        self._total_demotions += n_demotions

        # ── Write period results ───────────────────────────────────────────────
        # Pre-build helpers needed for membership_churn, drift_vector, velocity_alignment.
        prev_pcid_to_clid: Dict[str, int] = {v: k for k, v in pid_map_prev.items()}
        prev_drifts = self._period_drift_vectors.get(prev_period, {})
        this_period_drifts: Dict[str, Optional[np.ndarray]] = {}

        def _row_extras(
            ct_: int,
            pid_: str,
            cp_: Optional[int],
        ):
            """
            Return (churn_str, drift_str, va_str, drift_vec_or_none) for a
            matching row.  Captures period-local dicts from enclosing scope.

            membership_churn: fraction of T cluster chunks absent from T-1.
            drift_vector:     T centroid − T-1 centroid (space-separated floats).
            velocity_alignment: cosine sim between current and previous drift.
            """
            # membership_churn
            prev_cl = prev_pcid_to_clid.get(pid_)
            ct_chunks = cluster_chunks_T.get(ct_, set())
            n_ct = len(ct_chunks)
            if prev_cl is not None and n_ct > 0:
                prev_chs = cluster_chunks_prev.get(prev_cl, set())
                # str() coercion: chunk_ids are UUIDs (strings) throughout the
                # pipeline; explicit cast guards against any int/str mismatch in
                # the rolling buffer that would silently yield an empty intersection.
                ct_str   = {str(c) for c in ct_chunks}
                prev_str = {str(c) for c in prev_chs}
                churn = 1.0 - len(ct_str & prev_str) / n_ct
            else:
                churn = 1.0
            churn_str = f"{churn:.6f}"

            # drift_vector: T centroid − T-1 centroid
            t_cent = centroids_T.get(ct_)
            p_cent = centroids_prev.get(cp_) if cp_ is not None else None
            if t_cent is not None and p_cent is not None:
                # Compute in float64 to avoid cancellation, store as float32
                dv = (t_cent.astype(np.float64) - p_cent.astype(np.float64)).astype(np.float32)
                dv_str = " ".join(f"{v:.6f}" for v in dv)
            else:
                dv = None
                dv_str = "NULL"

            # velocity_alignment: cosine sim between this drift and previous drift
            if dv is not None:
                prev_dv = prev_drifts.get(pid_)
                if prev_dv is not None:
                    dot_ = float(np.dot(dv, prev_dv))
                    norm_ = float(np.linalg.norm(dv) * np.linalg.norm(prev_dv))
                    va_str = f"{dot_ / norm_:.6f}" if norm_ > 0.0 else "0.000000"
                else:
                    va_str = "NULL"
            else:
                va_str = "NULL"

            return churn_str, dv_str, va_str, dv

        pid_map = {}
        n_cont = n_weak = n_new = n_split = n_reborn = 0

        for ct in t_cluster_ids:
            if ct in split_assignments:
                prev_pid, comp, cs, co, cc = split_assignments[ct]
                pid_map[ct] = prev_pid
                _ch, _dv, _va, _dv_arr = _row_extras(ct, prev_pid, split_ct_to_cp.get(ct))
                this_period_drifts[prev_pid] = _dv_arr
                out_fh.write(
                    f"{corpus_id}\t{period_start}\t{ct}\t{prev_pid}\t"
                    f"split_child\t{comp:.6f}\t{cs:.6f}\t{co:.6f}\t{cc:.6f}\t"
                    f"NULL\tNULL\t{_ch}\t{_dv}\t{_va}\n"
                )
                n_split += 1

            elif ct in winners:
                cp, comp, cs, co, cc = winners[ct]
                prev_pid = pid_map_prev.get(cp)
                if prev_pid is None:
                    # Defensive: T-1 cluster existed but has no persistent ID.
                    # Should not happen after the first period, but guard gracefully.
                    prev_pid = self._next_pid()
                    log.warning(
                        "[match] %s cluster %d matched T-1 cluster %d "
                        "with no persistent ID — assigning new ID %s",
                        period_start, ct, cp, prev_pid,
                    )
                pid_map[ct] = prev_pid
                if comp >= self.continuation_threshold:
                    match_type = "continuation"
                    n_cont += 1
                else:
                    match_type = "weak"
                    n_weak += 1
                _ch, _dv, _va, _dv_arr = _row_extras(ct, prev_pid, cp)
                this_period_drifts[prev_pid] = _dv_arr
                out_fh.write(
                    f"{corpus_id}\t{period_start}\t{ct}\t{prev_pid}\t"
                    f"{match_type}\t{comp:.6f}\t{cs:.6f}\t{co:.6f}\t{cc:.6f}\t"
                    f"NULL\tNULL\t{_ch}\t{_dv}\t{_va}\n"
                )

            elif ct in reborn_assignments:
                # Reborn: inherits persistent ID from a prior lineage gap.
                prev_pid, sim, from_period, gap = reborn_assignments[ct]
                pid_map[ct] = prev_pid
                _ch, _dv, _va, _dv_arr = _row_extras(ct, prev_pid, None)
                this_period_drifts[prev_pid] = _dv_arr
                out_fh.write(
                    f"{corpus_id}\t{period_start}\t{ct}\t{prev_pid}\t"
                    f"reborn\t{sim:.6f}\t{sim:.6f}\t0.000000\t0.000000\t"
                    f"{gap}\t{from_period}\t{_ch}\t{_dv}\t{_va}\n"
                )
                n_reborn += 1

            else:
                # No qualifying match — genuinely new cluster.
                # Write the best score found (if any) so the output shows
                # WHY the cluster missed threshold.
                pid = self._next_pid()
                pid_map[ct] = pid
                _ch, _dv, _va, _dv_arr = _row_extras(ct, pid, None)
                this_period_drifts[pid] = _dv_arr
                best = best_for_t.get(ct)
                if best is not None:
                    comp, cs, co, cc = best
                    out_fh.write(
                        f"{corpus_id}\t{period_start}\t{ct}\t{pid}\t"
                        f"new\t{comp:.6f}\t{cs:.6f}\t{co:.6f}\t{cc:.6f}\t"
                        f"NULL\tNULL\t{_ch}\t{_dv}\t{_va}\n"
                    )
                else:
                    out_fh.write(
                        f"{corpus_id}\t{period_start}\t{ct}\t{pid}\t"
                        f"new\tNULL\tNULL\tNULL\tNULL\tNULL\tNULL\t"
                        f"{_ch}\t{_dv}\t{_va}\n"
                    )
                n_new += 1

            self._total_rows += 1

        self._totals["continuation"] += n_cont
        self._totals["weak"]         += n_weak
        self._totals["new"]          += n_new
        self._totals["split_child"]  += n_split
        self._totals["reborn"]       += n_reborn

        self._period_pid_map[period_start]       = pid_map
        self._period_drift_vectors[period_start] = this_period_drifts

        # Per-period lifecycle counts for f_period.tsv patching.
        # n_deaths = T-1 clusters not consumed by any winning or split match.
        _matched_t1 = (
            {cp for _ct, (cp, *_) in winners.items()} | split_source_prev
        )
        self._period_lifecycle[period_start] = (
            n_new,                                              # n_births
            max(0, len(prev_cluster_ids) - len(_matched_t1)),  # n_deaths
            n_reborn,                                           # n_reborn
            n_split,                                            # n_split_child
        )

        avg_aw = float(np.mean(aw_values)) if aw_values else 0.0
        log.info(
            "[match] %s  t_clusters=%d  prev_clusters=%d  "
            "continuation=%d  weak=%d  new=%d  split_child=%d  reborn=%d  "
            "avg_available_weight=%.3f",
            period_start, len(t_cluster_ids), len(prev_cluster_ids),
            n_cont, n_weak, n_new, n_split, n_reborn, avg_aw,
        )

        # Register this period as processed LAST so reborn lookback arithmetic
        # (which uses len(self._processed_periods) - 1 as T-1 index) is correct
        # throughout the method body above.
        self._processed_periods.append(period_start)

    # ── Finalization ──────────────────────────────────────────────────────────

    def finalize(self) -> Tuple[int, Dict[str, Tuple[int, int, int, int]]]:
        """
        Close the output file, write summary logs, and return results.
        Call once after the period loop completes.

        Returns (total_rows, period_lifecycle) where period_lifecycle maps
        period_start → (n_births, n_deaths, n_reborn, n_split_child).
        """
        self._out_fh.close()

        total_clusters = sum(self._totals.values())
        split_rate  = self._totals["split_child"] / max(total_clusters, 1)
        reborn_rate = self._totals["reborn"]      / max(total_clusters, 1)
        log.info(
            "match_clusters complete — total=%d  continuation=%d  weak=%d  "
            "new=%d  split_child=%d  reborn=%d  split_rate=%.3f  reborn_rate=%.3f  "
            "total_demotions=%d",
            total_clusters, self._totals["continuation"], self._totals["weak"],
            self._totals["new"], self._totals["split_child"], self._totals["reborn"],
            split_rate, reborn_rate, self._total_demotions,
        )
        if self._reborn_gaps_all:
            log.info(
                "match_clusters: reborn summary: %d total across all periods  "
                "avg gap=%.1f periods  max gap=%d periods",
                len(self._reborn_gaps_all),
                float(np.mean(self._reborn_gaps_all)),
                max(self._reborn_gaps_all),
            )
        log.info(
            "match_clusters: wrote %d rows → %s",
            self._total_rows, self._out_path,
        )
        return self._total_rows, self._period_lifecycle


# ---------------------------------------------------------------------------
# Phase 4: Upload outputs to S3 (or copy to local-output)
# ---------------------------------------------------------------------------

def phase_upload(
    work_dir: str,
    corpus_id: str,
    local_output: Optional[str],
    all_row_counts: Dict[str, int],
    completed_periods: List[str],
) -> None:
    log.set_phase("upload")
    output_dir = os.path.join(work_dir, "output")

    manifest = {
        "corpus_id":    corpus_id,
        "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
        "periods":      sorted(completed_periods),
        "files": [
            {"s3_key": k, "rows": v}
            for k, v in sorted(all_row_counts.items())
        ],
    }

    manifest_key   = f"import/{corpus_id}/manifest.json"
    manifest_local = os.path.join(output_dir, "import", corpus_id, "manifest.json")
    os.makedirs(os.path.dirname(manifest_local), exist_ok=True)
    with open(manifest_local, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    all_row_counts[manifest_key] = len(completed_periods)
    log.info("Manifest written: %d periods, %d files", len(completed_periods), len(all_row_counts))

    if local_output:
        os.makedirs(local_output, exist_ok=True)
        for root, _dirs, files in os.walk(output_dir):
            for fname in files:
                src = os.path.join(root, fname)
                rel = os.path.relpath(src, output_dir)
                dst = os.path.join(local_output, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
        log.info("Local mode: output written to %s", local_output)
        return

    s3 = _get_s3_client()
    bucket = os.environ["S3_BUCKET"]
    for root, _dirs, files in os.walk(output_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, output_dir)
            s3_key = rel.replace(os.sep, "/")
            _s3_upload(s3, bucket, s3_key, local_path)
    log.info("Upload complete — manifest at s3://%s/%s", bucket, manifest_key)

# ---------------------------------------------------------------------------
# Phase 5: Cleanup
# ---------------------------------------------------------------------------

def phase_cleanup(work_dir: str) -> None:
    log.set_phase("cleanup")
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
        log.info("Removed working directory: %s", work_dir)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    corpus_id = args.corpus_id
    work_dir  = "/tmp/arc_work"
    os.makedirs(work_dir, exist_ok=True)

    phase_times: Dict[str, float] = {}
    t_wall = time.monotonic()

    # ── Phase 0: Download ────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        chunks_path, config_path = phase_download(
            corpus_id, work_dir, args.local_input
        )
    except Exception as exc:
        log.error("Phase download failed: %s", exc)
        sys.exit(1)
    phase_times["download"] = time.monotonic() - t0

    # Detect optional incremental input files:
    #   embeddings_{corpus_id}.npz  — existing embeddings for incremental mode
    #   context_{corpus_id}_prev.tsv — T-1 cluster data for incremental matching
    if args.local_input:
        _local_in_dir = os.path.join(args.local_input, corpus_id)
        _npz_candidate = os.path.join(_local_in_dir, f"embeddings_{corpus_id}.npz")
        _ctx_candidate = os.path.join(_local_in_dir, f"context_{corpus_id}_prev.tsv")
        npz_input     = _npz_candidate if os.path.exists(_npz_candidate) else None
        context_input = _ctx_candidate if os.path.exists(_ctx_candidate) else None
        if npz_input:
            log.info("[embed] Found optional input npz: %s", npz_input)
        if context_input:
            log.info("[match] Found optional context file: %s", context_input)
    else:
        _s3 = _get_s3_client()
        _bucket = os.environ["S3_BUCKET"]
        _opt_dir = os.path.join(work_dir, "input", corpus_id)
        os.makedirs(_opt_dir, exist_ok=True)
        _npz_local = os.path.join(_opt_dir, f"embeddings_{corpus_id}.npz")
        _ctx_local = os.path.join(_opt_dir, f"context_{corpus_id}_prev.tsv")
        npz_input = _npz_local if _s3_download_optional(
            _s3, _bucket, f"input/{corpus_id}/embeddings_{corpus_id}.npz", _npz_local
        ) else None
        context_input = _ctx_local if _s3_download_optional(
            _s3, _bucket, f"input/{corpus_id}/context_{corpus_id}_prev.tsv", _ctx_local
        ) else None

    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)

    # CONFIG HYGIENE NOTE:
    # config.json should be auto-exported from sys_run_config in the DB,
    # not hand-edited. Hand-editing risks drift (e.g. year_from=NULL in DB
    # while config.json has 2000, or leiden_res=1.0 when DB has 2.0).
    # Future: add arc export-config --corpus {corpus_id} command to arc.sh
    # that generates config.json directly from sys_run_config.

    # Apply CLI overrides then cast all numeric fields immediately.
    # json.load() may return strings if the JSON was hand-edited with quoted numbers;
    # int()/float() coerce both str and numeric JSON values safely.
    if args.k is not None:
        config["k"] = args.k                        # argparse type=int, already int

    k             = int(config.get("k",                      16))
    leiden_res    = float(config.get("leiden_res",           1.0))
    leiden_seed   = int(config.get("leiden_seed",            42))
    year_from     = int(config.get("year_from",              0))
    void_threshold  = float(config.get("void_distance_threshold", 0.3))
    # junk_threshold: clusters with size <= this value are treated as junk clusters.
    # Junk chunks are counted as dark matter — ideas that didn't consolidate into any
    # coherent research topic.  Default 2 matches the DB-side junk_threshold convention.
    junk_threshold  = int(config.get("junk_threshold", 2))
    void_radius     = float(config.get("void_radius",   0.15))
    fringe_radius   = float(config.get("fringe_radius", 0.20))
    if "embedding_model" not in config:
        raise ValueError("config.json must specify 'embedding_model' — no default is safe")
    model_name      = config["embedding_model"]
    max_seq_length  = int(config.get("max_seq_length", 512))
    resolution    = config.get("resolution",                 "quarterly")
    batch_size    = args.batch_size

    # Config hygiene warnings
    if "year_from" not in config or config.get("year_from") is None:
        log.warning(
            "CONFIG: year_from not set — defaulting to 0 (all years). "
            "Export from sys_run_config to ensure correct value."
        )
    if "leiden_res" not in config or float(config.get("leiden_res", 1.0)) == 1.0:
        log.warning(
            "CONFIG: leiden_res=1.0 (default). DB may have a different value. "
            "Export from sys_run_config to verify."
        )

    log.info(
        "Config: corpus=%s k=%d leiden_res=%.2f model=%s resolution=%s "
        "year_from=%d void_threshold=%.3f",
        corpus_id, k, leiden_res, model_name, resolution, year_from, void_threshold,
    )

    # ── Phase 1: Period split ────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        chunk_ids, texts, period_map = phase_period_split(
            chunks_path, resolution, year_from
        )
    except Exception as exc:
        log.error("Phase period_split failed: %s", exc)
        sys.exit(1)
    phase_times["period_split"] = time.monotonic() - t0

    if not chunk_ids:
        log.error("No chunks loaded — aborting")
        sys.exit(1)

    # ── Phase 2: Embed ───────────────────────────────────────────────────────
    # Incremental mode: if an existing npz is provided, embed only new chunks
    # (those not already in the npz) and merge with the existing embeddings.
    # Backfill mode: embed all chunks from scratch.
    # emb_dict ({chunk_id: embedding}) is used by load_context_file() below to
    # compute centroids for the context period without a second embedding pass.
    t0 = time.monotonic()
    try:
        if npz_input:
            log.set_phase("embed")
            log.info("[embed] Incremental mode — loading existing embeddings from %s", npz_input)
            _existing_npz  = np.load(npz_input, allow_pickle=False)
            _existing_embs = _existing_npz["embeddings"].astype(np.float32)
            _existing_cids = _existing_npz["chunk_ids"].tolist()
            _existing_set  = set(_existing_cids)

            _new_mask   = [cid not in _existing_set for cid in chunk_ids]
            _new_texts  = [texts[i] for i, m in enumerate(_new_mask) if m]
            _new_cids   = [chunk_ids[i] for i, m in enumerate(_new_mask) if m]
            log.info(
                "[embed] %d existing embeddings, %d new chunks to embed",
                len(_existing_set), len(_new_texts),
            )

            emb_dict: Dict[str, np.ndarray] = dict(zip(_existing_cids, _existing_embs))
            if _new_texts:
                _new_embs = phase_embed(_new_texts, model_name, batch_size, max_seq_length)
                for _cid, _emb in zip(_new_cids, _new_embs):
                    emb_dict[_cid] = _emb
            else:
                log.info("[embed] No new chunks to embed")

            # Reconstruct correctly-ordered embeddings array (matches chunk_ids order)
            embeddings = np.stack(
                [emb_dict[cid] for cid in chunk_ids]
            ).astype(np.float32)
        else:
            log.set_phase("embed")
            log.info("[embed] Backfill mode — embedding all %d chunks from scratch", len(texts))
            embeddings = phase_embed(texts, model_name, batch_size, max_seq_length)
            # Only build emb_dict if context_input exists (incremental mode).
            # Comment: emb_dict maps chunk_id → embedding vector for load_context_file().
            # Building it costs ~96MB RAM for G06N. Skip in backfill mode where it's unused.
            if context_input or npz_input:
                emb_dict = dict(zip(chunk_ids, embeddings))
            else:
                emb_dict = {}  # not needed in backfill mode
    except Exception as exc:
        log.error("Phase embed failed: %s", exc)
        sys.exit(1)
    phase_times["embed"] = time.monotonic() - t0

    # ── Phase 3: Per-period loop ─────────────────────────────────────────────

    log.set_phase("period_loop")
    all_row_counts:    Dict[str, int] = {}
    completed_periods: List[str]      = []

    # Cumulative per-stage timing across all periods (for summary)
    _STAGES = ("knn", "leiden", "cugraph", "chunk_measures", "cluster_geometry",
               "fedge", "bridge_chunks", "fvoid", "fgap", "spectral",
               "attractors", "surprise", "phase_score", "write")
    stage_samples: Dict[str, List[float]] = {s: [] for s in _STAGES}

    # Rolling buffers for post-processing cluster matching.
    # Keyed by period_start so match_clusters() can access T and T-1 data
    # for each consecutive period pair.  Populated inside the period try-block
    # so only successfully completed periods contribute to matching.
    prev_chunk_cluster: Dict[str, Dict[str, int]]        = {}  # period → {chunk_id: cluster_id}
    prev_centroids:     Dict[str, Dict[int, np.ndarray]] = {}  # period → {cluster_id: centroid}
    prev_core_numbers:  Dict[str, Dict[str, int]]        = {}  # period → {chunk_id: core_number}

    # Persistent state for chunk-level cross-period measures.
    # These accumulate across ALL periods in chronological order and are NEVER reset.
    # chunk_cluster_change_count: total times a chunk has switched clusters so far.
    #   0 = always in the same cluster (stable, load-bearing idea).
    #   High = migrates frequently (idea in motion, potential bridge concept).
    # chunk_total_appearances: how many periods has this chunk appeared in total.
    # chunk_cluster_appearances: how many of those appearances were in the current cluster.
    #   belief_persistence_score = in_current / total — fraction of life spent here.
    chunk_cluster_change_count: Dict[str, int]              = {}  # chunk_id → cumulative changes
    chunk_total_appearances:    Dict[str, int]              = {}  # chunk_id → total appearances
    chunk_cluster_appearances:  Dict[Tuple[str, int], int]  = {}  # (chunk_id, cluster_id) → count

    # Tier 2 rolling scalars for phase_transition_score.
    # Updated at the end of each successfully completed period.
    prev_alg_conn:       Optional[float] = None  # T-1 algebraic_connectivity
    prev_system_entropy: Optional[float] = None  # T-1 Shannon entropy of cluster sizes
    prev_n_clusters_buf: Optional[int]   = None  # T-1 n_clusters

    # Load T-1 context file for incremental matching (if provided).
    # Centroids are computed from emb_dict (built in Phase 2).
    context_period_start: Optional[str] = None
    if context_input:
        try:
            _ctx_ps, _ctx_cc, _ctx_cn, _ctx_cents = load_context_file(
                context_input, emb_dict
            )
            prev_chunk_cluster[_ctx_ps] = _ctx_cc
            prev_centroids[_ctx_ps]     = _ctx_cents
            prev_core_numbers[_ctx_ps]  = _ctx_cn
            context_period_start = _ctx_ps
        except Exception as exc:
            log.warning(
                "Context file load failed (%s) — falling back to backfill matching", exc
            )

    # ── Initialise incremental cluster matcher ────────────────────────────────
    # ClusterMatcher.process_period() is called at the END of each period
    # iteration (after saving to the rolling buffer, before pruning) so T-1
    # is always present in the buffer when matching runs.
    matcher = ClusterMatcher(
        corpus_id, work_dir, config,
        context_period_start=context_period_start,
        context_chunk_cluster=prev_chunk_cluster if context_period_start else None,
        context_centroids=prev_centroids if context_period_start else None,
    )

    t0 = time.monotonic()

    for period_start in sorted(period_map.keys()):
        period_indices = period_map[period_start]
        log.info("=== Period %s (%d chunks) ===", period_start, len(period_indices))
        t_period = time.monotonic()

        try:
            # ---- kNN ----
            log.set_phase("knn")
            log.info("k=%r (type=%s)", k, type(k).__name__)  # confirm k is int
            ts = time.monotonic()
            period_emb, knn_edges = run_knn(period_indices, chunk_ids, embeddings, k)
            stage_samples["knn"].append(time.monotonic() - ts)

            # ---- Leiden ----
            log.set_phase("leiden")
            ts = time.monotonic()
            cluster_map, leiden_modularity = run_leiden(
                period_indices, chunk_ids, knn_edges, leiden_res, leiden_seed
            )
            stage_samples["leiden"].append(time.monotonic() - ts)

            # ---- GPU (cuGraph) + CPU (chunk_measures) — overlapped ----
            # Both depend only on outputs already available: knn_edges + cluster_map.
            # Centroids are needed by chunk_measures, so compute them first (fast).
            centroids = run_centroids(period_indices, chunk_ids, cluster_map, embeddings)

            # ---- GPU (cuGraph) + CPU (chunk_measures, cluster_geometry) ----
            # Also includes run_fedge and run_spectral — both depend only on
            # knn_edges and cluster_map (available before this block).
            # Comment: run_fedge and run_spectral have no dependency on cuGraph or
            # chunk_measures. Moving them into the parallel block saves ~40s per G06N
            # run and scales linearly with corpus size — fedge is the bottleneck for
            # large periods.
            log.set_phase("parallel_gpu_cpu")
            ts_parallel = time.monotonic()
            with ThreadPoolExecutor(max_workers=5) as pool:
                fut_cugraph  = pool.submit(run_cugraph, knn_edges)
                fut_chunk    = pool.submit(
                    run_chunk_measures,
                    period_indices, chunk_ids, period_emb,
                    centroids, cluster_map, knn_edges,
                )
                fut_geom     = pool.submit(
                    run_cluster_geometry,
                    cluster_map, embeddings, period_indices, chunk_ids,
                )
                fut_fedge    = pool.submit(run_fedge, knn_edges, cluster_map)
                fut_spectral = pool.submit(run_spectral, period_indices, chunk_ids, knn_edges)

                chunk_graph_df             = fut_cugraph.result()
                chunk_measures, cluster_stats = fut_chunk.result()
                cluster_geometry           = fut_geom.result()
                f_edge_df                  = fut_fedge.result()
                spectral                   = fut_spectral.result()
            _t_parallel = time.monotonic() - ts_parallel
            stage_samples["cugraph"].append(_t_parallel)
            stage_samples["chunk_measures"].append(_t_parallel)
            stage_samples["cluster_geometry"].append(_t_parallel)
            stage_samples["fedge"].append(_t_parallel)
            stage_samples["spectral"].append(_t_parallel)

            # ---- bridge_chunks (adds n_bridge_chunks to f_edge_df) ----
            log.set_phase("bridge_chunks")
            ts = time.monotonic()
            f_edge_df = run_bridge_chunks(knn_edges, cluster_map, f_edge_df)
            stage_samples["bridge_chunks"].append(time.monotonic() - ts)

            # ---- f_void ----
            log.set_phase("fvoid")
            ts = time.monotonic()
            f_void_rows = run_fvoid(centroids, f_edge_df, void_threshold)
            stage_samples["fvoid"].append(time.monotonic() - ts)

            # ---- f_gap ----
            log.set_phase("fgap")
            ts = time.monotonic()
            period_chunk_ids_p = [chunk_ids[i] for i in period_indices]
            f_gap_rows = run_fgap(
                centroids, cluster_map, period_emb, period_chunk_ids_p,
                knn_edges, chunk_measures, corpus_id, period_start, config,
                void_radius=void_radius, fringe_radius=fringe_radius,
                f_edge_df=f_edge_df,
            )
            stage_samples["fgap"].append(time.monotonic() - ts)

            # ---- Topology measures (attractors, saddle points, avg path length,
            #      propagation speed) — runs after parallel block so cluster
            #      assignments + chunk measures are available.            ----
            log.set_phase("attractors")
            ts = time.monotonic()
            topology_stats = run_topology_measures(
                cluster_map, knn_edges, period_indices, chunk_ids, period_emb, chunk_measures
            )
            stage_samples["attractors"].append(time.monotonic() - ts)

            # ---- field_surprise_index: linear-extrapolation prediction error ----
            # Requires T-2 centroid history.  Returns NULL for first two periods.
            log.set_phase("surprise")
            ts = time.monotonic()
            field_surprise: Dict[int, Optional[float]] = {}
            if len(completed_periods) >= 2:
                # Retrieve T-1 and T-2 centroid pools from the rolling history.
                # completed_periods is ordered and only contains successful periods,
                # so [-1] is T-1 and [-2] is T-2 relative to the current period.
                prev_cents_t1 = prev_centroids.get(completed_periods[-1], {})
                prev_cents_t2 = prev_centroids.get(completed_periods[-2], {})
                for clu, centroid in centroids.items():
                    field_surprise[clu] = _compute_surprise_for_cluster(
                        centroid, prev_cents_t1, prev_cents_t2
                    )
            else:
                # Not enough period history for trajectory extrapolation
                field_surprise = {clu: None for clu in centroids}
            stage_samples["surprise"].append(time.monotonic() - ts)

            # ---- phase_transition_score: composite structural instability ----
            log.set_phase("phase_score")
            ts = time.monotonic()
            system_entropy = _compute_system_entropy(cluster_map)
            # Estimate births and deaths using centroid distance threshold.
            # Uses T-1 centroids from the rolling buffer (prev_centroids).
            prev_cents_for_bd = (
                prev_centroids.get(completed_periods[-1], {})
                if completed_periods else {}
            )
            n_births_est, n_deaths_est = _estimate_births_deaths(
                centroids, prev_cents_for_bd
            )
            phase_score = _compute_phase_transition_score(
                n_births_est, n_deaths_est, len(centroids),
                spectral[0], prev_alg_conn,
                system_entropy, prev_system_entropy,
                prev_n_clusters_buf,
            )
            stage_samples["phase_score"].append(time.monotonic() - ts)

            # ---- Per-chunk cross-period measures ----
            # Computed after Leiden so cluster_map is final for this period.
            # Uses persistent dicts (never reset) to accumulate state across periods.
            log.set_phase("chunk_cross_period")

            # Identify junk clusters: too small to represent real research topics.
            # Chunks in junk clusters (or with no assignment) count as dark matter.
            cluster_sizes_p = collections.Counter(cluster_map.values())
            junk_clusters_p = {
                cl for cl, sz in cluster_sizes_p.items() if sz <= junk_threshold
            }

            # is_dark_matter: chunk belongs to no cluster, cluster_id == -1,
            # or belongs to a junk cluster.
            # High n_dark_matter_chunks per period = field in flux;
            # sudden spike often precedes a phase transition (new clusters forming).
            is_dark_matter: Dict[str, bool] = {}
            for gi in period_indices:
                cid_p = chunk_ids[gi]
                cl_p  = cluster_map.get(cid_p)
                is_dark_matter[cid_p] = (
                    cl_p is None or cl_p == -1 or cl_p in junk_clusters_p
                )
            n_dark_matter_chunks = sum(1 for v in is_dark_matter.values() if v)

            # T-1 cluster assignment for volatility comparison.
            # completed_periods[-1] is the most recently successful period.
            _prev_period_p  = completed_periods[-1] if completed_periods else None
            _prev_cmap_t1_p = prev_chunk_cluster.get(_prev_period_p, {}) if _prev_period_p else {}

            # membership_volatility: cumulative count of cluster changes across all
            # periods processed so far.
            #   0 = always in the same cluster (stable, load-bearing idea).
            #   High = migrates frequently (bridge concept or noise).
            #
            # belief_persistence_score: fraction of this chunk's appearances where it
            # was in its current cluster.  Range [0, 1].
            #   1.0 = always been in this cluster (perfectly settled idea).
            #   0.0 = just arrived, was always elsewhere before.
            membership_volatility:     Dict[str, int]   = {}
            belief_persistence_score_p: Dict[str, float] = {}

            for gi in period_indices:
                cid_p         = chunk_ids[gi]
                current_cl    = cluster_map.get(cid_p)
                prev_cl       = _prev_cmap_t1_p.get(cid_p)

                # Update cumulative change count
                if prev_cl is not None and prev_cl != current_cl:
                    chunk_cluster_change_count[cid_p] = (
                        chunk_cluster_change_count.get(cid_p, 0) + 1
                    )
                elif cid_p not in chunk_cluster_change_count:
                    chunk_cluster_change_count[cid_p] = 0
                membership_volatility[cid_p] = chunk_cluster_change_count[cid_p]

                # Update appearance trackers
                chunk_total_appearances[cid_p] = (
                    chunk_total_appearances.get(cid_p, 0) + 1
                )
                key_ca = (cid_p, current_cl)
                chunk_cluster_appearances[key_ca] = (
                    chunk_cluster_appearances.get(key_ca, 0) + 1
                )
                total_app  = chunk_total_appearances[cid_p]
                in_current = chunk_cluster_appearances[key_ca]
                belief_persistence_score_p[cid_p] = in_current / total_app

            log.info(
                "chunk_cross_period: n_dark_matter=%d (%.1f%%)  "
                "mean_volatility=%.2f  mean_bps=%.3f",
                n_dark_matter_chunks,
                100.0 * n_dark_matter_chunks / max(len(period_indices), 1),
                float(np.mean(list(membership_volatility.values()))) if membership_volatility else 0.0,
                float(np.mean(list(belief_persistence_score_p.values()))) if belief_persistence_score_p else 0.0,
            )

            # ---- Write ----
            log.set_phase("write")
            ts = time.monotonic()
            row_counts = write_period_outputs(
                work_dir,
                PeriodOutputs(
                    corpus_id=corpus_id,
                    period_start=period_start,
                    period_indices=period_indices,
                    chunk_ids=chunk_ids,
                    knn_edges=knn_edges,
                    cluster_map=cluster_map,
                    chunk_graph_df=chunk_graph_df,
                    f_edge_df=f_edge_df,
                    centroids=centroids,
                    chunk_measures=chunk_measures,
                    f_void_rows=f_void_rows,
                    f_gap_rows=f_gap_rows,
                    spectral=spectral,
                    cluster_geometry=cluster_geometry,
                    cluster_stats=cluster_stats,
                    topology_stats=topology_stats,
                    field_surprise=field_surprise,
                    phase_scores={
                        "phase_transition_score": phase_score,
                        "n_dark_matter_chunks":   n_dark_matter_chunks,
                        "is_dark_matter":         is_dark_matter,
                        "membership_volatility":  membership_volatility,
                        "belief_persistence_score": belief_persistence_score_p,
                    },
                    leiden_modularity=leiden_modularity,
                ),
            )
            stage_samples["write"].append(time.monotonic() - ts)

            # Cold keys are unique per period (update fine).
            # Import keys repeat across periods — accumulate row counts.
            # NOTE: loop var must NOT be named 'k' — that would shadow the kNN k integer.
            for s3_key, n_rows in row_counts.items():
                all_row_counts[s3_key] = all_row_counts.get(s3_key, 0) + n_rows

            # ---- Rolling buffer for post-processing cluster matching ----
            # Save chunk→cluster mapping, centroids, and core_number for this period.
            # Saved here (after write_period_outputs succeeds) so only successfully
            # completed periods contribute — matching never sees partial data.
            prev_chunk_cluster[period_start] = dict(cluster_map)
            prev_centroids[period_start]     = dict(centroids)
            if chunk_graph_df is not None:
                # Convert cuDF columns to a plain Python dict immediately so the
                # GPU-resident DataFrame can be garbage-collected and VRAM freed
                # before the next period begins.
                cg_sub = chunk_graph_df[["chunk_id", "core_number"]].to_pandas()
                prev_core_numbers[period_start] = dict(
                    zip(cg_sub["chunk_id"], cg_sub["core_number"].fillna(0).astype(int))
                )
            else:
                # cuGraph was unavailable (OOM or import failure) for this period.
                # core_continuity will default to 0.0 for all pairs using this period
                # as T-1, gracefully degrading the composite to centroid + chunk_overlap.
                prev_core_numbers[period_start] = {}

            # Update Tier 2 rolling scalars for the next period's phase_transition_score.
            # Saved after write succeeds to match the same consistency guarantee as
            # the cluster matching buffers above.
            prev_alg_conn       = spectral[0]
            prev_system_entropy = system_entropy
            prev_n_clusters_buf = len(centroids)

            completed_periods.append(period_start)

            # ── Incremental cluster matching ──────────────────────────────────
            # Run immediately after this period's data is in the rolling buffer
            # so T-1 is guaranteed present.  Must run BEFORE pruning below.
            log.set_phase("match")
            matcher.process_period(
                period_start, prev_chunk_cluster, prev_centroids, prev_core_numbers
            )
            log.set_phase("period_loop")

            # Prune rolling buffers to last MAX_BUFFER_PERIODS periods.
            # Comment: rolling buffers only need last 3 periods:
            # - T-1: chunk overlap, core_continuity, centroid matching, drift_vector
            # - T-2: field_surprise_index (linear extrapolation needs T-2 and T-1)
            # - T-3: safety margin for reborn detection lookback
            # Keeping all periods wastes ~5GB RAM at full USPTO scale (200K chunks × 200 periods).
            MAX_BUFFER_PERIODS = 3
            if len(prev_chunk_cluster) > MAX_BUFFER_PERIODS:
                _oldest_keys = sorted(prev_chunk_cluster.keys())[:-MAX_BUFFER_PERIODS]
                for _bk in _oldest_keys:
                    del prev_chunk_cluster[_bk]
                    prev_centroids.pop(_bk, None)
                    prev_core_numbers.pop(_bk, None)

        except Exception as exc:
            log.error(
                "Period %s FAILED — continuing to next period. Error: %s",
                period_start, exc, exc_info=True,
            )
            continue

        log.set_phase("period_loop")
        log.info("Period %s done in %.1fs", period_start, time.monotonic() - t_period)

    phase_times["period_loop"] = time.monotonic() - t0
    log.info(
        "Period loop complete: %d/%d periods succeeded",
        len(completed_periods), len(period_map),
    )

    if not completed_periods:
        log.error("No periods succeeded — aborting before upload")
        sys.exit(1)

    # ── Save embeddings npz to cold storage ──────────────────────────────────
    # Single compressed numpy binary replacing the per-period .tsv.gz files.
    # Contains float32 embedding matrix + string chunk_id array in same order.
    # In incremental mode this merges existing + new embeddings into one file.
    log.set_phase("save_embeddings")
    try:
        _cold_dir = os.path.join(work_dir, "output", "cold", corpus_id)
        os.makedirs(_cold_dir, exist_ok=True)
        _npz_cold = os.path.join(_cold_dir, f"embeddings_{corpus_id}.npz")
        np.savez_compressed(
            _npz_cold,
            embeddings=embeddings,
            chunk_ids=np.array(chunk_ids),
        )
        _n_emb = len(chunk_ids)
        all_row_counts[f"cold/{corpus_id}/embeddings_{corpus_id}.npz"] = _n_emb
        log.info(
            "Saved embeddings npz: %d chunks → %s (%.1f MB)",
            _n_emb, _npz_cold, os.path.getsize(_npz_cold) / 1e6,
        )
    except Exception as exc:
        log.error("Failed to save embeddings npz: %s", exc, exc_info=True)

    # ── Phase 3j: Cluster matching (finalize) ────────────────────────────────
    # process_period() was called inside the loop; finalize() closes the file
    # and returns totals.  Phase time accumulates loop-internal match calls too
    # (via log.set_phase("match") in the loop) so only add finalize overhead here.
    t0 = time.monotonic()
    try:
        n_match_rows, _lifecycle = matcher.finalize()
        all_row_counts[f"import/{corpus_id}/matching.tsv"] = n_match_rows
        patch_f_period_lifecycle(work_dir, corpus_id, _lifecycle)
    except Exception as exc:
        # Non-fatal: log and continue to upload whatever other outputs we have.
        log.error("Phase match finalize failed: %s", exc, exc_info=True)
    phase_times["match"] = phase_times.get("match", 0.0) + (time.monotonic() - t0)

    # ── Phase 4: Upload ──────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        phase_upload(
            work_dir, corpus_id, args.local_output,
            all_row_counts, completed_periods,
        )
    except Exception as exc:
        log.error("Phase upload failed: %s", exc)
        sys.exit(1)
    phase_times["upload"] = time.monotonic() - t0

    # ── Phase 5: Cleanup ─────────────────────────────────────────────────────
    t0 = time.monotonic()
    phase_cleanup(work_dir)
    phase_times["cleanup"] = time.monotonic() - t0

    # ── Summary ──────────────────────────────────────────────────────────────
    total = time.monotonic() - t_wall
    log.set_phase("done")
    log.info("=== arc_cloud_run complete — total %.1fs ===", total)

    # Top-level phase times
    for phase, elapsed in phase_times.items():
        log.info("  %-15s %.1fs", phase, elapsed)
    log.info("  periods_succeeded  %d / %d", len(completed_periods), len(period_map))
    log.info("  output_files       %d", len(all_row_counts))

    # Per-stage timing statistics across periods
    n_periods_ok = len(completed_periods)
    if n_periods_ok > 0:
        log.info("--- Per-stage timing (over %d periods) ---", n_periods_ok)
        log.info("  %-16s %8s %8s %8s", "stage", "avg(s)", "p50(s)", "p95(s)")
        for stage in _STAGES:
            samples = stage_samples[stage]
            if not samples:
                continue
            arr = np.array(samples)
            avg = float(arr.mean())
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            log.info("  %-16s %8.3f %8.3f %8.3f", stage, avg, p50, p95)
        loop_elapsed = phase_times.get("period_loop", 0.0)
        periods_per_sec = n_periods_ok / max(loop_elapsed, 1e-6)
        log.info(
            "  period_loop total=%.1fs  %.3f periods/sec",
            loop_elapsed, periods_per_sec,
        )


if __name__ == "__main__":
    main()

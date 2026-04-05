#!/usr/bin/env python3
"""
arc_backfill_worker.py — GPU worker for patent backfill embedding.

Lists files in R2 backfill/input/, processes the assigned range (--start to --end),
uploads embeddings + kNN results to R2 backfill/output/.

No DB, no job queue — just file ranges assigned at launch time.

Usage:
    # Process files 0-99:
    python3 arc_backfill_worker.py --start 0 --end 100

    # Process all files:
    python3 arc_backfill_worker.py --start 0 --end 9999

    # Local test with one file:
    python3 arc_backfill_worker.py --start 0 --end 1 --work-dir /tmp/test

Environment:
    R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging

csv.field_size_limit(10_000_000)  # 10MB — some patent abstracts exceed default 128KB
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import boto3
import numpy as np

# ─── Constants ───────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MAX_SEQ_LENGTH = 512
DEFAULT_BATCH_SIZE = 0  # 0 = auto-detect based on GPU VRAM
KNN_K = 16
R2_INPUT_PREFIX = "backfill/input/"
R2_OUTPUT_PREFIX = "backfill/output/"
R2_CLAIMED_PREFIX = "backfill/claimed/"
CLAIM_STALE_SEC = 1800  # 30 minutes — claims older than this are stale

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [backfill] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("backfill")

# Force line-buffered stdout
sys.stdout.reconfigure(line_buffering=True)

# ─── Graceful shutdown ──────────────────────────────────────────────────────

_shutdown = False

def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    log.info("Shutdown signal received — finishing current file")

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# ─── R2 helpers ──────────────────────────────────────────────────────────────

def make_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"],
    )


def list_r2_input_files(s3, bucket: str) -> list[tuple[str, int]]:
    """List all .tsv.gz files under backfill/input/. Returns [(key, size_bytes)]."""
    files = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=R2_INPUT_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".tsv.gz"):
                files.append((key, obj["Size"]))
    files.sort(key=lambda x: x[0])
    return files


def r2_key_exists(s3, bucket: str, key: str) -> bool:
    """Check if an R2 key exists (for skip-if-done)."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def claim_file(s3, bucket: str, filename: str, worker_id: str) -> bool:
    """Try to claim a file for processing. Returns True if claimed.
    Checks if claim exists and is fresh (< CLAIM_STALE_SEC old).
    If stale or missing, writes our claim and returns True."""
    claim_key = R2_CLAIMED_PREFIX + filename
    try:
        resp = s3.head_object(Bucket=bucket, Key=claim_key)
        age = (time.time() - resp["LastModified"].timestamp())
        if age < CLAIM_STALE_SEC:
            return False  # another worker has it, still fresh
        log.info("Stale claim on %s (%.0fs old) — reclaiming", filename, age)
    except s3.exceptions.ClientError:
        pass  # no claim exists

    # Write our claim
    body = f"{worker_id}\n{time.time()}\n".encode()
    s3.put_object(Bucket=bucket, Key=claim_key, Body=body)
    return True


def release_claim(s3, bucket: str, filename: str):
    """Remove claim marker after successful processing."""
    s3.delete_object(Bucket=bucket, Key=R2_CLAIMED_PREFIX + filename)


def is_done(s3, bucket: str, filename: str) -> bool:
    """Check if output already exists for this file."""
    return r2_key_exists(s3, bucket, R2_OUTPUT_PREFIX + filename + "/embeddings.tsv.gz")


BYTES_PER_DOC = 250  # conservative estimate for compressed TSV

def find_next_file(s3, bucket: str, worker_id: str,
                   max_docs: int = 0) -> str | None:
    """Scan R2 input files, return first unclaimed+undone file within size limit, or None.
    max_docs=0 means no limit."""
    max_size = max_docs * BYTES_PER_DOC if max_docs > 0 else 0
    all_files = list_r2_input_files(s3, bucket)
    for r2_key, size_bytes in all_files:
        filename = r2_key.split("/")[-1]
        if max_size > 0 and size_bytes > max_size:
            continue  # too big for this worker
        if is_done(s3, bucket, filename):
            continue
        if claim_file(s3, bucket, filename, worker_id):
            return r2_key
    return None


# ─── Embedding ───────────────────────────────────────────────────────────────

def load_model(model_name: str, max_seq_length: int):
    from sentence_transformers import SentenceTransformer
    log.info("Loading model: %s (max_seq_length=%d)", model_name, max_seq_length)
    t0 = time.monotonic()
    model = SentenceTransformer(model_name, model_kwargs={"torch_dtype": "float16"})
    # Cap sequence length — Qwen3 defaults to 32768 which causes OOM
    _default = model.max_seq_length
    model.max_seq_length = max_seq_length
    # Also set on tokenizer to ensure truncation
    if hasattr(model.tokenizer, 'model_max_length'):
        model.tokenizer.model_max_length = max_seq_length
    log.info("Model loaded in %.1fs (max_seq_length=%d, was %d, dim=%d, dtype=fp16)",
             time.monotonic() - t0, max_seq_length, _default,
             model.get_sentence_embedding_dimension())
    return model


def _auto_batch_size() -> int:
    """Pick batch size based on GPU VRAM. Conservative to avoid OOM."""
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            if vram_gb >= 20:
                return 256
            elif vram_gb >= 14:
                return 128
            elif vram_gb >= 10:
                return 96
            else:
                return 64
    except Exception:
        pass
    return 64


def embed_texts(model, texts: list[str], batch_size: int) -> np.ndarray:
    if batch_size <= 0:
        batch_size = _auto_batch_size()
    log.info("Embedding %d texts (batch_size=%d)...", len(texts), batch_size)
    t0 = time.monotonic()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    elapsed = time.monotonic() - t0
    log.info("Embedded %d texts in %.1fs (%.0f texts/sec), dim=%d",
             len(texts), elapsed, len(texts) / max(elapsed, 1e-6), emb.shape[1])
    return emb.astype(np.float32)


# ─── kNN ─────────────────────────────────────────────────────────────────────

def _gpu_compute_capability() -> float:
    """Return GPU compute capability (e.g. 7.5) or 0.0 if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            return float(f"{cap[0]}.{cap[1]}")
    except Exception:
        pass
    return 0.0


def build_faiss_index(dim: int):
    """Build FAISS IndexFlatIP — GPU if compute capability >= 7.0, else CPU.
    faiss-gpu-cu12 requires Volta (7.0) or newer; Pascal (6.x) crashes with
    CUDA error 209 (no kernel image) which is an unrecoverable C++ abort."""
    import faiss
    cc = _gpu_compute_capability()
    if cc >= 7.0:
        try:
            res = faiss.StandardGpuResources()
            cpu_index = faiss.IndexFlatIP(dim)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            log.info("FAISS: GPU IndexFlatIP (dim=%d, cc=%.1f)", dim, cc)
            return gpu_index
        except (AttributeError, RuntimeError) as exc:
            log.info("FAISS GPU init failed (%s) — CPU fallback", exc)
    else:
        log.info("FAISS: GPU compute capability %.1f < 7.0 — using CPU", cc)
    return faiss.IndexFlatIP(dim)


def run_knn(doc_ids: list[str], embeddings: np.ndarray, k: int) -> list[tuple]:
    n, dim = embeddings.shape
    log.info("Running kNN: %d docs, k=%d", n, k)
    t0 = time.monotonic()

    emb = embeddings.copy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    emb /= norms

    index = build_faiss_index(dim)
    index.add(emb)

    actual_k = min(k + 1, n)
    distances, nbr_indices = index.search(emb, actual_k)

    results = []
    for row_i, (dists, nbrs) in enumerate(zip(distances, nbr_indices)):
        src_id = doc_ids[row_i]
        rank = 0
        for dist, nbr_idx in zip(dists, nbrs):
            if nbr_idx == -1 or nbr_idx == row_i:
                continue
            rank += 1
            results.append((src_id, doc_ids[int(nbr_idx)], float(dist), rank))

    elapsed = time.monotonic() - t0
    log.info("kNN complete: %d docs → %d edges in %.1fs", n, len(results), elapsed)
    return results


# ─── Leiden clustering ───────────────────────────────────────────────────────

def run_leiden(
    doc_ids: list[str],
    knn_edges: list[tuple],
    leiden_res: float = 1.0,
    leiden_seed: int = 42,
) -> tuple[dict[str, int], float, int]:
    """
    Build undirected igraph from kNN edges, run Leiden with RBConfiguration.
    Returns (cluster_map, modularity, n_clusters).
    cluster_map: {doc_id: integer_cluster_id}

    Inlined from arc_cloud_run.py run_leiden (line 459).
    """
    import igraph as ig
    import leidenalg

    log.info("Running Leiden: %d docs, res=%.2f", len(doc_ids), leiden_res)
    t0 = time.monotonic()

    doc_to_idx = {did: i for i, did in enumerate(doc_ids)}

    # Deduplicate edges: take max similarity for each undirected pair
    best_weight: dict[tuple[int, int], float] = {}
    for src, dst, sim, _rank in knn_edges:
        si = doc_to_idx.get(src)
        di = doc_to_idx.get(dst)
        if si is None or di is None or si == di:
            continue
        key = (min(si, di), max(si, di))
        if sim > best_weight.get(key, -1.0):
            best_weight[key] = sim

    ig_edges = list(best_weight.keys())
    weights = [max(0.0, best_weight[e]) for e in ig_edges]

    g = ig.Graph(n=len(doc_ids), edges=ig_edges, directed=False)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=leiden_res,
        seed=leiden_seed,
        weights="weight",
    )

    cluster_map = {
        doc_ids[i]: int(partition.membership[i])
        for i in range(len(doc_ids))
    }
    n_clusters = len(set(cluster_map.values()))
    modularity = float(partition.modularity)

    elapsed = time.monotonic() - t0
    log.info("Leiden complete: %d docs → %d clusters in %.1fs (modularity=%.4f)",
             len(doc_ids), n_clusters, elapsed, modularity)
    return cluster_map, modularity, n_clusters


# ─── TSV I/O ─────────────────────────────────────────────────────────────────

def read_input_tsv(filepath: Path) -> tuple[list[str], list[str], bool]:
    """Read input TSV.gz via zcat (handles concatenated gzip streams).
    Returns (doc_ids, texts, is_cpc)."""
    doc_ids = []
    texts = []
    n_empty = 0
    is_cpc = False

    proc = subprocess.Popen(
        ["zcat", str(filepath)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    reader = csv.DictReader(
        (line.decode("utf-8", errors="replace") for line in proc.stdout),
        delimiter="\t",
    )
    for row in reader:
        text = (row.get("text_content") or "").strip()
        if not text:
            n_empty += 1
            continue
        doc_ids.append(row["doc_id"])
        texts.append(text)
        if row.get("primary_cpc"):
            is_cpc = True

    proc.wait()

    if n_empty > 0:
        log.info("Dropped %d rows with empty text_content", n_empty)
    log.info("Read %d docs from %s (type=%s)", len(doc_ids), filepath.name,
             "cpc" if is_cpc else "ipc_only")
    return doc_ids, texts, is_cpc


def write_embeddings_tsv_gz(doc_ids: list[str], embeddings: np.ndarray, path: Path):
    with gzip.open(path, "wt", compresslevel=6) as f:
        for doc_id, emb_row in zip(doc_ids, embeddings):
            floats = " ".join(f"{v:.6f}" for v in emb_row)
            f.write(f"{doc_id}\t{floats}\n")
    sz = path.stat().st_size / 1_048_576
    log.info("Wrote embeddings: %s (%.1f MB)", path.name, sz)


def write_knn_tsv_gz(knn_rows: list[tuple], path: Path):
    with gzip.open(path, "wt", compresslevel=6) as f:
        for doc_id, nbr_id, dist, rank in knn_rows:
            f.write(f"{doc_id}\t{nbr_id}\t{dist:.6f}\t{rank}\n")
    sz = path.stat().st_size / 1_048_576
    log.info("Wrote kNN: %s (%.1f MB)", path.name, sz)


def write_clusters_tsv_gz(cluster_map: dict[str, int], modularity: float, path: Path):
    """Write cluster assignments as TSV.gz: doc_id<TAB>cluster_id"""
    with gzip.open(path, "wt", compresslevel=6) as f:
        f.write(f"# modularity={modularity:.6f}\n")
        for doc_id, cluster_id in sorted(cluster_map.items()):
            f.write(f"{doc_id}\t{cluster_id}\n")
    sz = path.stat().st_size / 1_048_576
    log.info("Wrote clusters: %s (%.1f MB)", path.name, sz)


# ─── Process one file ───────────────────────────────────────────────────────

def process_file(
    r2_key: str,
    model,
    s3,
    bucket: str,
    work_dir: Path,
    batch_size: int,
) -> str:
    """Process a single input file. Returns summary string."""
    filename = r2_key.split("/")[-1]
    log.info("Processing: %s", filename)
    t0 = time.monotonic()

    # Download
    work_dir.mkdir(parents=True, exist_ok=True)
    local_input = work_dir / filename
    s3.download_file(bucket, r2_key, str(local_input))
    log.info("Downloaded %s (%.1f MB)", filename,
             local_input.stat().st_size / 1_048_576)

    # Read
    doc_ids, texts, is_cpc = read_input_tsv(local_input)
    if len(texts) == 0:
        local_input.unlink(missing_ok=True)
        return "0 docs (all empty) — skipped"

    # Embed
    embeddings = embed_texts(model, texts, batch_size)

    # Write + upload embeddings
    emb_path = work_dir / f"{filename}.embeddings.tsv.gz"
    write_embeddings_tsv_gz(doc_ids, embeddings, emb_path)
    emb_r2_key = R2_OUTPUT_PREFIX + filename + "/embeddings.tsv.gz"
    s3.upload_file(str(emb_path), bucket, emb_r2_key)
    log.info("Uploaded → %s", emb_r2_key)

    # kNN + Leiden for CPC files
    knn_summary = ""
    if is_cpc and len(doc_ids) >= 2:
        # kNN
        knn_rows = run_knn(doc_ids, embeddings, KNN_K)
        knn_path = work_dir / f"{filename}.knn.tsv.gz"
        write_knn_tsv_gz(knn_rows, knn_path)
        knn_r2_key = R2_OUTPUT_PREFIX + filename + "/knn.tsv.gz"
        s3.upload_file(str(knn_path), bucket, knn_r2_key)
        log.info("Uploaded → %s", knn_r2_key)
        knn_path.unlink(missing_ok=True)

        # Leiden clustering
        cluster_map, modularity, n_clusters = run_leiden(doc_ids, knn_rows)
        clust_path = work_dir / f"{filename}.clusters.tsv.gz"
        write_clusters_tsv_gz(cluster_map, modularity, clust_path)
        clust_r2_key = R2_OUTPUT_PREFIX + filename + "/clusters.tsv.gz"
        s3.upload_file(str(clust_path), bucket, clust_r2_key)
        log.info("Uploaded → %s", clust_r2_key)
        clust_path.unlink(missing_ok=True)

        knn_summary = f", {len(knn_rows)} kNN edges, {n_clusters} clusters (mod={modularity:.3f})"

    # Cleanup
    local_input.unlink(missing_ok=True)
    emb_path.unlink(missing_ok=True)

    elapsed = time.monotonic() - t0
    return (f"{len(doc_ids)} docs in {elapsed:.1f}s "
            f"({len(doc_ids)/max(elapsed,1e-6):.0f} docs/s{knn_summary})")


# ─── Main ────────────────────────────────────────────────────────────────────

def run_range_mode(args, s3, bucket, model, worker_id):
    """Process a fixed range of files [start, end)."""
    all_files = list_r2_input_files(s3, bucket)
    log.info("Total files in R2: %d", len(all_files))

    end = min(args.end, len(all_files))
    my_files = all_files[args.start:end]
    log.info("Assigned range: [%d, %d) → %d files", args.start, end, len(my_files))

    if not my_files:
        log.info("No files to process — exiting")
        return 0, 0, 0

    max_size = args.max_docs * BYTES_PER_DOC if args.max_docs > 0 else 0

    n_done = 0
    n_skipped = 0
    n_failed = 0
    t_start = time.time()

    for i, (r2_key, size_bytes) in enumerate(my_files):
        if _shutdown:
            log.info("Shutdown requested — stopping after %d files", n_done)
            break

        filename = r2_key.split("/")[-1]

        if max_size > 0 and size_bytes > max_size:
            log.info("[%d/%d] SKIP %s (too large: %.1f MB, max_docs=%d)",
                     i + 1, len(my_files), filename, size_bytes / 1e6, args.max_docs)
            n_skipped += 1
            continue

        if is_done(s3, bucket, filename):
            log.info("[%d/%d] SKIP %s (output exists)", i + 1, len(my_files), filename)
            n_skipped += 1
            continue

        log.info("[%d/%d] Processing %s", i + 1, len(my_files), filename)
        try:
            summary = process_file(r2_key, model, s3, bucket, args.work_dir, args.batch_size)
            n_done += 1
            elapsed_total = time.time() - t_start
            remaining = len(my_files) - (i + 1)
            rate = (i + 1) / elapsed_total if elapsed_total > 0 else 0
            eta_min = remaining / rate / 60 if rate > 0 else 0
            log.info("[%d/%d] Done: %s | ETA %.0fm", i + 1, len(my_files), summary, eta_min)
        except Exception as e:
            n_failed += 1
            log.error("[%d/%d] FAILED %s: %s", i + 1, len(my_files), filename, e,
                      exc_info=True)

    return n_done, n_skipped, n_failed


def run_auto_mode(args, s3, bucket, model, worker_id):
    """Auto mode: claim unclaimed files from R2, process until none left."""
    n_done = 0
    n_failed = 0
    idle_passes = 0
    max_docs = args.max_docs
    t_start = time.time()

    while not _shutdown:
        r2_key = find_next_file(s3, bucket, worker_id, max_docs=max_docs)

        if r2_key is None:
            idle_passes += 1
            max_idle = args.idle_passes
            if idle_passes >= max_idle:
                log.info("No unclaimed files after %d passes — all done", idle_passes)
                break
            log.info("No unclaimed files — waiting 60s (pass %d/%d)", idle_passes, max_idle)
            time.sleep(60)
            continue

        idle_passes = 0
        filename = r2_key.split("/")[-1]
        log.info("[auto] Processing %s (done=%d)", filename, n_done)

        try:
            summary = process_file(r2_key, model, s3, bucket, args.work_dir, args.batch_size)
            release_claim(s3, bucket, filename)
            n_done += 1
            elapsed = time.time() - t_start
            log.info("[auto] Done: %s | total=%d, %.1f files/hr",
                     summary, n_done, n_done / (elapsed / 3600))
        except Exception as e:
            n_failed += 1
            release_claim(s3, bucket, filename)
            log.error("[auto] FAILED %s: %s", filename, e, exc_info=True)

    return n_done, 0, n_failed


def main():
    ap = argparse.ArgumentParser(description="Patent backfill embedding worker")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--auto", action="store_true",
                      help="Auto mode: claim unclaimed files from R2, self-coordinate with other workers")
    mode.add_argument("--start", type=int,
                      help="Range mode: start index (inclusive)")
    ap.add_argument("--end", type=int, help="Range mode: end index (exclusive)")
    ap.add_argument("--max-docs", type=int, default=0,
                    help="Skip files estimated to have more than N docs (0 = no limit)")
    ap.add_argument("--idle-passes", type=int, default=20,
                    help="Exit after N idle passes with no work (default: 20, ~20 min)")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--work-dir", type=Path, default=Path("/workspace/arc/backfill"))
    args = ap.parse_args()

    if args.start is not None and args.end is None:
        ap.error("--end is required with --start")

    worker_id = os.environ.get("WORKER_ID", socket.gethostname())
    bucket = os.environ.get("R2_BUCKET", "arc-cloud")
    s3 = make_r2_client()

    log.info("Worker starting: id=%s mode=%s batch_size=%d",
             worker_id, "auto" if args.auto else f"range [{args.start},{args.end})",
             args.batch_size)

    # Load model once
    model = load_model(MODEL_NAME, MAX_SEQ_LENGTH)

    t_start = time.time()
    if args.auto:
        n_done, n_skipped, n_failed = run_auto_mode(args, s3, bucket, model, worker_id)
    else:
        n_done, n_skipped, n_failed = run_range_mode(args, s3, bucket, model, worker_id)

    elapsed = time.time() - t_start
    log.info("\n%s", "=" * 60)
    log.info("WORKER COMPLETE in %.1fm", elapsed / 60)
    log.info("  Done:    %d", n_done)
    log.info("  Skipped: %d", n_skipped)
    log.info("  Failed:  %d", n_failed)
    log.info("%s", "=" * 60)


if __name__ == "__main__":
    main()

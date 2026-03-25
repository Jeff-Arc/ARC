#!/usr/bin/env python3
"""
arc_cloud_orchestrate.py — Vast.ai GPU orchestrator for ARC cloud pipeline

Finds the cheapest qualifying GPU on vast.ai, provisions it, transfers data via
Cloudflare R2 object storage, runs arc_cloud_run.py, uploads results back to R2,
downloads locally, imports into arc_v5, then destroys the instance.

Data flow:
    local cloud_in/ ──→ R2  ──→ vast.ai instance  (input)
    vast.ai instance ──→ R2 ──→ local cloud_out_*/ (output)

Usage:
    python3 arc_cloud_orchestrate.py --corpus G06N_quarterly
    python3 arc_cloud_orchestrate.py --corpus all --max-price 0.12
    python3 arc_cloud_orchestrate.py --corpus G06N_quarterly --dry-run
    python3 arc_cloud_orchestrate.py --corpus G06N_quarterly --yes
    python3 arc_cloud_orchestrate.py --upload-to-r2          # seed R2 from local data

Flags:
    --corpus CORPUS_ID   corpus to run, or 'all' to run every corpus with cloud_in data
    --max-price FLOAT    max $/hr (default 0.15)
    --gpu-min-vram INT   minimum GPU VRAM in GB (default 12)
    --dry-run            find best GPU and print R2/run plan, then exit
    --yes                skip confirmation prompt
    --upload-to-r2       upload all local cloud_in data to R2 without running a job

GPU selection (in priority order):
    1. Minimum VRAM >= --gpu-min-vram (GB)
    2. CUDA >= 12.0
    3. Host reliability > 0.95
    4. Cheapest $/hr that meets above
    5. Prefer 16GB+ VRAM if price delta < $0.04/hr vs cheapest qualifying option

R2 key layout:
    {corpus_id}/chunks_{corpus_id}.tsv          input chunks (always required)
    {corpus_id}/config.json                     corpus config
    {corpus_id}/embeddings_{corpus_id}.npz      cached embeddings (skip embed if present)
    {corpus_id}/output/import/{corpus_id}/*.tsv pipeline output TSVs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# ─── Paths ────────────────────────────────────────────────────────────────────

ARC_DIR      = Path("/home/jeff/arc")
DATA_DIR     = ARC_DIR / "data"
CLOUD_IN_DIR = DATA_DIR / "cloud_in"
ENV_FILE     = ARC_DIR / ".env"

VASTAI_BIN   = "/home/jeff/miniconda3/bin/vastai"
PYTHON_BIN   = "/home/jeff/miniconda3/bin/python3"

# Docker image — pytorch base with CUDA 12.4 runtime
PYTORCH_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"

REMOTE_ROOT  = "/workspace/arc"
REMOTE_IN    = f"{REMOTE_ROOT}/cloud_in"
REMOTE_OUT   = f"{REMOTE_ROOT}/cloud_out"
REMOTE_CREDS = "/root/.r2creds"          # written once at instance setup

# ── Runtime model (calibrated against actual RTX 4090 runs) ───────────────────
# Actuals: H01L 350K=120min, G01N 60K=30min, C30B 15K=10min  (RTX 4090)
# Cloud RTX 3090 is ~1.1× slower → multiply by 1.1
# Formula: linear scale from H01L anchor, 10-min floor
_ANCHOR_CHUNKS   = 350_000
_ANCHOR_MINUTES  = 120
_CLOUD_SLOWDOWN  = 1.1
_MIN_RUNTIME_MIN = 10


# ─── Logging ──────────────────────────────────────────────────────────────────

def _setup_logging(label: str) -> logging.Logger:
    log_file = DATA_DIR / f"{label}_orchestrate.log"
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(fmt, datefmt))
        root.addHandler(ch)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        root.addHandler(fh)
    return logging.getLogger("orchestrate")


# ─── R2 / S3 helpers ──────────────────────────────────────────────────────────

def load_env(path: Path = ENV_FILE) -> dict[str, str]:
    """
    Parse a shell-style .env file, returning a dict of key→value pairs.
    Handles 'export KEY=VALUE' and 'KEY=VALUE' lines; strips quotes.
    """
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export").strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip().strip("'\"")
    return env


def r2_config() -> dict[str, str]:
    """
    Load R2 credentials from .env, falling back to environment variables.
    Returns dict with keys: endpoint, access_key, secret_key, bucket.
    Raises RuntimeError if any credential is missing.
    """
    env = {**load_env(), **os.environ}   # env vars take priority over .env
    missing = []
    cfg: dict[str, str] = {}
    for env_key, cfg_key in [
        ("R2_ENDPOINT",   "endpoint"),
        ("R2_ACCESS_KEY", "access_key"),
        ("R2_SECRET_KEY", "secret_key"),
        ("R2_BUCKET",     "bucket"),
    ]:
        val = env.get(env_key, "")
        if not val:
            missing.append(env_key)
        cfg[cfg_key] = val
    if missing:
        raise RuntimeError(
            f"Missing R2 credentials in .env or environment: {', '.join(missing)}"
        )
    return cfg


def make_r2_client(cfg: dict[str, str]):
    """Create a boto3 S3 client pointed at Cloudflare R2."""
    try:
        import boto3
    except ImportError:
        raise RuntimeError("boto3 not installed — run: pip install boto3")
    return boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"],
    )


# ── R2 key layout ──────────────────────────────────────────────────────────────

def _r2_key_chunks(corpus_id: str) -> str:
    return f"{corpus_id}/chunks_{corpus_id}.tsv"

def _r2_key_config(corpus_id: str) -> str:
    return f"{corpus_id}/config.json"

def _r2_key_embeddings(corpus_id: str) -> str:
    return f"{corpus_id}/embeddings_{corpus_id}.npz"

def _r2_key_output(corpus_id: str, filename: str) -> str:
    return f"{corpus_id}/output/import/{corpus_id}/{filename}"


# ── Upload / download helpers ──────────────────────────────────────────────────

def r2_upload_file(s3, bucket: str, local_path: Path, key: str,
                   log: logging.Logger) -> None:
    """Upload a local file to R2. Logs size and key."""
    size_mb = local_path.stat().st_size / 1_048_576
    log.info("R2 upload  %s  →  %s  (%.1f MB)", local_path.name, key, size_mb)
    s3.upload_file(str(local_path), bucket, key)


def r2_key_exists(s3, bucket: str, key: str) -> bool:
    """Return True if key exists in R2 bucket."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def upload_corpus_inputs(corpus_id: str, s3, cfg: dict,
                         log: logging.Logger) -> bool:
    """
    Upload chunks TSV, config.json, and (if present) embeddings.npz for a corpus.
    Returns True if embeddings were uploaded (incremental run possible).
    """
    bucket = cfg["bucket"]
    corpus_dir = CLOUD_IN_DIR / corpus_id

    # chunks TSV — required
    chunks_path = corpus_dir / f"chunks_{corpus_id}.tsv"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"chunks TSV not found: {chunks_path}\n"
            "Run arc_ingest.py to generate it before orchestrating."
        )
    r2_upload_file(s3, bucket, chunks_path, _r2_key_chunks(corpus_id), log)

    # config.json — required
    config_path = corpus_dir / "config.json"
    if config_path.exists():
        r2_upload_file(s3, bucket, config_path, _r2_key_config(corpus_id), log)
    else:
        log.warning("No config.json in %s — will use arc_cloud_run.py defaults", corpus_dir)

    # embeddings.npz — optional (enables skip-embed incremental mode)
    emb_path = corpus_dir / f"embeddings_{corpus_id}.npz"
    if emb_path.exists():
        r2_upload_file(s3, bucket, emb_path, _r2_key_embeddings(corpus_id), log)
        log.info("Embeddings uploaded — instance will run in incremental mode")
        return True
    else:
        log.info("No embeddings.npz — instance will run full embed")
        return False


def download_corpus_outputs(corpus_id: str, s3, cfg: dict,
                             log: logging.Logger) -> Path:
    """
    Download all output TSVs from R2 for a corpus to the local cloud_out_ directory.
    Returns the local output root path.
    """
    bucket    = cfg["bucket"]
    local_out = DATA_DIR / f"cloud_out_{corpus_id}"
    tsv_dir   = local_out / "import" / corpus_id
    tsv_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{corpus_id}/output/import/{corpus_id}/"
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    n_files = 0
    for page in pages:
        for obj in page.get("Contents", []):
            key      = obj["Key"]
            filename = Path(key).name
            dest     = tsv_dir / filename
            size_mb  = obj["Size"] / 1_048_576
            log.info("R2 download  %s  →  %s  (%.1f MB)", key, dest.name, size_mb)
            s3.download_file(bucket, key, str(dest))
            n_files += 1

    if n_files == 0:
        raise RuntimeError(
            f"No output TSVs found in R2 under prefix '{prefix}' — "
            "pipeline may have failed silently"
        )
    log.info("Downloaded %d output file(s) for %s", n_files, corpus_id)
    return local_out


def upload_all_to_r2(log: logging.Logger) -> tuple[int, float]:
    """
    Seed R2 with all local cloud_in data (chunks TSVs + embeddings.npz files).
    Used for initial setup or re-seeding.  Returns (n_files, total_bytes).
    """
    cfg = r2_config()
    s3  = make_r2_client(cfg)
    corpora = list_ready_corpora()

    if not corpora:
        log.warning("No corpora found in %s — nothing to upload", CLOUD_IN_DIR)
        return 0, 0.0

    total_files = 0
    total_bytes = 0.0

    for corpus_id in corpora:
        log.info("── Uploading inputs for %s ──", corpus_id)
        corpus_dir = CLOUD_IN_DIR / corpus_id
        for fname, key_fn in [
            (f"chunks_{corpus_id}.tsv",    _r2_key_chunks),
            ("config.json",                _r2_key_config),
            (f"embeddings_{corpus_id}.npz", _r2_key_embeddings),
        ]:
            local_path = corpus_dir / fname
            if not local_path.exists():
                continue
            key = key_fn(corpus_id)
            r2_upload_file(s3, cfg["bucket"], local_path, key, log)
            total_bytes += local_path.stat().st_size
            total_files += 1

    total_mb = total_bytes / 1_048_576
    log.info("Upload complete: %d files, %.1f MB total", total_files, total_mb)
    return total_files, total_mb


# ─── Vast.ai CLI wrapper ───────────────────────────────────────────────────────

def _vastai(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a vastai CLI command, return CompletedProcess."""
    return subprocess.run([VASTAI_BIN, *args],
                          capture_output=True, text=True, check=check)


def search_offers(min_vram_gb: int = 12, max_price: float = 0.15) -> list[dict]:
    """
    Search vast.ai for GPU offers matching criteria.

    Query is built from user-specified minimums; reliability and price are
    post-filtered in Python so we can apply richer logic (reliability2 field).

    Returns list of matching offers sorted by dph_total ascending.
    """
    # vast.ai query language: gpu_ram in GB (multiplied ×1000 internally → MB)
    query = (
        f"gpu_ram>={min_vram_gb} "
        "inet_up>200 "
        "disk_space>60 "
        "cuda_vers>=12.0 "
        "reliability>0.95"
    )
    result = _vastai("search", "offers", query,
                     "--order", "dph_total",
                     "--limit", "100",
                     "--raw")
    try:
        offers = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse vastai output: {exc}\n{result.stdout[:500]}")

    # Post-filter: reliability2 > 0.95, dph_total within budget,
    # confirmed VRAM threshold (response uses MB, ~1000MB per GB).
    filtered = [
        o for o in offers
        if o.get("reliability2", o.get("reliability", 0)) > 0.95
        and o.get("dph_total", 999) <= max_price
        and o.get("gpu_ram", 0) >= min_vram_gb * 1000
    ]
    filtered.sort(key=lambda o: o["dph_total"])
    return filtered


def get_min_vram(chunk_count: int) -> int:
    """
    Compute minimum GPU VRAM (GB) needed for a corpus based on chunk count.

    Memory model (Qwen3-0.6B embeddings):
        weights_gb   = 1.2 GB   (model weights in FP16)
        embedding_gb = chunk_count × 1024 dims × 4 bytes  (float32 output matrix)
        batch_gb     = 0.5 GB   (activations at batch_size=16, max_seq_length=512)
        total        = (weights + embedding + batch) × 2.0 safety margin

    The safety margin covers: kNN index build (FAISS IndexFlatIP copies the
    matrix), Leiden graph construction, and OS/framework overhead.
    """
    weights_gb   = 1.2
    embedding_gb = (chunk_count * 1024 * 4) / 1e9
    batch_gb     = 0.5
    total_gb     = (weights_gb + embedding_gb + batch_gb) * 2.0

    # Round up to nearest standard VRAM tier
    if total_gb <= 8:
        return 8
    elif total_gb <= 12:
        return 12
    elif total_gb <= 16:
        return 16
    elif total_gb <= 24:
        return 24
    else:
        return 40


def select_best_offer(offers: list[dict], min_vram_gb: int) -> dict:
    """
    Select the cheapest offer whose VRAM meets min_vram_gb.
    No premium logic — just cheapest viable option.
    """
    if not offers:
        raise RuntimeError("No qualifying GPU offers found.")
    return offers[0]


def print_top_offers(offers: list[dict], selected: dict, n: int = 5) -> None:
    """Print a formatted table of the top N offers."""
    sep = "─" * 84
    print(f"\n{sep}")
    print(f"  {'Rank':<4}  {'ID':<12}  {'GPU':<22}  {'VRAM':>5}  {'$/hr':>7}  "
          f"{'Reliability':>11}  {'CUDA':>5}")
    print(sep)
    for i, o in enumerate(offers[:n]):
        vram = round(o["gpu_ram"] / 1000)
        rel  = o.get("reliability2", o.get("reliability", 0))
        mark = "  ◄ selected" if o["id"] == selected["id"] else ""
        print(f"  {i+1:<4}  {o['id']:<12}  {o['gpu_name']:<22}  {vram:>4}G  "
              f"  ${o['dph_total']:>5.4f}  {rel:>11.4f}  {o['cuda_max_good']:>5}{mark}")
    print(sep)


# ─── Corpus helpers ───────────────────────────────────────────────────────────

def list_ready_corpora() -> list[str]:
    """Return corpus IDs that have cloud_in data prepared (config.json present)."""
    if not CLOUD_IN_DIR.exists():
        return []
    return sorted(
        d.name for d in CLOUD_IN_DIR.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    )


def count_chunks(corpus_id: str) -> int:
    """Count lines in the chunks TSV for the corpus."""
    tsv = CLOUD_IN_DIR / corpus_id / f"chunks_{corpus_id}.tsv"
    if not tsv.exists():
        return 0
    r = subprocess.run(["wc", "-l", str(tsv)], capture_output=True, text=True)
    try:
        return int(r.stdout.split()[0])
    except (IndexError, ValueError):
        return 0


def get_period_count(corpus_id: str) -> int:
    """
    Return number of periods for this corpus.

    Primary source: arc_v5.cloud_f_period (actual run history).
    Fallback: estimate from config.json year_from + resolution (for not-yet-run corpora).
    """
    try:
        r = subprocess.run(
            ["psql",
             "-h", "/var/run/postgresql", "-U", "jeff", "-d", "arc_v5",
             "-Atc",
             f"SELECT COUNT(*) FROM cloud_f_period WHERE corpus_id = '{corpus_id}'"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            n = int(r.stdout.strip())
            if n > 0:
                return n
    except Exception:
        pass

    # Fallback: estimate from config.json
    config_path = CLOUD_IN_DIR / corpus_id / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            year_from  = cfg.get("year_from", 2000)
            resolution = cfg.get("resolution", "quarterly")
            periods_per_year = {"quarterly": 4, "annual": 1, "biannual": 2}.get(resolution, 4)
            years = max(1, 2025 - year_from)
            return years * periods_per_year
        except Exception:
            pass

    return 100  # safe default


def estimate_runtime_minutes(chunk_count: int) -> float:
    """
    Estimate wall-clock minutes for one corpus run on a cloud RTX 3090.

    Calibrated against actual RTX 4090 runs (cloud 3090 is ~1.1× slower):
        H01L  350K chunks → 120 min  (anchor)
        G01N   60K chunks →  30 min
        C30B   15K chunks →  10 min

    Linear scale from the 350K anchor, 10-minute floor.
    """
    base = (chunk_count / _ANCHOR_CHUNKS) * _ANCHOR_MINUTES * _CLOUD_SLOWDOWN
    return max(base, _MIN_RUNTIME_MIN)


def estimate_cost(corpus_id: str, price_per_hr: float,
                  period_count: Optional[int] = None) -> tuple[float, float]:
    """Estimate (hours, cost) for a corpus run."""
    n_chunks = count_chunks(corpus_id)
    if n_chunks == 0:
        return _MIN_RUNTIME_MIN / 60, (_MIN_RUNTIME_MIN / 60) * price_per_hr
    hours = estimate_runtime_minutes(n_chunks) / 60
    return hours, hours * price_per_hr


# ─── SSH helpers ──────────────────────────────────────────────────────────────

_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=6",
    "-o", "LogLevel=ERROR",
]


def _ssh_base(host: str, port: int) -> list[str]:
    return ["ssh", *_SSH_OPTS, "-p", str(port), f"root@{host}"]


def ssh_run(host: str, port: int, command: str,
            timeout: Optional[int] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a remote command via SSH."""
    return subprocess.run(
        [*_ssh_base(host, port), command],
        timeout=timeout,
        check=check,
    )


def rsync_to(local: str | Path, host: str, port: int, remote: str,
             log: logging.Logger) -> None:
    """Rsync a file or directory from local to remote instance."""
    cmd = [
        "rsync", "-azP",
        "-e", f"ssh -p {port} " + " ".join(_SSH_OPTS),
        str(local),
        f"root@{host}:{remote}",
    ]
    log.info("rsync → instance:%s", remote)
    subprocess.run(cmd, check=True)


# ─── Instance lifecycle ───────────────────────────────────────────────────────

def create_instance(offer_id: int, log: logging.Logger) -> int:
    """
    Create a vast.ai instance.  Returns the new instance ID.
    Launched with --ssh --direct; no onstart script (deps installed via SSH).
    """
    log.info("Creating instance from offer %d  (image: %s)...", offer_id, PYTORCH_IMAGE)
    result = _vastai(
        "create", "instance", str(offer_id),
        "--image",  PYTORCH_IMAGE,
        "--disk",   "80",
        "--ssh",
        "--direct",
        "--raw",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"vastai create instance failed (rc={result.returncode}):\n{result.stderr}"
        )
    try:
        resp = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"Could not parse create-instance response:\n{result.stdout[:500]}")

    instance_id = resp.get("new_contract")
    if not instance_id:
        raise RuntimeError(f"No 'new_contract' in response: {resp}")

    log.info("Instance created: id=%d", instance_id)
    return int(instance_id)


def get_instance_info(instance_id: int) -> Optional[dict]:
    """Return current instance dict from vastai show instances, or None."""
    result = _vastai("show", "instances", "--raw", check=False)
    if result.returncode != 0:
        return None
    try:
        instances = json.loads(result.stdout)
        return next((i for i in instances if i["id"] == instance_id), None)
    except (json.JSONDecodeError, TypeError):
        return None


def wait_for_ssh(instance_id: int, log: logging.Logger,
                 timeout: int = 600) -> tuple[str, int]:
    """
    Poll until instance is running and SSH accepts connections.
    Returns (ssh_host, ssh_port).
    """
    log.info("Waiting for instance %d to boot (timeout %ds)...", instance_id, timeout)
    deadline = time.time() + timeout
    poll = 15

    while time.time() < deadline:
        info = get_instance_info(instance_id)
        if info is None:
            time.sleep(poll)
            continue

        status   = info.get("actual_status", info.get("status", "unknown"))
        ssh_host = info.get("ssh_host")
        ssh_port = info.get("ssh_port")
        log.info("  status=%-12s  ssh=%s:%s", status, ssh_host, ssh_port)

        if status == "running" and ssh_host and ssh_port:
            test = subprocess.run(
                [*_ssh_base(ssh_host, int(ssh_port)),
                 "-o", "ConnectTimeout=10",
                 "echo ready"],
                capture_output=True, text=True,
            )
            if test.returncode == 0 and "ready" in test.stdout:
                log.info("SSH ready on %s:%s", ssh_host, ssh_port)
                return ssh_host, int(ssh_port)

        time.sleep(poll)

    raise TimeoutError(f"Instance {instance_id} not SSH-ready after {timeout}s")


def install_deps(host: str, port: int, log: logging.Logger,
                 timeout: int = 900) -> None:
    """
    Install ARC runtime dependencies on the remote instance via SSH.

    Essential: sentence-transformers, faiss-gpu-cu12 (CPU fallback), leidenalg, boto3.
    Optional:  cugraph-cu12, cudf-cu12 (arc_cloud_run.py has CPU fallbacks).
    """
    log.info("Installing ARC dependencies on instance (5–10 min)...")
    script = r"""
set -e
apt-get update -qq
apt-get install -y -qq rsync 2>/dev/null || true

pip install -q --upgrade pip
pip install -q sentence-transformers leidenalg psycopg2-binary boto3

pip install -q faiss-gpu-cu12 2>/dev/null || pip install -q faiss-cpu

pip install -q --extra-index-url https://pypi.nvidia.com \
    cugraph-cu12 cudf-cu12 2>/dev/null \
    && echo "[deps] RAPIDS installed" \
    || echo "[deps] RAPIDS unavailable — CPU fallback active"

echo "[deps] ARC dependencies ready."
"""
    result = subprocess.run(
        [*_ssh_base(host, port), "bash -s"],
        input=script, text=True, timeout=timeout, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Dependency installation failed on remote instance.")
    log.info("Dependencies installed.")


def write_r2_creds(host: str, port: int, cfg: dict[str, str],
                   log: logging.Logger) -> None:
    """
    Write R2 credentials to REMOTE_CREDS on the instance so remote scripts
    can source them without exposing creds in process listings.
    """
    creds_content = (
        f"export R2_ENDPOINT={cfg['endpoint']}\n"
        f"export R2_ACCESS_KEY={cfg['access_key']}\n"
        f"export R2_SECRET_KEY={cfg['secret_key']}\n"
        f"export R2_BUCKET={cfg['bucket']}\n"
    )
    # Write via stdin to avoid creds in command-line args
    result = subprocess.run(
        [*_ssh_base(host, port),
         f"cat > {REMOTE_CREDS} && chmod 600 {REMOTE_CREDS}"],
        input=creds_content, text=True, check=True,
    )
    log.info("R2 credentials written to instance:%s", REMOTE_CREDS)


def destroy_instance(instance_id: int, log: logging.Logger) -> None:
    """Destroy a vast.ai instance (best-effort — never raises)."""
    try:
        result = _vastai("destroy", "instance", str(instance_id), "--raw", check=False)
        if result.returncode == 0:
            log.info("Instance %d destroyed.", instance_id)
        else:
            log.warning("destroy instance %d rc=%d: %s",
                        instance_id, result.returncode, result.stderr.strip())
    except Exception as exc:
        log.warning("destroy instance %d raised: %s", instance_id, exc)


@contextmanager
def managed_instance(offer: dict, cfg: dict, log: logging.Logger):
    """
    Context manager: create instance, install deps, write R2 creds.
    Guarantees destroy on exit regardless of success/failure.

    Yields (instance_id, ssh_host, ssh_port).
    """
    instance_id = None
    try:
        instance_id = create_instance(offer["id"], log)
        ssh_host, ssh_port = wait_for_ssh(instance_id, log)
        install_deps(ssh_host, ssh_port, log)
        write_r2_creds(ssh_host, ssh_port, cfg, log)
        yield instance_id, ssh_host, ssh_port
    finally:
        if instance_id is not None:
            destroy_instance(instance_id, log)


# ─── Remote R2 scripts ────────────────────────────────────────────────────────
#
# These Python snippets run on the vast.ai instance.  They source REMOTE_CREDS
# for R2 credentials, so no secrets appear in the SSH command line.
#

_REMOTE_DOWNLOAD_SCRIPT = """\
import boto3, os, sys
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY"],
    aws_secret_access_key=os.environ["R2_SECRET_KEY"],
)
bucket    = os.environ["R2_BUCKET"]
corpus_id = sys.argv[1]

os.makedirs(f"cloud_in/{corpus_id}", exist_ok=True)

# chunks TSV (required)
key = f"{corpus_id}/chunks_{corpus_id}.tsv"
dest = f"cloud_in/{corpus_id}/chunks_{corpus_id}.tsv"
s3.download_file(bucket, key, dest)
sz = os.path.getsize(dest) / 1_048_576
print(f"Downloaded chunks ({sz:.1f} MB)")

# config.json
try:
    s3.download_file(bucket, f"{corpus_id}/config.json",
                     f"cloud_in/{corpus_id}/config.json")
    print("Downloaded config.json")
except Exception:
    print("No config.json in R2 — pipeline will use defaults")

# embeddings.npz (optional — enables incremental / skip-embed mode)
try:
    emb_dest = f"cloud_in/{corpus_id}/embeddings_{corpus_id}.npz"
    s3.download_file(bucket, f"{corpus_id}/embeddings_{corpus_id}.npz", emb_dest)
    sz = os.path.getsize(emb_dest) / 1_048_576
    print(f"Downloaded embeddings ({sz:.1f} MB) — incremental run")
except Exception:
    print("No embeddings in R2 — full embed run")
"""

_REMOTE_UPLOAD_SCRIPT = """\
import boto3, glob, os, sys
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY"],
    aws_secret_access_key=os.environ["R2_SECRET_KEY"],
)
bucket    = os.environ["R2_BUCKET"]
corpus_id = sys.argv[1]

# Updated embeddings.npz → cache for next run
emb_path = f"cloud_in/{corpus_id}/embeddings_{corpus_id}.npz"
if os.path.exists(emb_path):
    key = f"{corpus_id}/embeddings_{corpus_id}.npz"
    sz  = os.path.getsize(emb_path) / 1_048_576
    s3.upload_file(emb_path, bucket, key)
    print(f"Uploaded updated embeddings ({sz:.1f} MB)")
else:
    print("No embeddings file found to upload")

# Output TSVs → R2 for local download + import
pattern = f"cloud_out/import/{corpus_id}/*.tsv"
tsv_files = glob.glob(pattern)
if not tsv_files:
    print(f"WARNING: no output TSVs found at {pattern}", flush=True)
    raise SystemExit(1)
for tsv in tsv_files:
    fname = os.path.basename(tsv)
    key   = f"{corpus_id}/output/import/{corpus_id}/{fname}"
    sz    = os.path.getsize(tsv) / 1_048_576
    s3.upload_file(tsv, bucket, key)
    print(f"Uploaded {fname} ({sz:.1f} MB)")
print(f"Upload complete: {len(tsv_files)} TSVs")
"""


def remote_download_corpus(host: str, port: int, corpus_id: str,
                            log: logging.Logger) -> None:
    """Download corpus input data from R2 onto the remote instance."""
    log.info("Instance downloading %s inputs from R2...", corpus_id)
    cmd = (
        f"cd {REMOTE_ROOT} && "
        f"source {REMOTE_CREDS} && "
        f"python3 - {corpus_id}"
    )
    result = subprocess.run(
        [*_ssh_base(host, port), cmd],
        input=_REMOTE_DOWNLOAD_SCRIPT,
        text=True,
        timeout=1800,    # 30 min — large chunks TSV can be slow
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Remote R2 download failed for {corpus_id}")


def remote_upload_results(host: str, port: int, corpus_id: str,
                           log: logging.Logger) -> None:
    """Upload pipeline results and updated embeddings from instance to R2."""
    log.info("Instance uploading %s results to R2...", corpus_id)
    cmd = (
        f"cd {REMOTE_ROOT} && "
        f"source {REMOTE_CREDS} && "
        f"python3 - {corpus_id}"
    )
    result = subprocess.run(
        [*_ssh_base(host, port), cmd],
        input=_REMOTE_UPLOAD_SCRIPT,
        text=True,
        timeout=1800,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Remote R2 upload failed for {corpus_id}")


# ─── Per-corpus pipeline ──────────────────────────────────────────────────────

def run_one_corpus(corpus_id: str, ssh_host: str, ssh_port: int,
                   s3, cfg: dict, log: logging.Logger) -> None:
    """
    Run the full pipeline for one corpus on an already-provisioned instance.

    Steps:
      1. Upload inputs (chunks, config, optional embeddings) to R2  [local → R2]
      2. Instance downloads inputs from R2                           [R2 → instance]
      3. Rsync arc_cloud_run.py to instance                         [local → instance]
      4. Run arc_cloud_run.py on instance
      5. Instance uploads results + updated embeddings to R2        [instance → R2]
      6. Local downloads output TSVs from R2                        [R2 → local]
      7. Run arc_import.py locally
    """
    log.info("=== corpus: %s ===", corpus_id)

    # ── 1. Upload inputs to R2 ─────────────────────────────────────────────────
    log.info("Uploading %s inputs to R2...", corpus_id)
    upload_corpus_inputs(corpus_id, s3, cfg, log)

    # ── 2. Instance downloads inputs from R2 ──────────────────────────────────
    ssh_run(host=ssh_host, port=ssh_port,
            command=f"mkdir -p {REMOTE_IN}/{corpus_id} {REMOTE_OUT}")
    remote_download_corpus(ssh_host, ssh_port, corpus_id, log)

    # ── 3. Rsync arc_cloud_run.py ──────────────────────────────────────────────
    rsync_to(ARC_DIR / "arc_cloud_run.py",
             ssh_host, ssh_port, f"{REMOTE_ROOT}/", log)

    # ── 4. Run remote pipeline ─────────────────────────────────────────────────
    remote_cmd = (
        f"cd {REMOTE_ROOT} && "
        f"python3 arc_cloud_run.py "
        f"--corpus-id {corpus_id} "
        f"--local-input {REMOTE_IN} "
        f"--local-output {REMOTE_OUT} "
        f"2>&1 | tee /tmp/arc_cloud_{corpus_id}.log; "
        f"exit ${{PIPESTATUS[0]}}"
    )
    log.info("Running arc_cloud_run.py on instance (corpus=%s)...", corpus_id)
    result = subprocess.run(
        [*_ssh_base(ssh_host, ssh_port), remote_cmd],
        timeout=4 * 3600,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"arc_cloud_run.py exited {result.returncode} for corpus={corpus_id}"
        )

    # ── 5. Instance uploads results to R2 ─────────────────────────────────────
    remote_upload_results(ssh_host, ssh_port, corpus_id, log)

    # ── 6. Local downloads output TSVs from R2 ────────────────────────────────
    log.info("Downloading %s output TSVs from R2...", corpus_id)
    local_out = download_corpus_outputs(corpus_id, s3, cfg, log)

    # ── 7. Local import into arc_v5 ───────────────────────────────────────────
    log.info("Importing results into arc_v5 (corpus=%s)...", corpus_id)
    env = {
        **os.environ,
        "PGHOST":    "/var/run/postgresql",
        "PGUSER":    "jeff",
        "PGDATABASE": "arc_v5",
    }
    imp = subprocess.run(
        [PYTHON_BIN, str(ARC_DIR / "arc_import.py"),
         "--corpus", corpus_id,
         "--input",  str(local_out)],
        env=env, check=False,
    )
    if imp.returncode != 0:
        log.warning("arc_import.py returned %d — results saved to %s but DB import incomplete",
                    imp.returncode, local_out)
    else:
        log.info("Import complete for corpus=%s", corpus_id)


# ─── Orchestration ────────────────────────────────────────────────────────────

# Result tuple: (corpus_id, success, elapsed_hr, actual_cost)
CorpusResult = tuple[str, bool, float, float]


def run_corpus_standalone(corpus_id: str, max_price: float, vram_floor: int,
                           cfg: dict, log: logging.Logger) -> CorpusResult:
    """
    Self-contained pipeline for one corpus: find GPU → provision → run → destroy.
    Designed to run in a thread for parallel execution.

    Retries once on failure (new instance each attempt).
    Returns (corpus_id, success, elapsed_hr, actual_cost).
    """
    n_chunks = count_chunks(corpus_id)
    min_vram = max(get_min_vram(n_chunks), vram_floor)

    try:
        offers = search_offers(min_vram, max_price)
    except Exception as exc:
        log.error("[%s] GPU search failed: %s", corpus_id, exc)
        return corpus_id, False, 0.0, 0.0

    if not offers:
        log.error("[%s] No offers found (VRAM>=%dGB, price<=$%.3f/hr)",
                  corpus_id, min_vram, max_price)
        return corpus_id, False, 0.0, 0.0

    offer    = select_best_offer(offers, min_vram)
    price    = offer["dph_total"]
    vram_got = round(offer["gpu_ram"] / 1000)
    log.info("[%s] Selected %s %dGB @ $%.4f/hr", corpus_id, offer["gpu_name"], vram_got, price)

    s3 = make_r2_client(cfg)
    t0 = time.time()

    for attempt in (1, 2):
        try:
            with managed_instance(offer, cfg, log) as (_, host, port):
                run_one_corpus(corpus_id, host, port, s3, cfg, log)
            elapsed = (time.time() - t0) / 3600
            cost    = elapsed * price
            log.info("[%s] DONE  %.2f hr  $%.3f", corpus_id, elapsed, cost)
            return corpus_id, True, elapsed, cost
        except Exception as exc:
            log.error("[%s] Attempt %d failed: %s", corpus_id, attempt, exc)

    elapsed = (time.time() - t0) / 3600
    return corpus_id, False, elapsed, elapsed * price


def run_parallel(corpora: list[str], max_price: float, vram_floor: int,
                 cfg: dict, log: logging.Logger) -> list[CorpusResult]:
    """
    Run all corpora in parallel, one instance per corpus.
    Prints a status line as each corpus completes.
    Returns list of CorpusResult tuples.
    """
    log.info("PARALLEL mode — spinning up %d instances simultaneously", len(corpora))

    results: list[CorpusResult] = []
    with ThreadPoolExecutor(max_workers=len(corpora)) as pool:
        futures = {
            pool.submit(run_corpus_standalone,
                        corpus_id, max_price, vram_floor, cfg, log): corpus_id
            for corpus_id in corpora
        }
        for future in as_completed(futures):
            corpus_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                log.error("[%s] Unhandled thread exception: %s", corpus_id, exc)
                result = (corpus_id, False, 0.0, 0.0)
            results.append(result)
            _, ok, elapsed, cost = result
            status = "OK " if ok else "ERR"
            print(f"  [{status}] {corpus_id:<30}  {elapsed:.2f}h  ${cost:.3f}")

    return results


def run_sequential(corpora: list[str], offer: dict, cfg: dict,
                   log: logging.Logger) -> list[CorpusResult]:
    """
    Run all corpora sequentially on a single shared instance, with one retry.
    Returns list of CorpusResult tuples.
    """
    log.info("SEQUENTIAL mode — one instance for %d corpus(es)", len(corpora))
    s3      = make_r2_client(cfg)
    done: dict[str, CorpusResult] = {}
    price   = offer["dph_total"]

    for attempt in (1, 2):
        pending = [c for c in corpora if c not in done or not done[c][1]]
        if not pending:
            break

        log.info("Attempt %d — %d corpus(es): %s",
                 attempt, len(pending), ", ".join(pending))
        t_inst = time.time()
        try:
            with managed_instance(offer, cfg, log) as (_, host, port):
                for corpus_id in pending:
                    t0 = time.time()
                    try:
                        run_one_corpus(corpus_id, host, port, s3, cfg, log)
                        elapsed = (time.time() - t0) / 3600
                        cost    = elapsed * price
                        done[corpus_id] = (corpus_id, True, elapsed, cost)
                    except Exception as exc:
                        elapsed = (time.time() - t0) / 3600
                        done[corpus_id] = (corpus_id, False, elapsed, elapsed * price)
                        log.error("corpus=%s failed on attempt %d: %s",
                                  corpus_id, attempt, exc)
        except Exception as exc:
            log.error("Instance-level failure on attempt %d: %s", attempt, exc)

    # Fill in any corpus that never got a result
    for corpus_id in corpora:
        if corpus_id not in done:
            done[corpus_id] = (corpus_id, False, 0.0, 0.0)

    return [done[c] for c in corpora]


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARC cloud orchestrator — vast.ai GPU pipeline runner with R2 storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--corpus",
                        help="Corpus ID or 'all'")
    parser.add_argument("--max-price", type=float, default=0.15,
                        metavar="USD/HR",
                        help="Maximum $/hr per GPU (default: 0.15)")
    parser.add_argument("--gpu-min-vram", type=int, default=8,
                        metavar="GB",
                        help="Hard VRAM floor in GB (default: 8); "
                             "dynamic model may require more")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit without starting anything")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompt")
    parser.add_argument("--upload-to-r2", action="store_true",
                        help="Upload all local cloud_in data to R2, then exit")

    # Parallel / sequential — mutually exclusive, default parallel
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--parallel", dest="parallel",
                            action="store_true", default=True,
                            help="Spin up one instance per corpus simultaneously (default)")
    mode_group.add_argument("--sequential", dest="parallel",
                            action="store_false",
                            help="Run corpora sequentially on one shared instance")

    args = parser.parse_args()

    if not args.corpus and not args.upload_to_r2:
        parser.error("--corpus is required unless --upload-to-r2 is used")

    # ── Handle --upload-to-r2 ─────────────────────────────────────────────────
    if args.upload_to_r2:
        log = _setup_logging("upload_to_r2")
        log.info("=== Seeding R2 with all local cloud_in data ===")
        try:
            cfg = r2_config()
        except RuntimeError as exc:
            log.error("%s", exc)
            sys.exit(1)
        n_files, total_mb = upload_all_to_r2(log)
        print(f"\nUploaded {n_files} files  ({total_mb:.1f} MB total)  →  R2 bucket: {cfg['bucket']}")
        return

    # ── Resolve corpus list ────────────────────────────────────────────────────
    if args.corpus == "all":
        corpora = list_ready_corpora()
        if not corpora:
            print("No corpora found in cloud_in/ — nothing to do.")
            sys.exit(1)
    else:
        corpora = [args.corpus]

    label = args.corpus.replace("/", "_")
    log   = _setup_logging(label)

    mode_str = "parallel" if args.parallel else "sequential"
    log.info("ARC Cloud Orchestrator  corpus=%s  mode=%s  max_price=$%.3f  "
             "vram_floor=%dGB  dry_run=%s",
             args.corpus, mode_str, args.max_price, args.gpu_min_vram, args.dry_run)

    # ── Validate R2 credentials early ─────────────────────────────────────────
    try:
        cfg = r2_config()
        log.info("R2 bucket: %s", cfg["bucket"])
    except RuntimeError as exc:
        log.error("R2 configuration error: %s", exc)
        sys.exit(1)

    # ── Per-corpus metadata: chunks, periods, VRAM, GPU offer ─────────────────
    # Use the same reference GPU for all corpora (cheapest for max VRAM need)
    # so dry-run doesn't require 11 API calls.  Parallel mode re-searches
    # per-corpus inside the thread.

    corpus_chunks   = {c: count_chunks(c) for c in corpora}
    corpus_min_vram = {c: max(get_min_vram(corpus_chunks[c]), args.gpu_min_vram)
                       for c in corpora}

    # Reference offer: cheapest GPU that satisfies the highest VRAM requirement
    ref_vram = max(corpus_min_vram.values())
    log.info("Searching vast.ai for reference offer (VRAM>=%dGB, max $%.3f/hr)...",
             ref_vram, args.max_price)
    try:
        ref_offers = search_offers(ref_vram, args.max_price)
    except Exception as exc:
        log.error("GPU search failed: %s", exc)
        sys.exit(1)

    if not ref_offers:
        print(f"\nNo offers found (VRAM>={ref_vram}GB, price<=${args.max_price}/hr, "
              "CUDA>=12, reliability>0.95).")
        print("Try --max-price 0.20")
        sys.exit(1)

    ref_offer = select_best_offer(ref_offers, ref_vram)
    print_top_offers(ref_offers, ref_offer)

    ref_price   = ref_offer["dph_total"]
    ref_vram_gb = round(ref_offer["gpu_ram"] / 1000)

    # ── Per-corpus estimates ───────────────────────────────────────────────────
    est_hrs: dict[str, float] = {}
    est_costs: dict[str, float] = {}
    for corpus_id in corpora:
        h, c = estimate_cost(corpus_id, ref_price)
        est_hrs[corpus_id]   = h
        est_costs[corpus_id] = c

    max_est_hr  = max(est_hrs.values())
    total_est_hr   = sum(est_hrs.values())
    total_est_cost = sum(est_costs.values())

    # ── Print plan ─────────────────────────────────────────────────────────────
    print(f"\n  R2 bucket  : {cfg['bucket']}")
    print(f"  Data flow  : local → R2 → instance → R2 → local")
    print(f"  Mode       : {mode_str.upper()}")
    print()
    print(f"  {'Corpus':<30}  {'Chunks':>8}  {'MinVRAM':>8}  "
          f"{'NPZ?':>5}  {'Est. hr':>8}  {'Est. cost':>10}")
    print("  " + "─" * 76)
    for corpus_id in corpora:
        n        = corpus_chunks[corpus_id]
        mv       = corpus_min_vram[corpus_id]
        has_npz  = (CLOUD_IN_DIR / corpus_id / f"embeddings_{corpus_id}.npz").exists()
        npz_mark = "yes" if has_npz else "no"
        print(f"  {corpus_id:<30}  {n:>8,}  {mv:>6}GB  {npz_mark:>5}  "
              f"{est_hrs[corpus_id]:>7.2f}h  ${est_costs[corpus_id]:>8.3f}")
    if len(corpora) > 1:
        print("  " + "─" * 76)
        print(f"  {'TOTAL':<30}  {'':>8}  {'':>8}  {'':>5}  "
              f"{total_est_hr:>7.2f}h  ${total_est_cost:>8.3f}")
    print()

    if len(corpora) > 1:
        if args.parallel:
            print(f"  Wall time (parallel)  : ~{max_est_hr:.2f}h  "
                  f"(longest corpus: {max(est_hrs, key=est_hrs.__getitem__)})")
            print(f"  Wall time (sequential): ~{total_est_hr:.2f}h")
        else:
            print(f"  Wall time (sequential): ~{total_est_hr:.2f}h")
            print(f"  Wall time (parallel)  : ~{max_est_hr:.2f}h  (if run with --parallel)")
        print(f"  Est. total cost       : ~${total_est_cost:.3f}")
        print()

    # ── Dry-run exit ───────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"[DRY RUN] No instances started.  R2 credentials verified.")
        if args.parallel and len(corpora) > 1:
            print(f"          Would start {len(corpora)} instances in parallel.")
        return

    # ── Confirmation ───────────────────────────────────────────────────────────
    if not args.yes:
        mode_note = (f"{len(corpora)} parallel instances"
                     if args.parallel and len(corpora) > 1
                     else f"{ref_offer['gpu_name']} @ ${ref_price:.4f}/hr")
        ans = input(f"Proceed? [{mode_note}] [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return

    # ── Run ────────────────────────────────────────────────────────────────────
    t0 = time.time()

    if args.parallel and len(corpora) > 1:
        print(f"\n  Spinning up {len(corpora)} instances in parallel...")
        print(f"  {'Corpus':<32}  {'Time':>6}  {'Cost':>8}")
        print("  " + "─" * 52)
        raw_results = run_parallel(corpora, args.max_price, args.gpu_min_vram, cfg, log)
    else:
        # Sequential: use the already-selected reference offer
        raw_results = run_sequential(corpora, ref_offer, cfg, log)

    elapsed_wall = (time.time() - t0) / 3600

    # ── Summary ────────────────────────────────────────────────────────────────
    ok_results  = [r for r in raw_results if r[1]]
    err_results = [r for r in raw_results if not r[1]]
    total_actual_cost = sum(r[3] for r in raw_results)

    print(f"\n{'═' * 60}")
    print(f"  SUMMARY  —  {len(ok_results)} OK  /  {len(err_results)} FAILED")
    print(f"  Mode    : {mode_str}")
    print(f"  Wall    : {elapsed_wall:.2f} hr")
    print(f"  Cost    : ${total_actual_cost:.3f}  (est. ${total_est_cost:.3f})")
    print()
    print(f"  {'Corpus':<30}  {'Status':<6}  {'Time':>6}  {'Cost':>8}")
    print("  " + "─" * 56)
    for corpus_id, ok, elapsed, cost in sorted(raw_results, key=lambda r: r[0]):
        status = "OK" if ok else "FAIL"
        print(f"  {corpus_id:<30}  {status:<6}  {elapsed:>5.2f}h  ${cost:>7.3f}")
    print("═" * 60)

    if err_results:
        sys.exit(1)


if __name__ == "__main__":
    main()

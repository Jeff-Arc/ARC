#!/usr/bin/env python3
"""
arc_cloud_sentinel.py — Vast.ai instance manager for ARC cloud job queue

Runs on the Hetzner server. Continuously polls Vast.ai offers, maintains
price history, launches GPU instances sized to pending jobs, monitors
running instances, and kills idle ones.

Usage:
    python3 arc_cloud_sentinel.py
    python3 arc_cloud_sentinel.py --max-price 0.12 --dry-run
    python3 arc_cloud_sentinel.py --max-instances 3

Environment:
    VASTAI_API_KEY   Vast.ai API key (or ~/.vastai config)
    HETZNER_DB_URL   postgres://user:pass@host:5432/arc_v5
    R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import mean, stdev
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras

# ─── Paths ────────────────────────────────────────────────────────────────────

ARC_DIR       = Path("/root/arc")
DATA_DIR      = ARC_DIR / "data"
PRICE_HISTORY = Path("/tmp/arc_sentinel") / "vastai_price_history.jsonl"
DECISION_LOG  = Path("/tmp/arc_sentinel") / "sentinel_decisions.log"
ENV_FILE      = ARC_DIR / ".env"

VASTAI_BIN    = os.environ.get("VASTAI_BIN", "/home/arc/.local/bin/vastai")
PYTORCH_IMAGE = "pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime"

# ─── Size bins ────────────────────────────────────────────────────────────────

SIZE_BINS = {
    "small":  {"max_chunks": 50_000,  "min_vram": 8,  "label": "<50K chunks"},
    "medium": {"max_chunks": 200_000, "min_vram": 12, "label": "50K-200K chunks"},
    "large":  {"max_chunks": 999_999_999, "min_vram": 24, "label": ">200K chunks"},
}

# ─── Constants ────────────────────────────────────────────────────────────────

POLL_INTERVAL_SEC  = 30
IDLE_KILL_SEC      = 600   # kill instances idle >10 min
PRICE_ANOMALY_MULT = 0.7   # flag prices < 70% of 7-day average
TOP_N_OFFERS       = 20    # log top N offers per poll

# Streaming parallel bootstrap: all installs + R2 download run simultaneously.
# Each pip install runs in its own background process so downloads overlap.
# R2 corpus download happens in parallel with all installs.
# Pipeline starts as soon as the last dependency finishes.
#
# Placeholders filled by provision_worker():
#   {corpus_id}  — corpus to process
#   {size_bin}   — worker size bin
# Streaming bootstrap: torch installed first (sequential), then remaining deps
# + R2 download all run in parallel. Pipeline starts when last one finishes.
#
# Why torch is sequential: sentence-transformers depends on torch — if pip
# installs both in parallel, sentence-transformers pulls latest torch and
# overwrites the cu124 version we need.
#
# Placeholders filled by provision_worker():
#   {corpus_id}  — corpus to pre-download from R2
#   {size_bin}   — worker size bin
# Install order: RAPIDS first (brings its CUDA libs), torch last (overwrites
# RAPIDS's nvidia-cusparse with the cu124 version torch needs).
# R2 download starts after numpy (no boto3 needed — uses urllib fallback or
# waits for boto3 from misc deps).
#
# Placeholders filled by provision_worker():
#   {corpus_id}  — corpus to pre-download from R2
#   {size_bin}   — worker size bin
_WORKER_BOOTSTRAP = """\
set -e
echo "[boot] $(date +%H:%M:%S) Starting bootstrap"
mkdir -p /workspace/arc/cloud_in/{corpus_id}

apt-get update -qq && apt-get install -y -qq rsync 2>/dev/null || true
pip install -q --upgrade pip
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# ── Step 1: numpy + boto3 (needed before parallel phase) ─────────────
pip install -q numpy==2.2.0 boto3
echo "[boot] $(date +%H:%M:%S) numpy + boto3 ready"

# ── Step 2: RAPIDS (installs its own CUDA libs — will be overwritten by torch) ─
pip install -q --extra-index-url https://pypi.nvidia.com \
    cugraph-cu12 cudf-cu12 2>/dev/null \
    && echo "[boot] $(date +%H:%M:%S) RAPIDS ready" \
    || echo "[boot] $(date +%H:%M:%S) RAPIDS unavailable (CPU fallback)"

# ── Step 3: Remaining deps (parallel) + R2 download ─────────────────
pip install -q leidenalg igraph faiss-gpu-cu12 psycopg2-binary \
    sentence-transformers texttable --no-deps &
PID_DEPS=$!

pip install -q transformers huggingface-hub tokenizers tqdm \
    scikit-learn scipy &
PID_ST_DEPS=$!

# R2 download in parallel (boto3 already installed in step 1)
(
    source /root/.worker_env
    python3 -c "
import boto3, os
s3 = boto3.client('s3', endpoint_url=os.environ['R2_ENDPOINT'],
    aws_access_key_id=os.environ['R2_ACCESS_KEY'],
    aws_secret_access_key=os.environ['R2_SECRET_KEY'])
bucket = os.environ['R2_BUCKET']
cid = '{corpus_id}'
s3.download_file(bucket, f'{{cid}}/chunks_{{cid}}.tsv',
    f'cloud_in/{{cid}}/chunks_{{cid}}.tsv')
try:
    s3.download_file(bucket, f'{{cid}}/config.json', f'cloud_in/{{cid}}/config.json')
except: pass
try:
    s3.download_file(bucket, f'{{cid}}/embeddings_{{cid}}.npz',
        f'cloud_in/{{cid}}/embeddings_{{cid}}.npz')
except: pass
import os as o; sz = o.path.getsize(f'cloud_in/{{cid}}/chunks_{{cid}}.tsv')/1e6
print(f'[r2] Downloaded {{cid}} ({{sz:.1f}} MB)')
"
) &
PID_R2=$!

echo "[boot] $(date +%H:%M:%S) Waiting for deps + R2..."
wait $PID_DEPS    && echo "[boot] $(date +%H:%M:%S) core deps ready" \
                   || echo "[boot] $(date +%H:%M:%S) core deps FAILED"
wait $PID_ST_DEPS && echo "[boot] $(date +%H:%M:%S) transformers ready" \
                   || echo "[boot] $(date +%H:%M:%S) transformers FAILED"
wait $PID_R2      && echo "[boot] $(date +%H:%M:%S) R2 ready" \
                   || echo "[boot] $(date +%H:%M:%S) R2 FAILED"

# ── Step 4: torch LAST — force-reinstall overwrites RAPIDS CUDA libs ─
pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
echo "[boot] $(date +%H:%M:%S) torch installed (last — CUDA libs pinned)"

# ── Verify ───────────────────────────────────────────────────────────
python3 -c "import torch; print('[boot] CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo "[boot] $(date +%H:%M:%S) disk: $(df -h / | tail -1)"
echo "[boot] $(date +%H:%M:%S) ALL READY"
"""

# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    fmt = "%(asctime)s  %(levelname)-8s  [sentinel] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(fmt, datefmt))
        root.addHandler(ch)
        fh = logging.FileHandler(DECISION_LOG)
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        root.addHandler(fh)
    return logging.getLogger("sentinel")


def decision(log: logging.Logger, msg: str) -> None:
    """Log a decision — goes to both console and decision log."""
    log.info("[DECISION] %s", msg)


# ─── Graceful shutdown ────────────────────────────────────────────────────────

_shutdown = False

def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# ─── Env / DB helpers ────────────────────────────────────────────────────────

def load_env(path: Path = ENV_FILE) -> dict[str, str]:
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


def get_db_conn(db_url: str):
    p = urlparse(db_url)
    return psycopg2.connect(
        host=p.hostname,
        port=p.port or 5432,
        dbname=p.path.lstrip("/") or "arc_v5",
        user=p.username or "jeff",
        password=p.password,
    )


def r2_env_block() -> str:
    """Build export block for R2 creds to inject into remote workers."""
    env = {**load_env(), **os.environ}
    return (
        f"export R2_ENDPOINT={env.get('R2_ENDPOINT', '')}\n"
        f"export R2_ACCESS_KEY={env.get('R2_ACCESS_KEY', '')}\n"
        f"export R2_SECRET_KEY={env.get('R2_SECRET_KEY', '')}\n"
        f"export R2_BUCKET={env.get('R2_BUCKET', '')}\n"
        f"export HETZNER_DB_URL={env.get('HETZNER_DB_URL', '')}\n"
    )


# ─── Vast.ai CLI ─────────────────────────────────────────────────────────────

def _vastai(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run([VASTAI_BIN, *args],
                          capture_output=True, text=True, check=check)


def search_offers(min_vram_gb: int, max_price: float) -> list[dict]:
    """Search Vast.ai for qualifying GPU offers."""
    query = (
        f"gpu_ram>={min_vram_gb} "
        "inet_up>200 disk_space>30 cuda_vers>=12.0 reliability>0.95"
    )
    result = _vastai("search", "offers", query,
                     "--order", "dph_total", "--limit", "100", "--raw")
    try:
        offers = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    return [
        o for o in offers
        if o.get("reliability2", o.get("reliability", 0)) > 0.95
        and o.get("dph_total", 999) <= max_price
        and o.get("gpu_ram", 0) >= min_vram_gb * 1000
    ]


def get_running_instances() -> list[dict]:
    """Get all running Vast.ai instances."""
    result = _vastai("show", "instances", "--raw")
    if result.returncode != 0:
        return []
    try:
        return json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return []


def create_instance(offer_id: int, label: str, log: logging.Logger) -> int | None:
    """Create a Vast.ai instance. Returns instance_id or None on failure."""
    result = _vastai(
        "create", "instance", str(offer_id),
        "--image", PYTORCH_IMAGE,
        "--disk", "25",
        "--ssh", "--direct",
        "--label", label,
        "--raw",
    )
    if result.returncode != 0:
        log.error("Failed to create instance from offer %d: %s",
                  offer_id, result.stderr[:200])
        return None
    try:
        resp = json.loads(result.stdout)
        iid = resp.get("new_contract")
        if iid:
            log.info("Instance created: id=%d  offer=%d  label=%s", iid, offer_id, label)
            return int(iid)
    except (json.JSONDecodeError, TypeError):
        pass
    log.error("Could not parse create response: %s", result.stdout[:200])
    return None


def destroy_instance(instance_id: int, log: logging.Logger) -> None:
    """Destroy instance (best-effort)."""
    result = _vastai("destroy", "instance", str(instance_id), "--raw")
    if result.returncode == 0:
        log.info("Instance %d destroyed", instance_id)
    else:
        log.warning("Failed to destroy instance %d: %s",
                    instance_id, result.stderr[:200])


def _ssh_base(host: str, port: int) -> list[str]:
    return [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-p", str(port),
        f"root@{host}",
    ]


def ssh_run(host: str, port: int, cmd: str, timeout: int = 60) -> tuple[bool, str]:
    """Run SSH command, return (success, output)."""
    try:
        result = subprocess.run(
            [*_ssh_base(host, port), cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except (subprocess.TimeoutExpired, Exception) as e:
        return False, str(e)


def wait_for_ssh(instance_id: int, log: logging.Logger,
                 timeout: int = 600) -> tuple[str, int] | None:
    """Poll until instance is SSH-ready. Returns (host, port) or None."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        instances = get_running_instances()
        info = next((i for i in instances if i["id"] == instance_id), None)
        if info and info.get("actual_status") == "running":
            host = info.get("ssh_host")
            port = info.get("ssh_port")
            if host and port:
                ok, _ = ssh_run(host, int(port), "echo ready", timeout=10)
                if ok:
                    return host, int(port)
        time.sleep(15)
    return None


def provision_worker(instance_id: int, size_bin: str,
                     log: logging.Logger,
                     corpus_id: str = None) -> bool:
    """
    Fully provision a worker instance with streaming parallel install.

    Steps (sequential prerequisites first, then everything in parallel):
    1. Wait for SSH
    2. Write R2/DB credentials (needed by R2 download in bootstrap)
    3. Rsync worker + pipeline scripts
    4. Run streaming bootstrap: all pip installs + R2 download in parallel
    5. Start worker process
    """
    ssh = wait_for_ssh(instance_id, log)
    if not ssh:
        log.error("Instance %d never became SSH-ready", instance_id)
        return False
    host, port = ssh

    # Write R2 + DB creds first (bootstrap needs them for R2 download)
    creds = r2_env_block()
    subprocess.run(
        [*_ssh_base(host, port), "cat > /root/.worker_env && chmod 600 /root/.worker_env"],
        input=creds, text=True, check=True,
    )

    # Create workspace and rsync worker + pipeline scripts
    ssh_run(host, port, "mkdir -p /workspace/arc", timeout=10)
    for script in ["arc_cloud_worker.py", "arc_cloud_run.py"]:
        src = ARC_DIR / script
        if src.exists():
            subprocess.run([
                "rsync", "-az", "-e",
                f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {port}",
                str(src), f"root@{host}:/workspace/arc/{script}",
            ], check=True)

    # Run streaming parallel bootstrap (installs + R2 download simultaneously)
    cid = corpus_id or "NONE"
    bootstrap = _WORKER_BOOTSTRAP.replace("{corpus_id}", cid).replace("{size_bin}", size_bin)
    log.info("Streaming install on instance %d (corpus=%s)...", instance_id, cid)
    result = subprocess.run(
        [*_ssh_base(host, port), "cd /workspace/arc && bash -s"],
        input=bootstrap,
        text=True, timeout=1200, check=False,
    )
    if result.returncode != 0:
        log.error("Bootstrap failed on instance %d (rc=%d)", instance_id, result.returncode)
        if result.stdout:
            log.error("stdout: %s", result.stdout[-500:])
        return False

    log.info("Instance %d provisioned", instance_id)

    # Start worker in background
    worker_cmd = (
        f"source /root/.worker_env && "
        f"cd /workspace/arc && "
        f"nohup python3 -u arc_cloud_worker.py "
        f"--size-bin {size_bin} "
        f"> /workspace/worker.log 2>&1 &"
    )
    ok, _ = ssh_run(host, port, worker_cmd, timeout=30)
    if ok:
        log.info("Worker started on instance %d (size_bin=%s)", instance_id, size_bin)
    else:
        log.error("Failed to start worker on instance %d", instance_id)
    return ok


# ─── Price history ────────────────────────────────────────────────────────────

def log_price_snapshot(offers_by_bin: dict[str, list[dict]]) -> None:
    """Append top 20 offers per size bin to price history JSONL."""
    ts = datetime.now(timezone.utc).isoformat()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(PRICE_HISTORY, "a") as f:
        for bin_name, offers in offers_by_bin.items():
            for o in offers[:TOP_N_OFFERS]:
                record = {
                    "timestamp": ts,
                    "size_bin": bin_name,
                    "gpu_name": o.get("gpu_name", "unknown"),
                    "price": o.get("dph_total", 0),
                    "vram_gb": round(o.get("gpu_ram", 0) / 1000, 1),
                    "reliability": o.get("reliability2", o.get("reliability", 0)),
                    "offer_id": o.get("id"),
                }
                f.write(json.dumps(record) + "\n")


def load_price_history(days: int = 7) -> dict[str, list[float]]:
    """Load price history from last N days, grouped by size bin."""
    if not PRICE_HISTORY.exists():
        return {}
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    prices: dict[str, list[float]] = defaultdict(list)

    for line in PRICE_HISTORY.read_text().splitlines():
        try:
            r = json.loads(line)
            ts = datetime.fromisoformat(r["timestamp"])
            if ts >= cutoff:
                prices[r["size_bin"]].append(r["price"])
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return dict(prices)


def detect_price_anomalies(offers: list[dict], history_prices: list[float],
                           bin_name: str, log: logging.Logger) -> list[dict]:
    """Flag offers significantly cheaper than historical average."""
    if len(history_prices) < 10:
        return []
    avg = mean(history_prices)
    threshold = avg * PRICE_ANOMALY_MULT
    anomalies = [o for o in offers if o.get("dph_total", 999) < threshold]
    for o in anomalies:
        log.info("PRICE ANOMALY [%s]: %s @ $%.3f/hr (7d avg: $%.3f, threshold: $%.3f)",
                 bin_name, o.get("gpu_name"), o["dph_total"], avg, threshold)
    return anomalies


# ─── Job queue queries ────────────────────────────────────────────────────────

def get_pending_jobs_by_bin(conn) -> dict[str, list[dict]]:
    """Get pending job counts and details grouped by size bin."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT job_id, corpus_id, size_bin, chunk_count, priority
            FROM arc_cloud_jobs
            WHERE status = 'pending'
            ORDER BY priority ASC, created_at ASC
        """)
        rows = cur.fetchall()

    by_bin: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bin[r["size_bin"] or "medium"].append(dict(r))
    return dict(by_bin)


def get_running_jobs(conn) -> list[dict]:
    """Get all currently running jobs."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT job_id, corpus_id, size_bin, assigned_instance, started_at
            FROM arc_cloud_jobs
            WHERE status = 'running'
        """)
        return [dict(r) for r in cur.fetchall()]


def reassign_failed_jobs(conn, failed_instance: str, log: logging.Logger) -> int:
    """Reset running jobs from a failed instance back to pending."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE arc_cloud_jobs
            SET status = 'pending',
                assigned_instance = NULL,
                started_at = NULL,
                error_text = 'instance failed — reassigned'
            WHERE status = 'running' AND assigned_instance = %s
            RETURNING job_id
        """, (failed_instance,))
        reassigned = cur.fetchall()
    conn.commit()
    count = len(reassigned)
    if count > 0:
        decision(log, f"Reassigned {count} jobs from failed instance {failed_instance}")
    return count


# ─── Instance tracking ────────────────────────────────────────────────────────

class InstanceTracker:
    """Track managed instances and their state."""

    def __init__(self):
        self.instances: dict[int, dict] = {}  # id → {size_bin, created_at, last_active}

    def add(self, instance_id: int, size_bin: str, gpu_name: str = ""):
        self.instances[instance_id] = {
            "size_bin": size_bin,
            "gpu_name": gpu_name,
            "created_at": time.time(),
            "last_active": time.time(),
        }

    def remove(self, instance_id: int):
        self.instances.pop(instance_id, None)

    def mark_active(self, instance_id: int):
        if instance_id in self.instances:
            self.instances[instance_id]["last_active"] = time.time()

    def idle_time(self, instance_id: int) -> float:
        info = self.instances.get(instance_id)
        return time.time() - info["last_active"] if info else 0

    def count_by_bin(self, size_bin: str) -> int:
        return sum(1 for v in self.instances.values() if v["size_bin"] == size_bin)

    def all_ids(self) -> set[int]:
        return set(self.instances.keys())


# ─── Launch logic ─────────────────────────────────────────────────────────────

def should_launch(pending_by_bin: dict[str, list],
                  tracker: InstanceTracker,
                  max_instances: int,
                  log: logging.Logger) -> list[tuple[str, int]]:
    """
    Decide which instances to launch. Returns list of (size_bin, min_vram_gb).

    Rules:
    - Never launch more instances than pending jobs per bin
    - Never exceed max_instances total
    - Small jobs can be packed: 1 instance per 3 pending small jobs
    - Medium/large: 1 instance per pending job
    """
    launches: list[tuple[str, int]] = []
    total_managed = len(tracker.instances)

    for bin_name in ["small", "medium", "large"]:
        pending = pending_by_bin.get(bin_name, [])
        if not pending:
            continue

        running_in_bin = tracker.count_by_bin(bin_name)
        n_pending = len(pending)

        # Small jobs: pack multiple per instance
        if bin_name == "small":
            needed = max(1, (n_pending + 2) // 3)  # ceil(n/3)
        else:
            needed = n_pending

        to_launch = max(0, needed - running_in_bin)
        to_launch = min(to_launch, max_instances - total_managed - len(launches))

        if to_launch > 0:
            vram = SIZE_BINS[bin_name]["min_vram"]
            for _ in range(to_launch):
                launches.append((bin_name, vram))
            decision(log,
                     f"Need {to_launch} instance(s) for {bin_name} "
                     f"({n_pending} pending, {running_in_bin} running)")

    return launches


def _get_learned_chunks_per_sec(gpu_name: str, db_url: str) -> float | None:
    """Query arc_cloud_instance_stats for historical embed speed of this GPU."""
    try:
        p = urlparse(db_url)
        conn = psycopg2.connect(host=p.hostname, port=p.port or 5432,
                                dbname=p.path.lstrip("/"), user=p.username,
                                password=p.password)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT AVG(embed_chunks_per_sec)
                FROM arc_cloud_instance_stats
                WHERE gpu_name = %s AND success = true AND embed_chunks_per_sec > 0
                HAVING COUNT(*) >= 3
            """, (gpu_name,))
            row = cur.fetchone()
        conn.close()
        return float(row[0]) if row and row[0] else None
    except Exception:
        return None


def estimate_total_cost(offer: dict, chunk_count: int = 100_000,
                        db_url: str = None) -> float:
    """
    Estimate total cost for a job on this offer.

    total = (gpu_cost + disk_cost) × hours + network_cost
    hours = embed_time + period_time + overhead
    """
    gpu_cost_hr = offer.get("dph_total", 0.10)
    disk_gb = 25
    storage_cost_gb_mo = offer.get("storage_cost", 0.20)
    disk_cost_hr = (disk_gb * storage_cost_gb_mo) / (30 * 24)  # $/hr
    net_cost_tb = offer.get("internet_up_cost_per_tb", 6.67)

    # Estimate embed speed: learned > DLPerf proxy > default
    gpu_name = offer.get("gpu_name", "")
    learned_cps = _get_learned_chunks_per_sec(gpu_name, db_url) if db_url else None

    if learned_cps:
        chunks_per_sec = learned_cps
    else:
        # DLPerf proxy: scale from RTX 3060 baseline (73 chunks/sec)
        dlperf = offer.get("dlperf", 10.0)
        # RTX 3060 has dlperf ~12-15, does 73 cps
        chunks_per_sec = max(10, (dlperf / 13.0) * 73.0)

    embed_hours = chunk_count / (chunks_per_sec * 3600)
    period_hours = 0.5  # ~30 min for period loop (varies by corpus)
    overhead_hours = 0.15  # bootstrap, R2 transfer
    total_hours = embed_hours + period_hours + overhead_hours

    # Network: ~200MB upload + download per corpus
    net_cost = (0.4 / 1000) * net_cost_tb  # 0.4 GB total transfer

    total = (gpu_cost_hr + disk_cost_hr) * total_hours + net_cost
    return total


def select_offer(offers: list[dict], history_prices: list[float],
                 chunk_count: int = 100_000, db_url: str = None) -> dict | None:
    """
    Select offer with lowest estimated total cost.
    Uses learned embed speed from arc_cloud_instance_stats if available,
    falls back to DLPerf proxy estimate.
    """
    if not offers:
        return None

    scored = []
    for o in offers:
        cost = estimate_total_cost(o, chunk_count, db_url)
        scored.append((cost, o))
    scored.sort(key=lambda x: x[0])

    # Log top 3 for visibility
    for cost, o in scored[:3]:
        log_msg = (f"  {o.get('gpu_name','?'):20s} ${o.get('dph_total',0):.3f}/hr "
                   f"est_total=${cost:.3f}")
        # Can't call log here (no logger), just pick best
    return scored[0][1]


# ─── Main loop ────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="ARC cloud sentinel — Vast.ai instance manager")
    ap.add_argument("--max-price", type=float, default=0.15,
                    help="Max $/hr per instance (default: 0.15)")
    ap.add_argument("--max-instances", type=int, default=5,
                    help="Max simultaneous instances (default: 5)")
    ap.add_argument("--poll-interval", type=int, default=POLL_INTERVAL_SEC,
                    help="Seconds between polls (default: 30)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Log decisions but don't create/destroy instances")
    return ap.parse_args()


def main():
    args = parse_args()
    log = setup_logging()
    log.info("Sentinel starting: max_price=$%.2f  max_instances=%d  poll=%ds  dry_run=%s",
             args.max_price, args.max_instances, args.poll_interval, args.dry_run)

    db_url = os.environ.get("HETZNER_DB_URL",
                            os.environ["HETZNER_DB_URL"])
    conn = get_db_conn(db_url)
    tracker = InstanceTracker()
    poll_count = 0

    while not _shutdown:
        poll_count += 1
        try:
            _poll_cycle(conn, tracker, args, log, poll_count)
        except psycopg2.OperationalError:
            log.warning("DB connection lost — reconnecting...")
            try:
                conn.close()
            except Exception:
                pass
            conn = get_db_conn(db_url)
        except Exception as e:
            log.error("Poll cycle error: %s", e, exc_info=True)

        time.sleep(args.poll_interval)

    # Shutdown: destroy all managed instances
    log.info("Sentinel shutting down — destroying %d managed instances",
             len(tracker.instances))
    if not args.dry_run:
        for iid in list(tracker.all_ids()):
            destroy_instance(iid, log)
    log.info("Sentinel stopped.")


def _poll_cycle(conn, tracker: InstanceTracker, args, log, poll_count: int):
    """Single poll iteration."""

    # ── 1. Fetch offers per size bin ──────────────────────────────────────
    offers_by_bin: dict[str, list[dict]] = {}
    for bin_name, cfg in SIZE_BINS.items():
        offers = search_offers(cfg["min_vram"], args.max_price)
        offers.sort(key=lambda o: o.get("dph_total", 999))
        offers_by_bin[bin_name] = offers

    # ── 2. Log price snapshot (every poll) ────────────────────────────────
    log_price_snapshot(offers_by_bin)

    # ── 3. Detect price anomalies ─────────────────────────────────────────
    history = load_price_history(days=7)
    for bin_name, offers in offers_by_bin.items():
        detect_price_anomalies(offers, history.get(bin_name, []), bin_name, log)

    # ── 4. Get job queue state ────────────────────────────────────────────
    pending_by_bin = get_pending_jobs_by_bin(conn)
    running_jobs = get_running_jobs(conn)
    total_pending = sum(len(v) for v in pending_by_bin.values())
    total_running = len(running_jobs)

    if poll_count % 10 == 1:  # log summary every ~5 min
        log.info("Queue: %d pending, %d running, %d instances managed",
                 total_pending, total_running, len(tracker.instances))
        for b, jobs in pending_by_bin.items():
            log.info("  %s: %d pending, %d instances",
                     b, len(jobs), tracker.count_by_bin(b))

    # ── 5. Monitor running instances ──────────────────────────────────────
    live_instances = get_running_instances()
    live_ids = {i["id"] for i in live_instances}

    # Periodic disk check on managed instances (~every 5 min)
    if poll_count % 10 == 1:
        for inst in live_instances:
            if inst["id"] in tracker.all_ids():
                h, p = inst.get("ssh_host"), inst.get("ssh_port")
                if h and p and inst.get("actual_status") == "running":
                    ok, out = ssh_run(h, int(p), "df -h / | tail -1", timeout=10)
                    if ok:
                        log.info("  disk [%d %s]: %s", inst["id"],
                                 inst.get("gpu_name", "?"), out.strip())

    # Check for failed managed instances
    for iid in list(tracker.all_ids()):
        if iid not in live_ids:
            log.warning("Managed instance %d disappeared — marking failed", iid)
            reassign_failed_jobs(conn, str(iid), log)
            tracker.remove(iid)

    # Track activity: instances with running jobs are active
    active_instances = {j["assigned_instance"] for j in running_jobs if j.get("assigned_instance")}
    for iid in tracker.all_ids():
        if str(iid) in active_instances:
            tracker.mark_active(iid)

    # ── 6. Kill idle instances ────────────────────────────────────────────
    if total_pending == 0:
        for iid in list(tracker.all_ids()):
            idle = tracker.idle_time(iid)
            if idle > IDLE_KILL_SEC:
                decision(log, f"Killing idle instance {iid} (idle {idle:.0f}s, no pending jobs)")
                if not args.dry_run:
                    destroy_instance(iid, log)
                tracker.remove(iid)

    # ── 7. Launch new instances if needed ─────────────────────────────────
    if total_pending > 0:
        launches = should_launch(pending_by_bin, tracker, args.max_instances, log)
        for size_bin, min_vram in launches:
            bin_offers = offers_by_bin.get(size_bin, [])
            bin_history = history.get(size_bin, [])
            # Estimate chunk count from first pending job in bin
            avg_chunks = 100_000
            bp = pending_by_bin.get(size_bin, [])
            if bp:
                avg_chunks = bp[0].get("chunk_count") or 100_000
            offer = select_offer(bin_offers, bin_history,
                                 chunk_count=avg_chunks, db_url=db_url)

            if not offer:
                log.warning("No qualifying offer for %s (vram>=%dGB, <=$%.2f/hr)",
                            size_bin, min_vram, args.max_price)
                continue

            gpu = offer.get("gpu_name", "?")
            price = offer.get("dph_total", 0)
            decision(log, f"Launching {size_bin} instance: {gpu} @ ${price:.3f}/hr "
                          f"(offer {offer['id']})")

            if args.dry_run:
                continue

            iid = create_instance(offer["id"], f"arc-{size_bin}", log)
            if iid:
                tracker.add(iid, size_bin, gpu)
                # Pick first pending corpus for this bin to pre-download during install
                first_corpus = None
                bin_pending = pending_by_bin.get(size_bin, [])
                if bin_pending:
                    first_corpus = bin_pending[0].get("corpus_id")
                # Provision in background (don't block the poll loop)
                import threading
                t = threading.Thread(
                    target=provision_worker,
                    args=(iid, size_bin, log, first_corpus),
                    daemon=True,
                )
                t.start()


if __name__ == "__main__":
    main()

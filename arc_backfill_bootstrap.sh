#!/bin/bash
# arc_backfill_bootstrap.sh — Vast.ai bootstrap for backfill workers
#
# Install order adapted from arc_cloud_sentinel.py _WORKER_BOOTSTRAP (lines 91-161)
# which ran full end-to-end on vast.ai. Key insight: torch must be installed LAST
# with --force-reinstall to pin cu124 CUDA libs, because sentence-transformers
# pulls its own torch version as a transitive dep.
#
# No RAPIDS (no Leiden cuGraph needed), but otherwise same proven order.

set -e

# Source credentials written by onstart
if [ -f /root/.worker_env ]; then
    source /root/.worker_env
    echo "[boot] $(date +%H:%M:%S) Loaded /root/.worker_env"
else
    echo "[boot] $(date +%H:%M:%S) WARNING: /root/.worker_env not found"
fi

echo "[boot] $(date +%H:%M:%S) Starting backfill bootstrap"
mkdir -p /workspace/arc

pip install -q --upgrade pip
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# ── Step 1: numpy + boto3 (needed before parallel phase) ──────────────────
pip install -q numpy==2.2.0 boto3
echo "[boot] $(date +%H:%M:%S) numpy + boto3 ready"

# ── Step 2: Deps (parallel) + R2 worker download ─────────────────────────
# Match sentinel: --no-deps on first group, transitive deps in second group
pip install -q leidenalg igraph faiss-gpu-cu12 sentence-transformers texttable --no-deps &
PID_DEPS=$!

pip install -q transformers huggingface-hub tokenizers tqdm scikit-learn scipy &
PID_ST_DEPS=$!

# Download worker script from R2 in parallel
(
    python3 -c "
import boto3, os
s3 = boto3.client('s3',
    endpoint_url=os.environ['R2_ENDPOINT'],
    aws_access_key_id=os.environ['R2_ACCESS_KEY'],
    aws_secret_access_key=os.environ['R2_SECRET_KEY'])
bucket = os.environ.get('R2_BUCKET', 'arc-cloud')
s3.download_file(bucket, 'backfill/scripts/arc_backfill_worker.py',
    '/workspace/arc/arc_backfill_worker.py')
print('[boot] Worker script downloaded')
"
) &
PID_R2=$!

echo "[boot] $(date +%H:%M:%S) Waiting for deps + R2..."
wait $PID_DEPS    && echo "[boot] $(date +%H:%M:%S) core deps ready" \
                   || { echo "[boot] $(date +%H:%M:%S) core deps FAILED"; exit 1; }
wait $PID_ST_DEPS && echo "[boot] $(date +%H:%M:%S) transformers ready" \
                   || { echo "[boot] $(date +%H:%M:%S) transformers FAILED"; exit 1; }
wait $PID_R2      && echo "[boot] $(date +%H:%M:%S) R2 ready" \
                   || { echo "[boot] $(date +%H:%M:%S) R2 FAILED"; exit 1; }

# ── Step 3: torch LAST — force-reinstall pins cu124 CUDA libs ─────────────
pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
echo "[boot] $(date +%H:%M:%S) torch installed (last — CUDA libs pinned)"

# ── Step 4: Verify ────────────────────────────────────────────────────────
python3 -c "
import torch
print(f'[boot] CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
import faiss; print('[boot] FAISS: OK')
from sentence_transformers import SentenceTransformer; print('[boot] sentence-transformers: OK')
import leidenalg, igraph; print(f'[boot] Leiden: OK')
"
echo "[boot] $(date +%H:%M:%S) disk: $(df -h / | tail -1)"
echo "[boot] $(date +%H:%M:%S) ALL READY"

# ── Step 5: Run worker ────────────────────────────────────────────────────
# Determine mode: --auto (default) or --start/--end range
if [ -n "${START_IDX:-}" ] && [ -n "${END_IDX:-}" ]; then
    MODE_ARGS="--start $START_IDX --end $END_IDX"
    echo "[boot] $(date +%H:%M:%S) Starting worker: range [$START_IDX, $END_IDX)"
else
    MODE_ARGS="--auto"
    echo "[boot] $(date +%H:%M:%S) Starting worker: auto mode"
fi

cd /workspace/arc
python3 -u arc_backfill_worker.py \
    $MODE_ARGS \
    --work-dir /workspace/arc/backfill \
    2>&1 | tee /workspace/worker.log

echo "[boot] $(date +%H:%M:%S) Worker finished"

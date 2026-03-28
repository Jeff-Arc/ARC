#!/usr/bin/env python3
"""
arc_cloud_worker.py — Vast.ai GPU worker for ARC cloud job queue

Runs on a Vast.ai instance. Connects to Hetzner arc_v5 database, claims
pending jobs via SELECT FOR UPDATE SKIP LOCKED, runs arc_cloud_run.py,
marks complete, and immediately claims the next job. Self-terminates
after 5 minutes idle with no pending jobs.

Environment variables:
    HETZNER_DB_URL   postgres://user:pass@host:5432/arc_v5 (set via env)
    R2_ENDPOINT      Cloudflare R2 endpoint
    R2_ACCESS_KEY    R2 access key
    R2_SECRET_KEY    R2 secret key
    R2_BUCKET        R2 bucket name
    WORKER_ID        Unique identifier for this instance (default: hostname)
    WORKER_SIZE_BIN  Size bin this worker handles: small/medium/large/any (default: any)

Usage (on vast.ai instance):
    export HETZNER_DB_URL="postgres://user:pass@host:5432/arc_v5 (set via env)"
    python3 arc_cloud_worker.py
    python3 arc_cloud_worker.py --size-bin small
    python3 arc_cloud_worker.py --idle-timeout 600
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import boto3
import psycopg2
import psycopg2.extras

# ─── Paths ────────────────────────────────────────────────────────────────────

WORK_ROOT  = Path("/workspace/arc")
CLOUD_IN   = WORK_ROOT / "cloud_in"
CLOUD_OUT  = WORK_ROOT / "cloud_out"
RUN_SCRIPT = WORK_ROOT / "arc_cloud_run.py"

# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    fmt = "%(asctime)s  %(levelname)-8s  [worker] %(message)s"
    logging.basicConfig(format=fmt, datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO, stream=sys.stdout)
    return logging.getLogger("worker")

log = setup_logging()

# ─── Graceful shutdown ────────────────────────────────────────────────────────

_shutdown_requested = False

def _handle_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    log.info("Shutdown signal received (sig=%d). Finishing current job...", signum)

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# ─── Database ─────────────────────────────────────────────────────────────────

def parse_db_url(url: str) -> dict:
    """Parse postgres://user:pass@host:port/dbname into connection kwargs."""
    p = urlparse(url)
    return {
        "host": p.hostname,
        "port": p.port or 5432,
        "dbname": p.path.lstrip("/") or "arc_v5",
        "user": p.username or "jeff",
        "password": p.password,
    }


def get_db_conn(db_url: str) -> psycopg2.extensions.connection:
    """Connect to Hetzner PostgreSQL with retry."""
    kwargs = parse_db_url(db_url)
    for attempt in range(5):
        try:
            conn = psycopg2.connect(**kwargs)
            conn.autocommit = False
            return conn
        except psycopg2.OperationalError as e:
            if attempt < 4:
                log.warning("DB connect failed (attempt %d/5): %s", attempt + 1, e)
                time.sleep(5 * (attempt + 1))
            else:
                raise


def claim_job(conn, worker_id: str, size_bin: str) -> dict | None:
    """
    Claim the next pending job using SELECT FOR UPDATE SKIP LOCKED.
    Returns job dict or None if no jobs available.
    """
    size_filter = ""
    params = [worker_id]
    if size_bin and size_bin != "any":
        size_filter = "AND size_bin = %s"
        params.append(size_bin)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"""
            UPDATE arc_cloud_jobs
            SET status = 'running',
                assigned_instance = %s,
                started_at = now()
            WHERE job_id = (
                SELECT job_id FROM arc_cloud_jobs
                WHERE status = 'pending'
                {size_filter}
                ORDER BY priority ASC, created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING *
        """, params)
        row = cur.fetchone()
    conn.commit()
    return dict(row) if row else None


def complete_job(conn, job_id: int, cost_actual: float = None) -> None:
    """Mark a job as completed."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE arc_cloud_jobs
            SET status = 'completed',
                completed_at = now(),
                cost_actual = %s
            WHERE job_id = %s
        """, (cost_actual, job_id))
    conn.commit()
    log.info("Job %d marked completed", job_id)


def fail_job(conn, job_id: int, error: str) -> None:
    """Mark a job as failed with error text."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE arc_cloud_jobs
            SET status = 'failed',
                completed_at = now(),
                error_text = %s
            WHERE job_id = %s
        """, (error[:2000], job_id))
    conn.commit()
    log.error("Job %d marked failed: %s", job_id, error[:200])


def check_terminate(conn) -> bool:
    """Check if there's a TERMINATE pseudo-job for this worker."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT job_id FROM arc_cloud_jobs
            WHERE status = 'pending' AND corpus_id = 'TERMINATE'
            LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            cur.execute("""
                UPDATE arc_cloud_jobs
                SET status = 'completed', completed_at = now()
                WHERE job_id = %s
            """, (row[0],))
            conn.commit()
            return True
    return False


def count_pending(conn, size_bin: str) -> int:
    """Count pending jobs, optionally filtered by size bin."""
    with conn.cursor() as cur:
        if size_bin and size_bin != "any":
            cur.execute(
                "SELECT count(*) FROM arc_cloud_jobs WHERE status='pending' AND size_bin=%s",
                (size_bin,))
        else:
            cur.execute("SELECT count(*) FROM arc_cloud_jobs WHERE status='pending'")
        return cur.fetchone()[0]


# ─── R2 helpers ───────────────────────────────────────────────────────────────

def make_r2_client():
    """Create boto3 S3 client for Cloudflare R2."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"],
    )


def r2_download_corpus(s3, bucket: str, corpus_id: str) -> None:
    """Download corpus inputs from R2 to local workspace."""
    corpus_dir = CLOUD_IN / corpus_id
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # chunks TSV (required)
    key = f"{corpus_id}/chunks_{corpus_id}.tsv"
    dest = corpus_dir / f"chunks_{corpus_id}.tsv"
    s3.download_file(bucket, key, str(dest))
    sz = dest.stat().st_size / 1_048_576
    log.info("Downloaded chunks_%s.tsv (%.1f MB)", corpus_id, sz)

    # config.json (optional)
    try:
        s3.download_file(bucket, f"{corpus_id}/config.json",
                         str(corpus_dir / "config.json"))
        log.info("Downloaded config.json")
    except Exception:
        log.info("No config.json in R2 — using defaults")

    # embeddings NPZ (optional — enables incremental mode)
    try:
        emb_dest = corpus_dir / f"embeddings_{corpus_id}.npz"
        s3.download_file(bucket, f"{corpus_id}/embeddings_{corpus_id}.npz",
                         str(emb_dest))
        sz = emb_dest.stat().st_size / 1_048_576
        log.info("Downloaded embeddings (%.1f MB) — incremental run", sz)
    except Exception:
        log.info("No embeddings in R2 — full embed run")


def r2_upload_results(s3, bucket: str, corpus_id: str) -> int:
    """Upload pipeline results and updated embeddings to R2. Returns file count."""
    uploaded = 0

    # Updated embeddings
    emb_path = CLOUD_IN / corpus_id / f"embeddings_{corpus_id}.npz"
    if emb_path.exists():
        key = f"{corpus_id}/embeddings_{corpus_id}.npz"
        s3.upload_file(str(emb_path), bucket, key)
        sz = emb_path.stat().st_size / 1_048_576
        log.info("Uploaded updated embeddings (%.1f MB)", sz)
        uploaded += 1

    # Output TSVs
    pattern = str(CLOUD_OUT / "import" / corpus_id / "*.tsv")
    tsv_files = glob.glob(pattern)
    if not tsv_files:
        raise RuntimeError(f"No output TSVs found at {pattern}")

    for tsv in tsv_files:
        fname = os.path.basename(tsv)
        key = f"{corpus_id}/output/import/{corpus_id}/{fname}"
        s3.upload_file(tsv, bucket, key)
        uploaded += 1

    log.info("Uploaded %d output TSVs", len(tsv_files))
    return uploaded


# ─── Pipeline execution ──────────────────────────────────────────────────────

def run_pipeline(corpus_id: str) -> tuple[bool, str]:
    """
    Run arc_cloud_run.py for a corpus. Returns (success, message).
    """
    cmd = [
        sys.executable, str(RUN_SCRIPT),
        "--corpus-id", corpus_id,
        "--local-input", str(CLOUD_IN),
        "--local-output", str(CLOUD_OUT),
    ]
    log.info("Running pipeline: %s", " ".join(cmd))
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=14400,  # 4 hour timeout
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            log.info("Pipeline completed in %.1f min", elapsed / 60)
            return True, f"completed in {elapsed/60:.1f} min"
        else:
            # Capture last 500 chars of stderr + stdout for error context
            err = result.stderr[-500:] if result.stderr else ""
            out = result.stdout[-500:] if result.stdout else ""
            combined = (err or out or "no output")
            log.error("Pipeline failed (rc=%d) in %.1f min:\n%s",
                      result.returncode, elapsed / 60, combined[:300])
            return False, f"rc={result.returncode}: {combined}"

    except subprocess.TimeoutExpired:
        return False, "timeout after 4 hours"
    except Exception as e:
        return False, str(e)[:500]


def execute_job(job: dict, s3, bucket: str) -> tuple[bool, str]:
    """
    Full job execution: download → run → upload.
    Returns (success, message).
    """
    corpus_id = job["corpus_id"]
    log.info("=== Starting job %d: %s (chunks=%s, bin=%s) ===",
             job["job_id"], corpus_id, job.get("chunk_count"), job.get("size_bin"))

    # Step 1: Download inputs from R2
    try:
        r2_download_corpus(s3, bucket, corpus_id)
    except Exception as e:
        return False, f"R2 download failed: {e}"

    # Step 2: Run pipeline
    success, msg = run_pipeline(corpus_id)
    if not success:
        return False, msg

    # Step 3: Upload results to R2
    try:
        r2_upload_results(s3, bucket, corpus_id)
    except Exception as e:
        return False, f"R2 upload failed: {e}"

    return True, msg


# ─── Main loop ────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="ARC cloud worker — runs on Vast.ai GPU instances")
    ap.add_argument("--size-bin", default="any",
                    choices=["small", "medium", "large", "any"],
                    help="Only claim jobs for this size bin (default: any)")
    ap.add_argument("--idle-timeout", type=int, default=300,
                    help="Self-terminate after N seconds with no pending jobs (default: 300)")
    ap.add_argument("--max-jobs", type=int, default=0,
                    help="Exit after N jobs (0 = unlimited)")
    return ap.parse_args()


def main():
    args = parse_args()
    worker_id = os.environ.get("WORKER_ID", socket.gethostname())
    db_url = os.environ["HETZNER_DB_URL"]  # required — no default with credentials
    bucket = os.environ.get("R2_BUCKET", "arc-cloud")
    size_bin = os.environ.get("WORKER_SIZE_BIN", args.size_bin)

    log.info("Worker starting: id=%s  size_bin=%s  idle_timeout=%ds",
             worker_id, size_bin, args.idle_timeout)

    # Connect to Hetzner DB
    conn = get_db_conn(db_url)
    log.info("Connected to Hetzner arc_v5")

    # Connect to R2
    s3 = make_r2_client()
    log.info("R2 client ready (bucket=%s)", bucket)

    # Ensure workspace exists
    CLOUD_IN.mkdir(parents=True, exist_ok=True)
    CLOUD_OUT.mkdir(parents=True, exist_ok=True)

    jobs_completed = 0
    idle_since = None

    while not _shutdown_requested:
        # Check for TERMINATE command
        if check_terminate(conn):
            log.info("TERMINATE job received — shutting down gracefully")
            break

        # Try to claim a job
        try:
            job = claim_job(conn, worker_id, size_bin)
        except psycopg2.OperationalError:
            log.warning("DB connection lost — reconnecting...")
            try:
                conn.close()
            except Exception:
                pass
            conn = get_db_conn(db_url)
            continue

        if job is None:
            # No jobs available
            if idle_since is None:
                idle_since = time.time()
                log.info("No pending jobs — entering idle (timeout %ds)", args.idle_timeout)
            elif time.time() - idle_since > args.idle_timeout:
                log.info("Idle timeout reached (%.0fs) — self-terminating",
                         time.time() - idle_since)
                break
            time.sleep(15)  # poll every 15 seconds
            continue

        # Got a job — reset idle timer
        idle_since = None
        t0 = time.time()

        # Execute the job
        success, msg = execute_job(job, s3, bucket)
        elapsed = time.time() - t0

        # Estimate cost (job stores cost_estimate based on offer price × time)
        cost_est = job.get("cost_estimate")
        cost_actual = None
        if cost_est and job.get("chunk_count"):
            # Scale estimate by actual vs expected time
            expected_min = (job["chunk_count"] / 350_000) * 120 * 1.1
            actual_min = elapsed / 60
            if expected_min > 0:
                cost_actual = cost_est * (actual_min / expected_min)

        if success:
            complete_job(conn, job["job_id"], cost_actual)
            jobs_completed += 1
        else:
            fail_job(conn, job["job_id"], msg)

        log.info("Job %d %s in %.1f min (total completed: %d)",
                 job["job_id"], "completed" if success else "FAILED",
                 elapsed / 60, jobs_completed)

        # Check max-jobs limit
        if args.max_jobs > 0 and jobs_completed >= args.max_jobs:
            log.info("Max jobs reached (%d) — exiting", args.max_jobs)
            break

    # Cleanup
    try:
        conn.close()
    except Exception:
        pass
    log.info("Worker shutdown. Jobs completed: %d", jobs_completed)


if __name__ == "__main__":
    main()

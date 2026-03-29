#!/usr/bin/env python3
"""
arc_cloud_worker.py — Vast.ai GPU worker for ARC cloud job queue

Runs on a Vast.ai instance. Connects to Hetzner arc_v5 database, claims
pending jobs via SELECT FOR UPDATE SKIP LOCKED, runs arc_cloud_run.py,
marks complete, and immediately claims the next job. Self-terminates
after 5 minutes idle with no pending jobs.

Resilience features:
  - Heartbeat thread: writes last_heartbeat + current_phase to DB every 60s
  - DB timeout: connect_timeout=10, statement_timeout=30s on all operations
  - DB reconnect: automatic reconnect on any OperationalError
  - Job done markers: if DB call fails after R2 upload, writes local marker
    file; on restart, retries marking jobs complete before claiming new ones

Environment variables:
    HETZNER_DB_URL   postgres://user:pass@host:5432/arc_v5
    R2_ENDPOINT      Cloudflare R2 endpoint
    R2_ACCESS_KEY    R2 access key
    R2_SECRET_KEY    R2 secret key
    R2_BUCKET        R2 bucket name
    WORKER_ID        Unique identifier for this instance (default: hostname)
    WORKER_SIZE_BIN  Size bin this worker handles: small/medium/large/any (default: any)

Usage (on vast.ai instance):
    python3 arc_cloud_worker.py
    python3 arc_cloud_worker.py --size-bin small
    python3 arc_cloud_worker.py --idle-timeout 120
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
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import boto3
import psycopg2
import psycopg2.extras

# ─── Paths ────────────────────────────────────────────────────────────────────

WORK_ROOT   = Path("/workspace/arc")
CLOUD_IN    = WORK_ROOT / "cloud_in"
CLOUD_OUT   = WORK_ROOT / "cloud_out"
RUN_SCRIPT  = WORK_ROOT / "arc_cloud_run.py"
DONE_MARKER = WORK_ROOT / ".done_markers"   # dir for job-done marker files

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

# ─── Database with timeout + reconnect ────────────────────────────────────────

DB_CONNECT_TIMEOUT = 10    # seconds to wait for TCP connection
DB_STATEMENT_TIMEOUT = 30  # seconds before query is killed


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
    """Connect to Hetzner PostgreSQL with retry, timeout, and keepalive."""
    kwargs = parse_db_url(db_url)
    kwargs["connect_timeout"] = DB_CONNECT_TIMEOUT
    # TCP keepalive: detect dead connections faster
    kwargs["keepalives"] = 1
    kwargs["keepalives_idle"] = 30
    kwargs["keepalives_interval"] = 10
    kwargs["keepalives_count"] = 3
    kwargs["options"] = f"-c statement_timeout={DB_STATEMENT_TIMEOUT * 1000}"

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


def _db_call(func):
    """Decorator: auto-reconnect on DB errors, retry once."""
    def wrapper(self, *args, **kwargs):
        for attempt in range(2):
            try:
                return func(self, *args, **kwargs)
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                log.warning("DB error in %s (attempt %d): %s", func.__name__, attempt + 1, e)
                try:
                    self.conn.close()
                except Exception:
                    pass
                try:
                    self.conn = get_db_conn(self.db_url)
                    log.info("DB reconnected")
                except Exception as re:
                    log.error("DB reconnect failed: %s", re)
                    if attempt == 1:
                        raise
        return None
    return wrapper


# ─── Heartbeat thread ────────────────────────────────────────────────────────

class Heartbeat:
    """Background thread that updates last_heartbeat + current_phase every 60s."""

    def __init__(self, db_url: str, job_id: int):
        self.db_url = db_url
        self.job_id = job_id
        self.phase = "starting"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def set_phase(self, phase: str):
        self.phase = phase

    def _run(self):
        while not self._stop.is_set():
            try:
                conn = get_db_conn(self.db_url)
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE arc_cloud_jobs
                        SET last_heartbeat = now(), current_phase = %s
                        WHERE job_id = %s
                    """, (self.phase, self.job_id))
                conn.commit()
                conn.close()
            except Exception as e:
                log.debug("Heartbeat failed: %s", e)
            self._stop.wait(60)


# ─── Worker class ─────────────────────────────────────────────────────────────

class Worker:
    def __init__(self, db_url: str, worker_id: str, size_bin: str,
                 bucket: str, idle_timeout: int, max_jobs: int):
        self.db_url = db_url
        self.worker_id = worker_id
        self.size_bin = size_bin
        self.bucket = bucket
        self.idle_timeout = idle_timeout
        self.max_jobs = max_jobs
        self.conn = get_db_conn(db_url)
        self.s3 = make_r2_client()
        self.jobs_completed = 0

    @_db_call
    def claim_job(self) -> dict | None:
        size_filter = ""
        params = [self.worker_id]
        if self.size_bin and self.size_bin != "any":
            size_filter = "AND size_bin = %s"
            params.append(self.size_bin)
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                UPDATE arc_cloud_jobs
                SET status = 'running',
                    assigned_instance = %s,
                    started_at = now(),
                    last_heartbeat = now(),
                    current_phase = 'claimed'
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
        self.conn.commit()
        return dict(row) if row else None

    @_db_call
    def complete_job(self, job_id: int, cost_actual: float = None) -> None:
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE arc_cloud_jobs
                SET status = 'completed', completed_at = now(),
                    cost_actual = %s, current_phase = 'done'
                WHERE job_id = %s
            """, (cost_actual, job_id))
        self.conn.commit()
        log.info("Job %d marked completed", job_id)

    @_db_call
    def fail_job(self, job_id: int, error: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE arc_cloud_jobs
                SET status = 'failed', completed_at = now(),
                    error_text = %s, current_phase = 'failed'
                WHERE job_id = %s
            """, (error[:2000], job_id))
        self.conn.commit()
        log.error("Job %d marked failed: %s", job_id, error[:200])

    @_db_call
    def check_terminate(self) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT job_id FROM arc_cloud_jobs
                WHERE status = 'pending' AND corpus_id = 'TERMINATE'
                LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                cur.execute(
                    "UPDATE arc_cloud_jobs SET status='completed', completed_at=now() WHERE job_id=%s",
                    (row[0],))
                self.conn.commit()
                return True
        return False

    def _write_done_marker(self, job_id: int, corpus_id: str, cost: float = None):
        """Write local marker so we can retry DB update on restart."""
        DONE_MARKER.mkdir(parents=True, exist_ok=True)
        marker = DONE_MARKER / f"job_{job_id}.json"
        marker.write_text(json.dumps({
            "job_id": job_id,
            "corpus_id": corpus_id,
            "cost_actual": cost,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }))
        log.info("Wrote done marker for job %d at %s", job_id, marker)

    def _retry_done_markers(self):
        """On startup, retry marking jobs complete from any leftover markers."""
        if not DONE_MARKER.exists():
            return
        for marker in DONE_MARKER.glob("job_*.json"):
            try:
                data = json.loads(marker.read_text())
                self.complete_job(data["job_id"], data.get("cost_actual"))
                marker.unlink()
                log.info("Retried done marker for job %d — success", data["job_id"])
            except Exception as e:
                log.warning("Failed to retry done marker %s: %s", marker.name, e)

    def execute_job(self, job: dict) -> tuple[bool, str]:
        corpus_id = job["corpus_id"]
        job_id = job["job_id"]
        log.info("=== Starting job %d: %s (chunks=%s, bin=%s) ===",
                 job_id, corpus_id, job.get("chunk_count"), job.get("size_bin"))

        # Start heartbeat
        hb = Heartbeat(self.db_url, job_id)
        hb.start()

        try:
            # Step 1: Download
            hb.set_phase("r2_download")
            try:
                r2_download_corpus(self.s3, self.bucket, corpus_id)
            except Exception as e:
                return False, f"R2 download failed: {e}"

            # Step 2: Pipeline
            hb.set_phase("pipeline")
            success, msg = run_pipeline(corpus_id)
            if not success:
                return False, msg

            # Step 3: Upload
            hb.set_phase("r2_upload")
            try:
                r2_upload_results(self.s3, self.bucket, corpus_id)
            except Exception as e:
                return False, f"R2 upload failed: {e}"

            return True, msg
        finally:
            hb.stop()

    def run(self):
        log.info("Worker starting: id=%s  size_bin=%s  idle_timeout=%ds",
                 self.worker_id, self.size_bin, self.idle_timeout)

        CLOUD_IN.mkdir(parents=True, exist_ok=True)
        CLOUD_OUT.mkdir(parents=True, exist_ok=True)

        # Retry any leftover done markers from a previous crash
        self._retry_done_markers()

        idle_since = None

        while not _shutdown_requested:
            if self.check_terminate():
                log.info("TERMINATE received — shutting down")
                break

            job = self.claim_job()

            if job is None:
                if idle_since is None:
                    idle_since = time.time()
                    log.info("No pending jobs — idle (timeout %ds)", self.idle_timeout)
                elif time.time() - idle_since > self.idle_timeout:
                    log.info("Idle timeout (%.0fs) — self-terminating",
                             time.time() - idle_since)
                    break
                time.sleep(15)
                continue

            idle_since = None
            t0 = time.time()
            success, msg = self.execute_job(job)
            elapsed = time.time() - t0

            cost_actual = None
            cost_est = job.get("cost_estimate")
            if cost_est and job.get("chunk_count"):
                expected_min = (job["chunk_count"] / 350_000) * 120 * 1.1
                actual_min = elapsed / 60
                if expected_min > 0:
                    cost_actual = cost_est * (actual_min / expected_min)

            if success:
                try:
                    self.complete_job(job["job_id"], cost_actual)
                except Exception:
                    # DB call failed — write marker so we can retry later
                    self._write_done_marker(job["job_id"], job["corpus_id"], cost_actual)
                self.jobs_completed += 1
            else:
                try:
                    self.fail_job(job["job_id"], msg)
                except Exception:
                    log.error("Could not mark job %d failed in DB", job["job_id"])

            log.info("Job %d %s in %.1f min (total: %d)",
                     job["job_id"], "completed" if success else "FAILED",
                     elapsed / 60, self.jobs_completed)

            if self.max_jobs > 0 and self.jobs_completed >= self.max_jobs:
                log.info("Max jobs (%d) — exiting", self.max_jobs)
                break

        try:
            self.conn.close()
        except Exception:
            pass
        log.info("Worker shutdown. Jobs completed: %d", self.jobs_completed)


# ─── R2 helpers ───────────────────────────────────────────────────────────────

def make_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"],
    )


def r2_download_corpus(s3, bucket: str, corpus_id: str) -> None:
    corpus_dir = CLOUD_IN / corpus_id
    corpus_dir.mkdir(parents=True, exist_ok=True)
    key = f"{corpus_id}/chunks_{corpus_id}.tsv"
    dest = corpus_dir / f"chunks_{corpus_id}.tsv"
    s3.download_file(bucket, key, str(dest))
    sz = dest.stat().st_size / 1_048_576
    log.info("Downloaded chunks_%s.tsv (%.1f MB)", corpus_id, sz)
    try:
        s3.download_file(bucket, f"{corpus_id}/config.json",
                         str(corpus_dir / "config.json"))
        log.info("Downloaded config.json")
    except Exception:
        log.info("No config.json in R2")
    try:
        emb_dest = corpus_dir / f"embeddings_{corpus_id}.npz"
        s3.download_file(bucket, f"{corpus_id}/embeddings_{corpus_id}.npz", str(emb_dest))
        log.info("Downloaded embeddings (%.1f MB)", emb_dest.stat().st_size / 1_048_576)
    except Exception:
        log.info("No embeddings in R2 — full embed run")


def r2_upload_results(s3, bucket: str, corpus_id: str) -> int:
    uploaded = 0
    emb_path = CLOUD_IN / corpus_id / f"embeddings_{corpus_id}.npz"
    if emb_path.exists():
        s3.upload_file(str(emb_path), bucket, f"{corpus_id}/embeddings_{corpus_id}.npz")
        log.info("Uploaded embeddings (%.1f MB)", emb_path.stat().st_size / 1_048_576)
        uploaded += 1
    pattern = str(CLOUD_OUT / "import" / corpus_id / "*.tsv")
    tsv_files = glob.glob(pattern)
    if not tsv_files:
        raise RuntimeError(f"No output TSVs at {pattern}")
    for tsv in tsv_files:
        fname = os.path.basename(tsv)
        s3.upload_file(tsv, bucket, f"{corpus_id}/output/import/{corpus_id}/{fname}")
        uploaded += 1
    log.info("Uploaded %d output TSVs", len(tsv_files))
    return uploaded


# ─── Pipeline execution ──────────────────────────────────────────────────────

def run_pipeline(corpus_id: str) -> tuple[bool, str]:
    cmd = [
        sys.executable, str(RUN_SCRIPT),
        "--corpus-id", corpus_id,
        "--local-input", str(CLOUD_IN),
        "--local-output", str(CLOUD_OUT),
    ]
    log.info("Running pipeline: %s", " ".join(cmd))
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
        elapsed = time.time() - t0
        if result.returncode == 0:
            log.info("Pipeline completed in %.1f min", elapsed / 60)
            return True, f"completed in {elapsed/60:.1f} min"
        else:
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


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="ARC cloud worker — Vast.ai GPU")
    ap.add_argument("--size-bin", default="any",
                    choices=["small", "medium", "large", "any"])
    ap.add_argument("--idle-timeout", type=int, default=300,
                    help="Self-terminate after N seconds idle (default: 300)")
    ap.add_argument("--max-jobs", type=int, default=0,
                    help="Exit after N jobs (0 = unlimited)")
    return ap.parse_args()


def main():
    args = parse_args()
    worker = Worker(
        db_url=os.environ["HETZNER_DB_URL"],
        worker_id=os.environ.get("WORKER_ID", socket.gethostname()),
        size_bin=os.environ.get("WORKER_SIZE_BIN", args.size_bin),
        bucket=os.environ.get("R2_BUCKET", "arc-embeddings"),
        idle_timeout=args.idle_timeout,
        max_jobs=args.max_jobs,
    )
    worker.run()


if __name__ == "__main__":
    main()

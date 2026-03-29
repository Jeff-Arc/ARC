#!/usr/bin/env python3
"""
run_epo_ingest.py — Download + ingest all 162 DOCDB ZIPs into normalized epo_* tables.

Pipeline:
  Thread-1 (downloader): prefetch next ZIP (queue maxsize=2)
  Main thread (ingestor): ingest current ZIP → delete → pull next from queue

Features:
  - Resume-safe: ingest tracker skips completed ZIPs
  - Download prefetch: 1 thread downloads ahead while main thread ingests
  - Delete-after-ingest: keeps disk footprint to ~3 GB of ZIPs
  - Crash-safe: LOGGED tables + synchronous_commit=off
  - Tag monitor: unknown XML paths logged to /tmp/epo_unknown_xml_paths.jsonl
  - Session settings: no parallel workers, conservative work_mem

Usage:
  python3 run_epo_ingest.py
  python3 run_epo_ingest.py --start-from 5       # resume from file #5
  python3 run_epo_ingest.py --no-delete           # keep ZIPs after ingest
  python3 run_epo_ingest.py --dry-run             # parse only, no DB writes
"""

import argparse
import functools
import json
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# Force unbuffered stdout
print = functools.partial(print, flush=True)  # type: ignore[assignment]

sys.path.insert(0, str(Path(__file__).parent))

from arc_docdb_download import (
    fetch_product, find_delivery, download_item, sha1_file,
    load_manifest, save_manifest, parse_size_bytes, human_size,
    download_url, DEFAULT_DELIVERY_NAME,
)
from arc_epo_ingest import (
    ingest_zip, PersonCache, HAVE_LXML,
)
import psycopg2

# ── Config ───────────────────────────────────────────────────────────────────

PROGRESS_FILE = Path("/tmp/epo_ingest_progress.txt")
DEST_DIR = Path.home() / "data" / "docdb_backfile"
ERROR_LOG = Path("/tmp/epo_ingest_errors.jsonl")
TRACKER_FILE = DEST_DIR / ".epo_ingest_tracker.json"
MIN_FREE_GB = 30  # pause if disk free drops below this

# ── Globals ──────────────────────────────────────────────────────────────────

_shutdown = threading.Event()


def _handle_signal(signum, frame):
    print(f"\n[SIGNAL {signum}] Shutting down gracefully...", flush=True)
    _shutdown.set()


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(PROGRESS_FILE, "a") as f:
        f.write(line + "\n")


# ── Tracker ──────────────────────────────────────────────────────────────────

def load_tracker() -> dict:
    if TRACKER_FILE.exists():
        try:
            return json.loads(TRACKER_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_tracker(tracker: dict):
    TRACKER_FILE.write_text(json.dumps(tracker, indent=2, default=str))


# ── Download prefetch thread ─────────────────────────────────────────────────

def downloader_thread(items: list[dict], delivery_id: int,
                      dl_queue: queue.Queue, tracker: dict):
    """Download ZIPs and put paths onto the queue. Runs in background thread."""
    for i, item in enumerate(items):
        if _shutdown.is_set():
            break

        name = item["itemName"]

        # Skip already ingested
        if name in tracker:
            continue

        item_id = item["itemId"]
        sha1_exp = item.get("fileChecksum", "").upper()
        size_b = int(parse_size_bytes(item.get("fileSize", "0")))
        local_path = DEST_DIR / name
        url = download_url(delivery_id, item_id)

        # Check if already downloaded AND file exists on disk
        manifest = load_manifest(DEST_DIR)
        if name not in manifest or not local_path.exists():
            log(f"  [DL] Downloading {name} ({item.get('fileSize', '?')})...")
            t_dl = time.time()
            try:
                result = download_item(url, local_path, sha1_exp, size_b, name)
            except Exception as e:
                log(f"  [DL] ERROR: {e} — retrying in 30s...")
                time.sleep(30)
                try:
                    result = download_item(url, local_path, sha1_exp, size_b, name)
                except Exception as e2:
                    log(f"  [DL] RETRY FAILED: {e2} — skipping {name}")
                    continue

            if result == "failed":
                log(f"  [DL] Download failed for {name}, skipping")
                continue

            # Verify SHA1
            if sha1_exp:
                actual = sha1_file(local_path)
                if actual != sha1_exp:
                    log(f"  [DL] CHECKSUM FAIL: {name} — expected {sha1_exp[:12]}, "
                        f"got {actual[:12]}")
                    local_path.unlink(missing_ok=True)
                    continue

            # Update manifest
            manifest[name] = {
                "item_id": item_id,
                "sha1": sha1_exp,
                "size": size_b,
                "result": result,
                "completed_at": datetime.utcnow().isoformat() + "Z",
            }
            save_manifest(DEST_DIR, manifest)
            dl_elapsed = time.time() - t_dl
            speed = human_size(size_b / dl_elapsed) if dl_elapsed > 0 else "?"
            log(f"  [DL] {name} downloaded in {dl_elapsed:.0f}s ({speed}/s)")

        # Verify file exists before queueing
        if not local_path.exists():
            log(f"  [DL] WARNING: {name} not on disk after download — skipping")
            continue

        # Put on queue (blocks if queue full — backpressure)
        dl_queue.put((str(local_path), name, item))

    # Sentinel: signal no more files
    dl_queue.put(None)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Download + ingest DOCDB into normalized epo_* tables")
    ap.add_argument("--start-from", type=int, default=1,
                    help="Start from file number N (1-indexed)")
    ap.add_argument("--no-delete", action="store_true",
                    help="Keep ZIPs after ingest (default: delete)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Download + parse only, no DB writes")
    args = ap.parse_args()

    if not HAVE_LXML:
        print("ERROR: lxml required"); sys.exit(1)

    delete_after = not args.no_delete

    # ── Fetch delivery info ──
    log("Fetching BDDS product catalogue...")
    product = fetch_product()
    delivery = find_delivery(product, DEFAULT_DELIVERY_NAME)
    if not delivery:
        log("ERROR: delivery not found"); sys.exit(1)

    items = sorted(delivery.get("items", []), key=lambda x: x["itemName"])
    d_id = delivery["deliveryId"]
    log(f"Delivery: {delivery['deliveryName']} — {len(items)} files")

    # Apply --start-from filter
    if args.start_from > 1:
        items = items[args.start_from - 1:]
        log(f"Starting from file #{args.start_from}, {len(items)} files remaining")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load tracker ──
    tracker = load_tracker()
    already_done = sum(1 for item in items if item["itemName"] in tracker)
    log(f"Tracker: {already_done} files already ingested, "
        f"{len(items) - already_done} remaining")

    # ── DB connection ──
    if not args.dry_run:
        conn = psycopg2.connect(
            host="/var/run/postgresql", dbname="arc_v4", user="jeff")
        # Session settings for safe, fast ingest
        with conn.cursor() as cur:
            cur.execute("SET synchronous_commit = off")
            cur.execute("SET work_mem = '64MB'")
            cur.execute("SET max_parallel_workers_per_gather = 0")
        conn.commit()
        log("DB connected — synchronous_commit=off, no parallel workers")
        person_cache = PersonCache()
    else:
        conn = None
        person_cache = None

    # ── Start download prefetch thread ──
    dl_queue: queue.Queue = queue.Queue(maxsize=2)
    dl_thread = threading.Thread(
        target=downloader_thread,
        args=(items, d_id, dl_queue, tracker),
        daemon=True,
    )
    dl_thread.start()
    log("Download prefetch thread started (queue size=2)")

    # ── Ingest loop ──
    grand_total = 0
    grand_counts: dict[str, int] = {}
    grand_errors = 0
    files_done = already_done
    t_start = time.time()

    log(f"=== EPO INGEST START === delete_after={delete_after}")

    while not _shutdown.is_set():
        # Pull next file from download queue
        try:
            item = dl_queue.get(timeout=5)
        except queue.Empty:
            continue

        if item is None:
            # Sentinel — all files processed
            break

        zip_path, zip_name, item_meta = item

        # Skip if already in tracker (downloader should filter, but double-check)
        if zip_name in tracker:
            log(f"  Skipping {zip_name} (already ingested)")
            continue

        # Check disk space
        stat = os.statvfs(str(DEST_DIR))
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        if free_gb < MIN_FREE_GB:
            log(f"  WARNING: Low disk space ({free_gb:.1f} GB free, "
                f"min {MIN_FREE_GB} GB). Pausing...")
            while free_gb < MIN_FREE_GB and not _shutdown.is_set():
                time.sleep(60)
                stat = os.statvfs(str(DEST_DIR))
                free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            if _shutdown.is_set():
                break

        log(f"--- [{files_done + 1}/{len(items)}] {zip_name} "
            f"({item_meta.get('fileSize', '?')}) --- disk: {free_gb:.0f} GB free")

        # ── Ingest ──
        if args.dry_run:
            log(f"  [DRY RUN] Would ingest {zip_name}")
            files_done += 1
            if delete_after and Path(zip_path).exists():
                Path(zip_path).unlink()
                log(f"  Deleted {zip_name}")
            continue

        t_ing = time.time()
        try:
            stats = ingest_zip(
                conn, zip_path, person_cache,
                error_log=ERROR_LOG,
                delete_after=delete_after,
                progress_file=PROGRESS_FILE,
            )
        except Exception as e:
            log(f"  INGEST ERROR: {e}")
            # Reconnect and retry
            try:
                conn.close()
            except Exception:
                pass
            conn = psycopg2.connect(
                host="/var/run/postgresql", dbname="arc_v4", user="jeff")
            with conn.cursor() as cur:
                cur.execute("SET synchronous_commit = off")
                cur.execute("SET work_mem = '64MB'")
                cur.execute("SET max_parallel_workers_per_gather = 0")
            conn.commit()
            log("  Reconnected to DB, retrying...")
            try:
                stats = ingest_zip(
                    conn, zip_path, person_cache,
                    error_log=ERROR_LOG,
                    delete_after=delete_after,
                    progress_file=PROGRESS_FILE,
                )
            except Exception as e2:
                log(f"  INGEST RETRY FAILED: {e2}")
                continue

        # Update tracker
        tracker[zip_name] = {
            "total": stats["total"],
            "counts": stats["counts"],
            "errors": stats["errors"],
            "elapsed": stats["elapsed"],
            "completed_at": datetime.utcnow().isoformat() + "Z",
        }
        save_tracker(tracker)

        # Update grand totals
        grand_total += stats["total"]
        grand_errors += stats["errors"]
        for k, v in stats["counts"].items():
            grand_counts[k] = grand_counts.get(k, 0) + v
        files_done += 1

        # ETA
        total_elapsed = time.time() - t_start
        remaining_files = len(items) - files_done
        if files_done > already_done:
            avg_per_file = total_elapsed / (files_done - already_done)
            eta_hours = (remaining_files * avg_per_file) / 3600
        else:
            eta_hours = 0

        doc_ins = grand_counts.get("document", 0)
        log(f"  RUNNING: {doc_ins:,} docs, {grand_errors:,} errors, "
            f"{files_done}/{len(items)} files, "
            f"persons: {person_cache.stats['inserts']:,} new / "
            f"{person_cache.stats['hits']:,} cached, "
            f"ETA {eta_hours:.1f}h")

    # ── Done ──
    total_elapsed = time.time() - t_start
    doc_ins = grand_counts.get("document", 0)

    log(f"=== ALL FILES PROCESSED === "
        f"{grand_total:,} seen, {doc_ins:,} docs, "
        f"{grand_errors:,} errors, {total_elapsed/3600:.1f}h")
    log(f"Persons: {person_cache.stats['inserts']:,} new / "
        f"{person_cache.stats['hits']:,} cached")
    log(f"Per-table totals: {json.dumps(grand_counts, indent=2)}")

    if conn:
        conn.close()

    # Check for unknown XML paths
    unknown_log = Path("/tmp/epo_unknown_xml_paths.jsonl")
    if unknown_log.exists() and unknown_log.stat().st_size > 0:
        n_unknown = sum(1 for _ in open(unknown_log))
        log(f"WARNING: {n_unknown} unknown XML paths detected — "
            f"review {unknown_log}")


if __name__ == "__main__":
    main()

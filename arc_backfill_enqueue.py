#!/usr/bin/env python3
"""
arc_backfill_enqueue.py — Upload patent TSVs to R2 and print indexed file list.

Uploads all TSV.gz files from ~/arc/exports/patents/ to R2 backfill/input/.
Prints the full file list with indexes for assigning ranges to workers.

Usage:
    python3 arc_backfill_enqueue.py                    # upload all + print list
    python3 arc_backfill_enqueue.py --list-only         # just print R2 file list
    python3 arc_backfill_enqueue.py --filter H01L       # upload matching only

Environment:
    R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

PATENTS_DIR = Path.home() / "arc" / "exports" / "patents"
R2_INPUT_PREFIX = "backfill/input/"


def make_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"],
    )


def list_r2_files(s3, bucket: str) -> list[str]:
    """List all files under backfill/input/ in R2."""
    files = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=R2_INPUT_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".tsv.gz"):
                files.append(key)
    files.sort()
    return files


def print_file_list(files: list[str]):
    """Print indexed file list for worker assignment."""
    print(f"\n{'='*70}")
    print(f"R2 file list: {len(files)} files under {R2_INPUT_PREFIX}")
    print(f"{'='*70}")
    for i, key in enumerate(files):
        fname = key.split("/")[-1]
        print(f"  {i:>4d}  {fname}")
    print(f"{'='*70}")
    print(f"Total: {len(files)} files")
    print(f"\nWorker assignment examples (4 workers):")
    chunk = len(files) // 4
    for w in range(4):
        s = w * chunk
        e = (w + 1) * chunk if w < 3 else len(files)
        print(f"  Worker {w+1}: --start {s} --end {e}  ({e-s} files)")
    print()


def main():
    ap = argparse.ArgumentParser(description="Upload patent TSVs to R2")
    ap.add_argument("--patents-dir", type=Path, default=PATENTS_DIR)
    ap.add_argument("--filter", type=str, help="Only upload files matching this substring")
    ap.add_argument("--list-only", action="store_true", help="Just list R2 files, no upload")
    ap.add_argument("--workers", type=int, default=8, help="Parallel upload threads (default: 8)")
    args = ap.parse_args()

    bucket = os.environ.get("R2_BUCKET", "arc-cloud")
    s3 = make_r2_client()

    if args.list_only:
        files = list_r2_files(s3, bucket)
        print_file_list(files)
        return

    # Get existing R2 files to skip re-uploads
    existing = set(list_r2_files(s3, bucket))
    print(f"Existing files in R2: {len(existing)}")

    # Scan local files
    local_files = sorted(args.patents_dir.glob("patents_*.tsv.gz"))
    if args.filter:
        local_files = [f for f in local_files if args.filter in f.name]

    # Filter out empty files and already-uploaded
    to_upload = []
    for fpath in local_files:
        if fpath.stat().st_size == 0:
            continue
        r2_key = R2_INPUT_PREFIX + fpath.name
        if r2_key in existing:
            continue
        to_upload.append(fpath)

    print(f"Local files: {len(local_files)}, to upload: {len(to_upload)}, "
          f"skipping: {len(local_files) - len(to_upload)}")

    if not to_upload:
        print("Nothing to upload.")
        files = list_r2_files(s3, bucket)
        print_file_list(files)
        return

    # Upload in parallel
    n_done = 0
    n_failed = 0
    t0 = time.time()

    def upload_one(fpath):
        r2_key = R2_INPUT_PREFIX + fpath.name
        s3_thread = make_r2_client()  # thread-safe: one client per thread
        s3_thread.upload_file(str(fpath), bucket, r2_key)
        return fpath.name

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(upload_one, f): f for f in to_upload}
        for future in as_completed(futures):
            try:
                fname = future.result()
                n_done += 1
                if n_done % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n_done}/{len(to_upload)} uploaded ({elapsed:.0f}s)")
            except Exception as e:
                n_failed += 1
                fpath = futures[future]
                print(f"  FAILED: {fpath.name}: {e}")

    elapsed = time.time() - t0
    print(f"\nUpload done in {elapsed:.0f}s: {n_done} uploaded, {n_failed} failed")

    # Print final file list
    files = list_r2_files(s3, bucket)
    print_file_list(files)


if __name__ == "__main__":
    main()

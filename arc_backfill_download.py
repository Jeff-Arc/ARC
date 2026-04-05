#!/usr/bin/env python3
"""
arc_backfill_download.py — Poll R2 for completed backfill outputs and download.

Watches R2 backfill/output/ for new embeddings.tsv.gz and knn.tsv.gz files.
Downloads them to ~/arc/exports/embeddings/ as they arrive.

Usage:
    python3 arc_backfill_download.py                    # poll until all done
    python3 arc_backfill_download.py --once              # single pass, no polling
    python3 arc_backfill_download.py --poll-interval 30  # check every 30s

Environment:
    R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

R2_INPUT_PREFIX = "backfill/input/"
R2_OUTPUT_PREFIX = "backfill/output/"
OUTPUT_DIR = Path.home() / "arc" / "exports" / "embeddings"

sys.stdout.reconfigure(line_buffering=True)

_shutdown = False

def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    print("\nShutdown signal — finishing current downloads")

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def make_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"],
    )


def list_r2_keys(s3, bucket: str, prefix: str) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def download_pass(s3, bucket: str, output_dir: Path, downloaded: set) -> int:
    """Download new output files from R2. Returns count of new downloads."""
    output_keys = list_r2_keys(s3, bucket, R2_OUTPUT_PREFIX)
    new_keys = [k for k in output_keys if k not in downloaded]

    if not new_keys:
        return 0

    n_new = 0
    for key in new_keys:
        # key: backfill/output/patents_H01L.tsv.gz/embeddings.tsv.gz
        # local: embeddings/patents_H01L.tsv.gz/embeddings.tsv.gz
        rel_path = key[len(R2_OUTPUT_PREFIX):]
        local_path = output_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists():
            downloaded.add(key)
            continue

        s3.download_file(bucket, key, str(local_path))
        downloaded.add(key)
        n_new += 1

    return n_new


def main():
    ap = argparse.ArgumentParser(description="Download backfill outputs from R2")
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--once", action="store_true", help="Single pass, no polling")
    ap.add_argument("--poll-interval", type=int, default=60, help="Seconds between polls")
    args = ap.parse_args()

    bucket = os.environ.get("R2_BUCKET", "arc-cloud")
    s3 = make_r2_client()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Count expected total from input files
    input_keys = list_r2_keys(s3, bucket, R2_INPUT_PREFIX)
    n_input = len([k for k in input_keys if k.endswith(".tsv.gz")])
    print(f"Input files in R2: {n_input}")

    # Track already-downloaded files
    downloaded = set()

    # Seed with files already on disk
    for local_file in args.output_dir.rglob("*.tsv.gz"):
        rel = str(local_file.relative_to(args.output_dir))
        r2_key = R2_OUTPUT_PREFIX + rel
        downloaded.add(r2_key)
    print(f"Already downloaded: {len(downloaded)} files")

    pass_num = 0
    while not _shutdown:
        pass_num += 1
        t0 = time.time()

        n_new = download_pass(s3, bucket, args.output_dir, downloaded)

        # Count completed (each input file produces 1-2 output files; count by directory)
        output_dirs = set()
        for key in downloaded:
            # backfill/output/patents_X.tsv.gz/embeddings.tsv.gz → patents_X.tsv.gz
            parts = key[len(R2_OUTPUT_PREFIX):].split("/")
            if len(parts) >= 2:
                output_dirs.add(parts[0])

        elapsed = time.time() - t0
        print(f"[pass {pass_num}] Downloaded {n_new} new files | "
              f"{len(output_dirs)}/{n_input} inputs complete | "
              f"{elapsed:.1f}s")

        if len(output_dirs) >= n_input:
            print(f"\nAll {n_input} inputs have outputs. Done!")
            break

        if args.once:
            break

        time.sleep(args.poll_interval)

    print(f"\nOutput directory: {args.output_dir}")
    print(f"Total files downloaded: {len(downloaded)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
arc_docdb_download.py — Download EPO DOCDB back file ZIPs from BDDS.

SOURCE:
  EPO Bulk Data Delivery Service (BDDS) — public, no authentication required.
  Product 14: "EPO worldwide bibliographic data (DOCDB) back file"
  Delivery 3071: "DOCDB Back file February 2026"
  162 files, ~219 GB total

API:
  Product info (all deliveries + items inline):
    GET https://publication-bdds.apps.epo.org/bdds/bdds-bff-service/prod/api/public/products/14
  Download one item:
    GET .../delivery/{delivery_id}/item/{item_id}/download
  Supports: Range requests (HTTP 206) for resumable downloads.
  No auth, no rate limit headers observed.

FILE NAMING:
  docdb_xml_bck_202607_NNN_X.zip
    202607 = version (YYYYWW: week 07 of 2026)
    NNN    = batch number (001-031, roughly by CPC section)
    X      = part letter (A-M within batch, ~1.5 GB each)

RESUME SAFETY:
  - Skip: local file size == expected content-length (download complete)
  - Resume: 0 < local size < expected → Range: bytes={size}- request
  - SHA1 checksum verified after every completed download
  - Manifest file records completed downloads with checksums

Usage:
  python3 arc_docdb_download.py --mode list
  python3 arc_docdb_download.py --mode download
  python3 arc_docdb_download.py --mode download --delivery 3071
  python3 arc_docdb_download.py --mode download --dest /home/jeff/data/docdb_backfile
  python3 arc_docdb_download.py --mode download --file docdb_xml_bck_202607_001_A.zip
  python3 arc_docdb_download.py --mode verify   # re-verify checksums of downloaded files
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ── Config ─────────────────────────────────────────────────────────────────────

BDDS_BASE    = "https://publication-bdds.apps.epo.org/bdds/bdds-bff-service/prod/api/public"
PRODUCT_ID   = 14
DEFAULT_DEST = Path.home() / "data" / "docdb_backfile"

# Target delivery name substring (case-insensitive match)
DEFAULT_DELIVERY_NAME = "back file february 2026"

# Chunk size for streaming downloads: 8 MB
DOWNLOAD_CHUNK = 8 * 1024 * 1024


# ── API ────────────────────────────────────────────────────────────────────────

def fetch_product(product_id: int = PRODUCT_ID) -> dict:
    """
    Fetch product metadata including all deliveries and items.
    Returns the full product dict.  All delivery items are inline — no
    separate /delivery/{id}/items call needed.
    """
    url  = f"{BDDS_BASE}/products/{product_id}"
    resp = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def find_delivery(product: dict, name_substr: str) -> dict | None:
    """
    Find a delivery by case-insensitive name substring.
    Returns the first matching delivery dict, or None.
    """
    needle = name_substr.lower()
    for d in product.get("deliveries", []):
        if needle in d.get("deliveryName", "").lower():
            return d
    return None


def get_delivery_by_id(product: dict, delivery_id: int) -> dict | None:
    """Return the delivery dict with the given ID, or None."""
    for d in product.get("deliveries", []):
        if d.get("deliveryId") == delivery_id:
            return d
    return None


def download_url(delivery_id: int, item_id: int) -> str:
    return (f"{BDDS_BASE}/products/{PRODUCT_ID}/delivery/{delivery_id}"
            f"/item/{item_id}/download")


# ── Size parsing ───────────────────────────────────────────────────────────────

def parse_size_bytes(size_str: str) -> float:
    """
    Convert BDDS file size string ('1.5 GB', '504.1 kB', '78.9 MB') to bytes.
    Returns 0.0 on parse failure.
    """
    try:
        val, unit = size_str.strip().split()
        val = float(val)
        unit = unit.upper()
        return val * {"KB": 1024, "MB": 1024**2, "GB": 1024**3}.get(unit, 1)
    except (ValueError, AttributeError):
        return 0.0


def human_size(n_bytes: float) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


# ── Checksum ───────────────────────────────────────────────────────────────────

def sha1_file(path: Path) -> str:
    """Compute SHA1 hex digest of a file (uppercase)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(DOWNLOAD_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest().upper()


# ── Manifest ───────────────────────────────────────────────────────────────────

def load_manifest(dest: Path) -> dict:
    """
    Load completed-downloads manifest from dest/.docdb_manifest.json.
    Returns {filename: {sha1, size, completed_at}}.
    """
    path = dest / ".docdb_manifest.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_manifest(dest: Path, manifest: dict) -> None:
    path = dest / ".docdb_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))


# ── List mode ──────────────────────────────────────────────────────────────────

def cmd_list(delivery: dict, dest: Path) -> None:
    """
    Print all files in the delivery with sizes and download status.
    """
    items     = sorted(delivery.get("items", []), key=lambda x: x["itemName"])
    manifest  = load_manifest(dest)
    total_b   = 0.0
    done_b    = 0.0
    done_n    = 0

    print(f"\n{'='*72}")
    print(f"Delivery: {delivery['deliveryName']}")
    print(f"Published: {delivery['deliveryPublicationDatetime']}")
    print(f"Files: {len(items)}")
    print(f"{'='*72}")
    print(f"{'#':>4}  {'Item ID':>8}  {'Filename':<45}  {'Size':>10}  Status")
    print(f"{'-'*4}  {'-'*8}  {'-'*45}  {'-'*10}  {'-'*10}")

    for i, item in enumerate(items, 1):
        name     = item["itemName"]
        item_id  = item["itemId"]
        size_str = item.get("fileSize", "?")
        size_b   = parse_size_bytes(size_str)
        total_b += size_b

        local_path = dest / name
        if name in manifest:
            status = "✓ complete"
            done_b += size_b
            done_n += 1
        elif local_path.exists():
            local_sz = local_path.stat().st_size
            if local_sz == 0:
                status = "empty"
            else:
                pct = local_sz / size_b * 100 if size_b > 0 else 0
                status = f"partial {pct:.0f}%"
            done_b += local_sz
        else:
            status = "missing"

        print(f"{i:>4}  {item_id:>8}  {name:<45}  {size_str:>10}  {status}")

    print(f"{'='*72}")
    print(f"Total:   {len(items)} files,  {human_size(total_b)} total")
    print(f"Done:    {done_n} complete,  {human_size(done_b)} downloaded so far")
    remaining = total_b - done_b
    print(f"Pending: {human_size(remaining)} remaining")
    print(f"{'='*72}\n")


# ── Download mode ─────────────────────────────────────────────────────────────

def download_item(
    url:          str,
    dest_path:    Path,
    expected_sha: str,
    expected_size: int,
    item_name:    str,
) -> str:
    """
    Download one item to dest_path with resume support.

    Returns: 'skipped' | 'resumed' | 'downloaded' | 'failed'
    """
    local_size = dest_path.stat().st_size if dest_path.exists() else 0

    # Already complete?
    if local_size == expected_size:
        return "skipped"

    headers = {}
    mode    = "wb"
    if 0 < local_size < expected_size:
        headers["Range"] = f"bytes={local_size}-"
        mode = "ab"
        status = "resumed"
    else:
        status = "downloaded"

    resp = requests.get(url, headers=headers, stream=True, timeout=60)

    if resp.status_code not in (200, 206):
        print(f"  [ERROR] HTTP {resp.status_code} for {item_name}", file=sys.stderr)
        return "failed"

    content_length = int(resp.headers.get("content-length", 0))
    total_expected = local_size + content_length if mode == "ab" else content_length
    downloaded     = local_size

    t0 = time.time()
    with open(dest_path, mode) as f:
        for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                elapsed     = time.time() - t0
                rate        = (downloaded - local_size) / elapsed if elapsed > 0 else 0
                pct         = downloaded / total_expected * 100 if total_expected > 0 else 0
                print(f"\r  {pct:5.1f}%  {human_size(downloaded):>10} / "
                      f"{human_size(total_expected):<10}  "
                      f"{human_size(rate)}/s    ", end="", flush=True)

    print()  # newline after progress line
    return status


def cmd_download(
    delivery:    dict,
    dest:        Path,
    file_filter: str | None = None,
) -> None:
    """
    Download all items in a delivery to dest/.
    Skips already-complete files; resumes partial files.
    Verifies SHA1 after each download.
    """
    dest.mkdir(parents=True, exist_ok=True)
    items    = sorted(delivery.get("items", []), key=lambda x: x["itemName"])
    manifest = load_manifest(dest)
    d_id     = delivery["deliveryId"]

    if file_filter:
        items = [i for i in items if file_filter in i["itemName"]]
        if not items:
            print(f"  No items match filter '{file_filter}'", file=sys.stderr)
            return

    total_b   = sum(parse_size_bytes(i.get("fileSize", "0")) for i in items)
    pending   = [i for i in items if i["itemName"] not in manifest]
    skippable = len(items) - len(pending)

    print(f"\n{'='*72}")
    print(f"Delivery:  {delivery['deliveryName']}")
    print(f"Dest:      {dest}")
    print(f"Files:     {len(items)} total,  {skippable} already complete,  "
          f"{len(pending)} to download")
    print(f"Remaining: {human_size(sum(parse_size_bytes(i.get('fileSize','0')) for i in pending))}")
    print(f"{'='*72}\n")

    downloaded_n = 0
    failed_n     = 0
    skipped_n    = 0
    session_b    = 0
    t_session    = time.time()

    for i, item in enumerate(items, 1):
        name       = item["itemName"]
        item_id    = item["itemId"]
        size_str   = item.get("fileSize", "?")
        sha1_exp   = item.get("fileChecksum", "").upper()
        size_b     = int(parse_size_bytes(size_str))
        dest_path  = dest / name
        url        = download_url(d_id, item_id)

        # Already verified in manifest
        if name in manifest:
            skipped_n += 1
            print(f"  [{i:>3}/{len(items)}] {name}  — already complete, skipping")
            continue

        print(f"  [{i:>3}/{len(items)}] {name}  ({size_str})")

        try:
            result = download_item(url, dest_path, sha1_exp, size_b, name)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}", file=sys.stderr)
            failed_n += 1
            continue

        if result == "failed":
            failed_n += 1
            continue

        if result == "skipped":
            # File appeared complete but wasn't in manifest — verify and add
            print(f"  [{i:>3}/{len(items)}] {name}  — size matches, verifying checksum...")
        else:
            session_b += size_b

        # Verify SHA1
        if sha1_exp:
            print(f"  Verifying SHA1...", end="", flush=True)
            actual_sha1 = sha1_file(dest_path)
            if actual_sha1 == sha1_exp:
                print(f" OK ({actual_sha1[:12]}...)")
            else:
                print(f"\n  [CHECKSUM FAIL] expected {sha1_exp}, got {actual_sha1}",
                      file=sys.stderr)
                # Delete corrupt file so it can be re-downloaded
                dest_path.unlink(missing_ok=True)
                failed_n += 1
                continue

        # Record in manifest
        manifest[name] = {
            "item_id":      item_id,
            "sha1":         sha1_exp,
            "size":         size_b,
            "result":       result,
            "completed_at": datetime.utcnow().isoformat() + "Z",
        }
        save_manifest(dest, manifest)

        if result in ("downloaded", "resumed"):
            downloaded_n += 1
        else:
            skipped_n += 1

        # Session progress
        elapsed = time.time() - t_session
        if elapsed > 0 and session_b > 0:
            rate = session_b / elapsed
            print(f"  Session: {downloaded_n} downloaded, "
                  f"{human_size(session_b)} in {elapsed/60:.1f} min "
                  f"({human_size(rate)}/s avg)")
        print()

    elapsed = time.time() - t_session
    print(f"\n{'='*72}")
    print(f"Download complete.")
    print(f"  Downloaded (new/resumed): {downloaded_n}")
    print(f"  Skipped (already done):   {skipped_n}")
    print(f"  Failed:                   {failed_n}")
    print(f"  Session data:             {human_size(session_b)}")
    print(f"  Session time:             {elapsed/60:.1f} min")
    if elapsed > 0 and session_b > 0:
        print(f"  Average speed:            {human_size(session_b/elapsed)}/s")
    print(f"{'='*72}\n")


# ── Verify mode ────────────────────────────────────────────────────────────────

def cmd_verify(delivery: dict, dest: Path) -> None:
    """
    Re-verify SHA1 checksums of all files present in dest.
    Updates manifest for any files not yet recorded.
    """
    items    = {i["itemName"]: i for i in delivery.get("items", [])}
    manifest = load_manifest(dest)
    files    = sorted(dest.glob("docdb_xml_bck_*.zip"))

    print(f"\nVerifying {len(files)} files in {dest}...\n")

    ok = fails = missing = 0
    for path in files:
        name = path.name
        item = items.get(name)
        if not item:
            print(f"  UNKNOWN  {name}  (not in delivery)")
            continue

        sha1_exp = item.get("fileChecksum", "").upper()
        if not sha1_exp:
            print(f"  NO_HASH  {name}")
            continue

        print(f"  Checking {name} ...", end="", flush=True)
        actual = sha1_file(path)
        if actual == sha1_exp:
            print(f" OK")
            ok += 1
            if name not in manifest:
                manifest[name] = {
                    "item_id":      item["itemId"],
                    "sha1":         sha1_exp,
                    "size":         path.stat().st_size,
                    "result":       "verified",
                    "completed_at": datetime.utcnow().isoformat() + "Z",
                }
        else:
            print(f" FAIL  expected {sha1_exp[:12]}...  got {actual[:12]}...")
            fails += 1

    save_manifest(dest, manifest)
    print(f"\nResults: {ok} OK,  {fails} FAILED,  {missing} missing")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Download EPO DOCDB back file ZIPs from BDDS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 arc_docdb_download.py --mode list
  python3 arc_docdb_download.py --mode download
  python3 arc_docdb_download.py --mode download --file docdb_xml_bck_202607_001_A.zip
  python3 arc_docdb_download.py --mode download --delivery 3071
  python3 arc_docdb_download.py --mode verify
        """,
    )
    ap.add_argument(
        "--mode",
        choices=["list", "download", "verify"],
        required=True,
        help=(
            "list: show all files and sizes, no downloads | "
            "download: fetch all files, resume-safe | "
            "verify: re-check SHA1 checksums of downloaded files"
        ),
    )
    ap.add_argument(
        "--delivery",
        type=int,
        default=None,
        help="Delivery ID to target (default: auto-find 'DOCDB Back file February 2026')",
    )
    ap.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})",
    )
    ap.add_argument(
        "--file",
        help="Download only a specific file by name (substring match)",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    print("Fetching BDDS product catalogue...", end=" ", flush=True)
    product  = fetch_product()
    print(f"OK  ({len(product.get('deliveries', []))} deliveries)")

    if args.delivery:
        delivery = get_delivery_by_id(product, args.delivery)
        if not delivery:
            print(f"ERROR: delivery ID {args.delivery} not found", file=sys.stderr)
            print("Available deliveries:")
            for d in product.get("deliveries", []):
                print(f"  {d['deliveryId']:6d}  {d['deliveryName']}")
            sys.exit(1)
    else:
        delivery = find_delivery(product, DEFAULT_DELIVERY_NAME)
        if not delivery:
            print(f"ERROR: no delivery matching '{DEFAULT_DELIVERY_NAME}'",
                  file=sys.stderr)
            print("Available deliveries:")
            for d in product.get("deliveries", []):
                print(f"  {d['deliveryId']:6d}  {d['deliveryName']}")
            sys.exit(1)

    if args.mode == "list":
        cmd_list(delivery, args.dest)

    elif args.mode == "download":
        cmd_download(delivery, args.dest, file_filter=args.file)

    elif args.mode == "verify":
        cmd_verify(delivery, args.dest)


if __name__ == "__main__":
    main()

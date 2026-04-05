#!/usr/bin/env python3
"""
arc_patent_export.py — Export 95M deduplicated patents to per-CPC gzipped TSV files.

Chunked by date range — each chunk is a short independent transaction.
No temp tables, no long-running cursors, no WAL accumulation.

Driving table: arc_v5.epo_document_dedup (permanent, 95M rows)
Output: ~/arc/exports/patents/patents_{CPC}.tsv.gz
        ~/arc/exports/patents/patents_no_cpc_{NNNNNN}.tsv.gz (5000 rows each)
"""

import gzip
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import psycopg2

DB_PARAMS = {
    "dbname": "arc_v4",
    "user": "jeff",
    "host": "/var/run/postgresql",
}

OUTPUT_DIR = Path("/home/jeff/arc/exports/patents")
FETCH_SIZE = 50_000
NO_CPC_BATCH_SIZE = 5_000
LOG_INTERVAL = 1_000_000

TSV_HEADER = "doc_id\tfamily_id\tdate_publ\tcountry\ttext_content\ttext_source\tprimary_cpc\tipcr_codes\n"

# Date chunks — each becomes one short transaction
CHUNKS = [
    ("pre-1970",  "date_publ < '1970-01-01'"),
    ("1970-1979", "date_publ >= '1970-01-01' AND date_publ < '1980-01-01'"),
    ("1980-1984", "date_publ >= '1980-01-01' AND date_publ < '1985-01-01'"),
    ("1985-1989", "date_publ >= '1985-01-01' AND date_publ < '1990-01-01'"),
    ("1990-1995", "date_publ >= '1990-01-01' AND date_publ < '1996-01-01'"),
    ("1996-2000", "date_publ >= '1996-01-01' AND date_publ < '2001-01-01'"),
    ("2001-2004", "date_publ >= '2001-01-01' AND date_publ < '2005-01-01'"),
    ("2005-2008", "date_publ >= '2005-01-01' AND date_publ < '2009-01-01'"),
    ("2009-2012", "date_publ >= '2009-01-01' AND date_publ < '2013-01-01'"),
    ("2013-2016", "date_publ >= '2013-01-01' AND date_publ < '2017-01-01'"),
    ("2017-2018", "date_publ >= '2017-01-01' AND date_publ < '2019-01-01'"),
    ("2019-2020", "date_publ >= '2019-01-01' AND date_publ < '2021-01-01'"),
    ("2021-2022", "date_publ >= '2021-01-01' AND date_publ < '2023-01-01'"),
    ("2023-2024", "date_publ >= '2023-01-01' AND date_publ < '2025-01-01'"),
    ("2025+",     "date_publ >= '2025-01-01'"),
    ("null-dates", "date_publ IS NULL"),
]

CHUNK_QUERY_TEMPLATE = """
SELECT
    d.doc_id,
    d.family_id,
    d.date_publ::text,
    d.country,
    a_en.abstract_text,
    a_any.abstract_text  AS abstract_any,
    t_en.title_text      AS title_en,
    t_any.title_text     AS title_any,
    pc.cpc_subclass      AS primary_cpc,
    ipcr.ipcr_codes
FROM arc_v5.epo_document_dedup d
LEFT JOIN LATERAL (
    SELECT abstract_text FROM arc_v5.epo_abstract
    WHERE doc_id = d.doc_id AND lang = 'en' AND abstract_text IS NOT NULL
    LIMIT 1
) a_en ON true
LEFT JOIN LATERAL (
    SELECT abstract_text FROM arc_v5.epo_abstract
    WHERE doc_id = d.doc_id AND abstract_text IS NOT NULL
    LIMIT 1
) a_any ON a_en.abstract_text IS NULL
LEFT JOIN LATERAL (
    SELECT title_text FROM arc_v5.epo_title
    WHERE doc_id = d.doc_id AND lang = 'en' AND title_text IS NOT NULL
    LIMIT 1
) t_en ON a_en.abstract_text IS NULL AND a_any.abstract_text IS NULL
LEFT JOIN LATERAL (
    SELECT title_text FROM arc_v5.epo_title
    WHERE doc_id = d.doc_id AND title_text IS NOT NULL
    LIMIT 1
) t_any ON a_en.abstract_text IS NULL AND a_any.abstract_text IS NULL AND t_en.title_text IS NULL
LEFT JOIN LATERAL (
    SELECT cpc_subclass FROM arc_v5.epo_patent_classification
    WHERE doc_id = d.doc_id AND cpc_subclass IS NOT NULL
    ORDER BY sequence LIMIT 1
) pc ON true
LEFT JOIN LATERAL (
    SELECT string_agg(ipc_subclass, '|') AS ipcr_codes
    FROM arc_v5.epo_classification_ipcr
    WHERE doc_id = d.doc_id AND ipc_subclass IS NOT NULL
) ipcr ON true
WHERE {date_filter}
"""


def clean_tsv(val):
    if val is None:
        return ""
    return str(val).replace("\t", " ").replace("\n", " ").replace("\r", "")


def resolve_text(abstract_en, abstract_any, title_en, title_any):
    if abstract_en:
        return abstract_en, "abstract_en"
    if abstract_any:
        return abstract_any, "abstract_xx"
    if title_en:
        return title_en, "title_en"
    if title_any:
        return title_any, "title_xx"
    return "", ""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Track completed chunks for resumability
    chunk_done_file = OUTPUT_DIR / ".chunks_done"
    chunks_done = set()
    if chunk_done_file.exists():
        chunks_done = set(chunk_done_file.read_text().strip().split("\n"))
        print(f"Resuming: {len(chunks_done)} chunks already done")

    # File handles — persist across chunks (append mode)
    writers = {}       # cpc -> gzip file handle
    no_cpc_rows = []
    no_cpc_file_num = 0

    # Count existing no_cpc files to resume numbering
    for f in OUTPUT_DIR.glob("patents_no_cpc_*.tsv.gz"):
        n = int(f.stem.split("_")[-1].replace(".tsv", ""))
        no_cpc_file_num = max(no_cpc_file_num, n)

    rows_total = 0
    rows_written = 0
    t_global_start = time.time()

    def get_writer(cpc):
        if cpc in writers:
            return writers[cpc]
        fpath = OUTPUT_DIR / f"patents_{cpc}.tsv.gz"
        # Append if file exists (from previous chunk), otherwise create with header
        if fpath.exists():
            f = gzip.open(fpath, "at", compresslevel=6)
        else:
            f = gzip.open(fpath, "wt", compresslevel=6)
            f.write(TSV_HEADER)
        writers[cpc] = f
        return f

    def flush_no_cpc_batch():
        nonlocal no_cpc_file_num, no_cpc_rows, rows_written
        if not no_cpc_rows:
            return
        no_cpc_file_num += 1
        fpath = OUTPUT_DIR / f"patents_no_cpc_{no_cpc_file_num:06d}.tsv.gz"
        with gzip.open(fpath, "wt", compresslevel=6) as f:
            f.write(TSV_HEADER)
            for line in no_cpc_rows:
                f.write(line)
        rows_written += len(no_cpc_rows)
        no_cpc_rows = []

    def format_row(row):
        doc_id, family_id, date_publ, country, \
            abstract_en, abstract_any, title_en, title_any, \
            primary_cpc, ipcr_codes = row
        text_content, text_source = resolve_text(
            abstract_en, abstract_any, title_en, title_any
        )
        return "\t".join([
            clean_tsv(doc_id),
            clean_tsv(family_id),
            clean_tsv(date_publ),
            clean_tsv(country),
            clean_tsv(text_content),
            clean_tsv(text_source),
            clean_tsv(primary_cpc),
            clean_tsv(ipcr_codes),
        ]) + "\n", primary_cpc

    try:
        for chunk_name, date_filter in CHUNKS:
            if chunk_name in chunks_done:
                print(f"  SKIP chunk {chunk_name} (already done)")
                continue

            print(f"\n{'='*60}")
            print(f"Chunk: {chunk_name}")
            print(f"{'='*60}")

            conn = psycopg2.connect(**DB_PARAMS)
            conn.autocommit = False
            cur = conn.cursor(name=f"chunk_{chunk_name}")
            cur.itersize = FETCH_SIZE

            query = CHUNK_QUERY_TEMPLATE.format(date_filter=date_filter)
            t_chunk = time.time()
            cur.execute(query)

            chunk_rows = 0
            while True:
                rows = cur.fetchmany(FETCH_SIZE)
                if not rows:
                    break
                for row in rows:
                    line, primary_cpc = format_row(row)
                    rows_total += 1
                    chunk_rows += 1

                    if primary_cpc:
                        writer = get_writer(primary_cpc)
                        writer.write(line)
                        rows_written += 1
                    else:
                        no_cpc_rows.append(line)
                        if len(no_cpc_rows) >= NO_CPC_BATCH_SIZE:
                            flush_no_cpc_batch()

                    if rows_total % LOG_INTERVAL == 0:
                        elapsed = time.time() - t_global_start
                        rate = rows_total / elapsed
                        print(
                            f"  {rows_total:>12,} total | "
                            f"{rows_written:,} written | "
                            f"{len(writers)} CPC files | "
                            f"{rate:,.0f} rows/s"
                        )

            cur.close()
            conn.commit()
            conn.close()

            chunk_elapsed = time.time() - t_chunk
            chunk_rate = chunk_rows / chunk_elapsed if chunk_elapsed > 0 else 0
            print(
                f"  Chunk {chunk_name} done: {chunk_rows:,} rows "
                f"in {chunk_elapsed/60:.1f}m ({chunk_rate:,.0f} rows/s)"
            )

            # Flush gzip buffers after each chunk so data is on disk
            for f in writers.values():
                f.flush()

            # Mark chunk as done
            with open(chunk_done_file, "a") as f:
                f.write(chunk_name + "\n")

        # Flush remaining no-CPC rows
        flush_no_cpc_batch()

    finally:
        for f in writers.values():
            f.close()

    elapsed = time.time() - t_global_start
    print(f"\n{'='*60}")
    print(f"EXPORT COMPLETE in {elapsed/60:.1f}m")
    print(f"  Total rows:   {rows_total:,}")
    print(f"  Written:      {rows_written:,}")
    print(f"  CPC files:    {len(writers)}")
    print(f"  No-CPC files: {no_cpc_file_num}")
    print(f"  Output:       {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

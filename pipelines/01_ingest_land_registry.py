"""
Stage 1 — Land Registry Price Paid Data
Ingestion, validation, and cleaning pipeline.

Input:  pp-complete.csv (raw download, ~5GB)
Output: data/processed/land_registry_clean.parquet

Run:
    python pipelines/01_ingest_land_registry.py --input data/raw/pp-complete.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Land Registry PP-Complete column names (no header row in the file)
COLUMNS = [
    "transaction_id",
    "price",
    "date_of_transfer",
    "postcode",
    "property_type",      # D=Detached, S=Semi, T=Terraced, F=Flat, O=Other
    "old_new",            # Y=New build, N=Established
    "duration",           # F=Freehold, L=Leasehold, U=Unknown
    "paon",               # Primary addressable object name (house number/name)
    "saon",               # Secondary addressable object name (flat/unit)
    "street",
    "locality",
    "town_city",
    "district",
    "county",
    "ppd_category",       # A=Standard, B=Additional (e.g. repossessions)
    "record_status",      # A=Addition, C=Change, D=Delete
]

PROPERTY_TYPES = {"D", "S", "T", "F", "O"}
DURATIONS = {"F", "L", "U"}

CHUNK_SIZE = 500_000  # rows per chunk — keeps memory under control

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_chunk(df: pd.DataFrame, chunk_num: int) -> dict:
    """Return a dict of quality metrics for one chunk."""
    return {
        "chunk": chunk_num,
        "rows_in": len(df),
        "null_price": df["price"].isna().sum(),
        "null_postcode": df["postcode"].isna().sum(),
        "null_date": df["date_of_transfer"].isna().sum(),
        "zero_price": (df["price"] == 0).sum(),
        "bad_property_type": (~df["property_type"].isin(PROPERTY_TYPES)).sum(),
    }


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning rules to one chunk. Returns cleaned DataFrame."""

    rows_before = len(df)

    # 1. Drop rows where core fields are null
    df = df.dropna(subset=["price", "postcode", "date_of_transfer"])

    # 2. Drop zero/negative prices (data errors)
    df = df[df["price"] > 0]

    # 3. Keep only standard transactions (category A) and active records
    df = df[df["ppd_category"] == "A"]
    df = df[df["record_status"] == "A"]

    # 4. Parse and type-cast
    df["price"] = df["price"].astype(int)
    df["date_of_transfer"] = pd.to_datetime(df["date_of_transfer"], errors="coerce")
    df = df.dropna(subset=["date_of_transfer"])  # drop unparseable dates

    # 5. Normalise strings
    df["postcode"] = df["postcode"].str.strip().str.upper()
    df["town_city"] = df["town_city"].str.strip().str.title()
    df["county"] = df["county"].str.strip().str.title()
    df["property_type"] = df["property_type"].str.strip().str.upper()

    # 6. Extract year/month for easier aggregation later
    df["year"] = df["date_of_transfer"].dt.year
    df["month"] = df["date_of_transfer"].dt.month

    rows_after = len(df)
    log.debug(f"  Chunk cleaned: {rows_before} → {rows_after} rows "
              f"({rows_before - rows_after} dropped)")

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(input_path: str, output_dir: str = "data/processed") -> None:

    input_file = Path(input_path)
    output_file = Path(output_dir) / "land_registry_clean.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Input:  {input_file}  ({input_file.stat().st_size / 1e9:.2f} GB)")
    log.info(f"Output: {output_file}")
    log.info(f"Chunk size: {CHUNK_SIZE:,} rows")

    all_chunks = []
    validation_log = []
    total_rows_in = 0
    total_rows_out = 0

    reader = pd.read_csv(
        input_file,
        names=COLUMNS,
        header=None,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding="utf-8",
        on_bad_lines="skip",   # Land Registry file has occasional malformed rows
    )

    for i, chunk in enumerate(reader):
        log.info(f"Processing chunk {i+1}  ({len(chunk):,} rows)...")

        # Validate before cleaning
        metrics = validate_chunk(chunk, i + 1)
        validation_log.append(metrics)

        if metrics["null_price"] > 0 or metrics["null_postcode"] > 0:
            log.warning(f"  Chunk {i+1}: {metrics['null_price']} null prices, "
                        f"{metrics['null_postcode']} null postcodes")

        # Clean
        cleaned = clean_chunk(chunk)
        total_rows_in += len(chunk)
        total_rows_out += len(cleaned)
        all_chunks.append(cleaned)

    # Combine and write
    log.info("Combining chunks...")
    df_all = pd.concat(all_chunks, ignore_index=True)

    log.info(f"Writing parquet to {output_file}...")
    df_all.to_parquet(output_file, index=False, engine="pyarrow", compression="snappy")

    # Summary
    val_df = pd.DataFrame(validation_log)
    log.info("=" * 55)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 55)
    log.info(f"  Total rows ingested : {total_rows_in:>12,}")
    log.info(f"  Total rows output   : {total_rows_out:>12,}")
    log.info(f"  Rows dropped        : {total_rows_in - total_rows_out:>12,}  "
             f"({(total_rows_in - total_rows_out) / total_rows_in * 100:.1f}%)")
    log.info(f"  Output file size    : {output_file.stat().st_size / 1e6:.1f} MB")
    log.info(f"  Date range          : {df_all['date_of_transfer'].min().date()} "
             f"→ {df_all['date_of_transfer'].max().date()}")
    log.info(f"  Unique postcodes    : {df_all['postcode'].nunique():,}")
    log.info(f"  Unique towns        : {df_all['town_city'].nunique():,}")
    log.info("=" * 55)

    # Save validation log for audit trail
    val_log_path = Path(output_dir) / "validation_log_land_registry.csv"
    val_df.to_csv(val_log_path, index=False)
    log.info(f"Validation log saved: {val_log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Land Registry PP-Complete data")
    parser.add_argument("--input", required=True, help="Path to pp-complete.csv")
    parser.add_argument("--output-dir", default="data/processed",
                        help="Directory for output files (default: data/processed)")
    args = parser.parse_args()

    run(args.input, args.output_dir)

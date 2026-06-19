"""
Stage 8 — Upload processed files to Azure Data Lake Gen2

Uploads:
    bronze/ ← raw source files (land registry, flood zones etc)
    silver/ ← cleaned parquet files (all postcodes_*.parquet)
    gold/   ← final scored outputs (la_scores.geojson, la_scores.parquet)

Run:
    python pipelines/08_upload_to_azure.py
"""

import logging
import os
from pathlib import Path

from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ACCOUNT_NAME    = "climatehousestorage"
CONTAINER_NAME  = "data"
OUTPUT_DIR      = Path("data/processed")

# Files to upload per layer
UPLOADS = {
    "silver": [
        "postcodes_flood_risk.parquet",
        "postcodes_coastal_risk.parquet",
        "postcodes_migration.parquet",
        "postcode_scores.parquet",
        "land_registry_clean.parquet",
        "flood_zones.parquet",
        "coastal_erosion.parquet",
    ],
    "gold": [
        "la_scores.parquet",
        "la_scores.geojson",
    ],
}


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

def get_client() -> DataLakeServiceClient:
    """
    Authenticate using DefaultAzureCredential — tries:
    1. Environment variables (AZURE_CLIENT_ID etc)
    2. Azure CLI login (what we're using)
    3. Managed identity (useful later if deployed to Azure)
    """
    account_url = f"https://{ACCOUNT_NAME}.dfs.core.windows.net"
    credential = DefaultAzureCredential()
    return DataLakeServiceClient(account_url=account_url, credential=credential)


def upload_file(client: DataLakeServiceClient,
                local_path: Path,
                remote_path: str) -> None:
    """Upload a single file to ADLS Gen2."""
    fs_client = client.get_file_system_client(CONTAINER_NAME)
    file_client = fs_client.get_file_client(remote_path)

    file_size = local_path.stat().st_size
    log.info(f"  Uploading {local_path.name} ({file_size / 1e6:.1f} MB) → {remote_path}")

    with open(local_path, "rb") as f:
        file_client.upload_data(f, overwrite=True, length=file_size)

    log.info(f"  ✓ {local_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    log.info("Connecting to Azure Data Lake...")
    client = get_client()
    log.info(f"  Account : {ACCOUNT_NAME}")
    log.info(f"  Container: {CONTAINER_NAME}")

    total_uploaded = 0
    total_size = 0

    for layer, files in UPLOADS.items():
        log.info(f"\nUploading to {layer}/...")
        for filename in files:
            local_path = OUTPUT_DIR / filename
            if not local_path.exists():
                log.warning(f"  SKIP — file not found: {local_path}")
                continue
            remote_path = f"{layer}/{filename}"
            upload_file(client, local_path, remote_path)
            total_uploaded += 1
            total_size += local_path.stat().st_size

    log.info("=" * 55)
    log.info("UPLOAD SUMMARY")
    log.info("=" * 55)
    log.info(f"  Files uploaded : {total_uploaded}")
    log.info(f"  Total size     : {total_size / 1e6:.1f} MB")
    log.info(f"  Storage URL    : https://{ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME}/")
    log.info("=" * 55)


if __name__ == "__main__":
    run()

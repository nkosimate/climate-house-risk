"""
Stage 2 — Environment Agency Flood Zone Data
Fetches flood zone polygons via the EA WFS API and saves as GeoParquet.

Flood zones:
    Zone 2 — land with between 1 in 100 and 1 in 1000 annual probability of flooding
    Zone 3a — land with 1 in 100 or greater annual probability of flooding
    Zone 3b — functional floodplain (highest risk)

Output:
    data/processed/flood_zones.parquet   ← polygon geometries + zone level
    data/processed/validation_log_flood_zones.csv

Run:
    python pipelines/02_ingest_flood_zones.py
"""

import logging
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# EA WFS endpoint — no API key required, open data
WFS_BASE = "https://environment.data.gov.uk/spatialdata/flood-map-for-planning-rivers-and-sea-flood-zone-3/wfs"

# We fetch in batches by bounding box to avoid timeout on full national dataset
# These are approximate bounding boxes for England regions (EPSG:4326 WGS84)
REGIONS = {
    "north_east":    (-2.5, 54.2, 0.1,  55.9),
    "north_west":    (-3.7, 53.2, -1.8, 54.9),
    "yorkshire":     (-2.7, 53.2, -0.5, 54.6),
    "east_midlands": (-2.0, 52.1, 0.8,  53.5),
    "west_midlands": (-3.3, 51.8, -1.0, 53.2),
    "east_england":  (-0.6, 51.4, 1.9,  53.1),
    "london":        (-0.6, 51.2, 0.4,  51.8),
    "south_east":    (-1.8, 50.7, 1.5,  51.8),
    "south_west":    (-5.8, 49.8, -1.5, 51.8),
}

# Layer name for Flood Zone 3 (highest risk — fetch this first)
# Zone 2 layer name below
FZ3_LAYER = "FMPSRSFZ3"
FZ2_LAYER = "FMPSRSFZ2"

MAX_FEATURES = 5000    # EA WFS hard limit per request
REQUEST_PAUSE = 1.0    # seconds between requests — be a good API citizen
MAX_RETRIES = 3

OUTPUT_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def build_wfs_url(layer: str, bbox: tuple, start_index: int = 0) -> str:
    """Build a WFS GetFeature URL for a bounding box page."""
    minx, miny, maxx, maxy = bbox
    return (
        f"{WFS_BASE}?"
        f"SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
        f"&TYPENAMES={layer}"
        f"&BBOX={miny},{minx},{maxy},{maxx},urn:ogc:def:crs:EPSG::4326"
        f"&SRSNAME=EPSG:27700"          # request in British National Grid
        f"&COUNT={MAX_FEATURES}"
        f"&STARTINDEX={start_index}"
        f"&OUTPUTFORMAT=application/json"
    )


def fetch_page(url: str, attempt: int = 1) -> dict | None:
    """Fetch one WFS page with retry logic."""
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        if attempt < MAX_RETRIES:
            wait = attempt * 5
            log.warning(f"  Request failed ({e}), retrying in {wait}s...")
            time.sleep(wait)
            return fetch_page(url, attempt + 1)
        else:
            log.error(f"  Request failed after {MAX_RETRIES} attempts: {e}")
            return None


def fetch_region(layer: str, region_name: str, bbox: tuple) -> gpd.GeoDataFrame | None:
    """Fetch all pages for one region, return combined GeoDataFrame."""
    all_features = []
    start_index = 0

    log.info(f"  Fetching {region_name}...")

    while True:
        url = build_wfs_url(layer, bbox, start_index)
        data = fetch_page(url)

        if data is None:
            break

        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        log.info(f"    {region_name}: {len(all_features):,} features so far...")

        # If we got fewer than MAX_FEATURES, we've reached the last page
        if len(features) < MAX_FEATURES:
            break

        start_index += MAX_FEATURES
        time.sleep(REQUEST_PAUSE)

    if not all_features:
        log.warning(f"  No features returned for {region_name}")
        return None

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:27700")
    gdf["region"] = region_name
    return gdf


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def fetch_flood_zones(layer: str, zone_label: str) -> gpd.GeoDataFrame:
    """Fetch all regions for a flood zone layer."""
    log.info(f"Fetching {zone_label} polygons for all regions...")
    region_gdfs = []

    for region_name, bbox in REGIONS.items():
        gdf = fetch_region(layer, region_name, bbox)
        if gdf is not None:
            gdf["flood_zone"] = zone_label
            region_gdfs.append(gdf)
        time.sleep(REQUEST_PAUSE)

    if not region_gdfs:
        raise RuntimeError(f"No data returned for {zone_label}")

    combined = gpd.GeoDataFrame(
        pd.concat(region_gdfs, ignore_index=True),
        crs="EPSG:27700"
    )

    # Drop duplicate geometries from overlapping bounding boxes
    before = len(combined)
    combined = combined.drop_duplicates(subset=["geometry"])
    after = len(combined)
    log.info(f"{zone_label}: {before:,} raw polygons → {after:,} after deduplication")

    return combined


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "flood_zones.parquet"

    log.info("Loading flood zones from local file...")
    gdf = gpd.read_file("data/raw/flood_zones.geojson")

    # Ensure BNG projection
    if gdf.crs.to_epsg() != 27700:
        gdf = gdf.to_crs("EPSG:27700")

    # Fix any invalid geometries
    gdf["geometry"] = gdf["geometry"].buffer(0)

    # Standardise zone column name — check what columns came back
    print("Columns:", gdf.columns.tolist())
    print("Sample:", gdf.head(2))

    gdf.to_parquet(output_file, index=False)
    log.info(f"Saved {len(gdf):,} polygons to {output_file}")


if __name__ == "__main__":
    run()

"""
Stage 3 — Spatial Join: Postcodes → Flood Zones
Assigns a flood risk level to every postcode in the Land Registry dataset.

Requires:
    data/processed/land_registry_clean.parquet
    data/processed/flood_zones.parquet
    data/raw/postcodes.csv  ← ONS postcode directory (see note below)

ONS Postcode Directory (free):
    https://geoportal.statistics.gov.uk/
    Search: "ONS Postcode Directory"
    Download the CSV — you need columns: pcds, lat, long

Output:
    data/processed/postcodes_flood_risk.parquet
    — one row per postcode with flood_zone (zone_2 / zone_3 / none)

Run:
    python pipelines/03_spatial_join_flood.py --postcodes data/raw/postcodes.csv
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_postcodes(path: str) -> gpd.GeoDataFrame:
    """
    Load ONS postcode directory, keep only postcodes in our Land Registry data,
    convert to GeoDataFrame in EPSG:27700 (British National Grid).
    """
    log.info("Loading Land Registry postcodes...")
    lr = pd.read_parquet(OUTPUT_DIR / "land_registry_clean.parquet", columns=["postcode"])
    lr_postcodes = set(lr["postcode"].dropna().unique())
    log.info(f"  {len(lr_postcodes):,} unique postcodes in Land Registry")

    log.info(f"Loading ONS postcode directory from {path}...")
    # ONS postcode directory has many columns — we only need these three
    pcd = pd.read_csv(
        path,
        usecols=["pcds", "lat", "long"],   # pcds = postcode, lat/long = WGS84
        dtype={"pcds": str},
        low_memory=False,
    )
    pcd = pcd.rename(columns={"pcds": "postcode"})
    pcd["postcode"] = pcd["postcode"].str.strip().str.upper()

    # Keep only postcodes present in Land Registry
    pcd = pcd[pcd["postcode"].isin(lr_postcodes)].copy()
    log.info(f"  {len(pcd):,} postcodes matched to Land Registry")

    # Drop missing coordinates
    pcd = pcd.dropna(subset=["lat", "long"])

    # Convert to GeoDataFrame (WGS84 first, then reproject to BNG)
    geometry = [Point(lon, lat) for lon, lat in zip(pcd["long"], pcd["lat"])]
    gdf = gpd.GeoDataFrame(pcd[["postcode"]], geometry=geometry, crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:27700")   # match flood zone CRS

    log.info(f"  Reprojected to EPSG:27700 (British National Grid)")
    return gdf


def load_flood_zones() -> gpd.GeoDataFrame:
    log.info("Loading flood zones...")
    fz = gpd.read_parquet(OUTPUT_DIR / "flood_zones.parquet")
    log.info(f"  {len(fz):,} flood zone polygons, CRS: {fz.crs}")
    return fz[["flood_zone", "geometry"]]


# ---------------------------------------------------------------------------
# Spatial join
# ---------------------------------------------------------------------------

def assign_flood_risk(postcodes: gpd.GeoDataFrame,
                      flood_zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Point-in-polygon join. Each postcode centroid gets the highest flood zone
    it falls within (zone_3 > zone_2 > none).
    """
    log.info("Running spatial join (point-in-polygon)...")
    log.info(f"  {len(postcodes):,} postcode points × {len(flood_zones):,} flood polygons")

    # Spatial join — finds all flood zones each postcode falls within
    joined = gpd.sjoin(
        postcodes,
        flood_zones,
        how="left",
        predicate="within",
    )

    # A postcode can fall in both zone_2 and zone_3 polygons — keep highest risk
    # Priority: zone_3 > zone_2 > none
    risk_order = {"FZ3": 2, "FZ2": 1}
    joined["risk_rank"] = joined["flood_zone"].map(risk_order).fillna(0)

    result = (
        joined
        .sort_values("risk_rank", ascending=False)
        .drop_duplicates(subset=["postcode"])
        .copy()
    )

    result["flood_zone"] = result["flood_zone"].fillna("none")
    result = result[["postcode", "flood_zone", "geometry"]].reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(postcodes_path: str):
    output_file = OUTPUT_DIR / "postcodes_flood_risk.parquet"

    postcodes = load_postcodes(postcodes_path)
    flood_zones = load_flood_zones()
    result = assign_flood_risk(postcodes, flood_zones)

    # Save
    log.info(f"Writing to {output_file}...")
    result.to_parquet(output_file, index=False)

    # Summary
    counts = result["flood_zone"].value_counts()
    total = len(result)

    log.info("=" * 55)
    log.info("SPATIAL JOIN SUMMARY")
    log.info("=" * 55)
    log.info(f"  Total postcodes joined : {total:>10,}")
    for zone, count in counts.items():
        pct = count / total * 100
        log.info(f"  {zone:<22} : {count:>10,}  ({pct:.1f}%)")
    log.info(f"  Output file            : {output_file}")
    log.info("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spatial join: postcodes → flood zones"
    )
    parser.add_argument(
        "--postcodes",
        required=True,
        help="Path to ONS postcode directory CSV"
    )
    args = parser.parse_args()
    run(args.postcodes)

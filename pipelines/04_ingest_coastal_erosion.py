"""
Stage 4 — Coastal Erosion Data
Ingests DEFRA/EA shoreline erosion data and assigns erosion risk to coastal postcodes.

Input:  data/raw/coastal_erosion/*.shp
Output: data/processed/coastal_erosion.parquet
        data/processed/postcodes_coastal_risk.parquet

Run:
    python pipelines/04_ingest_coastal_erosion.py
"""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/processed")
RAW_DIR = Path("data/raw/coastal_erosion")


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def find_shapefile() -> Path:
    """Find the .shp file in the coastal erosion directory."""
    shapefiles = list(RAW_DIR.glob("**/*.shp"))
    if not shapefiles:
        raise FileNotFoundError(
            f"No .shp file found in {RAW_DIR}. "
            "Please unzip the downloaded shapefile into data/raw/coastal_erosion/"
        )
    if len(shapefiles) > 1:
        log.warning(f"Multiple shapefiles found, using first: {shapefiles[0]}")
    log.info(f"Found shapefile: {shapefiles[0]}")
    return shapefiles[0]


def load_erosion_data(shp_path: Path) -> gpd.GeoDataFrame:
    """Load and standardise the erosion shapefile."""
    log.info(f"Loading shapefile: {shp_path}")
    gdf = gpd.read_file(shp_path)

    log.info(f"  Shape: {gdf.shape}")
    log.info(f"  CRS: {gdf.crs}")
    log.info(f"  Columns: {gdf.columns.tolist()}")
    log.info(f"  Sample:\n{gdf.head(2).to_string()}")

    # Reproject to British National Grid if needed
    if gdf.crs.to_epsg() != 27700:
        log.info("  Reprojecting to EPSG:27700...")
        gdf = gdf.to_crs("EPSG:27700")

    # Fix invalid geometries
    gdf["geometry"] = gdf["geometry"].buffer(0)

    return gdf


def standardise_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Normalise column names to a standard schema.
    Shoreline Management Plan data can have varying column names
    depending on the specific dataset version downloaded.
    We'll print what we have and map to a standard set.
    """
    cols = [c.lower() for c in gdf.columns]
    gdf.columns = [c.lower() for c in gdf.columns]

    log.info(f"Columns (lowercased): {cols}")

    # Common column name variants across SMP dataset versions
    # Map whatever exists to our standard names
    name_map = {}

    # Erosion rate / epoch columns — look for these patterns
    rate_candidates = [c for c in cols if any(x in c for x in
                       ["rate", "erosion", "epoch", "change", "retreat"])]
    log.info(f"  Potential erosion rate columns: {rate_candidates}")

    # Epoch/period candidates (SMP data often has epoch1, epoch2, epoch3
    # representing short/medium/long term)
    epoch_candidates = [c for c in cols if "epoch" in c or "ep" in c]
    log.info(f"  Potential epoch columns: {epoch_candidates}")

    return gdf


def assign_erosion_risk(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    NCERM dataset: nfi2055_95 = predicted erosion in metres by 2055 (95th percentile).
    Negative values = erosion (land loss). Positive = accretion (land gain).
    We classify into risk bands for the composite score.
    """
    col = "nfi2055_95"
    gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    log.info(f"  Erosion column '{col}' stats:")
    log.info(f"    min: {gdf[col].min():.1f}m  max: {gdf[col].max():.1f}m  "
             f"mean: {gdf[col].mean():.1f}m  nulls: {gdf[col].isna().sum()}")

    # Classify by predicted erosion (metres lost by 2055)
    # Negative = erosion, so we flip sign for intuitive bucketing
    gdf["erosion_metres"] = gdf[col]
    gdf["erosion_risk"] = pd.cut(
    gdf[col],
    bins=[-float("inf"), 0.0, 0.5, 2.0, 10.0, float("inf")],
    labels=["none", "low", "medium", "high", "critical"]
)
    gdf["erosion_risk"] = gdf["erosion_risk"].astype(str).replace("nan", "unknown")

    counts = gdf["erosion_risk"].value_counts()
    log.info(f"  Erosion risk distribution:\n{counts.to_string()}")

    return gdf


# ---------------------------------------------------------------------------
# Spatial join to postcodes
# ---------------------------------------------------------------------------

def join_to_postcodes(erosion_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Assign coastal erosion risk to postcodes via spatial join.
    Postcodes not near the coast get 'none'.
    """
    log.info("Loading postcodes...")
    postcodes = gpd.read_parquet(OUTPUT_DIR / "postcodes_flood_risk.parquet")

    log.info(f"Joining {len(postcodes):,} postcodes to {len(erosion_gdf):,} erosion features...")

    # Keep only the columns we need for the join
    erosion_slim = erosion_gdf[["erosion_risk", "geometry"]].copy()

    joined = gpd.sjoin(
        postcodes[["postcode", "geometry"]],
        erosion_slim,
        how="left",
        predicate="within",
    )

    # Keep highest risk if multiple matches
    risk_order = {"high": 3, "medium": 2, "low": 1, "negligible": 0, "unknown": 0}
    joined["risk_rank"] = joined["erosion_risk"].map(risk_order).fillna(-1)

    result = (
        joined
        .sort_values("risk_rank", ascending=False)
        .drop_duplicates(subset=["postcode"])
        .copy()
    )

    result["erosion_risk"] = result["erosion_risk"].fillna("none")
    result = result[["postcode", "erosion_risk", "geometry"]].reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shp_path = find_shapefile()
    gdf = load_erosion_data(shp_path)
    gdf = standardise_columns(gdf)
    gdf = assign_erosion_risk(gdf)

    # Save raw processed erosion data
    erosion_out = OUTPUT_DIR / "coastal_erosion.parquet"
    gdf.to_parquet(erosion_out, index=False)
    log.info(f"Saved erosion data: {erosion_out}")

    # Join to postcodes
    postcode_erosion = join_to_postcodes(gdf)

    postcode_out = OUTPUT_DIR / "postcodes_coastal_risk.parquet"
    postcode_erosion.to_parquet(postcode_out, index=False)

    # Summary
    counts = postcode_erosion["erosion_risk"].value_counts()
    total = len(postcode_erosion)

    log.info("=" * 55)
    log.info("COASTAL EROSION PIPELINE SUMMARY")
    log.info("=" * 55)
    log.info(f"  Erosion features loaded : {len(gdf):>8,}")
    log.info(f"  Total postcodes         : {total:>8,}")
    for risk, count in counts.items():
        pct = count / total * 100
        log.info(f"  {risk:<24} : {count:>8,}  ({pct:.1f}%)")
    log.info("=" * 55)


if __name__ == "__main__":
    run()

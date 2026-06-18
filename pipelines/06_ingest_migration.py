"""
Stage 6 — ONS Internal Migration
Computes net migration per Local Authority and joins to postcodes.

Input format: origin-destination flows
    outla, inla, age, sex, moves
    E06000001, E06000002, 0, F, 0.6619

Net migration per LA = sum(inflows) - sum(outflows)
Positive = net destination (people moving in)
Negative = net origin (people moving out)

Inputs:
    data/raw/internal_migration.csv
    data/raw/local_authority_boundaries/*.shp
Output:
    data/processed/postcodes_migration.parquet

Run:
    python pipelines/06_ingest_migration.py
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/processed")
MIGRATION_PATH = Path("data/raw/internal_migration.csv")
LA_BOUNDARIES_DIR = Path("data/raw/local_authority_boundaries")


# ---------------------------------------------------------------------------
# Compute net migration per LA
# ---------------------------------------------------------------------------

def compute_net_migration() -> pd.DataFrame:
    log.info(f"Loading migration flows from {MIGRATION_PATH}...")
    df = pd.read_csv(MIGRATION_PATH, dtype={"outla": str, "inla": str})

    log.info(f"  Rows: {len(df):,}")
    log.info(f"  Columns: {df.columns.tolist()}")
    log.info(f"  Total moves: {df['moves'].sum():,.0f}")

    # Sum total outflows per LA (people leaving)
    outflows = (
        df.groupby("outla")["moves"]
        .sum()
        .reset_index()
        .rename(columns={"outla": "la_code", "moves": "outflow"})
    )

    # Sum total inflows per LA (people arriving)
    inflows = (
        df.groupby("inla")["moves"]
        .sum()
        .reset_index()
        .rename(columns={"inla": "la_code", "moves": "inflow"})
    )

    # Merge and compute net
    net = outflows.merge(inflows, on="la_code", how="outer").fillna(0)
    net["net_migration"] = net["inflow"] - net["outflow"]
    net["net_migration_rate"] = (
        net["net_migration"] / (net["outflow"] + net["inflow"]) * 100
    ).round(2)

    net = net.sort_values("net_migration", ascending=False).reset_index(drop=True)

    log.info(f"  Total LAs: {len(net):,}")
    log.info(f"  Top 10 net destinations (people moving IN):")
    log.info(net.head(10)[["la_code", "inflow", "outflow", "net_migration"]].to_string())
    log.info(f"  Top 10 net origins (people moving OUT):")
    log.info(net.tail(10)[["la_code", "inflow", "outflow", "net_migration"]].to_string())

    return net


# ---------------------------------------------------------------------------
# Spatial join: postcodes → LA → migration score
# ---------------------------------------------------------------------------

def load_la_boundaries() -> gpd.GeoDataFrame:
    shapefiles = list(LA_BOUNDARIES_DIR.glob("**/*.shp"))
    if not shapefiles:
        raise FileNotFoundError(
            f"No shapefile found in {LA_BOUNDARIES_DIR}. "
            "Please download ONS Local Authority boundaries."
        )
    log.info(f"Loading LA boundaries: {shapefiles[0]}")
    gdf = gpd.read_file(shapefiles[0])
    log.info(f"  Shape: {gdf.shape}")
    log.info(f"  Columns: {gdf.columns.tolist()}")
    log.info(f"  CRS: {gdf.crs}")

    if gdf.crs.to_epsg() != 27700:
        gdf = gdf.to_crs("EPSG:27700")

    return gdf


def find_la_code_column(gdf: gpd.GeoDataFrame) -> str:
    candidates = [c for c in gdf.columns if any(x in c.lower() for x in
                  ["cd", "code", "lad"])]
    log.info(f"  LA code column candidates: {candidates}")
    code_col = next((c for c in candidates if c.upper().endswith("CD")), None)
    if not code_col:
        code_col = candidates[0] if candidates else gdf.columns[0]
    log.info(f"  Using LA code column: {code_col}")
    return code_col


def join_migration_to_postcodes(net: pd.DataFrame,
                                 la_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    la_code_col = find_la_code_column(la_gdf)

    la_gdf = la_gdf.merge(
        net[["la_code", "net_migration", "net_migration_rate"]],
        left_on=la_code_col,
        right_on="la_code",
        how="left",
    )

    unmatched = la_gdf["net_migration"].isna().sum()
    if unmatched > 0:
        log.warning(f"  {unmatched} LAs had no migration data — setting to 0")
    la_gdf["net_migration"] = la_gdf["net_migration"].fillna(0)
    la_gdf["net_migration_rate"] = la_gdf["net_migration_rate"].fillna(0)

    log.info("Loading postcodes...")
    postcodes = gpd.read_parquet(OUTPUT_DIR / "postcodes_flood_risk.parquet")
    log.info(f"  {len(postcodes):,} postcodes to join")

    log.info("Joining postcodes to LA polygons...")
    joined = gpd.sjoin(
        postcodes[["postcode", "geometry"]],
        la_gdf[[la_code_col, "net_migration", "net_migration_rate", "geometry"]],
        how="left",
        predicate="within",
    )

    joined = joined.drop_duplicates(subset=["postcode"])

    joined["migration_trend"] = pd.cut(
        joined["net_migration_rate"],
        bins=[-float("inf"), -5, -1, 1, 5, float("inf")],
        labels=["strong_decline", "mild_decline", "stable", "mild_growth", "strong_growth"]
    ).astype(str).replace("nan", "unknown")

    result = joined[["postcode", "net_migration", "net_migration_rate",
                     "migration_trend", "geometry"]].reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "postcodes_migration.parquet"

    net = compute_net_migration()
    la_gdf = load_la_boundaries()
    result = join_migration_to_postcodes(net, la_gdf)

    log.info(f"Writing to {output_file}...")
    result.to_parquet(output_file, index=False)

    counts = result["migration_trend"].value_counts()
    total = len(result)

    log.info("=" * 55)
    log.info("MIGRATION PIPELINE SUMMARY")
    log.info("=" * 55)
    log.info(f"  Total postcodes scored : {total:>10,}")
    for trend, count in counts.items():
        pct = count / total * 100
        log.info(f"  {trend:<24} : {count:>10,}  ({pct:.1f}%)")
    log.info(f"  Output: {output_file}")
    log.info("=" * 55)


if __name__ == "__main__":
    run()

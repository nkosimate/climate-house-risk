"""
Stage 7 — Composite Climate Attractiveness Scoring Model
Combines all risk layers into a single score per postcode, then aggregates
to Local Authority level for the front end map.

Scoring factors (weights sum to 1.0):
    flood_risk        0.25  — EA flood zones 2/3
    coastal_erosion   0.20  — NCERM predicted erosion by 2055
    migration_trend   0.20  — ONS net migration (revealed preference)
    price_trend       0.20  — Land Registry 5-year price growth
    heat_risk         0.15  — UKCP18 heat days (placeholder until data arrives)

Score range: 0 (least attractive) → 100 (most attractive)
Higher = better place to buy under climate change scenario

Inputs:
    data/processed/land_registry_clean.parquet
    data/processed/postcodes_flood_risk.parquet
    data/processed/postcodes_coastal_risk.parquet
    data/processed/postcodes_migration.parquet
Output:
    data/processed/postcode_scores.parquet
    data/processed/la_scores.parquet          ← used by front end map

Run:
    python pipelines/07_build_scores.py
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# Weights — must sum to 1.0
# ---------------------------------------------------------------------------
WEIGHTS = {
    "flood_score":     0.25,
    "erosion_score":   0.20,
    "migration_score": 0.20,
    "price_score":     0.20,
    "heat_score":      0.15,   # placeholder until UKCP18 data arrives
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ---------------------------------------------------------------------------
# Load and merge all layers
# ---------------------------------------------------------------------------

def load_flood_risk() -> pd.DataFrame:
    log.info("Loading flood risk...")
    df = gpd.read_parquet(OUTPUT_DIR / "postcodes_flood_risk.parquet")
    df = pd.DataFrame(df[["postcode", "flood_zone"]])

    # Convert to numeric penalty: zone_3=high risk, zone_2=medium, none=0
    flood_map = {"FZ3": 1.0, "FZ2": 0.5, "none": 0.0}
    df["flood_penalty"] = df["flood_zone"].map(flood_map).fillna(0.0)

    log.info(f"  {len(df):,} postcodes, flood distribution:")
    log.info(f"  {df['flood_zone'].value_counts().to_string()}")
    return df[["postcode", "flood_penalty"]]


def load_coastal_risk() -> pd.DataFrame:
    log.info("Loading coastal erosion risk...")
    df = gpd.read_parquet(OUTPUT_DIR / "postcodes_coastal_risk.parquet")
    df = pd.DataFrame(df[["postcode", "erosion_risk"]])

    erosion_map = {"critical": 1.0, "high": 0.75, "medium": 0.5,
                   "low": 0.25, "none": 0.0, "unknown": 0.0}
    df["erosion_penalty"] = df["erosion_risk"].map(erosion_map).fillna(0.0)

    log.info(f"  {len(df):,} postcodes")
    return df[["postcode", "erosion_penalty"]]


def load_migration() -> pd.DataFrame:
    log.info("Loading migration data...")
    df = gpd.read_parquet(OUTPUT_DIR / "postcodes_migration.parquet")
    df = pd.DataFrame(df[["postcode", "net_migration_rate"]])
    df["net_migration_rate"] = df["net_migration_rate"].fillna(0.0)

    log.info(f"  {len(df):,} postcodes")
    return df[["postcode", "net_migration_rate"]]


def load_price_trend() -> pd.DataFrame:
    """
    Compute 5-year price growth rate per postcode district.
    Uses Land Registry data — compares median price 2019-2021 vs 2022-2024.
    """
    log.info("Computing price trends from Land Registry...")
    df = pd.read_parquet(
        OUTPUT_DIR / "land_registry_clean.parquet",
        columns=["postcode", "price", "year"]
    )

    # Use postcode district (e.g. 'SW1A') for more stable price estimates
    df["postcode_district"] = df["postcode"].str.extract(r"^([A-Z]{1,2}[0-9]{1,2})")

    # Baseline period vs recent period
    baseline = df[df["year"].between(2017, 2019)].groupby("postcode_district")["price"].median()
    recent   = df[df["year"].between(2022, 2024)].groupby("postcode_district")["price"].median()

    trend = pd.DataFrame({"baseline": baseline, "recent": recent}).dropna()
    trend["price_growth_pct"] = ((trend["recent"] - trend["baseline"]) /
                                  trend["baseline"] * 100).round(2)

    log.info(f"  {len(trend):,} postcode districts with price trend data")
    log.info(f"  Median growth: {trend['price_growth_pct'].median():.1f}%")
    log.info(f"  Top districts by growth:\n"
             f"{trend.nlargest(5, 'price_growth_pct')[['price_growth_pct']].to_string()}")

    # Map back to full postcodes
    postcode_district = df[["postcode", "postcode_district"]].drop_duplicates()
    result = postcode_district.merge(
        trend["price_growth_pct"].reset_index(),
        on="postcode_district",
        how="left"
    )
    result["price_growth_pct"] = result["price_growth_pct"].fillna(
        trend["price_growth_pct"].median()  # fill missing with national median
    )

    return result[["postcode", "price_growth_pct"]].drop_duplicates(subset=["postcode"])


# ---------------------------------------------------------------------------
# Build scores
# ---------------------------------------------------------------------------

def normalise_0_to_1(series: pd.Series) -> pd.Series:
    """Min-max normalise a series to [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw risk values into component scores (0=worst, 1=best),
    then combine with weights into a final attractiveness score 0-100.
    """
    log.info("Building component scores...")

    # Flood: high penalty = low score (invert)
    df["flood_score"] = 1.0 - df["flood_penalty"]

    # Erosion: high penalty = low score (invert)
    df["erosion_score"] = 1.0 - df["erosion_penalty"]

    # Migration: higher net migration = better (normalise, don't invert)
    df["migration_score"] = normalise_0_to_1(df["net_migration_rate"])

    # Price trend: higher growth = better (normalise, don't invert)
    # Cap extreme outliers at 5th/95th percentile first
    p5  = df["price_growth_pct"].quantile(0.05)
    p95 = df["price_growth_pct"].quantile(0.95)
    df["price_growth_capped"] = df["price_growth_pct"].clip(p5, p95)
    df["price_score"] = normalise_0_to_1(df["price_growth_capped"])

    # Heat: placeholder — neutral score (0.5) until UKCP18 data arrives
    df["heat_score"] = 0.5
    log.warning("  Heat score set to 0.5 (neutral) — update when UKCP18 data arrives")

    # Composite weighted score → scale to 0-100
    df["climate_score"] = (
        df["flood_score"]     * WEIGHTS["flood_score"]     +
        df["erosion_score"]   * WEIGHTS["erosion_score"]   +
        df["migration_score"] * WEIGHTS["migration_score"] +
        df["price_score"]     * WEIGHTS["price_score"]     +
        df["heat_score"]      * WEIGHTS["heat_score"]
    ) * 100

    df["climate_score"] = df["climate_score"].round(1)

    log.info(f"  Score distribution:")
    log.info(f"  {df['climate_score'].describe().round(1).to_string()}")

    return df


def score_band(score: float) -> str:
    if score >= 75:   return "A — Very Attractive"
    if score >= 60:   return "B — Attractive"
    if score >= 45:   return "C — Average"
    if score >= 30:   return "D — At Risk"
    return                   "E — High Risk"


# ---------------------------------------------------------------------------
# Aggregate to Local Authority for map
# ---------------------------------------------------------------------------

def aggregate_to_la(postcode_scores: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Aggregate postcode scores to Local Authority level.
    The front end map renders at LA level — 361 polygons is fast,
    1.27M postcode points would be slow.
    """
    log.info("Aggregating to Local Authority level...")

    la_boundaries = gpd.read_file(
        list(Path("data/raw/local_authority_boundaries").glob("**/*.shp"))[0]
    )
    if la_boundaries.crs.to_epsg() != 27700:
        la_boundaries = la_boundaries.to_crs("EPSG:27700")

    # Detect LA code and name columns
    code_col = next(c for c in la_boundaries.columns if c.upper().endswith("CD"))
    name_col = next(c for c in la_boundaries.columns if c.upper().endswith("NM") and
                    "W" not in c.upper()[-3:])
    log.info(f"  LA code: {code_col}, name: {name_col}")

    # Load postcode → LA mapping from migration file (already has this join)
    postcode_la = gpd.read_parquet(OUTPUT_DIR / "postcodes_migration.parquet")
    postcode_la = pd.DataFrame(postcode_la[["postcode"]])

    # Spatial join postcodes to LA
    postcodes_gdf = gpd.read_parquet(OUTPUT_DIR / "postcodes_flood_risk.parquet")
    postcodes_gdf = postcodes_gdf[["postcode", "geometry"]]

    joined = gpd.sjoin(
        postcodes_gdf,
        la_boundaries[[code_col, name_col, "geometry"]],
        how="left",
        predicate="within",
    ).drop_duplicates(subset=["postcode"])

    # Merge scores
    scored = joined.merge(
        postcode_scores[["postcode", "climate_score", "flood_score",
                         "erosion_score", "migration_score", "price_score"]],
        on="postcode",
        how="left"
    )

    # Aggregate to LA
    la_scores = (
        scored.groupby([code_col, name_col])
        .agg(
            climate_score    =("climate_score",    "mean"),
            flood_score      =("flood_score",      "mean"),
            erosion_score    =("erosion_score",    "mean"),
            migration_score  =("migration_score",  "mean"),
            price_score      =("price_score",      "mean"),
            postcode_count   =("postcode",         "count"),
        )
        .round(2)
        .reset_index()
    )

    la_scores["score_band"] = la_scores["climate_score"].apply(score_band)
    la_scores = la_scores.sort_values("climate_score", ascending=False)

    log.info(f"  {len(la_scores):,} LAs scored")
    log.info(f"\n  TOP 15 MOST ATTRACTIVE LAs:")
    log.info(la_scores.head(15)[[name_col, "climate_score", "score_band"]].to_string())
    log.info(f"\n  BOTTOM 15 LEAST ATTRACTIVE LAs:")
    log.info(la_scores.tail(15)[[name_col, "climate_score", "score_band"]].to_string())

    # Merge back with geometry for GeoJSON export
    la_scores_geo = la_boundaries[[code_col, name_col, "geometry"]].merge(
        la_scores.drop(columns=[name_col]),
        on=code_col,
        how="left"
    )

    return la_scores_geo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all layers
    flood    = load_flood_risk()
    erosion  = load_coastal_risk()
    migration = load_migration()
    prices   = load_price_trend()

    # Merge on postcode
    log.info("Merging all layers...")
    df = flood.merge(erosion,   on="postcode", how="left")
    df = df.merge(migration,    on="postcode", how="left")
    df = df.merge(prices,       on="postcode", how="left")

    # Fill any remaining nulls
    df["net_migration_rate"] = df["net_migration_rate"].fillna(0.0)
    df["price_growth_pct"]   = df["price_growth_pct"].fillna(0.0)

    log.info(f"  Merged dataset: {len(df):,} postcodes")

    # Build scores
    df = build_scores(df)

    # Save postcode-level scores
    postcode_out = OUTPUT_DIR / "postcode_scores.parquet"
    df.to_parquet(postcode_out, index=False)
    log.info(f"  Postcode scores saved: {postcode_out}")

    # Aggregate to LA and save as GeoJSON (for front end)
    la_scores = aggregate_to_la(df)
    la_parquet = OUTPUT_DIR / "la_scores.parquet"
    la_geojson = OUTPUT_DIR / "la_scores.geojson"

    la_scores.to_parquet(la_parquet, index=False)
    la_scores.to_file(la_geojson, driver="GeoJSON")

    log.info(f"  LA scores saved: {la_parquet}")
    log.info(f"  LA GeoJSON saved: {la_geojson}")

    log.info("=" * 55)
    log.info("SCORING MODEL SUMMARY")
    log.info("=" * 55)
    band_counts = la_scores["score_band"].value_counts()
    for band, count in band_counts.items():
        log.info(f"  {band:<30} : {count:>4} LAs")
    log.info("=" * 55)


if __name__ == "__main__":
    run()

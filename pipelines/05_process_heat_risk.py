"""
Stage 5 — UKCP18 Heat Risk Processing
Converts Met Office UKCP18 temperature projections (NetCDF) to postcode-level
heat risk scores.

Metric: projected number of days per year above 30°C by 2040-2060 (RCP8.5)
This is a meaningful threshold for:
    - Heat-related health risk
    - Property cooling demand
    - Urban heat island amplification

Input:  data/raw/ukcp18_temperature.nc
Output: data/processed/postcodes_heat_risk.parquet

Run:
    python pipelines/05_process_heat_risk.py
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/processed")
NC_PATH = Path("data/raw/ukcp18_temperature.nc")

# Future scenario window — days above threshold in this period
FUTURE_START = "2040-01-01"
FUTURE_END   = "2060-12-31"
HEAT_THRESHOLD_C = 30.0   # degrees Celsius


# ---------------------------------------------------------------------------
# Load and inspect NetCDF
# ---------------------------------------------------------------------------

def load_netcdf() -> xr.Dataset:
    log.info(f"Loading NetCDF: {NC_PATH}")
    ds = xr.open_dataset(NC_PATH, engine="netcdf4")

    log.info(f"  Variables : {list(ds.data_vars)}")
    log.info(f"  Dimensions: {dict(ds.dims)}")
    log.info(f"  Coords    : {list(ds.coords)}")

    # Print time range
    time = ds["time"]
    log.info(f"  Time range: {str(time.values[0])[:10]} → {str(time.values[-1])[:10]}")

    return ds


def get_temperature_variable(ds: xr.Dataset) -> str:
    """
    UKCP18 temperature variable is usually 'tas' (near-surface air temperature).
    Detect it from the dataset.
    """
    candidates = [v for v in ds.data_vars if "tas" in v.lower() or "temp" in v.lower()]
    if candidates:
        log.info(f"  Temperature variable: {candidates[0]}")
        return candidates[0]
    # Fallback — just use first variable
    var = list(ds.data_vars)[0]
    log.warning(f"  Could not detect temperature variable, using: {var}")
    return var


def get_coordinate_names(ds: xr.Dataset) -> tuple[str, str]:
    """
    UKCP18 uses rotated pole grid coordinates.
    Detect x/y or lat/lon coordinate names.
    """
    coords = list(ds.coords)
    log.info(f"  All coordinate names: {coords}")

    # Look for projection_x/y_coordinate (UKCP18 standard)
    x_candidates = [c for c in coords if any(x in c.lower() for x in
                    ["projection_x", "grid_lon", "x", "lon", "east"])]
    y_candidates = [c for c in coords if any(y in c.lower() for y in
                    ["projection_y", "grid_lat", "y", "lat", "north"])]

    # Prefer projection coordinates
    x_coord = next((c for c in x_candidates if "projection_x" in c.lower()), None)
    y_coord = next((c for c in y_candidates if "projection_y" in c.lower()), None)

    # Fall back to any x/y
    if not x_coord:
        x_coord = x_candidates[0] if x_candidates else coords[0]
    if not y_coord:
        y_coord = y_candidates[0] if y_candidates else coords[1]

    log.info(f"  Using coordinates: x={x_coord}, y={y_coord}")
    return x_coord, y_coord


# ---------------------------------------------------------------------------
# Compute heat days
# ---------------------------------------------------------------------------

def compute_heat_days(ds: xr.Dataset, temp_var: str) -> pd.DataFrame:
    log.info(f"Selecting future window {FUTURE_START} → 2060-11-30...")

    # UKCP18 uses 360-day calendar — Dec 31 doesn't exist, use Nov 30
    future = ds[temp_var].sel(time=slice(FUTURE_START, "2060-11-30"))
    log.info(f"  Future slice shape: {future.shape}")

    # Check units
    units = future.attrs.get("units", "")
    log.info(f"  Temperature units: '{units}'")
    if "K" in units or float(future.mean()) > 200:
        log.info("  Converting Kelvin → Celsius")
        future = future - 273.15
    else:
        log.info("  Already in Celsius")

    # This dataset is regional averages (geo_region dimension), not gridded
    # Average across ensemble members first, then compute hot days
    log.info("  Averaging across ensemble members...")
    future_mean = future.mean(dim="ensemble_member")

    hot_days = (future_mean > HEAT_THRESHOLD_C)
    total_hot_days = hot_days.sum(dim="time")
    n_years = 20
    mean_hot_days = total_hot_days / n_years

    log.info(f"  Regions: {ds['geo_region'].values.tolist()}")
    log.info(f"  Hot days per region:\n{mean_hot_days.values}")

    # Build DataFrame — one row per region
    df = pd.DataFrame({
        "geo_region": ds["geo_region"].values,
        "heat_days_per_year": mean_hot_days.values,
    })
    log.info(f"  Region heat data:\n{df.to_string()}")
    return df


# ---------------------------------------------------------------------------
# Reproject grid to BNG and join to postcodes
# ---------------------------------------------------------------------------

def reproject_grid_to_bng(heat_df: pd.DataFrame, ds: xr.Dataset) -> pd.DataFrame:
    """
    UKCP18 uses a rotated pole projection. We need to convert grid coordinates
    to EPSG:27700 (BNG) for joining with postcodes.

    If the dataset has lat/lon coordinates we use those directly.
    Otherwise we use the projection info from the dataset.
    """
    from pyproj import Transformer, CRS

    coords = list(ds.coords)

    # Check if we have latitude/longitude variables directly
    lat_col = next((c for c in coords if "lat" in c.lower() and "grid" not in c.lower()), None)
    lon_col = next((c for c in coords if "lon" in c.lower() and "grid" not in c.lower()), None)

    if lat_col and lon_col:
        log.info(f"  Using lat/lon coordinates directly: {lat_col}, {lon_col}")
        lat_vals = ds[lat_col].values
        lon_vals = ds[lon_col].values

        if lat_vals.ndim == 1:
            lons, lats = np.meshgrid(lon_vals, lat_vals)
        else:
            lons, lats = lon_vals, lat_vals

        # Flatten and match to heat_df length
        lats_flat = lats.flatten()
        lons_flat = lons.flatten()

        # Filter to non-NaN cells (same mask as heat_df)
        # We need to reconstruct the full grid first
        x_coord, y_coord = get_coordinate_names(ds)
        x_vals = ds[x_coord].values
        y_vals = ds[y_coord].values
        if x_vals.ndim == 1:
            xx, yy = np.meshgrid(x_vals, y_vals)
        else:
            xx, yy = x_vals, y_vals

        full_df = pd.DataFrame({
            "x": xx.flatten(),
            "y": yy.flatten(),
            "lat": lats_flat,
            "lon": lons_flat,
        })
        heat_df = heat_df.merge(full_df, on=["x", "y"], how="left")

        # Convert WGS84 to BNG
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
        bng_x, bng_y = transformer.transform(heat_df["lon"].values, heat_df["lat"].values)
        heat_df["bng_x"] = bng_x
        heat_df["bng_y"] = bng_y

    else:
        log.info("  No direct lat/lon found — using CRS from dataset")
        # Try to get CRS from dataset
        crs_wkt = None
        for var in ds.data_vars:
            if hasattr(ds[var], "attrs"):
                for attr in ["grid_mapping_name", "crs_wkt", "proj4"]:
                    if attr in ds[var].attrs:
                        crs_wkt = ds[var].attrs[attr]
                        break

        # Fall back to rotated pole (UKCP18 standard)
        log.info("  Assuming UKCP18 rotated pole projection")
        from pyproj import Transformer
        # UKCP18 rotated pole: pole at 37.5°N, 177.5°E
        transformer = Transformer.from_crs(
            "+proj=ob_tran +o_proj=longlat +o_lon_p=0 +o_lat_p=37.5 +lon_0=177.5",
            "EPSG:27700",
            always_xy=True
        )
        bng_x, bng_y = transformer.transform(heat_df["x"].values, heat_df["y"].values)
        heat_df["bng_x"] = bng_x
        heat_df["bng_y"] = bng_y

    log.info(f"  BNG extent: x={heat_df['bng_x'].min():.0f}–{heat_df['bng_x'].max():.0f}, "
             f"y={heat_df['bng_y'].min():.0f}–{heat_df['bng_y'].max():.0f}")

    return heat_df


def join_to_postcodes(heat_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Nearest-neighbour join from postcode centroids to climate grid cells.
    Since the grid is 12km resolution, we use nearest-neighbour rather than
    point-in-polygon — each postcode gets the value of its nearest grid cell.
    """
    log.info("Loading postcodes...")
    postcodes = gpd.read_parquet(OUTPUT_DIR / "postcodes_flood_risk.parquet")

    # Extract postcode coordinates from geometry
    postcode_coords = np.column_stack([
        postcodes.geometry.x.values,
        postcodes.geometry.y.values,
    ])

    # Build KD-tree from grid cell centres
    grid_coords = heat_df[["bng_x", "bng_y"]].values
    log.info(f"  Building KD-tree from {len(grid_coords):,} grid cells...")
    tree = cKDTree(grid_coords)

    # Find nearest grid cell for each postcode
    log.info(f"  Querying nearest grid cell for {len(postcode_coords):,} postcodes...")
    distances, indices = tree.query(postcode_coords, k=1)

    log.info(f"  Max distance to nearest grid cell: {distances.max():.0f}m "
             f"(expected ~6,000m for 12km grid)")

    # Assign heat days
    postcodes["heat_days_per_year"] = heat_df["heat_days_per_year"].values[indices]

    # Classify into risk bands
    postcodes["heat_risk"] = pd.cut(
        postcodes["heat_days_per_year"],
        bins=[-0.001, 1, 5, 15, 30, float("inf")],
        labels=["negligible", "low", "moderate", "high", "extreme"]
    ).astype(str)

    log.info(f"  Heat risk distribution:\n"
             f"{postcodes['heat_risk'].value_counts().to_string()}")

    return postcodes[["postcode", "heat_days_per_year", "heat_risk", "geometry"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "postcodes_heat_risk.parquet"

    ds = load_netcdf()
    temp_var = get_temperature_variable(ds)
    heat_df = compute_heat_days(ds, temp_var)
    heat_df = reproject_grid_to_bng(heat_df, ds)
    result = join_to_postcodes(heat_df)

    log.info(f"Writing to {output_file}...")
    result.to_parquet(output_file, index=False)

    log.info("=" * 55)
    log.info("HEAT RISK PIPELINE SUMMARY")
    log.info("=" * 55)
    log.info(f"  Total postcodes scored : {len(result):>10,}")
    log.info(f"  Mean heat days/year    : {result['heat_days_per_year'].mean():>10.1f}")
    log.info(f"  Max heat days/year     : {result['heat_days_per_year'].max():>10.1f}")
    for risk, count in result["heat_risk"].value_counts().items():
        pct = count / len(result) * 100
        log.info(f"  {risk:<24} : {count:>10,}  ({pct:.1f}%)")
    log.info("=" * 55)

    ds.close()


if __name__ == "__main__":
    run()

"""
Stage 1b — Land Registry Exploration
Run this after 01_ingest_land_registry.py to validate the output
and get a feel for the data before building the scoring model.

Run:
    jupyter notebook  (then open this file)
    OR: python pipelines/01b_explore_land_registry.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PARQUET_PATH = "data/processed/land_registry_clean.parquet"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading parquet...")
df = pd.read_parquet(PARQUET_PATH)
print(f"Shape: {df.shape}")
print(f"\nColumns:\n{df.dtypes}")
print(f"\nSample rows:")
print(df.head(3).to_string())

# ---------------------------------------------------------------------------
# Basic stats
# ---------------------------------------------------------------------------
print("\n--- Price stats ---")
print(df["price"].describe().apply(lambda x: f"{x:,.0f}"))

print("\n--- Property types ---")
print(df["property_type"].value_counts())

print("\n--- Transactions per year (last 10 years) ---")
recent = df[df["year"] >= df["year"].max() - 10]
print(recent.groupby("year")["price"].agg(["count", "median"])
      .rename(columns={"count": "transactions", "median": "median_price"}))

# ---------------------------------------------------------------------------
# Price trend over time — sanity check
# ---------------------------------------------------------------------------
annual = (df.groupby("year")["price"]
            .median()
            .reset_index()
            .rename(columns={"price": "median_price"}))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(annual["year"], annual["median_price"] / 1000, linewidth=2, color="#2563eb")
ax.set_title("UK Median House Price by Year", fontsize=14, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Median Price (£ thousands)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:.0f}k"))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("data/processed/price_trend.png", dpi=150)
print("\nPrice trend chart saved to data/processed/price_trend.png")

# ---------------------------------------------------------------------------
# Top 20 towns by transaction volume — confirms data coverage
# ---------------------------------------------------------------------------
print("\n--- Top 20 towns by volume ---")
print(df.groupby("town_city")["transaction_id"]
        .count()
        .sort_values(ascending=False)
        .head(20)
        .to_string())

# ---------------------------------------------------------------------------
# Postcode coverage check — important for the spatial join later
# ---------------------------------------------------------------------------
print(f"\n--- Postcode coverage ---")
print(f"Total unique postcodes: {df['postcode'].nunique():,}")
print(f"Missing postcodes: {df['postcode'].isna().sum():,}")

# Postcode district (first part, e.g. 'EH1', 'SW1A')
df["postcode_district"] = df["postcode"].str.extract(r"^([A-Z]{1,2}[0-9]{1,2})")
print(f"Unique postcode districts: {df['postcode_district'].nunique():,}")

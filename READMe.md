# Where Should You Buy a House if Climate Change Continues?

> A scenario modelling tool that scores every Local Authority in the UK on climate attractiveness combining flood risk, coastal erosion, heat projections, population migration and property price trends into a single index, updated from live open data sources.

**Live demo:** https://victorious-field-0cfc98c03.7.azurestaticapps.net

---

## What it does

Most house-buying guidance ignores climate risk. This project asks a different question: if RCP8.5 warming continues to 2040, which UK areas become *more* or *less* attractive to live in — before the market has fully priced it in?

It isn't a prediction. It's a scenario model a structured way to combine five risk factors into a comparable index across 319 Local Authorities, so the relative attractiveness of Hull vs Highland becomes a data question rather than a guess.

---

## Highlights

- **29.4 million** Land Registry transactions cleaned and analysed for price trend signals
- **813,000** Environment Agency flood zone polygons joined to 1.27 million UK postcodes via point-in-polygon spatial join
- **1,423** postcodes identified at coastal erosion risk using NCERM predictions to 2055
- Full **bronze → silver → gold** medallion data pipeline on Azure Data Lake Gen2
- Interactive choropleth map — click any LA to see its score breakdown across all five factors

---

## Architecture

```
Open Data Sources
    ├── Land Registry Price Paid (~5GB, 29.4M rows)
    ├── Environment Agency Flood Map (WFS API → 813k polygons)
    ├── NCERM Coastal Erosion (DEFRA Shapefile)
    ├── UKCP18 Climate Projections (Met Office NetCDF)
    └── ONS Internal Migration (775k origin-destination flows)
            │
            ▼
    Local Pipeline (Python / GeoPandas)
    ├── 01_ingest_land_registry.py    — chunked CSV ingestion, validation
    ├── 02_ingest_flood_zones.py      — EA flood zone polygons
    ├── 03_spatial_join_flood.py      — point-in-polygon join at scale
    ├── 04_ingest_coastal_erosion.py  — NCERM shapefile + spatial join
    ├── 05_process_heat_risk.py       — UKCP18 NetCDF raster processing
    ├── 06_ingest_migration.py        — ONS O-D flows → net migration per LA
    ├── 07_build_scores.py            — composite scoring model
    └── 08_upload_to_azure.py         — ADLS Gen2 upload
            │
            ▼
    Azure Data Lake Gen2 (ADLS Gen2)
    ├── bronze/   raw source files
    ├── silver/   cleaned parquet files (postcode-level risk layers)
    └── gold/     la_scores.geojson — final scored output
            │
            ▼
    Azure Static Web Apps
    └── Interactive choropleth map (Mapbox GL JS)
```

---

## Scoring model

Each postcode is scored across five factors, then aggregated to Local Authority level:

| Factor | Weight | Source | Signal |
|---|---|---|---|
| Flood safety | 25% | EA Flood Map | FZ2/FZ3 zone membership |
| Coastal erosion | 20% | NCERM 2055 | Predicted metres lost |
| Migration trend | 20% | ONS Internal Migration | Net migration rate per LA |
| Price trend | 20% | Land Registry | 5-year median growth |
| Heat risk | 15% | UKCP18 RCP8.5 | Days above 30°C by 2040–2060 |

Scores run 0–100. Higher = more climate-attractive.

| Band | Range | Description |
|---|---|---|
| A | 75–100 | Very Attractive |
| B | 60–75 | Attractive |
| C | 45–60 | Average |
| D | 30–45 | At Risk |
| E | 0–30 | High Risk |

---

## Key findings

- **Kingston upon Hull** and **Boston (Lincolnshire)** rank lowest — flood exposure, coastal proximity, and population outflow combine into a clear risk signal
- **South Holland** and **Spelthorne** are the most at-risk in southern England — low-lying terrain and flood zone exposure
- **Greater Manchester** LAs (Oldham, Rochdale, Tameside) score highly — inland, no erosion risk, net migration inflows
- **Welsh Valleys** (Blaenau Gwent, Rhondda Cynon Taf) score well on risk factors but migration patterns are mixed — an interesting divergence worth watching

---

## Stack

| Layer | Technology |
|---|---|
| Data pipeline | Python, Pandas, GeoPandas, xarray |
| Geospatial joins | GeoPandas (point-in-polygon, nearest-neighbour) |
| Cloud storage | Azure Data Lake Gen2 (ADLS Gen2) |
| Cloud upload | Azure SDK (`azure-storage-file-datalake`) |
| Front end | Mapbox GL JS, vanilla HTML/CSS/JS |
| Hosting | Azure Static Web Apps (free tier) |
| Version control | Git / GitHub |

---

## Installation

```bash
git clone https://github.com/nkosimate/climate-house-risk
cd climate-house-risk
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data sources (download separately)

| File | Source | Save to |
|---|---|---|
| Land Registry Price Paid | [gov.uk](http://prod.publicdata.landregistry.gov.uk/pp-complete.csv) | `data/raw/pp-complete.csv` |
| EA Flood Zones | [environment.data.gov.uk](https://environment.data.gov.uk/dataset/04532375-a198-476e-985e-0579a0a11b47) | `data/raw/flood_zones.geojson` |
| NCERM Coastal Erosion | [data.gov.uk](https://www.data.gov.uk) | `data/raw/coastal_erosion/` |
| ONS Postcode Directory | [geoportal.statistics.gov.uk](https://geoportal.statistics.gov.uk) | `data/raw/postcodes.csv` |
| ONS Internal Migration | [ons.gov.uk](https://www.ons.gov.uk) | `data/raw/internal_migration.csv` |
| LA Boundaries | [geoportal.statistics.gov.uk](https://geoportal.statistics.gov.uk) | `data/raw/local_authority_boundaries/` |

### Azure setup

```bash
az login
az group create --name climate-house-risk-rg --location uksouth
az storage account create \
  --name climatehousestorage \
  --resource-group climate-house-risk-rg \
  --sku Standard_LRS --kind StorageV2 --hns true
```

---

## Running the pipeline

Run scripts in order:

```bash
python pipelines/01_ingest_land_registry.py --input data/raw/pp-complete.csv
python pipelines/02_ingest_flood_zones.py
python pipelines/03_spatial_join_flood.py --postcodes data/raw/postcodes.csv
python pipelines/04_ingest_coastal_erosion.py
python pipelines/05_process_heat_risk.py
python pipelines/06_ingest_migration.py
python pipelines/07_build_scores.py
python pipelines/08_upload_to_azure.py
```

Each script logs a summary on completion. Validation logs are written to `data/processed/`.

---

## Project structure

```
climate-house-risk/
├── pipelines/
│   ├── 01_ingest_land_registry.py
│   ├── 02_ingest_flood_zones.py
│   ├── 03_spatial_join_flood.py
│   ├── 04_ingest_coastal_erosion.py
│   ├── 05_process_heat_risk.py
│   ├── 06_ingest_migration.py
│   ├── 07_build_scores.py
│   └── 08_upload_to_azure.py
├── app/
│   └── index.html
├── data/
│   ├── raw/          (gitignored — download separately)
│   └── processed/    (gitignored — generated by pipeline)
├── requirements.txt
└── README.md
```

---

## Roadmap

- [ ] Plug in UKCP18 12km gridded heat data when available (request submitted to Met Office)
- [ ] Add insurance risk layer using FloodRe postcode-level data
- [ ] RCP4.5 vs RCP8.5 scenario toggle on the map
- [ ] 2030 / 2040 / 2050 time horizon slider
- [ ] Move pipeline to Azure Data Factory for scheduled monthly refresh
- [ ] Scale geospatial joins to Apache Sedona on Azure Databricks

---

## Data sources and licences

- Land Registry Price Paid Data — [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/)
- Environment Agency Flood Map — [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/)
- NCERM Coastal Erosion — [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/)
- UKCP18 Climate Projections — [Met Office / CEDA](https://www.metoffice.gov.uk/research/approach/collaboration/ukcp)
- ONS Internal Migration — [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/)

---

*Built with Python, GeoPandas, Azure, and open government data.*

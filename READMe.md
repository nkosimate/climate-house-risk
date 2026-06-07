# Climate House Risk — UK Property & Climate Scenario Modeller

## What it does
Scores ~35,000 UK geographic zones (LSOAs) on climate attractiveness 
across 2030/2040 scenarios, combining property prices, flood risk, 
coastal erosion, heat projections and population migration.

## Architecture


## Data Sources
- UK Land Registry Price Paid Data
- Environment Agency Flood Zone data
- DEFRA Shoreline Management Plans
- Met Office UKCP18 climate projections
- ONS Internal Migration statistics

## Pipeline Stages
1. Ingestion (Azure Data Factory)
2. Storage (Azure Data Lake Gen2 — bronze/silver/gold)
3. Processing (Azure Databricks + GeoPandas/Sedona)
4. Scoring (composite risk model)
5. Serving (Azure Maps + Static Web App)


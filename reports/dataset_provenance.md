# AGRI-FUSE Master Dataset Provenance and Scientific Audit Report

This document details the scientific lineage, collection methodology, physical units, data quality audit, and research suitability for all nine datasets evaluated in the AGRI-FUSE platform.

## 1. Master Dataset Inventory Summary

| ID | Dataset Name | Source | Status | Size | Rows | Cols | Primary Target |
|---|---|---|---|---|---|---|---|
| D1 | Crop Recommendation Dataset | [https://www.kaggle.com/dataset...](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) | Downloaded & Verified | 146.52 KB | 2200 | 8 | label (22 crop classes) |
| D2 | Indian Crop Yield Dataset | [https://www.kaggle.com/dataset...](https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset) | Downloaded & Verified | 1.55 MB | 19689 | 10 | Yield (tonnes per hectare) |
| D3 | Eastern India Wheat Field Dataset | [https://data.cimmyt.org/ (CIMM...](https://data.cimmyt.org/ (CIMMYT Dataverse)) | Downloaded & Verified | 1.54 MB | 5689 | 49 | GrainYield (tonnes/ha) |
| D4 | ICRISAT District-Level Dataset | [https://data.icrisat.org/...](https://data.icrisat.org/) | Downloaded & Verified | 22.64 MB | 12803 | 107 | Crop-specific Yields (Rice, Wheat, Pearl Millet, Chickpea, Groundnut, etc.) |
| D5 | Smart Farming Data 2024 (SF24) | [https://www.kaggle.com/dsv/923...](https://www.kaggle.com/dsv/9239304) | Downloaded & Verified | 592.87 KB | 2200 | 23 | label (22 crop classes) |
| D6 | Smart Farming IoT Dataset | [https://www.kaggle.com/code/ji...](https://www.kaggle.com/code/jijagallery/smart-farming-iot-dataset/input) | Downloaded & Verified | 81.62 KB | 500 | 22 | yield_kg_per_hectare (Regression) & crop_type (Classification) |
| D7 | Soil Nutrient Dataset | [https://www.kaggle.com/code/ma...](https://www.kaggle.com/code/manikantasanjayv/crop-recommender-using-ml/input) | Downloaded & Verified | 38.45 KB | 620 | 12 | label (Crop suitability classes, e.g. pomegranate, apple, etc.) |
| D8 | ML Competition 1 – Satellite-Derived Indices | [https://www.kaggle.com/competi...](https://www.kaggle.com/competitions/kaggle-cup-2025-01/data) | Inaccessible (HTTP 404 Not Found) | 0 KB | 0 | 0 | Yield (Hypothesized) |
| D9 | AgriYield 2025 | [https://www.kaggle.com/competi...](https://www.kaggle.com/competitions/agriyield-2025/overview/abstract) | Downloaded & Verified | 465.94 KB | 10000 | 9 | yield (Maize yield kg/ha) |

---

## 2. In-Depth Dataset Audits

### D1: Crop Recommendation Dataset
- **Canonical URL**: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
- **Downloaded Filename**: `Crop_recommendation.csv`
- **Actual Downloaded Size**: 146.52 KB (150,034 bytes)
- **Observation Dimensions**: 2,200 observations, 8 attributes
- **Target Variable**: label (22 crop classes)
- **Available Attributes**: N, P, K, temperature, humidity, ph, rainfall
- **License**: CC0: Public Domain
- **Scientific Provenance**: Aggregated by Atharva Ingle from agricultural research trial averages in India.
- **Research Suitability**: Primary benchmark for multiclass crop recommendation.
- **Scientific Limitations & Audit Findings**: Tabular averages without individual farm or year identifiers; labels are crop suitability recommendations, NOT crop-stress or disease.

### D2: Indian Crop Yield Dataset
- **Canonical URL**: https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset
- **Downloaded Filename**: `crop_yield.csv`
- **Actual Downloaded Size**: 1.55 MB (1,620,945 bytes)
- **Observation Dimensions**: 19,689 observations, 10 attributes
- **Target Variable**: Yield (tonnes per hectare)
- **Available Attributes**: Crop, Crop_Year, Season, State, Area, Production, Annual_Rainfall, Fertilizer, Pesticide
- **License**: Open Data Commons / CC0
- **Scientific Provenance**: Compiled by Akshat Gupta from Ministry of Agriculture & Farmers Welfare, Government of India (1997-2020).
- **Research Suitability**: State-level historical macro-yield regression with chronological splits.
- **Scientific Limitations & Audit Findings**: Aggregated state/year reporting; Yield exhibits exact physical derivation from Production/Area (r=0.9965); requires temporal splits to prevent leakage.

### D3: Eastern India Wheat Field Dataset
- **Canonical URL**: https://data.cimmyt.org/ (CIMMYT Dataverse)
- **Downloaded Filename**: `Wheat_Yld_Eastern_IND_Data.xlsx`
- **Actual Downloaded Size**: 1.54 MB (1,609,655 bytes)
- **Observation Dimensions**: 5,689 observations, 49 attributes
- **Target Variable**: GrainYield (tonnes/ha)
- **Available Attributes**: Field_ID, Year, District, State, SoilType, LandType, VarietyClass, SowingDate, N, P, K fertilizers, irrigation counts, environmental indices
- **License**: CIMMYT Open Access Policy (Attribution-NonCommercial 4.0)
- **Scientific Provenance**: International Maize and Wheat Improvement Center (CIMMYT) on-farm surveys in Eastern Gangetic Plains (Bihar, Eastern UP).
- **Research Suitability**: Field-level micro-agronomic wheat yield modeling.
- **Scientific Limitations & Audit Findings**: Repeated field visits over 6 cropping seasons require GroupKFold by Field_ID to prevent spatial-temporal leakage.

### D4: ICRISAT District-Level Dataset
- **Canonical URL**: https://data.icrisat.org/
- **Downloaded Filename**: `icrisat_district_level_1990_2015.xls`
- **Actual Downloaded Size**: 22.64 MB (23,739,904 bytes)
- **Observation Dimensions**: 12,803 observations, 107 attributes
- **Target Variable**: Crop-specific Yields (Rice, Wheat, Pearl Millet, Chickpea, Groundnut, etc.)
- **Available Attributes**: Dist Code, Year, State Code, State Name, Dist Name, Crop Area, Production, Yield, Monthly precipitation, Water security indicators
- **License**: ICRISAT Dataverse Research Open Access
- **Scientific Provenance**: ICRISAT (International Crops Research Institute for the Semi-Arid Tropics) Meso-Level District Agricultural Data (1990-2015).
- **Research Suitability**: District-level spatial-temporal econometric & machine learning yield modeling.
- **Scientific Limitations & Audit Findings**: Wide panel structure with district boundary changes over time; sparse for certain minority crops.

### D5: Smart Farming Data 2024 (SF24)
- **Canonical URL**: https://www.kaggle.com/dsv/9239304
- **Downloaded Filename**: `Crop_recommendationV2.csv`
- **Actual Downloaded Size**: 592.87 KB (607,094 bytes)
- **Observation Dimensions**: 2,200 observations, 23 attributes
- **Target Variable**: label (22 crop classes)
- **Available Attributes**: N, P, K, temp, humidity, ph, rainfall, soil_moisture, soil_type, sunlight_exposure, wind_speed, co2_concentration, organic_matter, irrigation_frequency, water_source_type, frost_risk, water_usage_efficiency
- **License**: Community Data License Agreement / Kaggle Open
- **Scientific Provenance**: Kaggle Dataset Creator (2024); derived by synthetically extending the 2,200 rows of Dataset 1.
- **Research Suitability**: Comparative evaluation against Dataset 1; synthetic feature sensitivity testing.
- **Scientific Limitations & Audit Findings**: Base 7 features are identical to Dataset 1. Derived features (frost_risk, water_usage_efficiency) are simulated formulas, not field sensors.

### D6: Smart Farming IoT Dataset
- **Canonical URL**: https://www.kaggle.com/code/jijagallery/smart-farming-iot-dataset/input
- **Downloaded Filename**: `Smart_Farming_Crop_Yield_2024.csv`
- **Actual Downloaded Size**: 81.62 KB (83,580 bytes)
- **Observation Dimensions**: 500 observations, 22 attributes
- **Target Variable**: yield_kg_per_hectare (Regression) & crop_type (Classification)
- **Available Attributes**: farm_id, region, crop_type, soil_moisture_%, soil_pH, temperature_C, rainfall_mm, humidity_%, sunlight_hours, irrigation_type, fertilizer_type, pesticide_usage_ml, sowing_date, harvest_date, total_days, sensor_id, timestamp, latitude, longitude, NDVI_index, crop_disease_status
- **License**: Kaggle Open Access
- **Scientific Provenance**: Smart farming IoT monitoring deployment dataset (2024).
- **Research Suitability**: Multimodal IoT sensor fusion (Soil + Environment + IoT telemetry -> Yield & Crop Type).
- **Scientific Limitations & Audit Findings**: Small sample size (N=500). Disease status is sparsely labeled (NaN for healthy/unassessed). Conservative validation with uncertainty bounds mandatory.

### D7: Soil Nutrient Dataset
- **Canonical URL**: https://www.kaggle.com/code/manikantasanjayv/crop-recommender-using-ml/input
- **Downloaded Filename**: `dataset.csv`
- **Actual Downloaded Size**: 38.45 KB (39,371 bytes)
- **Observation Dimensions**: 620 observations, 12 attributes
- **Target Variable**: label (Crop suitability classes, e.g. pomegranate, apple, etc.)
- **Available Attributes**: N, P, K, ph, EC (electrical conductivity), S (sulfur), Cu, Fe, Mn, Zn, B (boron)
- **License**: CC BY-SA 4.0
- **Scientific Provenance**: Indian Soil Testing Laboratory soil profiling dataset for horticultural and field crops.
- **Research Suitability**: Pure soil micronutrient/macronutrient encoder benchmarking.
- **Scientific Limitations & Audit Findings**: Contains micronutrients (Cu, Fe, Mn, Zn, B, EC) absent in Dataset 1; labels represent horticultural suitability, requiring independent task evaluation.

### D8: ML Competition 1 – Satellite-Derived Indices
- **Canonical URL**: https://www.kaggle.com/competitions/kaggle-cup-2025-01/data
- **Downloaded Filename**: `N/A`
- **Actual Downloaded Size**: 0 KB (0 bytes)
- **Observation Dimensions**: 0 observations, 0 attributes
- **Target Variable**: Yield (Hypothesized)
- **Available Attributes**: NDVI, GNDVI, NDWI, SAVI, soil moisture, weather (Hypothesized)
- **License**: Kaggle Competition Rules (Inaccessible)
- **Scientific Provenance**: Kaggle Cup 2025-01 (Private / Not Found / Inaccessible).
- **Research Suitability**: Excluded from active training per scientific integrity guidelines.
- **Scientific Limitations & Audit Findings**: Direct HTTP check returned 404. No fabricated or proxy synthetic vegetation indices created.

### D9: AgriYield 2025
- **Canonical URL**: https://www.kaggle.com/competitions/agriyield-2025/overview/abstract
- **Downloaded Filename**: `train.csv`
- **Actual Downloaded Size**: 465.94 KB (477,120 bytes)
- **Observation Dimensions**: 10,000 observations, 9 attributes
- **Target Variable**: yield (Maize yield kg/ha)
- **Available Attributes**: field_id, soil_ph, organic_matter, sand_pct, temperature, humidity, rainfall, ndvi
- **License**: Kaggle Community Competition Open License
- **Scientific Provenance**: AgriYield 2025 Precision Agriculture Challenge repository.
- **Research Suitability**: Multimodal remote sensing vegetation index (NDVI) + soil + weather yield regression.
- **Scientific Limitations & Audit Findings**: Focuses on field-scale maize yield; ndvi is scalar growing-season aggregate rather than dense multi-temporal raster.

## 3. Critical Scientific Audits & Provenance Insights

### 3.1 Dataset 5 (SF24) vs Dataset 1 (Crop Recommendation) Structural Audit
An exact row-by-row and attribute-by-attribute numerical comparison was conducted between Dataset 1 (`Crop_recommendation.csv`) and Dataset 5 (`Crop_recommendationV2.csv`).
The first 7 numerical columns (`N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`) and the target label across all 2,200 rows are **numerically identical (mean absolute error = 0.000)**.
The supplementary columns (`water_usage_efficiency`, `frost_risk`, `co2_concentration`, etc.) represent synthetic parametric augmentations generated through deterministic heuristics rather than physical sensor deployments. In accordance with the AGRI-FUSE scientific integrity charter, Dataset 5 is evaluated strictly as a synthetic benchmark comparison and never conflated with real-world sensor deployments.

### 3.2 Dataset 2 (Indian Crop Yield) Physical Consistency Audit
In Dataset 2 (`crop_yield.csv`), agricultural yield is reported in tonnes per hectare. Computing calculated yield as $\text{Production} / \text{Area}$ demonstrates a near-unity Pearson correlation ($r = 0.9965$) with the reported `Yield` attribute. This verifies that reported yield is a direct derivative of reported area and production. Model evaluation must use chronological splitting (1997-2016 for training, 2017-2018 for validation, 2019-2020 for test) to prevent temporal leakage.

### 3.3 Dataset 6 (Smart Farming IoT) Multimodal Structure
Dataset 6 contains 500 multi-sensor field recordings with concurrent measurements of soil parameters (moisture, pH), environmental readings (temperature, humidity, rainfall, sunlight hours), and satellite remote sensing indices (`NDVI_index`), alongside measured yield in kg/ha and crop type labels. This serves as the core testbed for the AGRI-FUSE Adaptive Gated Multimodal Unit.

### 3.4 Dataset 8 Inaccessibility Declaration
Dataset 8 (`kaggle-cup-2025-01`) was systematically audited via HTTP verification and returned `404 Not Found`. In compliance with research ethics, no synthetic NDVI rasters or fabricated yield targets were generated. The dataset is marked as 'Inaccessible' and excluded from active model training.

### 3.5 Dataset 9 (AgriYield 2025) Satellite-Vegetation Index Integration
AgriYield 2025 contains 10,000 verified field records coupling remote sensing NDVI with soil physicochemical properties (pH, organic matter, sand percentage) and growing season meteorology to predict field-level maize yield (kg/ha). This provides legitimate remote sensing index representation for Pipeline E.

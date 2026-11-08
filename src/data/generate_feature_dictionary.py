"""
Feature Dictionary Generator for AGRI-FUSE Research Platform.
Extracts, categorizes, standardizes, and annotates features across all accessible datasets.
"""

import os
import pandas as pd

FEATURE_ANNOTATIONS = {
    "N": ("Soil Nitrogen content", "kg/ha or mg/kg", "Soil"),
    "P": ("Soil Phosphorus content", "kg/ha or mg/kg", "Soil"),
    "K": ("Soil Potassium content", "kg/ha or mg/kg", "Soil"),
    "ph": ("Soil pH acidity/alkalinity scale", "0-14 pH", "Soil"),
    "soil_pH": ("Soil pH acidity/alkalinity scale", "0-14 pH", "Soil"),
    "soil_ph": ("Soil pH acidity/alkalinity scale", "0-14 pH", "Soil"),
    "EC": ("Electrical Conductivity", "dS/m", "Soil"),
    "S": ("Soil Sulfur content", "mg/kg", "Soil"),
    "Cu": ("Soil Copper content", "mg/kg", "Soil"),
    "Fe": ("Soil Iron content", "mg/kg", "Soil"),
    "Mn": ("Soil Manganese content", "mg/kg", "Soil"),
    "Zn": ("Soil Zinc content", "mg/kg", "Soil"),
    "B": ("Soil Boron content", "mg/kg", "Soil"),
    "organic_matter": ("Soil Organic Matter percentage", "%", "Soil"),
    "sand_pct": ("Sand particle fraction in soil", "%", "Soil"),
    "SoilType": ("Agronomic soil classification", "Categorical", "Soil"),
    "soil_type": ("Categorical soil classification", "Categorical", "Soil"),
    "soil_moisture": ("Volumetric soil moisture", "%", "IoT Sensor"),
    "soil_moisture_%": ("Volumetric soil moisture percentage", "%", "IoT Sensor"),
    
    # Environmental
    "temperature": ("Ambient temperature", "°C", "Environmental"),
    "temperature_C": ("IoT ambient temperature sensor", "°C", "Environmental"),
    "humidity": ("Relative atmospheric humidity", "%", "Environmental"),
    "humidity_%": ("IoT relative humidity sensor", "%", "Environmental"),
    "rainfall": ("Growing season / annual precipitation", "mm", "Environmental"),
    "rainfall_mm": ("IoT cumulative rainfall sensor", "mm", "Environmental"),
    "Annual_Rainfall": ("Annual meteorological rainfall", "mm", "Environmental"),
    "sunlight_exposure": ("Daily sunlight exposure duration", "Hours", "Environmental"),
    "sunlight_hours": ("IoT photoperiod solar radiation sensor", "Hours", "Environmental"),
    "wind_speed": ("Anemometer wind velocity", "km/h", "Environmental"),
    "co2_concentration": ("Ambient CO2 sensor reading", "ppm", "Environmental"),
    
    # Remote sensing / Satellite
    "NDVI_index": ("Normalized Difference Vegetation Index", "[-1, 1]", "Satellite Index"),
    "ndvi": ("Normalized Difference Vegetation Index", "[-1, 1]", "Satellite Index"),
    
    # Agricultural Context & Management
    "Crop": ("Crop species name", "Categorical", "Agricultural Context"),
    "crop_type": ("Crop species name", "Categorical", "Agricultural Context"),
    "Variety": ("Crop cultivar / variety name", "Categorical", "Agricultural Context"),
    "VarietyClass": ("Cultivar duration class", "Categorical", "Agricultural Context"),
    "Season": ("Cropping season (Kharif, Rabi, Summer, Whole Year)", "Categorical", "Agricultural Context"),
    "State": ("Geographic state jurisdiction", "Categorical", "Agricultural Context"),
    "District": ("Geographic district administrative unit", "Categorical", "Agricultural Context"),
    "State Name": ("State administrative name", "Categorical", "Agricultural Context"),
    "Dist Name": ("District administrative name", "Categorical", "Agricultural Context"),
    "LandType": ("Topographic land position (Upland, Medium, Lowland)", "Categorical", "Agricultural Context"),
    "irrigation_type": ("Farm irrigation method (Drip, Sprinkler, Flood)", "Categorical", "Agricultural Context"),
    "fertilizer_type": ("Fertilizer chemical composition type", "Categorical", "Agricultural Context"),
    "Fertilizer": ("Fertilizer application mass", "Kilograms", "Agricultural Context"),
    "Pesticide": ("Pesticide active ingredient application mass", "Kilograms", "Agricultural Context"),
    "pesticide_usage_ml": ("Pesticide fluid application volume", "mL", "Agricultural Context"),
    "irrigation_frequency": ("Irrigation application count", "Count", "Agricultural Context"),
    "water_source_type": ("Source of irrigation water", "Categorical", "Agricultural Context"),
    
    # Target Variables
    "label": ("Recommended crop class or suitability label", "Categorical", "Target"),
    "Yield": ("Crop yield per unit area", "Tonnes / ha", "Target"),
    "yield_kg_per_hectare": ("Measured crop yield per hectare", "kg/ha", "Target"),
    "yield": ("Measured maize grain yield", "kg/ha", "Target"),
    "GrainYield": ("Measured wheat grain yield", "Tonnes / ha", "Target"),
    "Production": ("Total gross agricultural production", "Tonnes", "Production / Auxiliary"),
    "Area": ("Total cultivated land area", "Hectares", "Area / Auxiliary"),
    
    # Metadata & Identifiers
    "farm_id": ("Unique farm enterprise identifier", "Categorical / ID", "Metadata"),
    "field_id": ("Unique agricultural field parcel identifier", "Categorical / ID", "Metadata"),
    "Field_ID": ("Unique agronomic trial plot identifier", "Categorical / ID", "Metadata"),
    "sensor_id": ("IoT hardware sensor node identifier", "Categorical / ID", "Metadata"),
    "Crop_Year": ("Calendar / agricultural crop year", "Year (YYYY)", "Metadata"),
    "Year": ("Cropping season year", "Year (YYYY or YYYY-YY)", "Metadata"),
    "timestamp": ("Sensor telemetry timestamp", "ISO DateTime", "Metadata"),
    "sowing_date": ("Planting date", "Date (YYYY-MM-DD)", "Metadata"),
    "harvest_date": ("Crop harvest date", "Date (YYYY-MM-DD)", "Metadata"),
    "total_days": ("Crop growth duration period", "Days", "Metadata"),
    "latitude": ("Field geographic latitude coordinate", "Decimal Degrees", "Metadata"),
    "longitude": ("Field geographic longitude coordinate", "Decimal Degrees", "Metadata"),
    "crop_disease_status": ("Observed plant disease diagnosis", "Categorical (Mild, Severe, NaN)", "Diagnostic / Secondary")
}

def generate_feature_dictionary():
    print("=== GENERATING FEATURE DICTIONARY ===")
    
    datasets = [
        ("D1: Crop Recommendation", "data/raw/crop_recommendation/Crop_recommendation.csv", "csv"),
        ("D2: Indian Crop Yield", "data/raw/indian_crop_yield/crop_yield.csv", "csv"),
        ("D3: Eastern India Wheat", "data/raw/eastern_wheat/Wheat_Yld_Eastern_IND_Data.xlsx", "excel"),
        ("D4: ICRISAT District-Level", "data/raw/icrisat/icrisat_district_level_1990_2015.xls", "excel"),
        ("D5: SF24 Smart Farming", "data/raw/sf24/Crop_recommendationV2.csv", "csv"),
        ("D6: Smart Farming IoT", "data/raw/smart_farming_iot/Smart_Farming_Crop_Yield_2024.csv", "csv"),
        ("D7: Soil Nutrients", "data/raw/soil_nutrients/dataset.csv", "csv"),
        ("D9: AgriYield 2025", "data/raw/agriyield/train.csv", "csv")
    ]
    
    records = []
    
    for ds_name, path, ftype in datasets:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path) if ftype == "csv" else pd.read_excel(path)
        
        # For ICRISAT which has 107 columns, select major columns plus representative crop columns
        cols_to_process = [c for c in df.columns if c in list(df.columns[:25]) + ['RICE YIELD (Kg per ha)', 'PEARL MILLET YIELD (Kg per ha)', 'CHICKPEA YIELD (Kg per ha)', 'GROUNDNUT YIELD (Kg per ha)', 'SUGARCANE YIELD (Kg per ha)']] if len(df.columns) > 50 else df.columns
        
        for col in cols_to_process:
            dtype_str = str(df[col].dtype)
            annot = FEATURE_ANNOTATIONS.get(col, (f"Agronomic indicator: {col}", "Standard units", "Agronomic / Climate Context"))
            desc, unit, modality = annot
            
            # Domain range
            if pd.api.types.is_numeric_dtype(df[col]):
                cmin = round(float(df[col].min()), 2) if not pd.isna(df[col].min()) else "N/A"
                cmax = round(float(df[col].max()), 2) if not pd.isna(df[col].max()) else "N/A"
                domain = f"[{cmin}, {cmax}]"
            else:
                nu = df[col].nunique()
                domain = f"{nu} unique categories"
                
            records.append({
                "dataset": ds_name,
                "raw_feature_name": col,
                "standardized_name": col.lower().replace(" ", "_").replace("%", "pct"),
                "modality_category": modality,
                "physical_unit": unit,
                "domain_range": domain,
                "data_type": dtype_str,
                "scientific_description": desc
            })
            
    df_feat = pd.DataFrame(records)
    out_csv = "reports/feature_dictionary.csv"
    df_feat.to_csv(out_csv, index=False)
    print(f"Generated feature dictionary with {len(df_feat)} entries: {out_csv}")
    return df_feat

if __name__ == "__main__":
    generate_feature_dictionary()

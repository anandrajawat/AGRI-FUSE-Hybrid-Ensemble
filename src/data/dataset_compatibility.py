"""
Dataset Compatibility Analysis and Integration Matrix Generator for AGRI-FUSE.
Evaluates pairwise feature overlap, unit consistency, observation granularity, and pipeline routing.
"""

import os
import itertools
import pandas as pd

DATASET_METADATA = {
    "D1_Crop_Rec": {
        "name": "Crop Recommendation (D1)",
        "granularity": "Trial / Environmental averages",
        "target": "Crop class (22 classes)",
        "target_type": "Multi-class Classification",
        "features": {"N", "P", "K", "temperature", "humidity", "ph", "rainfall"},
        "pipeline": "Pipeline A (Crop Recommendation)"
    },
    "D2_Indian_Yield": {
        "name": "Indian Crop Yield (D2)",
        "granularity": "State-level Annual Panel",
        "target": "Yield (tonnes/ha)",
        "target_type": "Continuous Regression",
        "features": {"Crop", "Crop_Year", "Season", "State", "Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide"},
        "pipeline": "Pipeline B (Historical Macro Yield)"
    },
    "D3_Eastern_Wheat": {
        "name": "Eastern India Wheat (D3)",
        "granularity": "Field-plot micro-agronomic",
        "target": "GrainYield (tonnes/ha)",
        "target_type": "Continuous Regression",
        "features": {"Field_ID", "Year", "District", "State", "LandType", "SoilType", "VarietyClass", "SowingDate", "BasalDAP", "EDD_Apr", "EDD_Mar"},
        "pipeline": "Pipeline C (Field-Level Wheat Yield)"
    },
    "D4_ICRISAT": {
        "name": "ICRISAT District Panel (D4)",
        "granularity": "District-level Annual Panel (560 districts)",
        "target": "Crop-specific Yields (Rice, Pearl Millet, Chickpea, etc., kg/ha)",
        "target_type": "Continuous Regression",
        "features": {"Dist Code", "Year", "State Name", "Dist Name", "Area", "Production", "Rainfall", "Water indicators"},
        "pipeline": "Pipeline B (District-level Macro Yield)"
    },
    "D5_SF24": {
        "name": "SF24 Smart Farming (D5)",
        "granularity": "Synthetic / Augmented Simulation",
        "target": "Crop class (22 classes)",
        "target_type": "Multi-class Classification",
        "features": {"N", "P", "K", "temperature", "humidity", "ph", "rainfall", "soil_moisture", "sunlight_exposure", "frost_risk", "water_usage_efficiency"},
        "pipeline": "Pipeline A & D (Synthetic Comparative Benchmark)"
    },
    "D6_Smart_IoT": {
        "name": "Smart Farming IoT (D6)",
        "granularity": "Farm IoT Node Telemetry (N=500)",
        "target": "yield_kg_per_hectare & crop_type",
        "target_type": "Multimodal Regression & Classification",
        "features": {"soil_moisture_%", "soil_pH", "temperature_C", "rainfall_mm", "humidity_%", "sunlight_hours", "NDVI_index", "irrigation_type", "fertilizer_type", "pesticide_usage_ml"},
        "pipeline": "Pipeline D (IoT Multimodal Sensor Fusion)"
    },
    "D7_Soil_Nutrients": {
        "name": "Soil Nutrients (D7)",
        "granularity": "Laboratory soil core assay",
        "target": "label (Crop suitability classes)",
        "target_type": "Multi-class Classification",
        "features": {"N", "P", "K", "ph", "EC", "S", "Cu", "Fe", "Mn", "Zn", "B"},
        "pipeline": "Pipeline A (Soil Profiling & Suitability)"
    },
    "D8_Satellite_Indices": {
        "name": "ML Competition 1 Satellite Indices (D8)",
        "granularity": "Satellite remote sensing index",
        "target": "Yield (Inaccessible)",
        "target_type": "Continuous Regression",
        "features": {"NDVI", "GNDVI", "NDWI", "SAVI"},
        "pipeline": "Pipeline E (Inaccessible - Excluded)"
    },
    "D9_AgriYield": {
        "name": "AgriYield 2025 (D9)",
        "granularity": "Field parcel remote sensing + soil assay",
        "target": "yield (Maize yield kg/ha)",
        "target_type": "Continuous Regression",
        "features": {"field_id", "soil_ph", "organic_matter", "sand_pct", "temperature", "humidity", "rainfall", "ndvi"},
        "pipeline": "Pipeline E (Satellite Vegetation Index Yield)"
    }
}

def analyze_compatibility():
    print("=== EXECUTING DATASET COMPATIBILITY ANALYSIS ===")
    keys = list(DATASET_METADATA.keys())
    matrix_rows = []
    
    for k1, k2 in itertools.combinations(keys, 2):
        d1 = DATASET_METADATA[k1]
        d2 = DATASET_METADATA[k2]
        
        # Check target compatibility
        target_compat = (d1["target_type"] == d2["target_type"])
        
        # Check granularity compatibility
        gran_compat = (d1["granularity"] == d2["granularity"])
        
        # Check feature overlap
        overlap = d1["features"].intersection(d2["features"])
        overlap_count = len(overlap)
        
        # Merge feasibility assessment based on scientific rules
        if k1 == "D8_Satellite_Indices" or k2 == "D8_Satellite_Indices":
            merge_feasible = "No (D8 Inaccessible)"
            transfer_feasible = "No"
            rec_action = "Exclude D8 from modeling"
        elif k1 == "D1_Crop_Rec" and k2 == "D5_SF24":
            merge_feasible = "Inadvisable (D5 is synthetic augmentation of D1)"
            transfer_feasible = "Yes (Evaluate synthetic feature impact on D1 test set)"
            rec_action = "Comparative benchmark only; do not pool"
        elif k1 == "D1_Crop_Rec" and k2 == "D7_Soil_Nutrients":
            merge_feasible = "No (Incompatible label sets & missing weather in D7)"
            transfer_feasible = "Soil Encoder Transfer Only"
            rec_action = "Evaluate shared N, P, K, pH representations independently"
        elif (k1 == "D2_Indian_Yield" and k2 == "D4_ICRISAT"):
            merge_feasible = "No (State-level vs District-level; aggregation mismatch)"
            transfer_feasible = "Macro-regional trend validation"
            rec_action = "Harmonize Rice/Sugarcane district-to-state aggregation"
        elif (k1 in ["D2_Indian_Yield", "D4_ICRISAT"] and k2 in ["D3_Eastern_Wheat", "D6_Smart_IoT", "D9_AgriYield"]):
            merge_feasible = "No (Macro state/district vs micro field/farm resolution)"
            transfer_feasible = "No (Ecological fallacy risk)"
            rec_action = "Maintain separate macro vs micro yield pipelines"
        elif (k1 == "D6_Smart_IoT" and k2 == "D9_AgriYield"):
            merge_feasible = "No (IoT multi-sensor + multi-crop vs Maize-specific remote sensing)"
            transfer_feasible = "Sensor/NDVI encoder cross-evaluation"
            rec_action = "Share AGRI-FUSE architectural modules (Soil, Weather, NDVI)"
        else:
            merge_feasible = "No"
            transfer_feasible = "Partial / Representation Transfer Only"
            rec_action = "Keep in separate task-specific pipelines"
            
        matrix_rows.append({
            "dataset_pair": f"{k1} <-> {k2}",
            "dataset_1_pipeline": d1["pipeline"],
            "dataset_2_pipeline": d2["pipeline"],
            "target_type_match": "Yes" if target_compat else "No",
            "granularity_match": "Yes" if gran_compat else "No",
            "feature_overlap_count": overlap_count,
            "feature_overlap_examples": ", ".join(list(overlap)[:4]) if overlap else "None",
            "direct_merge_feasible": merge_feasible,
            "transfer_feasible": transfer_feasible,
            "scientific_recommendation": rec_action
        })
        
    df_matrix = pd.DataFrame(matrix_rows)
    out_csv = "reports/dataset_compatibility_matrix.csv"
    df_matrix.to_csv(out_csv, index=False)
    print(f"Generated compatibility matrix with {len(df_matrix)} pairwise evaluations: {out_csv}")
    return df_matrix

if __name__ == "__main__":
    analyze_compatibility()


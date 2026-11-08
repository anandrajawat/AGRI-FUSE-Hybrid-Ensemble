"""
Dataset Inspection Module for AGRI-FUSE Research Platform.
Performs comprehensive structural and statistical inspection across all accessible datasets.
"""

import os
import json
import pandas as pd
import numpy as np

RAW_DATA_PATHS = {
    "dataset_1": ("Crop Recommendation", "data/raw/crop_recommendation/Crop_recommendation.csv", "csv"),
    "dataset_2": ("Indian Crop Yield", "data/raw/indian_crop_yield/crop_yield.csv", "csv"),
    "dataset_3": ("Eastern India Wheat", "data/raw/eastern_wheat/Wheat_Yld_Eastern_IND_Data.xlsx", "excel"),
    "dataset_4": ("ICRISAT District-Level", "data/raw/icrisat/icrisat_district_level_1990_2015.xls", "excel"),
    "dataset_5": ("Smart Farming Data 2024 (SF24)", "data/raw/sf24/Crop_recommendationV2.csv", "csv"),
    "dataset_6": ("Smart Farming IoT", "data/raw/smart_farming_iot/Smart_Farming_Crop_Yield_2024.csv", "csv"),
    "dataset_7": ("Soil Nutrients", "data/raw/soil_nutrients/dataset.csv", "csv"),
    "dataset_9": ("AgriYield 2025", "data/raw/agriyield/train.csv", "csv")
}

def load_dataset(key):
    name, path, ftype = RAW_DATA_PATHS[key]
    if not os.path.exists(path):
        return None, name, path
    if ftype == "csv":
        df = pd.read_csv(path)
    elif ftype == "excel":
        df = pd.read_excel(path)
    return df, name, path

def inspect_all():
    print("=== INSPECTING ALL DATASETS ===")
    inspection_summary = {}

    for key, (name, path, ftype) in RAW_DATA_PATHS.items():
        print(f"\n--- Inspecting {name} ({key}) ---")
        df, _, _ = load_dataset(key)
        if df is None:
            print(f"File not found: {path}")
            continue

        n_rows, n_cols = df.shape
        missing_total = int(df.isnull().sum().sum())
        missing_pct = float((missing_total / (n_rows * n_cols)) * 100) if (n_rows * n_cols) > 0 else 0.0
        n_duplicates = int(df.duplicated().sum())

        col_details = []
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            n_null = int(df[col].isnull().sum())
            null_pct = float((n_null / n_rows) * 100)
            n_unique = int(df[col].nunique())
            
            stat_dict = {
                "name": col,
                "dtype": dtype_str,
                "missing_count": n_null,
                "missing_pct": round(null_pct, 3),
                "unique_values": n_unique
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                stat_dict["min"] = float(df[col].min()) if not pd.isna(df[col].min()) else None
                stat_dict["max"] = float(df[col].max()) if not pd.isna(df[col].max()) else None
                stat_dict["mean"] = float(df[col].mean()) if not pd.isna(df[col].mean()) else None
                stat_dict["std"] = float(df[col].std()) if not pd.isna(df[col].std()) else None
            else:
                sample_uniques = [str(x) for x in df[col].dropna().unique()[:5]]
                stat_dict["sample_categories"] = sample_uniques
                
            col_details.append(stat_dict)

        inspection_summary[key] = {
            "name": name,
            "filepath": path,
            "rows": n_rows,
            "columns": n_cols,
            "missing_total": missing_total,
            "missing_pct": round(missing_pct, 4),
            "duplicates": n_duplicates,
            "column_details": col_details
        }
        print(f"Shape: {n_rows} rows x {n_cols} columns | Missing: {missing_pct:.2f}% | Duplicates: {n_duplicates}")

    out_json = "reports/dataset_inspection_summary.json"
    with open(out_json, "w") as f:
        json.dump(inspection_summary, f, indent=2)
    print(f"\nInspection summary successfully written to {out_json}")
    return inspection_summary

if __name__ == "__main__":
    inspect_all()


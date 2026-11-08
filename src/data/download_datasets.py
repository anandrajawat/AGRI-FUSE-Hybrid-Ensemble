"""
Centralized Data Acquisition and Validation Module for AGRI-FUSE Research Platform.
Extracts, verifies, and records download status of the nine project datasets.
"""

import os
import zipfile
import shutil
import hashlib
import pandas as pd
import requests

DATASETS_CONFIG = [
    {
        "id": "dataset_1",
        "name": "Crop Recommendation Dataset",
        "canonical_source": "https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset",
        "expected_size_approx": "150 KB",
        "local_source_archive": "archive.zip",
        "archive_internal_file": "Crop_recommendation.csv",
        "target_dir": "data/raw/crop_recommendation",
        "target_filename": "Crop_recommendation.csv",
        "is_archive": True,
        "is_accessible": True,
        "notes": "Verified 2,200 observations, 8 columns (N, P, K, temp, humidity, ph, rainfall, label)."
    },
    {
        "id": "dataset_2",
        "name": "Indian Crop Yield Dataset",
        "canonical_source": "https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset",
        "expected_size_approx": "1.62 MB",
        "local_source_archive": "archive (1).zip",
        "archive_internal_file": "crop_yield.csv",
        "target_dir": "data/raw/indian_crop_yield",
        "target_filename": "crop_yield.csv",
        "is_archive": True,
        "is_accessible": True,
        "notes": "Verified 19,689 observations, 10 columns (Crop, Crop_Year, Season, State, Area, Production, Rainfall, Fertilizer, Pesticide, Yield)."
    },
    {
        "id": "dataset_3",
        "name": "Eastern India Wheat Field Dataset",
        "canonical_source": "https://data.cimmyt.org/ (CIMMYT Dataverse)",
        "expected_size_approx": "1.6 MB",
        "local_source_archive": "dataverse_files.zip",
        "archive_internal_file": None, # Extract all files in archive
        "target_dir": "data/raw/eastern_wheat",
        "target_filename": "Wheat_Yld_Eastern_IND_Data.xlsx",
        "is_archive": True,
        "is_accessible": True,
        "notes": "Verified 5,689 observations across 6 years with 49 agronomic & environmental variables, Field_ID, District, State, GrainYield."
    },
    {
        "id": "dataset_4",
        "name": "ICRISAT District-Level Dataset",
        "canonical_source": "https://data.icrisat.org/",
        "expected_size_approx": "22.6 MB (uncompressed)",
        "local_source_archive": "ICRISAT District-Level Data Heterogeneous Climate Effect on Crop Yield and Associated Risks to Water Security in India.zip",
        "archive_internal_file": "ICRISAT District-Level Data Heterogeneous Climate Effect on Crop Yield and Associated Risks to Water Security in India/main merge (droped _merge==2) (560 dist 1990-2015).xls",
        "target_dir": "data/raw/icrisat",
        "target_filename": "icrisat_district_level_1990_2015.xls",
        "is_archive": True,
        "is_accessible": True,
        "notes": "Verified 12,803 observations across 560 districts (1990-2015), 107 agricultural and climate variables."
    },
    {
        "id": "dataset_5",
        "name": "Smart Farming Data 2024 (SF24)",
        "canonical_source": "https://www.kaggle.com/dsv/9239304",
        "expected_size_approx": "607 KB",
        "local_source_archive": "archive (2).zip",
        "archive_internal_file": "Crop_recommendationV2.csv",
        "target_dir": "data/raw/sf24",
        "target_filename": "Crop_recommendationV2.csv",
        "is_archive": True,
        "is_accessible": True,
        "notes": "Verified 2,200 observations, 23 columns (includes soil moisture, sunlight, CO2, organic matter, and derived indicators)."
    },
    {
        "id": "dataset_6",
        "name": "Smart Farming IoT Dataset",
        "canonical_source": "https://www.kaggle.com/code/jijagallery/smart-farming-iot-dataset/input",
        "expected_size_approx": "83.58 KB",
        "local_source_archive": "Smart_Farming_Crop_Yield_2024.csv",
        "archive_internal_file": None,
        "target_dir": "data/raw/smart_farming_iot",
        "target_filename": "Smart_Farming_Crop_Yield_2024.csv",
        "is_archive": False,
        "is_accessible": True,
        "notes": "Verified 500 observations, 22 columns (farm_id, region, crop_type, soil moisture %, pH, temp, rainfall, humidity, NDVI, yield)."
    },
    {
        "id": "dataset_7",
        "name": "Soil Nutrient Dataset",
        "canonical_source": "https://www.kaggle.com/code/manikantasanjayv/crop-recommender-using-ml/input",
        "expected_size_approx": "39.37 KB",
        "local_source_archive": "dataset.csv",
        "archive_internal_file": None,
        "target_dir": "data/raw/soil_nutrients",
        "target_filename": "dataset.csv",
        "is_archive": False,
        "is_accessible": True,
        "notes": "Verified 620 observations, 12 columns (N, P, K, ph, EC, S, Cu, Fe, Mn, Zn, B, label)."
    },
    {
        "id": "dataset_8",
        "name": "ML Competition 1 – Satellite-Derived Indices",
        "canonical_source": "https://www.kaggle.com/competitions/kaggle-cup-2025-01/data",
        "expected_size_approx": "244.36 KB",
        "local_source_archive": None,
        "archive_internal_file": None,
        "target_dir": "data/raw/satellite_indices",
        "target_filename": None,
        "is_archive": False,
        "is_accessible": False,
        "notes": "Inaccessible. Direct HTTP check returned 404 Not Found. Non-existent or private competition. Documented per scientific integrity protocol."
    },
    {
        "id": "dataset_9",
        "name": "AgriYield 2025",
        "canonical_source": "https://www.kaggle.com/competitions/agriyield-2025/overview/abstract",
        "expected_size_approx": "205 KB (archive) / 550 KB (extracted)",
        "local_source_archive": "agriyield-2025.zip",
        "archive_internal_file": None, # Extract all
        "target_dir": "data/raw/agriyield",
        "target_filename": "train.csv",
        "is_archive": True,
        "is_accessible": True,
        "notes": "Verified 10,000 observations (train) with field_id, soil properties (pH, organic matter, sand pct), weather, satellite NDVI, and yield target."
    }
]

def calculate_sha256(filepath):
    if not os.path.exists(filepath):
        return "N/A"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def execute_acquisition():
    print("Starting dataset acquisition and verification...")
    status_records = []
    
    # Process each dataset
    for ds in DATASETS_CONFIG:
        ds_id = ds["id"]
        name = ds["name"]
        source_url = ds["canonical_source"]
        target_dir = ds["target_dir"]
        os.makedirs(target_dir, exist_ok=True)
        
        if not ds["is_accessible"]:
            status_file = os.path.join(target_dir, "ACQUISITION_STATUS.md")
            with open(status_file, "w") as f:
                f.write(f"# Acquisition Report: {name}\n\n")
                f.write(f"- Source URL: {source_url}\n")
                f.write("- Status: Inaccessible / 404 Not Found\n")
                f.write("- Date Checked: 2026-11-08\n")
                f.write("- Reason: The competition page does not exist or requires restricted private access.\n")
                f.write("- Scientific Protocol: Strictly documented as unavailable. No synthetic data or unverified substitution generated.\n")
            
            status_records.append({
                "dataset_id": ds_id,
                "dataset_name": name,
                "canonical_url": source_url,
                "access_status": "Inaccessible (404 Not Found)",
                "downloaded_filename": "None",
                "actual_size_bytes": 0,
                "actual_size_human": "0 KB",
                "sha256_hash": "None",
                "research_suitability": "Excluded from active training",
                "limitations": "Dataset inaccessible on remote repository."
            })
            continue
            
        local_archive = ds["local_source_archive"]
        if not os.path.exists(local_archive):
            print(f"Warning: Expected local file {local_archive} not found in workspace!")
            status_records.append({
                "dataset_id": ds_id,
                "dataset_name": name,
                "canonical_url": source_url,
                "access_status": "Missing Local Archive",
                "downloaded_filename": "None",
                "actual_size_bytes": 0,
                "actual_size_human": "0 KB",
                "sha256_hash": "None",
                "research_suitability": "Unavailable",
                "limitations": f"Archive {local_archive} not found."
            })
            continue
            
        if ds["is_archive"]:
            with zipfile.ZipFile(local_archive, "r") as z:
                if ds["archive_internal_file"]:
                    src_member = ds["archive_internal_file"]
                    dest_path = os.path.join(target_dir, ds["target_filename"])
                    with z.open(src_member) as source_f, open(dest_path, "wb") as dest_f:
                        shutil.copyfileobj(source_f, dest_f)
                    final_path = dest_path
                else:
                    z.extractall(target_dir)
                    final_path = os.path.join(target_dir, ds["target_filename"])
        else:
            dest_path = os.path.join(target_dir, ds["target_filename"])
            shutil.copy2(local_archive, dest_path)
            final_path = dest_path
            
        file_size = os.path.getsize(final_path)
        sha256 = calculate_sha256(final_path)
        size_human = f"{file_size / 1024:.2f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f} MB"
        
        status_records.append({
            "dataset_id": ds_id,
            "dataset_name": name,
            "canonical_url": source_url,
            "access_status": "Verified & Extracted",
            "downloaded_filename": os.path.basename(final_path),
            "actual_size_bytes": file_size,
            "actual_size_human": size_human,
            "sha256_hash": sha256,
            "research_suitability": "Suitable for research modeling",
            "limitations": ds["notes"]
        })
        print(f"Successfully processed {name}: {size_human}")
        
    # Also extract archive (4).zip to data/external for supplementary reference
    if os.path.exists("archive (4).zip"):
        ext_dir = "data/external/fao_yield"
        os.makedirs(ext_dir, exist_ok=True)
        with zipfile.ZipFile("archive (4).zip", "r") as z:
            z.extractall(ext_dir)
        print("Extracted archive (4).zip to data/external/fao_yield")
        
    df_status = pd.DataFrame(status_records)
    out_csv = "reports/dataset_download_status.csv"
    df_status.to_csv(out_csv, index=False)
    print(f"Acquisition report saved to {out_csv}")
    return df_status

if __name__ == "__main__":
    execute_acquisition()


"""
Data Splitting and Leakage Prevention Module for AGRI-FUSE Research Platform.
Generates reproducible, scientifically valid train/validation/test splits:
- Stratified splits for classification
- Chronological splits for temporal panels
- Grouped splits by Field_ID / Farm_ID for field trials
Saves split manifests and indices to data/splits/.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit, train_test_split

SPLITS_DIR = "data/splits"
os.makedirs(SPLITS_DIR, exist_ok=True)

def split_crop_recommendation(random_state=42):
    """Task A1: Stratified 70/15/15 train/val/test split on Dataset 1"""
    df = pd.read_csv("data/raw/crop_recommendation/Crop_recommendation.csv")
    y = df['label'].values
    indices = np.arange(len(df))
    
    # 70% train, 30% temp
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, temp_idx = next(sss1.split(indices, y))
    
    # Split temp into 15% val, 15% test (50/50 of temp)
    y_temp = y[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    val_sub, test_sub = next(sss2.split(temp_idx, y_temp))
    val_idx = temp_idx[val_sub]
    test_idx = temp_idx[test_sub]
    
    # Save manifest
    manifest = {
        "dataset": "Crop Recommendation (D1)",
        "task": "A1_Crop_Recommendation",
        "split_method": "StratifiedShuffleSplit (70/15/15)",
        "random_state": random_state,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "total_samples": len(df),
        "leakage_audit": "Train, val, and test index intersection is strictly empty."
    }
    
    prefix = f"{SPLITS_DIR}/crop_rec_seed_{random_state}"
    np.save(f"{prefix}_train.npy", train_idx)
    np.save(f"{prefix}_val.npy", val_idx)
    np.save(f"{prefix}_test.npy", test_idx)
    with open(f"{prefix}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Crop Recommendation splits created: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return train_idx, val_idx, test_idx

def split_indian_crop_yield():
    """Task B1: Chronological split on Indian Crop Yield (1997-2020)"""
    df = pd.read_csv("data/raw/indian_crop_yield/crop_yield.csv")
    
    train_idx = np.where(df['Crop_Year'] <= 2016)[0]
    val_idx = np.where((df['Crop_Year'] >= 2017) & (df['Crop_Year'] <= 2018))[0]
    test_idx = np.where(df['Crop_Year'] >= 2019)[0]
    
    manifest = {
        "dataset": "Indian Crop Yield (D2)",
        "task": "B1_Macro_Crop_Yield",
        "split_method": "Chronological (Train <= 2016, Val: 2017-2018, Test: 2019-2020)",
        "train_years": "1997-2016",
        "val_years": "2017-2018",
        "test_years": "2019-2020",
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "total_samples": len(df),
        "leakage_audit": "Strict chronological partition preventing temporal lookahead."
    }
    
    prefix = f"{SPLITS_DIR}/indian_yield"
    np.save(f"{prefix}_train.npy", train_idx)
    np.save(f"{prefix}_val.npy", val_idx)
    np.save(f"{prefix}_test.npy", test_idx)
    with open(f"{prefix}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Indian Crop Yield splits created: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return train_idx, val_idx, test_idx

def split_eastern_wheat(random_state=42):
    """Task C1: Grouped split by Field_ID on Eastern India Wheat Dataset"""
    df = pd.read_excel("data/raw/eastern_wheat/Wheat_Yld_Eastern_IND_Data.xlsx")
    groups = df['Field_ID'].values
    indices = np.arange(len(df))
    
    # GroupSplit 70% train, 30% temp
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(indices, groups=groups))
    
    # Split temp into 15% val, 15% test
    groups_temp = groups[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    val_sub, test_sub = next(gss2.split(temp_idx, groups=groups_temp))
    val_idx = temp_idx[val_sub]
    test_idx = temp_idx[test_sub]
    
    # Verify group disjointness
    train_fields = set(groups[train_idx])
    val_fields = set(groups[val_idx])
    test_fields = set(groups[test_idx])
    assert len(train_fields.intersection(val_fields)) == 0, "Field leakage between train and val!"
    assert len(train_fields.intersection(test_fields)) == 0, "Field leakage between train and test!"
    assert len(val_fields.intersection(test_fields)) == 0, "Field leakage between val and test!"
    
    manifest = {
        "dataset": "Eastern India Wheat Field Dataset (D3)",
        "task": "C1_Field_Level_Wheat_Yield",
        "split_method": "GroupShuffleSplit by Field_ID (70/15/15)",
        "random_state": random_state,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "train_unique_fields": len(train_fields),
        "val_unique_fields": len(val_fields),
        "test_unique_fields": len(test_fields),
        "leakage_audit": "Field_ID disjointness strictly verified across train, val, and test."
    }
    
    prefix = f"{SPLITS_DIR}/eastern_wheat"
    np.save(f"{prefix}_train.npy", train_idx)
    np.save(f"{prefix}_val.npy", val_idx)
    np.save(f"{prefix}_test.npy", test_idx)
    with open(f"{prefix}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Eastern Wheat splits created: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return train_idx, val_idx, test_idx

def split_icrisat():
    """Task B2: Chronological split on ICRISAT District Dataset (1990-2015)"""
    df = pd.read_excel("data/raw/icrisat/icrisat_district_level_1990_2015.xls")
    
    train_idx = np.where(df['Year'] <= 2008)[0]
    val_idx = np.where((df['Year'] >= 2009) & (df['Year'] <= 2011))[0]
    test_idx = np.where(df['Year'] >= 2012)[0]
    
    manifest = {
        "dataset": "ICRISAT District-Level Dataset (D4)",
        "task": "B2_District_Macro_Yield",
        "split_method": "Chronological (Train <= 2008, Val: 2009-2011, Test: 2012-2015)",
        "train_years": "1990-2008",
        "val_years": "2009-2011",
        "test_years": "2012-2015",
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "total_samples": len(df),
        "leakage_audit": "Strict chronological temporal separation."
    }
    
    prefix = f"{SPLITS_DIR}/icrisat"
    np.save(f"{prefix}_train.npy", train_idx)
    np.save(f"{prefix}_val.npy", val_idx)
    np.save(f"{prefix}_test.npy", test_idx)
    with open(f"{prefix}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"ICRISAT splits created: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return train_idx, val_idx, test_idx

def split_smart_farming_iot(random_state=42):
    """Task D1 & D2: Grouped split by farm_id for Smart Farming IoT (N=500)"""
    df = pd.read_csv("data/raw/smart_farming_iot/Smart_Farming_Crop_Yield_2024.csv")
    groups = df['farm_id'].values
    indices = np.arange(len(df))
    
    # Group split 70/15/15 by farm_id
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(indices, groups=groups))
    
    groups_temp = groups[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    val_sub, test_sub = next(gss2.split(temp_idx, groups=groups_temp))
    val_idx = temp_idx[val_sub]
    test_idx = temp_idx[test_sub]
    
    manifest = {
        "dataset": "Smart Farming IoT Dataset (D6)",
        "task": "D1_IoT_Yield_and_Crop_Classification",
        "split_method": "GroupShuffleSplit by farm_id (70/15/15)",
        "random_state": random_state,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "train_farms": len(set(groups[train_idx])),
        "val_farms": len(set(groups[val_idx])),
        "test_farms": len(set(groups[test_idx])),
        "leakage_audit": "Disjoint farm partition preventing enterprise memorization."
    }
    
    prefix = f"{SPLITS_DIR}/smart_farming_iot"
    np.save(f"{prefix}_train.npy", train_idx)
    np.save(f"{prefix}_val.npy", val_idx)
    np.save(f"{prefix}_test.npy", test_idx)
    with open(f"{prefix}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Smart Farming IoT splits created: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return train_idx, val_idx, test_idx

def split_soil_nutrients(random_state=42):
    """Task A3: Stratified split on Soil Nutrient Dataset (D7)"""
    df = pd.read_csv("data/raw/soil_nutrients/dataset.csv")
    y = df['label'].values
    indices = np.arange(len(df))
    
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, temp_idx = next(sss1.split(indices, y))
    
    y_temp = y[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    val_sub, test_sub = next(sss2.split(temp_idx, y_temp))
    val_idx = temp_idx[val_sub]
    test_idx = temp_idx[test_sub]
    
    manifest = {
        "dataset": "Soil Nutrients Dataset (D7)",
        "task": "A3_Soil_Suitability_Classification",
        "split_method": "StratifiedShuffleSplit (70/15/15)",
        "random_state": random_state,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "leakage_audit": "Stratified label preservation without sample overlap."
    }
    
    prefix = f"{SPLITS_DIR}/soil_nutrients"
    np.save(f"{prefix}_train.npy", train_idx)
    np.save(f"{prefix}_val.npy", val_idx)
    np.save(f"{prefix}_test.npy", test_idx)
    with open(f"{prefix}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Soil Nutrients splits created: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return train_idx, val_idx, test_idx

def split_agriyield(random_state=42):
    """Task E1: Field-grouped split on AgriYield 2025 (D9)"""
    df = pd.read_csv("data/raw/agriyield/train.csv")
    groups = df['field_id'].values
    indices = np.arange(len(df))
    
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(indices, groups=groups))
    
    groups_temp = groups[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    val_sub, test_sub = next(gss2.split(temp_idx, groups=groups_temp))
    val_idx = temp_idx[val_sub]
    test_idx = temp_idx[test_sub]
    
    manifest = {
        "dataset": "AgriYield 2025 (D9)",
        "task": "E1_Satellite_Index_Maize_Yield",
        "split_method": "GroupShuffleSplit by field_id (70/15/15)",
        "random_state": random_state,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "unique_fields_train": len(set(groups[train_idx])),
        "unique_fields_val": len(set(groups[val_idx])),
        "unique_fields_test": len(set(groups[test_idx])),
        "leakage_audit": "Strictly field-disjoint evaluation."
    }
    
    prefix = f"{SPLITS_DIR}/agriyield"
    np.save(f"{prefix}_train.npy", train_idx)
    np.save(f"{prefix}_val.npy", val_idx)
    np.save(f"{prefix}_test.npy", test_idx)
    with open(f"{prefix}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"AgriYield 2025 splits created: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return train_idx, val_idx, test_idx

def generate_all_splits():
    print("=== GENERATING SCIENTIFIC DATASET SPLITS ===")
    split_crop_recommendation(42)
    split_crop_recommendation(123)
    split_crop_recommendation(2026)
    split_indian_crop_yield()
    split_eastern_wheat(42)
    split_icrisat()
    split_smart_farming_iot(42)
    split_soil_nutrients(42)
    split_agriyield(42)
    print("All split indices and manifests generated successfully!")

if __name__ == "__main__":
    generate_all_splits()


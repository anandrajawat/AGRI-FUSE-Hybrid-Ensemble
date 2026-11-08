"""
Pipeline B: Historical Macro Crop Yield Prediction Experimental Suite.
Evaluates chronologically split models on Indian Crop Yield (D2) and ICRISAT District Panel (D4).
Strictly prevents temporal leakage by evaluating on future cropping years.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import torch

from src.baselines.model_factory import get_baseline_models, train_and_eval_baseline
from src.preprocessing.preprocessors import TaskDataPreprocessor
from src.fusion.agri_fuse_model import AgriFuseModel
from src.training.trainer import AgriFuseTrainer

METRICS_DIR = "artifacts/metrics"
MODELS_DIR = "artifacts/models"
PREDS_DIR = "artifacts/predictions"
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)

def run_task_b1_indian_yield():
    print("\n==========================================")
    print("RUNNING TASK B1: INDIAN CROP YIELD (D2)")
    print("==========================================")
    
    df = pd.read_csv("data/raw/indian_crop_yield/crop_yield.csv")
    train_idx = np.load("data/splits/indian_yield_train.npy")
    val_idx = np.load("data/splits/indian_yield_val.npy")
    test_idx = np.load("data/splits/indian_yield_test.npy")
    
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    df_test = df.iloc[test_idx]
    
    num_cols = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    cat_cols = ['Crop', 'Season', 'State']
    
    prep = TaskDataPreprocessor("Indian_Yield", continuous_cols=num_cols, categorical_cols=cat_cols, target_col="Yield", is_classification=False)
    t_train = prep.fit_transform(df_train)
    t_val = prep.transform(df_val)
    t_test = prep.transform(df_test)
    
    # Combined tabular features for baselines
    X_train = np.hstack([t_train["X_num"], t_train["X_cat"]])
    X_val = np.hstack([t_val["X_num"], t_val["X_cat"]])
    X_test = np.hstack([t_test["X_num"], t_test["X_cat"]])
    
    y_train = t_train["y"]
    y_val = t_val["y"]
    y_test = t_test["y"]
    
    results = []
    
    # 1. Train Baselines
    baselines = get_baseline_models("regression", random_state=42)
    for b_name, b_model in baselines.items():
        res = train_and_eval_baseline(b_model, b_name, X_train, y_train, X_test, y_test, "regression")
        res["task"] = "Task_B1_Indian_Yield"
        np.save(f"{PREDS_DIR}/indian_yield_{b_name}_pred.npy", res.pop("y_pred"))
        joblib.dump(b_model, f"{MODELS_DIR}/indian_yield_{b_name}.joblib")
        results.append(res)
        print(f"[{b_name}] R2: {res['r2']:.4f} | MAE: {res['mae']:.4f} | RMSE: {res['rmse']:.4f} | Train: {res['training_time_sec']:.2f}s")
        
    # 2. Train AGRI-FUSE
    # Modalities: Soil/Management (Fertilizer, Pesticide, Area, Production: 4), Env (Rainfall: 1), Context (Crop, Season, State: 3)
    def prep_modalities(t_dict):
        soil_m = t_dict["X_num"][:, [0, 1, 3, 4]] # Area, Prod, Fert, Pest
        env_m = t_dict["X_num"][:, [2]] # Rainfall
        ctx_m = t_dict["X_cat"].astype(np.float32) # Crop, Season, State
        b_size = len(t_dict["X_num"])
        mask = np.tile([1.0, 1.0, 0.0, 1.0, 0.0], (b_size, 1)).astype(np.float32)
        return {
            "soil": soil_m,
            "env": env_m,
            "iot": np.zeros((b_size, 1), dtype=np.float32),
            "context": ctx_m,
            "satellite": np.zeros((b_size, 1), dtype=np.float32),
            "mask": mask
        }
        
    mod_train = prep_modalities(t_train)
    mod_val = prep_modalities(t_val)
    mod_test = prep_modalities(t_test)
    
    model = AgriFuseModel(
        soil_dim=4, env_dim=1, iot_dim=1, context_dim=3, satellite_dim=1,
        latent_dim=128, fusion_type="gated", task_type="regression", dropout_rate=0.2
    )
    trainer = AgriFuseTrainer(model, task_type="regression", lr=0.0003, max_epochs=60, patience=15)
    train_stats = trainer.fit(mod_train, y_train, mod_val, y_val)
    eval_stats = trainer.evaluate(mod_test, y_test)
    
    eval_stats["task"] = "Task_B1_Indian_Yield"
    eval_stats["training_time_sec"] = train_stats["training_time_sec"]
    np.save(f"{PREDS_DIR}/indian_yield_AGRI_FUSE_pred.npy", eval_stats.pop("y_pred"))
    np.save(f"{PREDS_DIR}/indian_yield_AGRI_FUSE_gates.npy", eval_stats.pop("gate_weights"))
    torch.save(model.state_dict(), f"{MODELS_DIR}/indian_yield_AGRI_FUSE.pth")
    joblib.dump(prep, f"{MODELS_DIR}/indian_yield_preprocessor.joblib")
    results.append(eval_stats)
    print(f"[AGRI-FUSE Gated] R2: {eval_stats['r2']:.4f} | MAE: {eval_stats['mae']:.4f} | RMSE: {eval_stats['rmse']:.4f} | Train: {train_stats['training_time_sec']:.2f}s")
    
    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{METRICS_DIR}/task_b1_indian_yield_summary.csv", index=False)
    print(f"Task B1 completed! Summary saved to {METRICS_DIR}/task_b1_indian_yield_summary.csv")
    return df_res

def run_task_b2_icrisat():
    print("\n==========================================")
    print("RUNNING TASK B2: ICRISAT DISTRICT YIELD (D4)")
    print("==========================================")
    
    df = pd.read_excel("data/raw/icrisat/icrisat_district_level_1990_2015.xls")
    train_idx = np.load("data/splits/icrisat_train.npy")
    val_idx = np.load("data/splits/icrisat_val.npy")
    test_idx = np.load("data/splits/icrisat_test.npy")
    
    target_col = 'RICE YIELD (Kg per ha)'
    # Filter rows where target is valid positive
    valid_mask = (df[target_col] > 0) & (df['RICE AREA (1000 ha)'] > 0)
    
    train_idx = [i for i in train_idx if valid_mask.iloc[i]]
    test_idx = [i for i in test_idx if valid_mask.iloc[i]]
    
    num_cols = ['RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)', 'Year']
    # Add available rainfall/climate cols if present
    rf_cols = [c for c in df.columns if 'RAIN' in c.upper() or 'PRECIP' in c.upper() or 'WATER' in c.upper()]
    num_cols.extend(rf_cols[:5])
    
    cat_cols = ['State Name', 'Dist Name']
    prep = TaskDataPreprocessor("ICRISAT_Rice", continuous_cols=num_cols, categorical_cols=cat_cols, target_col=target_col, is_classification=False)
    
    t_train = prep.fit_transform(df.iloc[train_idx])
    t_test = prep.transform(df.iloc[test_idx])
    
    X_train = np.hstack([t_train["X_num"], t_train["X_cat"]])
    X_test = np.hstack([t_test["X_num"], t_test["X_cat"]])
    y_train = t_train["y"]
    y_test = t_test["y"]
    
    results = []
    # Test major baselines
    for b_name in ["RandomForest", "ExtraTrees", "LightGBM", "CatBoost", "Ridge"]:
        baselines = get_baseline_models("regression", random_state=42)
        b_model = baselines[b_name]
        res = train_and_eval_baseline(b_model, b_name, X_train, y_train, X_test, y_test, "regression")
        res["task"] = "Task_B2_ICRISAT_Rice_Yield"
        np.save(f"{PREDS_DIR}/icrisat_{b_name}_pred.npy", res.pop("y_pred"))
        results.append(res)
        print(f"[{b_name}] R2: {res['r2']:.4f} | MAE: {res['mae']:.4f} | RMSE: {res['rmse']:.4f}")
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{METRICS_DIR}/task_b2_icrisat_summary.csv", index=False)
    print(f"Task B2 completed! Summary saved to {METRICS_DIR}/task_b2_icrisat_summary.csv")

if __name__ == "__main__":
    run_task_b1_indian_yield()
    run_task_b2_icrisat()

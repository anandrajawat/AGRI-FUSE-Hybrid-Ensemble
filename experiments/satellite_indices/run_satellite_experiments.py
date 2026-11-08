"""
Pipeline E: Satellite-Derived Vegetation Index & Multimodal Agricultural Prediction.
Evaluates AgriYield 2025 (D9) combining remote sensing NDVI, soil properties, and weather.
Evaluates baselines and AGRI-FUSE Gated Architecture.
Documents inaccessible status of D8 per scientific integrity protocol.
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
from src.preprocessing.preprocessors import extract_modalities_agriyield
from src.fusion.agri_fuse_model import AgriFuseModel
from src.training.trainer import AgriFuseTrainer

METRICS_DIR = "artifacts/metrics"
MODELS_DIR = "artifacts/models"
PREDS_DIR = "artifacts/predictions"
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)

def run_task_e1_agriyield():
    print("\n==========================================")
    print("RUNNING TASK E1: SATELLITE-INDEX AGRIYIELD 2025 (D9)")
    print("==========================================")
    
    df = pd.read_csv("data/raw/agriyield/train.csv")
    train_idx = np.load("data/splits/agriyield_train.npy")
    val_idx = np.load("data/splits/agriyield_val.npy")
    test_idx = np.load("data/splits/agriyield_test.npy")
    
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    df_test = df.iloc[test_idx]
    
    y_train = df_train['yield'].values.astype(np.float32)
    y_val = df_val['yield'].values.astype(np.float32)
    y_test = df_test['yield'].values.astype(np.float32)
    
    mod_train = extract_modalities_agriyield(df_train)
    mod_val = extract_modalities_agriyield(df_val)
    mod_test = extract_modalities_agriyield(df_test)
    
    # Tabular features: soil (3) + env (3) + sat (1) = 7 features
    X_train = np.hstack([mod_train["soil"], mod_train["env"], mod_train["satellite"]])
    X_val = np.hstack([mod_val["soil"], mod_val["env"], mod_val["satellite"]])
    X_test = np.hstack([mod_test["soil"], mod_test["env"], mod_test["satellite"]])
    
    results = []
    
    # 1. Train Baselines
    baselines = get_baseline_models("regression", random_state=42)
    for b_name, b_model in baselines.items():
        res = train_and_eval_baseline(b_model, b_name, X_train, y_train, X_test, y_test, "regression")
        res["task"] = "Task_E1_AgriYield_Satellite"
        np.save(f"{PREDS_DIR}/agriyield_{b_name}_pred.npy", res.pop("y_pred"))
        joblib.dump(b_model, f"{MODELS_DIR}/agriyield_{b_name}.joblib")
        results.append(res)
        print(f"[{b_name}] R2: {res['r2']:.4f} | MAE: {res['mae']:.4f} | RMSE: {res['rmse']:.4f} | Train: {res['training_time_sec']:.2f}s")
        
    # 2. Train AGRI-FUSE Gated Model (Soil=3, Env=3, Sat=1)
    model = AgriFuseModel(
        soil_dim=3, env_dim=3, iot_dim=1, context_dim=1, satellite_dim=1,
        latent_dim=128, fusion_type="gated", task_type="regression", dropout_rate=0.2
    )
    trainer = AgriFuseTrainer(model, task_type="regression", lr=0.0003, max_epochs=60, patience=15)
    train_stats = trainer.fit(mod_train, y_train, mod_val, y_val)
    eval_stats = trainer.evaluate(mod_test, y_test)
    
    eval_stats["task"] = "Task_E1_AgriYield_Satellite"
    eval_stats["training_time_sec"] = train_stats["training_time_sec"]
    np.save(f"{PREDS_DIR}/agriyield_AGRI_FUSE_pred.npy", eval_stats.pop("y_pred"))
    np.save(f"{PREDS_DIR}/agriyield_AGRI_FUSE_gates.npy", eval_stats.pop("gate_weights"))
    torch.save(model.state_dict(), f"{MODELS_DIR}/agriyield_AGRI_FUSE.pth")
    results.append(eval_stats)
    print(f"[AGRI-FUSE Gated] R2: {eval_stats['r2']:.4f} | MAE: {eval_stats['mae']:.4f} | RMSE: {eval_stats['rmse']:.4f} | Train: {train_stats['training_time_sec']:.2f}s")
    
    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{METRICS_DIR}/task_e1_agriyield_summary.csv", index=False)
    print(f"Task E1 completed! Summary saved to {METRICS_DIR}/task_e1_agriyield_summary.csv")
    return df_res

if __name__ == "__main__":
    run_task_e1_agriyield()

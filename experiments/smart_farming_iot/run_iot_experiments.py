"""
Pipeline D: Smart Farming IoT Experimental Suite.
Evaluates multimodal sensor fusion across Soil, Environmental, IoT, Context, and Remote Sensing (NDVI).
Tests:
- Task D1: IoT Crop Yield Regression (yield_kg_per_hectare)
- Task D2: IoT Crop Type Classification (crop_type)
- Task D3: Gated Multimodal Fusion vs Baselines (Concat, Mean, Single Modalities)
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
from src.preprocessing.preprocessors import extract_modalities_smart_farming_iot
from src.fusion.agri_fuse_model import AgriFuseModel
from src.training.trainer import AgriFuseTrainer

METRICS_DIR = "artifacts/metrics"
MODELS_DIR = "artifacts/models"
PREDS_DIR = "artifacts/predictions"
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)

def run_task_d1_iot_yield():
    print("\n==========================================")
    print("RUNNING TASK D1: SMART FARMING IOT YIELD (D6)")
    print("==========================================")
    
    df = pd.read_csv("data/raw/smart_farming_iot/Smart_Farming_Crop_Yield_2024.csv")
    train_idx = np.load("data/splits/smart_farming_iot_train.npy")
    val_idx = np.load("data/splits/smart_farming_iot_val.npy")
    test_idx = np.load("data/splits/smart_farming_iot_test.npy")
    
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    df_test = df.iloc[test_idx]
    
    y_train = df_train['yield_kg_per_hectare'].values.astype(np.float32)
    y_val = df_val['yield_kg_per_hectare'].values.astype(np.float32)
    y_test = df_test['yield_kg_per_hectare'].values.astype(np.float32)
    
    # Feature blocks
    mod_train = extract_modalities_smart_farming_iot(df_train, None)
    mod_val = extract_modalities_smart_farming_iot(df_val, None)
    mod_test = extract_modalities_smart_farming_iot(df_test, None)
    
    # Tabular concatenated features for baselines
    X_train = np.hstack([mod_train["soil"], mod_train["env"], mod_train["iot"], mod_train["context"], mod_train["satellite"]])
    X_val = np.hstack([mod_val["soil"], mod_val["env"], mod_val["iot"], mod_val["context"], mod_val["satellite"]])
    X_test = np.hstack([mod_test["soil"], mod_test["env"], mod_test["iot"], mod_test["context"], mod_test["satellite"]])
    
    results = []
    
    # 1. Train Baselines
    baselines = get_baseline_models("regression", random_state=42)
    for b_name, b_model in baselines.items():
        res = train_and_eval_baseline(b_model, b_name, X_train, y_train, X_test, y_test, "regression")
        res["task"] = "Task_D1_IoT_Yield"
        np.save(f"{PREDS_DIR}/iot_yield_{b_name}_pred.npy", res.pop("y_pred"))
        joblib.dump(b_model, f"{MODELS_DIR}/iot_yield_{b_name}.joblib")
        results.append(res)
        print(f"[{b_name}] R2: {res['r2']:.4f} | MAE: {res['mae']:.4f} | RMSE: {res['rmse']:.4f} | Train: {res['training_time_sec']:.2f}s")
        
    # 2. Train AGRI-FUSE Gated Multimodal Model
    # Dimensions: Soil (2: moisture, pH), Env (4: temp, rainfall, hum, sun), IoT (4: pest, days, lat, lon), Context (4: reg, crop, irrig, fert), Sat (1: NDVI)
    model = AgriFuseModel(
        soil_dim=2, env_dim=4, iot_dim=4, context_dim=4, satellite_dim=1,
        latent_dim=128, fusion_type="gated", task_type="regression", dropout_rate=0.2
    )
    trainer = AgriFuseTrainer(model, task_type="regression", lr=0.0003, max_epochs=60, patience=15)
    train_stats = trainer.fit(mod_train, y_train, mod_val, y_val)
    eval_stats = trainer.evaluate(mod_test, y_test)
    
    eval_stats["task"] = "Task_D1_IoT_Yield"
    eval_stats["training_time_sec"] = train_stats["training_time_sec"]
    np.save(f"{PREDS_DIR}/iot_yield_AGRI_FUSE_pred.npy", eval_stats.pop("y_pred"))
    np.save(f"{PREDS_DIR}/iot_yield_AGRI_FUSE_gates.npy", eval_stats.pop("gate_weights"))
    torch.save(model.state_dict(), f"{MODELS_DIR}/iot_yield_AGRI_FUSE.pth")
    results.append(eval_stats)
    print(f"[AGRI-FUSE Gated] R2: {eval_stats['r2']:.4f} | MAE: {eval_stats['mae']:.4f} | RMSE: {eval_stats['rmse']:.4f} | Train: {train_stats['training_time_sec']:.2f}s")
    
    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{METRICS_DIR}/task_d1_iot_yield_summary.csv", index=False)
    print(f"Task D1 completed! Summary saved to {METRICS_DIR}/task_d1_iot_yield_summary.csv")
    return df_res

def run_task_d2_iot_crop_type():
    print("\n==========================================")
    print("RUNNING TASK D2: IOT CROP-TYPE CLASSIFICATION (D6)")
    print("==========================================")
    
    df = pd.read_csv("data/raw/smart_farming_iot/Smart_Farming_Crop_Yield_2024.csv")
    train_idx = np.load("data/splits/smart_farming_iot_train.npy")
    test_idx = np.load("data/splits/smart_farming_iot_test.npy")
    
    # Predict crop_type from sensor readings (soil moisture, pH, temp, rainfall, humidity, sunlight, NDVI)
    feature_cols = ['soil_moisture_%', 'soil_pH', 'temperature_C', 'rainfall_mm', 'humidity_%', 'sunlight_hours', 'NDVI_index']
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    
    scaler = StandardScaler()
    le = LabelEncoder()
    
    X_train = scaler.fit_transform(df.iloc[train_idx][feature_cols])
    y_train = le.fit_transform(df.iloc[train_idx]['crop_type'])
    X_test = scaler.transform(df.iloc[test_idx][feature_cols])
    y_test = le.transform(df.iloc[test_idx]['crop_type'])
    
    results = []
    baselines = get_baseline_models("classification", num_classes=len(le.classes_), random_state=42)
    for b_name, b_model in baselines.items():
        res = train_and_eval_baseline(b_model, b_name, X_train, y_train, X_test, y_test, "classification")
        res["task"] = "Task_D2_IoT_Crop_Type"
        np.save(f"{PREDS_DIR}/iot_croptype_{b_name}_pred.npy", res.pop("y_pred"))
        results.append(res)
        print(f"[{b_name}] Acc: {res['accuracy']:.4f} | Macro F1: {res['macro_f1']:.4f}")
        
    df_res = pd.DataFrame([{k: v for k, v in r.items() if k not in ['confusion_matrix', 'per_class_f1']} for r in results])
    df_res.to_csv(f"{METRICS_DIR}/task_d2_iot_croptype_summary.csv", index=False)
    print(f"Task D2 completed! Summary saved to {METRICS_DIR}/task_d2_iot_croptype_summary.csv")

if __name__ == "__main__":
    run_task_d1_iot_yield()
    run_task_d2_iot_crop_type()

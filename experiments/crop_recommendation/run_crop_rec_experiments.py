"""
Pipeline A: Crop Recommendation Experimental Suite.
Evaluates Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, HistGradientBoosting,
Logistic Regression, MLP, and AGRI-FUSE across multiple random seeds (42, 123, 2026).
Also evaluates SF24 synthetic benchmark and Soil Nutrient dataset.
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
from src.preprocessing.preprocessors import TaskDataPreprocessor, extract_modalities_crop_recommendation
from src.fusion.agri_fuse_model import AgriFuseModel
from src.training.trainer import AgriFuseTrainer

METRICS_DIR = "artifacts/metrics"
MODELS_DIR = "artifacts/models"
PREDS_DIR = "artifacts/predictions"
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)

def run_task_a1_crop_rec():
    print("\n==========================================")
    print("RUNNING TASK A1: CROP RECOMMENDATION (D1)")
    print("==========================================")
    
    df = pd.read_csv("data/raw/crop_recommendation/Crop_recommendation.csv")
    seeds = [42, 123, 2026]
    all_results = []
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        train_idx = np.load(f"data/splits/crop_rec_seed_{seed}_train.npy")
        val_idx = np.load(f"data/splits/crop_rec_seed_{seed}_val.npy")
        test_idx = np.load(f"data/splits/crop_rec_seed_{seed}_test.npy")
        
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        df_test = df.iloc[test_idx]
        
        num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        prep = TaskDataPreprocessor("Crop_Rec", continuous_cols=num_cols, target_col="label", is_classification=True)
        t_train = prep.fit_transform(df_train)
        t_val = prep.transform(df_val)
        t_test = prep.transform(df_test)
        
        X_train, y_train = t_train["X_num"], t_train["y"]
        X_val, y_val = t_val["X_num"], t_val["y"]
        X_test, y_test = t_test["X_num"], t_test["y"]
        
        # 1. Train and evaluate baselines
        baselines = get_baseline_models("classification", num_classes=22, random_state=seed)
        for b_name, b_model in baselines.items():
            res = train_and_eval_baseline(b_model, b_name, X_train, y_train, X_test, y_test, "classification")
            res["seed"] = seed
            res["task"] = "Task_A1_Crop_Rec"
            # save predictions
            np.save(f"{PREDS_DIR}/crop_rec_{b_name}_seed_{seed}_pred.npy", res.pop("y_pred"))
            if seed == 42:
                joblib.dump(b_model, f"{MODELS_DIR}/crop_rec_{b_name}.joblib")
            all_results.append(res)
            print(f"[{b_name}] Acc: {res['accuracy']:.4f} | Macro F1: {res['macro_f1']:.4f} | Train: {res['training_time_sec']:.2f}s")
            
        # 2. Train AGRI-FUSE Model
        mod_train = extract_modalities_crop_recommendation(X_train)
        mod_val = extract_modalities_crop_recommendation(X_val)
        mod_test = extract_modalities_crop_recommendation(X_test)
        
        torch.manual_seed(seed)
        model = AgriFuseModel(
            soil_dim=4, env_dim=3, iot_dim=1, context_dim=1, satellite_dim=1,
            latent_dim=128, fusion_type="gated", task_type="classification", num_classes=22
        )
        trainer = AgriFuseTrainer(model, task_type="classification", lr=0.0003, max_epochs=60, patience=15)
        train_stats = trainer.fit(mod_train, y_train, mod_val, y_val)
        eval_stats = trainer.evaluate(mod_test, y_test)
        
        eval_stats["seed"] = seed
        eval_stats["task"] = "Task_A1_Crop_Rec"
        eval_stats["training_time_sec"] = train_stats["training_time_sec"]
        np.save(f"{PREDS_DIR}/crop_rec_AGRI_FUSE_seed_{seed}_pred.npy", eval_stats.pop("y_pred"))
        np.save(f"{PREDS_DIR}/crop_rec_AGRI_FUSE_seed_{seed}_gates.npy", eval_stats.pop("gate_weights"))
        if seed == 42:
            torch.save(model.state_dict(), f"{MODELS_DIR}/crop_rec_AGRI_FUSE.pth")
            joblib.dump(prep, f"{MODELS_DIR}/crop_rec_preprocessor.joblib")
        all_results.append(eval_stats)
        print(f"[AGRI-FUSE Gated] Acc: {eval_stats['accuracy']:.4f} | Macro F1: {eval_stats['macro_f1']:.4f} | Train: {train_stats['training_time_sec']:.2f}s")

    # Aggregate results across seeds
    df_res = pd.DataFrame([{k: v for k, v in r.items() if k not in ['confusion_matrix', 'per_class_f1']} for r in all_results])
    summary = df_res.groupby("model_name").agg({
        "accuracy": ["mean", "std"],
        "macro_f1": ["mean", "std"],
        "weighted_f1": ["mean", "std"],
        "training_time_sec": "mean",
        "latency_ms_per_sample": "mean"
    }).reset_index()
    
    summary.to_csv(f"{METRICS_DIR}/task_a1_crop_rec_summary.csv", index=False)
    with open(f"{METRICS_DIR}/task_a1_crop_rec_all_runs.json", "w") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\nTask A1 completed! Summary saved to {METRICS_DIR}/task_a1_crop_rec_summary.csv")
    return summary

def run_task_a3_soil_nutrients():
    print("\n==========================================")
    print("RUNNING TASK A3: SOIL NUTRIENT SUITABILITY (D7)")
    print("==========================================")
    
    df = pd.read_csv("data/raw/soil_nutrients/dataset.csv")
    train_idx = np.load("data/splits/soil_nutrients_train.npy")
    val_idx = np.load("data/splits/soil_nutrients_val.npy")
    test_idx = np.load("data/splits/soil_nutrients_test.npy")
    
    num_cols = ['N', 'P', 'K', 'ph', 'EC', 'S', 'Cu', 'Fe', 'Mn', 'Zn', 'B']
    prep = TaskDataPreprocessor("Soil_Nutrients", continuous_cols=num_cols, target_col="label", is_classification=True)
    t_train = prep.fit_transform(df.iloc[train_idx])
    t_test = prep.transform(df.iloc[test_idx])
    
    X_train, y_train = t_train["X_num"], t_train["y"]
    X_test, y_test = t_test["X_num"], t_test["y"]
    num_classes = len(prep.label_encoder.classes_)
    
    results = []
    baselines = get_baseline_models("classification", num_classes=num_classes, random_state=42)
    for b_name, b_model in baselines.items():
        res = train_and_eval_baseline(b_model, b_name, X_train, y_train, X_test, y_test, "classification")
        res["task"] = "Task_A3_Soil_Suitability"
        np.save(f"{PREDS_DIR}/soil_nutrients_{b_name}_pred.npy", res.pop("y_pred"))
        results.append(res)
        print(f"[{b_name}] Acc: {res['accuracy']:.4f} | Macro F1: {res['macro_f1']:.4f}")
        
    df_res = pd.DataFrame([{k: v for k, v in r.items() if k not in ['confusion_matrix', 'per_class_f1']} for r in results])
    df_res.to_csv(f"{METRICS_DIR}/task_a3_soil_nutrients_summary.csv", index=False)
    print(f"Task A3 completed! Summary saved to {METRICS_DIR}/task_a3_soil_nutrients_summary.csv")

if __name__ == "__main__":
    run_task_a1_crop_rec()
    run_task_a3_soil_nutrients()

"""
Pipeline C: Field-Level Wheat Yield Prediction Experimental Suite.
Evaluates field-disjoint models on CIMMYT Eastern India Wheat Field Dataset (D3).
Tests baselines and AGRI-FUSE (Soil/Management + Environmental branches).
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

def run_task_c1_wheat_yield():
    print("\n==========================================")
    print("RUNNING TASK C1: FIELD-LEVEL WHEAT YIELD (D3)")
    print("==========================================")
    
    df = pd.read_excel("data/raw/eastern_wheat/Wheat_Yld_Eastern_IND_Data.xlsx")
    train_idx = np.load("data/splits/eastern_wheat_train.npy")
    val_idx = np.load("data/splits/eastern_wheat_val.npy")
    test_idx = np.load("data/splits/eastern_wheat_test.npy")
    
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    df_test = df.iloc[test_idx]
    
    # Agronomic continuous management & environmental variables
    num_cols = [
        'SeedRate', 'SowingDay', 'BasalDAP', 'BasalNPK', 'Split1Urea', 'Split2Urea',
        'IrrigationNumber', 'Prcp_Monsoon', 'Prcp_Oct', 'Nov_Tmax', 'EDD_Nov',
        'Dec_Tmax', 'EDD_Dec', 'Jan_Srad', 'Feb_Srad', 'Mar_Tmax', 'EDD_Mar', 'Apr_Tmax', 'EDD_Apr'
    ]
    cat_cols = ['District', 'State', 'LandType', 'SoilType', 'VarietyClass']
    
    prep = TaskDataPreprocessor("Eastern_Wheat", continuous_cols=num_cols, categorical_cols=cat_cols, target_col="GrainYield", is_classification=False)
    t_train = prep.fit_transform(df_train)
    t_val = prep.transform(df_val)
    t_test = prep.transform(df_test)
    
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
        res["task"] = "Task_C1_Eastern_Wheat_Yield"
        np.save(f"{PREDS_DIR}/wheat_yield_{b_name}_pred.npy", res.pop("y_pred"))
        joblib.dump(b_model, f"{MODELS_DIR}/wheat_yield_{b_name}.joblib")
        results.append(res)
        print(f"[{b_name}] R2: {res['r2']:.4f} | MAE: {res['mae']:.4f} | RMSE: {res['rmse']:.4f} | Train: {res['training_time_sec']:.2f}s")
        
    # 2. Train AGRI-FUSE
    # Modalities: Soil/Management (cols 0-6: 7), Environmental (cols 7-18: 12), Context (cat_cols: 5)
    def prep_modalities(t_dict):
        soil_m = t_dict["X_num"][:, :7] # SeedRate, SowingDay, DAP, NPK, Urea, Irrig
        env_m = t_dict["X_num"][:, 7:] # Weather & EDD
        ctx_m = t_dict["X_cat"].astype(np.float32) # District, State, LandType, SoilType, Variety
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
        soil_dim=7, env_dim=12, iot_dim=1, context_dim=5, satellite_dim=1,
        latent_dim=128, fusion_type="gated", task_type="regression", dropout_rate=0.2
    )
    trainer = AgriFuseTrainer(model, task_type="regression", lr=0.0003, max_epochs=60, patience=15)
    train_stats = trainer.fit(mod_train, y_train, mod_val, y_val)
    eval_stats = trainer.evaluate(mod_test, y_test)
    
    eval_stats["task"] = "Task_C1_Eastern_Wheat_Yield"
    eval_stats["training_time_sec"] = train_stats["training_time_sec"]
    np.save(f"{PREDS_DIR}/wheat_yield_AGRI_FUSE_pred.npy", eval_stats.pop("y_pred"))
    np.save(f"{PREDS_DIR}/wheat_yield_AGRI_FUSE_gates.npy", eval_stats.pop("gate_weights"))
    torch.save(model.state_dict(), f"{MODELS_DIR}/wheat_yield_AGRI_FUSE.pth")
    joblib.dump(prep, f"{MODELS_DIR}/wheat_yield_preprocessor.joblib")
    results.append(eval_stats)
    print(f"[AGRI-FUSE Gated] R2: {eval_stats['r2']:.4f} | MAE: {eval_stats['mae']:.4f} | RMSE: {eval_stats['rmse']:.4f} | Train: {train_stats['training_time_sec']:.2f}s")
    
    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{METRICS_DIR}/task_c1_wheat_yield_summary.csv", index=False)
    print(f"Task C1 completed! Summary saved to {METRICS_DIR}/task_c1_wheat_yield_summary.csv")
    return df_res

if __name__ == "__main__":
    run_task_c1_wheat_yield()

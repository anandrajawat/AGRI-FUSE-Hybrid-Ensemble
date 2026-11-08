"""
AGRI-FUSE Hybrid Ensemble (AF-HE) Experimental Suite.
Executes end-to-end benchmark of the AF-HE Architecture:
1. Base Learners: Random Forest, XGBoost, CatBoost, AGRI-FUSE Gated Multimodal Unit
2. 5-Fold Leakage-Safe Out-Of-Fold (OOF) Cross-Validation
3. Stacking Meta-Learner (Non-Negative Least Squares / RidgeCV / Logistic Regression)
4. Validation-Calibrated Weighted Averaging (SLSQP Optimization)
5. Adaptive Reliability-Weighted Blending
6. Full 14-Configuration Ablation Suite
7. Uncertainty Quantification: Conformal Prediction (90% coverage) & Brier Calibration
8. Empirical Improvement Table: Hybrid Score - Best Individual Score
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch

from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.catboost_model import CatBoostModel
from src.models.agri_fuse import AgriFuseEstimator
from src.ensemble.stacking import AFHEStackedEnsemble
from src.ensemble.averaging import WeightedAveragingEnsemble
from src.ensemble.adaptive_blending import AdaptiveReliabilityBlender
from src.ensemble.ablation_suite import run_14_ablation_benchmarks
from src.calibration.calibrator import compute_brier_score, get_calibration_curve_data, ConformalPredictor
from src.preprocessing.preprocessors import (
    TaskDataPreprocessor,
    extract_modalities_crop_recommendation,
    extract_modalities_smart_farming_iot,
    extract_modalities_agriyield
)
from src.evaluation.metrics import compute_classification_metrics, compute_regression_metrics

METRICS_DIR = "artifacts/metrics"
MODELS_DIR = "artifacts/models"
PREDS_DIR = "artifacts/predictions"
OOF_DIR = "artifacts/oof_predictions"

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)
os.makedirs(OOF_DIR, exist_ok=True)

# =========================================================================
# EXPERIMENT 1: TASK A - CROP RECOMMENDATION (DATASET 1)
# =========================================================================
def run_af_he_crop_recommendation():
    print("\n" + "="*70)
    print("RUNNING AF-HE: CROP RECOMMENDATION (DATASET 1)")
    print("="*70)
    
    df = pd.read_csv("data/raw/crop_recommendation/Crop_recommendation.csv")
    train_idx = np.load("data/splits/crop_rec_seed_42_train.npy")
    val_idx = np.load("data/splits/crop_rec_seed_42_val.npy")
    test_idx = np.load("data/splits/crop_rec_seed_42_test.npy")
    
    num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    prep = TaskDataPreprocessor("Crop_Rec", continuous_cols=num_cols, target_col="label", is_classification=True)
    t_train = prep.fit_transform(df.iloc[train_idx])
    t_val = prep.transform(df.iloc[val_idx])
    t_test = prep.transform(df.iloc[test_idx])
    
    X_train, y_train = t_train["X_num"], t_train["y"]
    X_val, y_val = t_val["X_num"], t_val["y"]
    X_test, y_test = t_test["X_num"], t_test["y"]
    
    mod_train = extract_modalities_crop_recommendation(X_train)
    mod_val = extract_modalities_crop_recommendation(X_val)
    mod_test = extract_modalities_crop_recommendation(X_test)
    
    def train_mod_fn(indices):
        return {k: v[indices] for k, v in mod_train.items()}
        
    base_factories = {
        "RF": lambda: RandomForestModel(task_type="classification", n_estimators=150, max_depth=16, random_state=42),
        "XGBoost": lambda: XGBoostModel(task_type="classification", n_estimators=150, max_depth=6, random_state=42),
        "CatBoost": lambda: CatBoostModel(task_type="classification", iterations=150, depth=6, random_seed=42),
        "AGRI_FUSE": lambda: AgriFuseEstimator(
            soil_dim=4, env_dim=3, iot_dim=1, context_dim=1, satellite_dim=1,
            latent_dim=128, fusion_type="gated", task_type="classification", num_classes=22,
            max_epochs=25, patience=8, random_state=42
        )
    }
    
    # 1. Fit AF-HE Stacking Ensemble with 5-fold OOF Cross-Validation
    t0 = time.time()
    stacker = AFHEStackedEnsemble(task_type="classification", n_splits=5, random_state=42)
    stacker.fit_oof_and_meta(
        base_factories=base_factories,
        X_train=X_train,
        y_train=y_train,
        train_modalities_fn=train_mod_fn,
        task_id="crop_rec"
    )
    fit_time = time.time() - t0
    
    # 2. Base Predictions on Validation and Test Splits
    base_val = stacker.get_base_predictions(X_val, mod_val)
    base_test = stacker.get_base_predictions(X_test, mod_test)
    
    # Stacked Predictions
    y_pred_stack = stacker.predict(X_test, mod_test)
    y_proba_stack = stacker.predict_proba(X_test, mod_test)
    np.save(f"{PREDS_DIR}/af_he_crop_rec_stacked_pred.npy", y_pred_stack)
    np.save(f"{PREDS_DIR}/af_he_crop_rec_stacked_proba.npy", y_proba_stack)
    
    # Save Stacked Model
    stacker.save(f"{MODELS_DIR}/af_he_crop_rec")
    
    # 3. Full 14-Configuration Ablation Benchmarks
    print("\nRunning 14-Configuration Ablation Benchmarks...")
    df_ablations = run_14_ablation_benchmarks(
        base_predictions_val=base_val,
        base_predictions_test=base_test,
        y_val=y_val,
        y_test=y_test,
        task_type="classification",
        stacked_test_pred=y_pred_stack,
        task_id="Task_A1_Crop_Rec"
    )
    df_ablations.to_csv(f"{METRICS_DIR}/af_he_crop_rec_ablations.csv", index=False)
    
    # 4. Uncertainty & Calibration Metrics
    brier_rf = compute_brier_score(y_test, base_test["RF"])
    brier_xgb = compute_brier_score(y_test, base_test["XGBoost"])
    brier_cat = compute_brier_score(y_test, base_test["CatBoost"])
    brier_af = compute_brier_score(y_test, base_test["AGRI_FUSE"])
    brier_stack = compute_brier_score(y_test, y_proba_stack)
    
    calib_curve = get_calibration_curve_data(y_test, y_proba_stack, n_bins=10)
    with open(f"{METRICS_DIR}/af_he_crop_rec_calibration.json", "w") as f:
        json.dump({
            "brier_rf": brier_rf,
            "brier_xgb": brier_xgb,
            "brier_cat": brier_cat,
            "brier_agri_fuse": brier_af,
            "brier_stack": brier_stack,
            "calibration_curve": calib_curve
        }, f, indent=2)
        
    # 5. Improvement Calculation
    best_single_acc = max(
        df_ablations[df_ablations["ensemble_strategy"] == "single"]["accuracy"].max(),
        0.0
    )
    best_single_f1 = max(
        df_ablations[df_ablations["ensemble_strategy"] == "single"]["macro_f1"].max(),
        0.0
    )
    stack_row = df_ablations[df_ablations["ablation_id"].str.contains("Stacking")].iloc[0]
    stack_acc = stack_row["accuracy"]
    stack_f1 = stack_row["macro_f1"]
    
    summary_df = pd.DataFrame([{
        "task": "Crop Recommendation (D1)",
        "best_single_model": df_ablations[df_ablations["accuracy"] == best_single_acc]["models_included"].iloc[0],
        "best_single_accuracy": best_single_acc,
        "best_single_macro_f1": best_single_f1,
        "af_he_stacking_accuracy": stack_acc,
        "af_he_stacking_macro_f1": stack_f1,
        "accuracy_improvement": stack_acc - best_single_acc,
        "f1_improvement": stack_f1 - best_single_f1,
        "brier_score": brier_stack,
        "total_training_time_sec": fit_time
    }])
    summary_df.to_csv(f"{METRICS_DIR}/af_he_crop_rec_summary.csv", index=False)
    print(f"Task A completed in {fit_time:.2f}s | AF-HE Stack Acc: {stack_acc:.4f} (Best Single: {best_single_acc:.4f}, Delta: {stack_acc - best_single_acc:+.4f})")
    return summary_df, df_ablations


# =========================================================================
# EXPERIMENT 2: TASK B - HISTORICAL INDIAN CROP YIELD (DATASET 2)
# =========================================================================
def run_af_he_indian_yield():
    print("\n" + "="*70)
    print("RUNNING AF-HE: INDIAN CROP YIELD (DATASET 2)")
    print("="*70)
    
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
    
    X_train = np.hstack([t_train["X_num"], t_train["X_cat"]])
    X_val = np.hstack([t_val["X_num"], t_val["X_cat"]])
    X_test = np.hstack([t_test["X_num"], t_test["X_cat"]])
    
    y_train = t_train["y"]
    y_val = t_val["y"]
    y_test = t_test["y"]
    
    def prep_modalities(t_dict):
        soil_m = t_dict["X_num"][:, [0, 1, 3, 4]]
        env_m = t_dict["X_num"][:, [2]]
        ctx_m = t_dict["X_cat"].astype(np.float32)
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
    
    def train_mod_fn(indices):
        return {k: v[indices] for k, v in mod_train.items()}
        
    base_factories = {
        "RF": lambda: RandomForestModel(task_type="regression", n_estimators=150, max_depth=16, random_state=42),
        "XGBoost": lambda: XGBoostModel(task_type="regression", n_estimators=150, max_depth=6, random_state=42),
        "CatBoost": lambda: CatBoostModel(task_type="regression", iterations=150, depth=6, random_seed=42),
        "AGRI_FUSE": lambda: AgriFuseEstimator(
            soil_dim=4, env_dim=1, iot_dim=1, context_dim=3, satellite_dim=1,
            latent_dim=128, fusion_type="gated", task_type="regression",
            max_epochs=20, patience=6, random_state=42
        )
    }
    
    t0 = time.time()
    stacker = AFHEStackedEnsemble(task_type="regression", n_splits=5, random_state=42)
    stacker.fit_oof_and_meta(
        base_factories=base_factories,
        X_train=X_train,
        y_train=y_train,
        train_modalities_fn=train_mod_fn,
        task_id="indian_yield"
    )
    fit_time = time.time() - t0
    
    base_val = stacker.get_base_predictions(X_val, mod_val)
    base_test = stacker.get_base_predictions(X_test, mod_test)
    
    y_pred_stack = stacker.predict(X_test, mod_test)
    np.save(f"{PREDS_DIR}/af_he_indian_yield_stacked_pred.npy", y_pred_stack)
    stacker.save(f"{MODELS_DIR}/af_he_indian_yield")
    
    # 3. 14 Ablations
    print("\nRunning 14-Configuration Ablation Benchmarks...")
    df_ablations = run_14_ablation_benchmarks(
        base_predictions_val=base_val,
        base_predictions_test=base_test,
        y_val=y_val,
        y_test=y_test,
        task_type="regression",
        stacked_test_pred=y_pred_stack,
        task_id="Task_B1_Indian_Yield"
    )
    df_ablations.to_csv(f"{METRICS_DIR}/af_he_indian_yield_ablations.csv", index=False)
    
    # 4. Conformal Prediction Intervals (90% coverage)
    y_pred_val_stack = stacker.predict(X_val, mod_val)
    conformal = ConformalPredictor(alpha=0.10)
    conformal.fit_calibration(y_val, y_pred_val_stack)
    intervals = conformal.predict_intervals(y_pred_stack)
    actual_coverage = float(np.mean((y_test >= intervals["y_lower"]) & (y_test <= intervals["y_upper"])))
    
    # Non-coconut subgroup analysis
    non_coco_mask = (df_test['Crop'].astype(str).str.strip().str.lower() != 'coconut').values
    non_coco_r2 = compute_regression_metrics(y_test[non_coco_mask], y_pred_stack[non_coco_mask])
    
    with open(f"{METRICS_DIR}/af_he_indian_yield_uncertainty.json", "w") as f:
        json.dump({
            "conformal_q_val": intervals["q_val"],
            "target_coverage": 0.90,
            "empirical_coverage": actual_coverage,
            "mean_interval_width": intervals["interval_width"],
            "non_coconut_r2": non_coco_r2["r2"],
            "non_coconut_mae": non_coco_r2["mae"],
            "non_coconut_rmse": non_coco_r2["rmse"]
        }, f, indent=2)
        
    best_single_r2 = df_ablations[df_ablations["ensemble_strategy"] == "single"]["r2"].max()
    best_single_mae = df_ablations[df_ablations["ensemble_strategy"] == "single"]["mae"].min()
    stack_row = df_ablations[df_ablations["ablation_id"].str.contains("Stacking")].iloc[0]
    stack_r2 = stack_row["r2"]
    stack_mae = stack_row["mae"]
    
    summary_df = pd.DataFrame([{
        "task": "Indian Crop Yield (D2)",
        "best_single_model": df_ablations[df_ablations["r2"] == best_single_r2]["models_included"].iloc[0],
        "best_single_r2": best_single_r2,
        "best_single_mae": best_single_mae,
        "af_he_stacking_r2": stack_r2,
        "af_he_stacking_mae": stack_mae,
        "r2_improvement": stack_r2 - best_single_r2,
        "mae_reduction": best_single_mae - stack_mae,
        "conformal_coverage": actual_coverage,
        "non_coconut_mae": non_coco_r2["mae"],
        "total_training_time_sec": fit_time
    }])
    summary_df.to_csv(f"{METRICS_DIR}/af_he_indian_yield_summary.csv", index=False)
    print(f"Task B completed in {fit_time:.2f}s | AF-HE Stack R2: {stack_r2:.4f} | Non-coconut MAE: {non_coco_r2['mae']:.2f}")
    return summary_df, df_ablations


# =========================================================================
# EXPERIMENT 3: TASK C - FIELD-LEVEL WHEAT YIELD (DATASET 3)
# =========================================================================
def run_af_he_wheat_yield():
    print("\n" + "="*70)
    print("RUNNING AF-HE: FIELD-LEVEL WHEAT YIELD (DATASET 3)")
    print("="*70)
    
    df = pd.read_excel("data/raw/eastern_wheat/Wheat_Yld_Eastern_IND_Data.xlsx")
    train_idx = np.load("data/splits/eastern_wheat_train.npy")
    val_idx = np.load("data/splits/eastern_wheat_val.npy")
    test_idx = np.load("data/splits/eastern_wheat_test.npy")
    
    num_cols = [
        'SeedRate', 'SowingDay', 'BasalDAP', 'BasalNPK', 'Split1Urea', 'Split2Urea',
        'IrrigationNumber', 'Prcp_Monsoon', 'Prcp_Oct', 'Nov_Tmax', 'EDD_Nov',
        'Dec_Tmax', 'EDD_Dec', 'Jan_Srad', 'Feb_Srad', 'Mar_Tmax', 'EDD_Mar', 'Apr_Tmax', 'EDD_Apr'
    ]
    cat_cols = ['District', 'State', 'LandType', 'SoilType', 'VarietyClass']
    
    prep = TaskDataPreprocessor("Eastern_Wheat", continuous_cols=num_cols, categorical_cols=cat_cols, target_col="GrainYield", is_classification=False)
    t_train = prep.fit_transform(df.iloc[train_idx])
    t_val = prep.transform(df.iloc[val_idx])
    t_test = prep.transform(df.iloc[test_idx])
    
    X_train = np.hstack([t_train["X_num"], t_train["X_cat"]])
    X_val = np.hstack([t_val["X_num"], t_val["X_cat"]])
    X_test = np.hstack([t_test["X_num"], t_test["X_cat"]])
    
    y_train = t_train["y"]
    y_val = t_val["y"]
    y_test = t_test["y"]
    
    def prep_modalities(t_dict):
        soil_m = t_dict["X_num"][:, :7]
        env_m = t_dict["X_num"][:, 7:]
        ctx_m = t_dict["X_cat"].astype(np.float32)
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
    
    def train_mod_fn(indices):
        return {k: v[indices] for k, v in mod_train.items()}
        
    base_factories = {
        "RF": lambda: RandomForestModel(task_type="regression", n_estimators=150, max_depth=16, random_state=42),
        "XGBoost": lambda: XGBoostModel(task_type="regression", n_estimators=150, max_depth=6, random_state=42),
        "CatBoost": lambda: CatBoostModel(task_type="regression", iterations=150, depth=6, random_seed=42),
        "AGRI_FUSE": lambda: AgriFuseEstimator(
            soil_dim=7, env_dim=12, iot_dim=1, context_dim=5, satellite_dim=1,
            latent_dim=128, fusion_type="gated", task_type="regression",
            max_epochs=20, patience=6, random_state=42
        )
    }
    
    t0 = time.time()
    stacker = AFHEStackedEnsemble(task_type="regression", n_splits=5, random_state=42)
    stacker.fit_oof_and_meta(
        base_factories=base_factories,
        X_train=X_train,
        y_train=y_train,
        train_modalities_fn=train_mod_fn,
        task_id="eastern_wheat"
    )
    fit_time = time.time() - t0
    
    base_val = stacker.get_base_predictions(X_val, mod_val)
    base_test = stacker.get_base_predictions(X_test, mod_test)
    
    y_pred_stack = stacker.predict(X_test, mod_test)
    np.save(f"{PREDS_DIR}/af_he_wheat_stacked_pred.npy", y_pred_stack)
    stacker.save(f"{MODELS_DIR}/af_he_eastern_wheat")
    
    df_ablations = run_14_ablation_benchmarks(
        base_predictions_val=base_val,
        base_predictions_test=base_test,
        y_val=y_val,
        y_test=y_test,
        task_type="regression",
        stacked_test_pred=y_pred_stack,
        task_id="Task_C1_Eastern_Wheat"
    )
    df_ablations.to_csv(f"{METRICS_DIR}/af_he_wheat_ablations.csv", index=False)
    
    # Conformal
    y_pred_val_stack = stacker.predict(X_val, mod_val)
    conformal = ConformalPredictor(alpha=0.10)
    conformal.fit_calibration(y_val, y_pred_val_stack)
    intervals = conformal.predict_intervals(y_pred_stack)
    actual_coverage = float(np.mean((y_test >= intervals["y_lower"]) & (y_test <= intervals["y_upper"])))
    
    best_single_r2 = df_ablations[df_ablations["ensemble_strategy"] == "single"]["r2"].max()
    best_single_mae = df_ablations[df_ablations["ensemble_strategy"] == "single"]["mae"].min()
    stack_row = df_ablations[df_ablations["ablation_id"].str.contains("Stacking")].iloc[0]
    stack_r2 = stack_row["r2"]
    stack_mae = stack_row["mae"]
    
    summary_df = pd.DataFrame([{
        "task": "Field-Level Wheat (D3)",
        "best_single_model": df_ablations[df_ablations["r2"] == best_single_r2]["models_included"].iloc[0],
        "best_single_r2": best_single_r2,
        "best_single_mae": best_single_mae,
        "af_he_stacking_r2": stack_r2,
        "af_he_stacking_mae": stack_mae,
        "r2_improvement": stack_r2 - best_single_r2,
        "mae_reduction": best_single_mae - stack_mae,
        "conformal_coverage": actual_coverage,
        "total_training_time_sec": fit_time
    }])
    summary_df.to_csv(f"{METRICS_DIR}/af_he_wheat_summary.csv", index=False)
    print(f"Task C completed in {fit_time:.2f}s | AF-HE Stack R2: {stack_r2:.4f} (Best Single: {best_single_r2:.4f})")
    return summary_df, df_ablations


# =========================================================================
# EXPERIMENT 4: TASK D - SMART FARMING IOT (DATASET 6)
# =========================================================================
def run_af_he_iot_yield():
    print("\n" + "="*70)
    print("RUNNING AF-HE: SMART FARMING IOT YIELD (DATASET 6)")
    print("="*70)
    
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
    
    mod_train = extract_modalities_smart_farming_iot(df_train, None)
    mod_val = extract_modalities_smart_farming_iot(df_val, None)
    mod_test = extract_modalities_smart_farming_iot(df_test, None)
    
    X_train = np.hstack([mod_train["soil"], mod_train["env"], mod_train["iot"], mod_train["context"], mod_train["satellite"]])
    X_val = np.hstack([mod_val["soil"], mod_val["env"], mod_val["iot"], mod_val["context"], mod_val["satellite"]])
    X_test = np.hstack([mod_test["soil"], mod_test["env"], mod_test["iot"], mod_test["context"], mod_test["satellite"]])
    
    def train_mod_fn(indices):
        return {k: v[indices] for k, v in mod_train.items()}
        
    base_factories = {
        "RF": lambda: RandomForestModel(task_type="regression", n_estimators=150, max_depth=16, random_state=42),
        "XGBoost": lambda: XGBoostModel(task_type="regression", n_estimators=150, max_depth=6, random_state=42),
        "CatBoost": lambda: CatBoostModel(task_type="regression", iterations=150, depth=6, random_seed=42),
        "AGRI_FUSE": lambda: AgriFuseEstimator(
            soil_dim=2, env_dim=4, iot_dim=4, context_dim=4, satellite_dim=1,
            latent_dim=128, fusion_type="gated", task_type="regression",
            max_epochs=20, patience=6, random_state=42
        )
    }
    
    t0 = time.time()
    stacker = AFHEStackedEnsemble(task_type="regression", n_splits=5, random_state=42)
    stacker.fit_oof_and_meta(
        base_factories=base_factories,
        X_train=X_train,
        y_train=y_train,
        train_modalities_fn=train_mod_fn,
        task_id="smart_farming_iot"
    )
    fit_time = time.time() - t0
    
    base_val = stacker.get_base_predictions(X_val, mod_val)
    base_test = stacker.get_base_predictions(X_test, mod_test)
    
    y_pred_stack = stacker.predict(X_test, mod_test)
    np.save(f"{PREDS_DIR}/af_he_iot_stacked_pred.npy", y_pred_stack)
    stacker.save(f"{MODELS_DIR}/af_he_smart_farming_iot")
    
    df_ablations = run_14_ablation_benchmarks(
        base_predictions_val=base_val,
        base_predictions_test=base_test,
        y_val=y_val,
        y_test=y_test,
        task_type="regression",
        stacked_test_pred=y_pred_stack,
        task_id="Task_D1_IoT_Yield"
    )
    df_ablations.to_csv(f"{METRICS_DIR}/af_he_iot_ablations.csv", index=False)
    
    y_pred_val_stack = stacker.predict(X_val, mod_val)
    conformal = ConformalPredictor(alpha=0.10)
    conformal.fit_calibration(y_val, y_pred_val_stack)
    intervals = conformal.predict_intervals(y_pred_stack)
    actual_coverage = float(np.mean((y_test >= intervals["y_lower"]) & (y_test <= intervals["y_upper"])))
    
    best_single_r2 = df_ablations[df_ablations["ensemble_strategy"] == "single"]["r2"].max()
    best_single_mae = df_ablations[df_ablations["ensemble_strategy"] == "single"]["mae"].min()
    stack_row = df_ablations[df_ablations["ablation_id"].str.contains("Stacking")].iloc[0]
    stack_r2 = stack_row["r2"]
    stack_mae = stack_row["mae"]
    
    summary_df = pd.DataFrame([{
        "task": "Smart Farming IoT Yield (D6)",
        "best_single_model": df_ablations[df_ablations["r2"] == best_single_r2]["models_included"].iloc[0],
        "best_single_r2": best_single_r2,
        "best_single_mae": best_single_mae,
        "af_he_stacking_r2": stack_r2,
        "af_he_stacking_mae": stack_mae,
        "r2_improvement": stack_r2 - best_single_r2,
        "mae_reduction": best_single_mae - stack_mae,
        "conformal_coverage": actual_coverage,
        "total_training_time_sec": fit_time
    }])
    summary_df.to_csv(f"{METRICS_DIR}/af_he_iot_summary.csv", index=False)
    print(f"Task D completed in {fit_time:.2f}s | AF-HE Stack R2: {stack_r2:.4f} (Best Single: {best_single_r2:.4f})")
    return summary_df, df_ablations


# =========================================================================
# EXPERIMENT 5: TASK E - SOIL NUTRIENT SUITABILITY (DATASET 7)
# =========================================================================
def run_af_he_soil_nutrients():
    print("\n" + "="*70)
    print("RUNNING AF-HE: SOIL NUTRIENT SUITABILITY (DATASET 7)")
    print("="*70)
    
    df = pd.read_csv("data/raw/soil_nutrients/dataset.csv")
    train_idx = np.load("data/splits/soil_nutrients_train.npy")
    val_idx = np.load("data/splits/soil_nutrients_val.npy")
    test_idx = np.load("data/splits/soil_nutrients_test.npy")
    
    num_cols = ['N', 'P', 'K', 'ph', 'EC', 'S', 'Cu', 'Fe', 'Mn', 'Zn', 'B']
    prep = TaskDataPreprocessor("Soil_Nutrients", continuous_cols=num_cols, target_col="label", is_classification=True)
    t_train = prep.fit_transform(df.iloc[train_idx])
    t_val = prep.transform(df.iloc[val_idx])
    t_test = prep.transform(df.iloc[test_idx])
    
    X_train, y_train = t_train["X_num"], t_train["y"]
    X_val, y_val = t_val["X_num"], t_val["y"]
    X_test, y_test = t_test["X_num"], t_test["y"]
    num_classes = len(prep.label_encoder.classes_)
    
    def extract_soil_modalities(X):
        N = len(X)
        return {
            "soil": X[:, :6].astype(np.float32),
            "env": X[:, 6:].astype(np.float32),
            "iot": np.zeros((N, 1), dtype=np.float32),
            "context": np.zeros((N, 1), dtype=np.float32),
            "satellite": np.zeros((N, 1), dtype=np.float32),
            "mask": np.tile([1.0, 1.0, 0.0, 0.0, 0.0], (N, 1)).astype(np.float32)
        }
        
    mod_train = extract_soil_modalities(X_train)
    mod_val = extract_soil_modalities(X_val)
    mod_test = extract_soil_modalities(X_test)
    
    def train_mod_fn(indices):
        return {k: v[indices] for k, v in mod_train.items()}
        
    base_factories = {
        "RF": lambda: RandomForestModel(task_type="classification", n_estimators=150, max_depth=16, random_state=42),
        "XGBoost": lambda: XGBoostModel(task_type="classification", n_estimators=150, max_depth=6, random_state=42),
        "CatBoost": lambda: CatBoostModel(task_type="classification", iterations=150, depth=6, random_seed=42),
        "AGRI_FUSE": lambda: AgriFuseEstimator(
            soil_dim=6, env_dim=5, iot_dim=1, context_dim=1, satellite_dim=1,
            latent_dim=128, fusion_type="gated", task_type="classification", num_classes=num_classes,
            max_epochs=25, patience=8, random_state=42
        )
    }
    
    t0 = time.time()
    stacker = AFHEStackedEnsemble(task_type="classification", n_splits=5, random_state=42)
    stacker.fit_oof_and_meta(
        base_factories=base_factories,
        X_train=X_train,
        y_train=y_train,
        train_modalities_fn=train_mod_fn,
        task_id="soil_nutrients"
    )
    fit_time = time.time() - t0
    
    base_val = stacker.get_base_predictions(X_val, mod_val)
    base_test = stacker.get_base_predictions(X_test, mod_test)
    
    y_pred_stack = stacker.predict(X_test, mod_test)
    y_proba_stack = stacker.predict_proba(X_test, mod_test)
    np.save(f"{PREDS_DIR}/af_he_soil_stacked_pred.npy", y_pred_stack)
    np.save(f"{PREDS_DIR}/af_he_soil_stacked_proba.npy", y_proba_stack)
    stacker.save(f"{MODELS_DIR}/af_he_soil_nutrients")
    
    df_ablations = run_14_ablation_benchmarks(
        base_predictions_val=base_val,
        base_predictions_test=base_test,
        y_val=y_val,
        y_test=y_test,
        task_type="classification",
        stacked_test_pred=y_pred_stack,
        task_id="Task_A3_Soil_Suitability"
    )
    df_ablations.to_csv(f"{METRICS_DIR}/af_he_soil_ablations.csv", index=False)
    
    brier_stack = compute_brier_score(y_test, y_proba_stack)
    best_single_acc = df_ablations[df_ablations["ensemble_strategy"] == "single"]["accuracy"].max()
    best_single_f1 = df_ablations[df_ablations["ensemble_strategy"] == "single"]["macro_f1"].max()
    stack_row = df_ablations[df_ablations["ablation_id"].str.contains("Stacking")].iloc[0]
    stack_acc = stack_row["accuracy"]
    stack_f1 = stack_row["macro_f1"]
    
    summary_df = pd.DataFrame([{
        "task": "Soil Nutrient Suitability (D7)",
        "best_single_model": df_ablations[df_ablations["accuracy"] == best_single_acc]["models_included"].iloc[0],
        "best_single_accuracy": best_single_acc,
        "best_single_macro_f1": best_single_f1,
        "af_he_stacking_accuracy": stack_acc,
        "af_he_stacking_macro_f1": stack_f1,
        "accuracy_improvement": stack_acc - best_single_acc,
        "f1_improvement": stack_f1 - best_single_f1,
        "brier_score": brier_stack,
        "total_training_time_sec": fit_time
    }])
    summary_df.to_csv(f"{METRICS_DIR}/af_he_soil_summary.csv", index=False)
    print(f"Task E completed in {fit_time:.2f}s | AF-HE Stack Acc: {stack_acc:.4f} (Best Single: {best_single_acc:.4f})")
    return summary_df, df_ablations


# =========================================================================
# EXPERIMENT 6: TASK F - AGRIYIELD 2025 SATELLITE (DATASET 9)
# =========================================================================
def run_af_he_agriyield():
    print("\n" + "="*70)
    print("RUNNING AF-HE: SATELLITE-INDEX AGRIYIELD 2025 (DATASET 9)")
    print("="*70)
    
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
    
    X_train = np.hstack([mod_train["soil"], mod_train["env"], mod_train["satellite"]])
    X_val = np.hstack([mod_val["soil"], mod_val["env"], mod_val["satellite"]])
    X_test = np.hstack([mod_test["soil"], mod_test["env"], mod_test["satellite"]])
    
    def train_mod_fn(indices):
        return {k: v[indices] for k, v in mod_train.items()}
        
    base_factories = {
        "RF": lambda: RandomForestModel(task_type="regression", n_estimators=150, max_depth=16, random_state=42),
        "XGBoost": lambda: XGBoostModel(task_type="regression", n_estimators=150, max_depth=6, random_state=42),
        "CatBoost": lambda: CatBoostModel(task_type="regression", iterations=150, depth=6, random_seed=42),
        "AGRI_FUSE": lambda: AgriFuseEstimator(
            soil_dim=3, env_dim=3, iot_dim=1, context_dim=1, satellite_dim=1,
            latent_dim=128, fusion_type="gated", task_type="regression",
            max_epochs=20, patience=6, random_state=42
        )
    }
    
    t0 = time.time()
    stacker = AFHEStackedEnsemble(task_type="regression", n_splits=5, random_state=42)
    stacker.fit_oof_and_meta(
        base_factories=base_factories,
        X_train=X_train,
        y_train=y_train,
        train_modalities_fn=train_mod_fn,
        task_id="agriyield"
    )
    fit_time = time.time() - t0
    
    base_val = stacker.get_base_predictions(X_val, mod_val)
    base_test = stacker.get_base_predictions(X_test, mod_test)
    
    y_pred_stack = stacker.predict(X_test, mod_test)
    np.save(f"{PREDS_DIR}/af_he_agriyield_stacked_pred.npy", y_pred_stack)
    stacker.save(f"{MODELS_DIR}/af_he_agriyield")
    
    df_ablations = run_14_ablation_benchmarks(
        base_predictions_val=base_val,
        base_predictions_test=base_test,
        y_val=y_val,
        y_test=y_test,
        task_type="regression",
        stacked_test_pred=y_pred_stack,
        task_id="Task_E1_AgriYield"
    )
    df_ablations.to_csv(f"{METRICS_DIR}/af_he_agriyield_ablations.csv", index=False)
    
    y_pred_val_stack = stacker.predict(X_val, mod_val)
    conformal = ConformalPredictor(alpha=0.10)
    conformal.fit_calibration(y_val, y_pred_val_stack)
    intervals = conformal.predict_intervals(y_pred_stack)
    actual_coverage = float(np.mean((y_test >= intervals["y_lower"]) & (y_test <= intervals["y_upper"])))
    
    best_single_r2 = df_ablations[df_ablations["ensemble_strategy"] == "single"]["r2"].max()
    best_single_mae = df_ablations[df_ablations["ensemble_strategy"] == "single"]["mae"].min()
    stack_row = df_ablations[df_ablations["ablation_id"].str.contains("Stacking")].iloc[0]
    stack_r2 = stack_row["r2"]
    stack_mae = stack_row["mae"]
    
    summary_df = pd.DataFrame([{
        "task": "AgriYield 2025 (D9)",
        "best_single_model": df_ablations[df_ablations["r2"] == best_single_r2]["models_included"].iloc[0],
        "best_single_r2": best_single_r2,
        "best_single_mae": best_single_mae,
        "af_he_stacking_r2": stack_r2,
        "af_he_stacking_mae": stack_mae,
        "r2_improvement": stack_r2 - best_single_r2,
        "mae_reduction": best_single_mae - stack_mae,
        "conformal_coverage": actual_coverage,
        "total_training_time_sec": fit_time
    }])
    summary_df.to_csv(f"{METRICS_DIR}/af_he_agriyield_summary.csv", index=False)
    print(f"Task F completed in {fit_time:.2f}s | AF-HE Stack R2: {stack_r2:.4f} (Best Single: {best_single_r2:.4f})")
    return summary_df, df_ablations


# =========================================================================
# MASTER PIPELINE CONSOLIDATION
# =========================================================================
def run_all_af_he_experiments():
    start_all = time.time()
    print("\n" + "#"*70)
    print("STARTING FULL AF-HE HYBRID ENSEMBLE BENCHMARK SUITE")
    print("#"*70)
    
    s_a, _ = run_af_he_crop_recommendation()
    s_b, _ = run_af_he_indian_yield()
    s_c, _ = run_af_he_wheat_yield()
    s_d, _ = run_af_he_iot_yield()
    s_e, _ = run_af_he_soil_nutrients()
    s_f, _ = run_af_he_agriyield()
    
    master_df = pd.concat([s_a, s_b, s_c, s_d, s_e, s_f], ignore_index=True)
    master_df.to_csv(f"{METRICS_DIR}/af_he_master_comparison.csv", index=False)
    
    total_time = time.time() - start_all
    print("\n" + "#"*70)
    print(f"ALL AF-HE EXPERIMENTS FINISHED IN {total_time/60:.2f} MINUTES!")
    print(f"Consolidated results saved to: {METRICS_DIR}/af_he_master_comparison.csv")
    print("#"*70)

if __name__ == "__main__":
    run_all_af_he_experiments()


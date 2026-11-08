"""
Explainable AI (XAI) and Interpretability Module for AGRI-FUSE.
Generates SHAP values, Permutation Feature Importance, Gate-Weight Distributions,
and Error Analysis visualizations.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import shap

PLOTS_DIR = "artifacts/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def generate_crop_rec_explainability():
    print("--- Generating Explainability for Crop Recommendation ---")
    df = pd.read_csv("data/raw/crop_recommendation/Crop_recommendation.csv")
    test_idx = np.load("data/splits/crop_rec_seed_42_test.npy")
    df_test = df.iloc[test_idx]
    
    prep = joblib.load("artifacts/models/crop_rec_preprocessor.joblib")
    rf_model = joblib.load("artifacts/models/crop_rec_RandomForest.joblib")
    
    t_test = prep.transform(df_test)
    X_test, y_test = t_test["X_num"], t_test["y"]
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    # 1. Permutation Feature Importance
    perm = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=42)
    df_perm = pd.DataFrame({
        "feature": features,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values("importance_mean", ascending=False)
    
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=df_perm, x="importance_mean", y="feature", palette="viridis")
    plt.title("Task A1: Permutation Feature Importance (Random Forest)", fontsize=12, fontweight='bold')
    plt.xlabel("Mean Accuracy Decrease Upon Shuffling")
    plt.ylabel("Agronomic / Environmental Feature")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/crop_rec_permutation_importance.png", dpi=200)
    plt.close()
    
    # 2. Tree SHAP values
    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(X_test[:100]) # Sample for speed
    
    # Average absolute SHAP across classes
    if isinstance(shap_vals, list):
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
    elif shap_vals.ndim == 3:
        mean_shap = np.abs(shap_vals).mean(axis=(0, 2))
    else:
        mean_shap = np.abs(shap_vals).mean(axis=0)
        
    df_shap = pd.DataFrame({"feature": features, "mean_abs_shap": mean_shap}).sort_values("mean_abs_shap", ascending=False)
    
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=df_shap, x="mean_abs_shap", y="feature", palette="rocket")
    plt.title("Task A1: Global Mean |SHAP| Values (Soil & Environmental Drivers)", fontsize=12, fontweight='bold')
    plt.xlabel("Mean |SHAP value| (Impact on Model Log-Odds)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/crop_rec_shap_summary.png", dpi=200)
    plt.close()
    
    # 3. Gate Weight Analysis for AGRI-FUSE
    gates_path = "artifacts/predictions/crop_rec_AGRI_FUSE_seed_42_gates.npy"
    if os.path.exists(gates_path):
        gates = np.load(gates_path)
        # gates: [Soil, Env, IoT, Context, Satellite]
        df_gates = pd.DataFrame({
            "Soil Encoder (alpha_1)": gates[:, 0],
            "Environmental Encoder (alpha_2)": gates[:, 1]
        })
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=df_gates, palette="Blues")
        plt.title("AGRI-FUSE Adaptive Gating Weight Distributions (Crop Recommendation)", fontsize=11, fontweight='bold')
        plt.ylabel("Availability-Aware Gate Weight (alpha_i)")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/crop_rec_gating_weights.png", dpi=200)
        plt.close()
        
    print("Crop Recommendation explainability figures saved!")

def generate_satellite_yield_explainability():
    print("--- Generating Explainability for Satellite Index AgriYield 2025 ---")
    rf_model = joblib.load("artifacts/models/agriyield_RandomForest.joblib")
    gates_path = "artifacts/predictions/agriyield_AGRI_FUSE_gates.npy"
    
    features = ['soil_ph', 'organic_matter', 'sand_pct', 'temperature', 'humidity', 'rainfall', 'ndvi']
    importances = rf_model.feature_importances_
    df_imp = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)
    
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=df_imp, x="importance", y="feature", palette="crest")
    plt.title("Task E1: AgriYield 2025 Feature Importances (Random Forest)", fontsize=12, fontweight='bold')
    plt.xlabel("Gini Impurity Feature Importance")
    plt.ylabel("Remote Sensing, Soil & Weather Features")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/agriyield_feature_importance.png", dpi=200)
    plt.close()
    
    if os.path.exists(gates_path):
        gates = np.load(gates_path)
        df_gates = pd.DataFrame({
            "Soil Branch (alpha_1)": gates[:, 0],
            "Weather Branch (alpha_2)": gates[:, 1],
            "Satellite NDVI (alpha_5)": gates[:, 4]
        })
        plt.figure(figsize=(8, 4.5))
        sns.boxplot(data=df_gates, palette="Set2")
        plt.title("AGRI-FUSE Gating Distribution Across Soil, Weather & Satellite NDVI", fontsize=11, fontweight='bold')
        plt.ylabel("Gating Weight (alpha_i)")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/agriyield_gating_weights.png", dpi=200)
        plt.close()
        
    print("AgriYield explainability figures saved!")

def generate_iot_explainability():
    print("--- Generating Explainability for Smart Farming IoT ---")
    gates_path = "artifacts/predictions/iot_yield_AGRI_FUSE_gates.npy"
    if os.path.exists(gates_path):
        gates = np.load(gates_path)
        df_gates = pd.DataFrame({
            "Soil Sensor (alpha_1)": gates[:, 0],
            "Weather Sensor (alpha_2)": gates[:, 1],
            "IoT Telemetry (alpha_3)": gates[:, 2],
            "Agri Context (alpha_4)": gates[:, 3],
            "Remote Sensing NDVI (alpha_5)": gates[:, 4]
        })
        plt.figure(figsize=(9, 4.5))
        sns.boxplot(data=df_gates, palette="magma")
        plt.title("AGRI-FUSE 5-Modality Gating Allocation (Smart Farming IoT)", fontsize=11, fontweight='bold')
        plt.ylabel("Availability-Aware Gate Weight (alpha_i)")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/iot_5modality_gating_weights.png", dpi=200)
        plt.close()
        print("IoT explainability figures saved!")

if __name__ == "__main__":
    generate_crop_rec_explainability()
    generate_satellite_yield_explainability()
    generate_iot_explainability()


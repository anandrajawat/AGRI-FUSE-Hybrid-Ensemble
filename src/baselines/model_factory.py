"""
Baseline Model Suite for AGRI-FUSE Research Platform.
Provides standard implementations of Random Forest, Extra Trees, XGBoost, LightGBM,
CatBoost, HistGradientBoosting, Ridge/Logistic Regression, and MLP.
"""

import time
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from src.evaluation.metrics import (
    compute_classification_metrics, compute_regression_metrics, benchmark_inference_speed
)

def get_baseline_models(task_type="classification", num_classes=22, random_state=42):
    """
    Returns dictionary of instantiated baseline models for classification or regression.
    """
    if task_type == "classification":
        return {
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1),
            "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=random_state, eval_metric='mlogloss', n_jobs=-1),
            "LightGBM": lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=random_state, verbose=-1, n_jobs=-1),
            "CatBoost": cb.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_seed=random_state, verbose=False),
            "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, random_state=random_state),
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=random_state),
            "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=random_state, early_stopping=True)
        }
    elif task_type == "regression":
        return {
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1),
            "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=random_state, n_jobs=-1),
            "LightGBM": lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=random_state, verbose=-1, n_jobs=-1),
            "CatBoost": cb.CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, random_seed=random_state, verbose=False),
            "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=100, random_state=random_state),
            "Ridge": Ridge(alpha=1.0, random_state=random_state),
            "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200, random_state=random_state, early_stopping=True)
        }
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

def train_and_eval_baseline(model, name, X_train, y_train, X_test, y_test, task_type="classification"):
    """
    Fits baseline model, times training, measures inference latency, and computes full metrics.
    """
    start_t = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = round(time.perf_counter() - start_t, 3)
    
    start_inf = time.perf_counter()
    y_pred = model.predict(X_test)
    inf_time = time.perf_counter() - start_inf
    latency_ms = round((inf_time / len(X_test)) * 1000.0, 3)
    
    if task_type == "classification":
        metrics = compute_classification_metrics(y_test, y_pred)
    else:
        metrics = compute_regression_metrics(y_test, y_pred)
        
    metrics["model_name"] = name
    metrics["training_time_sec"] = train_time
    metrics["latency_ms_per_sample"] = latency_ms
    metrics["y_pred"] = y_pred
    
    return metrics


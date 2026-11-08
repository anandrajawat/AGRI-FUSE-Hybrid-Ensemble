"""
Comprehensive Agricultural Evaluation Metrics Module for AGRI-FUSE.
Computes full classification, regression, and computational efficiency metrics.
"""

import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)

def compute_classification_metrics(y_true, y_pred, y_prob=None, labels=None):
    """
    Computes complete classification benchmark metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_f1": [round(x, 4) for x in per_class_f1],
        "confusion_matrix": cm
    }

def compute_regression_metrics(y_true, y_pred):
    """
    Computes complete regression benchmark metrics.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    
    # Mean Bias Error (MBE): mean(y_pred - y_true)
    mbe = float(np.mean(y_pred - y_true))
    
    # MAPE (filter non-zero ground truth)
    nonzero_mask = np.abs(y_true) > 1e-5
    if np.sum(nonzero_mask) > 0:
        mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100)
    else:
        mape = None
        
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mbe": round(mbe, 4),
        "mape_pct": round(mape, 2) if mape is not None else None
    }

def benchmark_inference_speed(predict_fn, sample_input, n_warmup=10, n_iter=50):
    """
    Measures inference latency (ms per sample) and throughput (samples/sec).
    """
    # Warmup
    for _ in range(n_warmup):
        _ = predict_fn(sample_input)
        
    batch_size = len(sample_input) if hasattr(sample_input, '__len__') and not isinstance(sample_input, dict) else len(next(iter(sample_input.values())))
    
    start_t = time.perf_counter()
    for _ in range(n_iter):
        _ = predict_fn(sample_input)
    total_time = time.perf_counter() - start_t
    
    avg_latency_ms = (total_time / (n_iter * batch_size)) * 1000.0
    throughput = (n_iter * batch_size) / total_time
    
    return {
        "latency_ms_per_sample": round(avg_latency_ms, 3),
        "throughput_samples_per_sec": round(throughput, 1)
    }


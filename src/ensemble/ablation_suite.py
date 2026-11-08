"""
14-Configuration Ablation Suite for AF-HE Hybrid Architecture.
Implements and evaluates:
1. Random Forest alone
2. XGBoost alone
3. CatBoost alone
4. AGRI-FUSE alone
5. RF + XGBoost
6. RF + CatBoost
7. XGBoost + CatBoost
8. AGRI-FUSE + RF
9. AGRI-FUSE + XGBoost
10. AGRI-FUSE + CatBoost
11. All 4 Simple Averaging
12. All 4 Weighted Averaging
13. All 4 Stacking (AF-HE Stacked Fusion)
14. All 4 Adaptive Reliability Weighting
"""

import numpy as np
import pandas as pd
from src.ensemble.averaging import WeightedAveragingEnsemble
from src.ensemble.adaptive_blending import AdaptiveReliabilityBlender
from src.evaluation.metrics import compute_classification_metrics, compute_regression_metrics

def evaluate_predictions(y_test, y_pred, task_type="classification"):
    if task_type == "classification":
        return compute_classification_metrics(y_test, y_pred)
    else:
        return compute_regression_metrics(y_test, y_pred)

def run_14_ablation_benchmarks(
    base_predictions_val, # dict: { "RF": val_p, "XGB": val_p, "CatBoost": val_p, "AGRI_FUSE": val_p }
    base_predictions_test, # dict: { "RF": test_p, "XGB": test_p, "CatBoost": test_p, "AGRI_FUSE": test_p }
    y_val,
    y_test,
    task_type="classification",
    stacked_test_pred=None,
    task_id="Task"
):
    """
    Executes and compares all 14 ablation configurations on identical validation & test splits.
    """
    ablations = [
        ("1. Random Forest alone", ["RF"], "single"),
        ("2. XGBoost alone", ["XGBoost"], "single"),
        ("3. CatBoost alone", ["CatBoost"], "single"),
        ("4. AGRI-FUSE alone", ["AGRI_FUSE"], "single"),
        ("5. RF + XGBoost", ["RF", "XGBoost"], "average"),
        ("6. RF + CatBoost", ["RF", "CatBoost"], "average"),
        ("7. XGBoost + CatBoost", ["XGBoost", "CatBoost"], "average"),
        ("8. AGRI-FUSE + RF", ["AGRI_FUSE", "RF"], "average"),
        ("9. AGRI-FUSE + XGBoost", ["AGRI_FUSE", "XGBoost"], "average"),
        ("10. AGRI-FUSE + CatBoost", ["AGRI_FUSE", "CatBoost"], "average"),
        ("11. All 4 Simple Averaging", ["RF", "XGBoost", "CatBoost", "AGRI_FUSE"], "simple_avg"),
        ("12. All 4 Weighted Averaging", ["RF", "XGBoost", "CatBoost", "AGRI_FUSE"], "weighted_avg"),
        ("13. All 4 Stacking (AF-HE Stacked Fusion)", ["RF", "XGBoost", "CatBoost", "AGRI_FUSE"], "stacking"),
        ("14. All 4 Adaptive Reliability Weighting", ["RF", "XGBoost", "CatBoost", "AGRI_FUSE"], "adaptive")
    ]
    
    results = []
    
    for config_name, models, mode in ablations:
        if mode == "single":
            m_key = models[0]
            p_test = base_predictions_test[m_key]
            if task_type == "classification" and p_test.ndim > 1:
                p_label = np.argmax(p_test, axis=1)
            else:
                p_label = p_test
                
        elif mode == "average" or mode == "simple_avg":
            # Unweighted average of probabilities or predictions
            sub_preds = [base_predictions_test[m] for m in models]
            mean_p = np.mean(sub_preds, axis=0)
            if task_type == "classification" and mean_p.ndim > 1:
                p_label = np.argmax(mean_p, axis=1)
            else:
                p_label = mean_p
                
        elif mode == "weighted_avg":
            sub_val = [base_predictions_val[m] for m in models]
            sub_test = [base_predictions_test[m] for m in models]
            wa = WeightedAveragingEnsemble(task_type=task_type, num_models=len(models))
            wa.fit_weights(sub_val, y_val)
            p_label = wa.predict(sub_test)
            
        elif mode == "stacking":
            if stacked_test_pred is not None:
                p_label = stacked_test_pred
            else:
                # Fallback to weighted averaging if stacking pred not supplied
                sub_test = [base_predictions_test[m] for m in models]
                p_label = np.mean(sub_test, axis=0)
                if task_type == "classification" and p_label.ndim > 1:
                    p_label = np.argmax(p_label, axis=1)
                    
        elif mode == "adaptive":
            blender = AdaptiveReliabilityBlender(task_type=task_type)
            val_dict = {m: base_predictions_val[m] for m in models}
            test_dict = {m: base_predictions_test[m] for m in models}
            blender.fit_reliability(models, val_dict, y_val)
            p_label = blender.predict(test_dict)

        metrics = evaluate_predictions(y_test, p_label, task_type=task_type)
        metrics["ablation_id"] = config_name
        metrics["models_included"] = "+".join(models)
        metrics["ensemble_strategy"] = mode
        metrics["task_id"] = task_id
        results.append(metrics)
        
    df_res = pd.DataFrame([{k: v for k, v in r.items() if k not in ['confusion_matrix', 'per_class_f1']} for r in results])
    return df_res


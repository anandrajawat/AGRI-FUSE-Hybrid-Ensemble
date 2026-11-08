"""
Simple and Validation-Calibrated Weighted Averaging Ensemble for AF-HE Architecture.
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, log_loss

class WeightedAveragingEnsemble:
    """
    Computes simple or validation-optimized weighted averages of base model predictions.
    Weights are constrained to w_i >= 0 and sum(w_i) = 1.0.
    """
    def __init__(self, task_type="classification", num_models=4):
        self.task_type = task_type
        self.num_models = num_models
        self.weights = np.ones(num_models) / num_models

    def fit_weights(self, val_preds_list, y_val):
        """
        Optimizes weights w_i on validation partition out-of-fold predictions.
        
        Args:
            val_preds_list: List of K prediction arrays (logits/probabilities for classification, continuous for regression)
            y_val: Ground truth validation targets
        """
        num_m = len(val_preds_list)
        init_w = np.ones(num_m) / num_m
        bounds = [(0.0, 1.0) for _ in range(num_m)]
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        if self.task_type == "regression":
            def loss_func(w):
                blended = np.zeros_like(val_preds_list[0])
                for i in range(num_m):
                    blended += w[i] * val_preds_list[i]
                return mean_squared_error(y_val, blended)
        else:
            num_classes = val_preds_list[0].shape[1]
            all_labels = list(range(num_classes))
            def loss_func(w):
                blended = np.zeros_like(val_preds_list[0])
                for i in range(num_m):
                    blended += w[i] * val_preds_list[i]
                # clip for numerical stability
                blended = np.clip(blended, 1e-7, 1.0 - 1e-7)
                blended /= blended.sum(axis=1, keepdims=True)
                return log_loss(y_val, blended, labels=all_labels)

        res = minimize(loss_func, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success:
            self.weights = res.x
        else:
            self.weights = init_w
        return self.weights

    def predict(self, preds_list):
        if self.task_type == "regression":
            blended = np.zeros_like(preds_list[0], dtype=np.float64)
            for i, p in enumerate(preds_list):
                blended += self.weights[i] * p
            return blended
        else:
            probs = self.predict_proba(preds_list)
            return np.argmax(probs, axis=1)

    def predict_proba(self, probs_list):
        if self.task_type != "classification":
            raise ValueError("predict_proba is only for classification.")
        blended = np.zeros_like(probs_list[0], dtype=np.float64)
        for i, p in enumerate(probs_list):
            blended += self.weights[i] * p
        blended /= blended.sum(axis=1, keepdims=True)
        return blended

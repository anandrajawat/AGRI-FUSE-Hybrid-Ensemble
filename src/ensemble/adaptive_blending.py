"""
Adaptive Reliability-Weighted Ensemble for AF-HE Architecture.
Dynamically calibrates base model weights using validation loss and error distributions.
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss

class AdaptiveReliabilityBlender:
    """
    Computes validation-calibrated reliability weights:
        w_i = exp(-lambda * Error_i) / sum(exp(-lambda * Error_j))
    or via constrained quadratic optimization on validation predictions.
    """
    def __init__(self, task_type="classification", temperature=1.0):
        self.task_type = task_type
        self.temperature = temperature
        self.weights = None
        self.model_names = []

    def fit_reliability(self, model_names, val_preds_dict, y_val):
        """
        Estimates reliability weights based on validation error.
        
        Args:
            model_names: List of model identifiers
            val_preds_dict: Dict mapping name -> validation predictions (probabilities or continuous)
            y_val: Ground truth validation targets
        """
        self.model_names = model_names
        num_m = len(model_names)
        
        if self.task_type == "regression":
            # Estimate validation MSE for each model
            errors = []
            for name in model_names:
                p = val_preds_dict[name]
                mse = mean_squared_error(y_val, p)
                errors.append(mse)
            errors = np.array(errors, dtype=np.float64)
            
            # Constrained optimization to minimize ensemble validation error
            def obj(w):
                pred = sum(w[i] * val_preds_dict[model_names[i]] for i in range(num_m))
                return mean_squared_error(y_val, pred)
                
            init_w = np.ones(num_m) / num_m
            bounds = [(0.0, 1.0) for _ in range(num_m)]
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            res = minimize(obj, init_w, method='SLSQP', bounds=bounds, constraints=cons)
            
            if res.success:
                self.weights = res.x
            else:
                # Softmin fallback based on RMSE
                scaled = -errors / (np.std(errors) + 1e-8) / self.temperature
                self.weights = np.exp(scaled - np.max(scaled))
                self.weights /= np.sum(self.weights)
        else:
            # Classification: optimize log-loss / Brier score
            num_classes = val_preds_dict[model_names[0]].shape[1]
            all_labels = list(range(num_classes))
            def obj(w):
                pred = sum(w[i] * val_preds_dict[model_names[i]] for i in range(num_m))
                pred = np.clip(pred, 1e-7, 1.0 - 1e-7)
                pred /= pred.sum(axis=1, keepdims=True)
                return log_loss(y_val, pred, labels=all_labels)
                
            init_w = np.ones(num_m) / num_m
            bounds = [(0.0, 1.0) for _ in range(num_m)]
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            res = minimize(obj, init_w, method='SLSQP', bounds=bounds, constraints=cons)
            
            if res.success:
                self.weights = res.x
            else:
                self.weights = init_w
                
        return {name: round(float(self.weights[i]), 4) for i, name in enumerate(model_names)}

    def predict(self, test_preds_dict):
        if self.task_type == "regression":
            blended = np.zeros_like(test_preds_dict[self.model_names[0]], dtype=np.float64)
            for i, name in enumerate(self.model_names):
                blended += self.weights[i] * test_preds_dict[name]
            return blended
        else:
            probs = self.predict_proba(test_preds_dict)
            return np.argmax(probs, axis=1)

    def predict_proba(self, test_preds_dict):
        if self.task_type != "classification":
            raise ValueError("predict_proba only valid for classification.")
        blended = np.zeros_like(test_preds_dict[self.model_names[0]], dtype=np.float64)
        for i, name in enumerate(self.model_names):
            blended += self.weights[i] * test_preds_dict[name]
        blended /= blended.sum(axis=1, keepdims=True)
        return blended

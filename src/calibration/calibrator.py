"""
Calibration and Uncertainty Quantification Module for AF-HE Architecture.
Provides probability calibration, Brier score, reliability diagrams,
residual analysis, and conformal prediction intervals.
"""

import numpy as np
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

def compute_brier_score(y_true, y_probs):
    """Computes multi-class Brier score."""
    num_classes = y_probs.shape[1]
    one_hot = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((y_probs - one_hot) ** 2, axis=1)))

def get_calibration_curve_data(y_true, y_probs, n_bins=10):
    """Computes calibration curve for primary class or one-vs-rest."""
    prob_true, prob_pred = calibration_curve(
        (y_true == np.argmax(y_probs, axis=1)).astype(int),
        np.max(y_probs, axis=1),
        n_bins=n_bins
    )
    return {
        "prob_pred": prob_pred.tolist(),
        "prob_true": prob_true.tolist()
    }

class ConformalPredictor:
    """
    Inductive Conformal Prediction for continuous yield regressions.
    Generates non-parametric, finite-sample valid prediction intervals [y_lower, y_upper].
    """
    def __init__(self, alpha=0.10):
        # 1 - alpha is coverage (default 90% coverage)
        self.alpha = alpha
        self.q_val = None

    def fit_calibration(self, y_cal, y_pred_cal):
        """Computes conformity non-conformity scores |y_cal - y_pred_cal|."""
        residuals = np.abs(y_cal - y_pred_cal)
        n = len(residuals)
        # Empirical quantile at level (1 - alpha)(1 + 1/n)
        q_level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / n))
        self.q_val = float(np.quantile(residuals, q_level))
        return self.q_val

    def predict_intervals(self, y_pred_test):
        """Constructs conformal prediction intervals [y_pred - q, y_pred + q]."""
        if self.q_val is None:
            raise RuntimeError("Must fit calibration partition first.")
        lower = y_pred_test - self.q_val
        upper = y_pred_test + self.q_val
        return {
            "q_val": self.q_val,
            "y_lower": lower,
            "y_upper": upper,
            "interval_width": 2.0 * self.q_val
        }


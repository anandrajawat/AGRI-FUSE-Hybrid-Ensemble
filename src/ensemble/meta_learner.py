"""
Stacking Meta-Learners for AF-HE Architecture.
Provides regularized Ridge, Non-Negative Least Squares, and Logistic Regression Meta-Learners.
"""

import numpy as np
from sklearn.linear_model import RidgeCV, LogisticRegression
from scipy.optimize import nnls

class RegressionMetaLearner:
    """Meta-learner for stacking continuous yield regressions."""
    def __init__(self, use_non_negative=True):
        self.use_non_negative = use_non_negative
        self.ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_meta, y):
        if self.use_non_negative:
            # Add bias column
            X_b = np.hstack([np.ones((len(X_meta), 1)), X_meta])
            w, _ = nnls(X_b, y)
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.ridge.fit(X_meta, y)
            self.coef_ = self.ridge.coef_
            self.intercept_ = self.ridge.intercept_
        return self

    def predict(self, X_meta):
        if self.use_non_negative:
            return self.intercept_ + np.dot(X_meta, self.coef_)
        else:
            return self.ridge.predict(X_meta)


class ClassificationMetaLearner:
    """Meta-learner for stacking multi-class crop recommendations and crop types."""
    def __init__(self, C=1.0, random_state=42):
        self.clf = LogisticRegression(C=C, max_iter=500, random_state=random_state)

    def fit(self, X_meta, y):
        self.clf.fit(X_meta, y)
        return self

    def predict(self, X_meta):
        return self.clf.predict(X_meta)

    def predict_proba(self, X_meta):
        return self.clf.predict_proba(X_meta)


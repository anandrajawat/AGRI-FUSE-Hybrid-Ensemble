"""
XGBoost Model Wrapper for AF-HE Architecture.
Supports tuned XGBoost Classifier and Regressor with early stopping.
"""

import joblib
import xgboost as xgb

class XGBoostModel:
    def __init__(
        self,
        task_type="classification",
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    ):
        self.task_type = task_type
        self.random_state = random_state
        if task_type == "classification":
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=random_state,
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=1
            )
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=random_state,
                tree_method="hist",
                n_jobs=1
            )

    def fit(self, X, y, eval_set=None, verbose=False):
        if eval_set:
            self.model.fit(X, y, eval_set=eval_set, verbose=verbose)
        else:
            self.model.fit(X, y, verbose=verbose)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.task_type == "classification":
            return self.model.predict_proba(X)
        raise ValueError("predict_proba is only available for classification.")

    def save(self, filepath):
        joblib.dump(self.model, filepath)

    @classmethod
    def load(cls, filepath, task_type="classification"):
        instance = cls(task_type=task_type)
        instance.model = joblib.load(filepath)
        return instance


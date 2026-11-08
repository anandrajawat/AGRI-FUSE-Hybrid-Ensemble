"""
Random Forest Model Wrapper for AF-HE Architecture.
Supports tuned Random Forest Classifier and Regressor.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class RandomForestModel:
    def __init__(self, task_type="classification", n_estimators=150, max_depth=16, min_samples_split=4, min_samples_leaf=2, max_features="sqrt", random_state=42):
        self.task_type = task_type
        self.random_state = random_state
        if task_type == "classification":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state,
                n_jobs=1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state,
                n_jobs=1
            )
            
    def fit(self, X, y):
        self.model.fit(X, y)
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


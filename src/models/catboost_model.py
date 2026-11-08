"""
CatBoost Model Wrapper for AF-HE Architecture.
Supports tuned CatBoost Classifier and Regressor with native categorical support.
"""

import joblib
import catboost as cb

class CatBoostModel:
    def __init__(
        self,
        task_type="classification",
        iterations=150,
        depth=6,
        learning_rate=0.08,
        l2_leaf_reg=3.0,
        random_strength=1.0,
        random_seed=42
    ):
        self.task_type = task_type
        self.random_seed = random_seed
        if task_type == "classification":
            self.model = cb.CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                random_strength=random_strength,
                random_seed=random_seed,
                thread_count=1,
                verbose=False
            )
        else:
            self.model = cb.CatBoostRegressor(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                random_strength=random_strength,
                random_seed=random_seed,
                thread_count=1,
                verbose=False
            )

    def fit(self, X, y, cat_features=None, eval_set=None):
        if eval_set:
            self.model.fit(X, y, cat_features=cat_features, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X, y, cat_features=cat_features, verbose=False)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.task_type == "classification":
            return self.model.predict_proba(X)
        raise ValueError("predict_proba is only available for classification.")

    def save(self, filepath):
        self.model.save_model(filepath)

    @classmethod
    def load(cls, filepath, task_type="classification"):
        instance = cls(task_type=task_type)
        if task_type == "classification":
            instance.model = cb.CatBoostClassifier()
        else:
            instance.model = cb.CatBoostRegressor()
        instance.model.load_model(filepath)
        return instance


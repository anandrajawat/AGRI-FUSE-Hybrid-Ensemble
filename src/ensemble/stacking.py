"""
AF-HE Stacked Fusion Ensemble Engine.
Implements leakage-safe Out-Of-Fold (OOF) prediction generation across Random Forest,
XGBoost, CatBoost, and AGRI-FUSE, followed by meta-learner optimization.
"""

import os
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from src.ensemble.meta_learner import RegressionMetaLearner, ClassificationMetaLearner

FOLDS_DIR = "data/folds" if os.path.exists("data/folds") else "agri-fuse/data/folds"
os.makedirs(FOLDS_DIR, exist_ok=True)

OOF_DIR = "artifacts/oof_predictions" if os.path.exists("artifacts/oof_predictions") else "agri-fuse/artifacts/oof_predictions"
os.makedirs(OOF_DIR, exist_ok=True)

class AFHEStackedEnsemble:
    """
    AGRI-FUSE Hybrid Ensemble (AF-HE) Stacking Architecture.
    Combines:
        1. AGRI-FUSE Neural Multimodal Branch
        2. Random Forest
        3. XGBoost
        4. CatBoost
    Through a regularized meta-learner fitted strictly on out-of-fold predictions.
    """
    def __init__(self, task_type="classification", n_splits=5, random_state=42):
        self.task_type = task_type
        self.n_splits = n_splits
        self.random_state = random_state
        self.meta_learner = (
            ClassificationMetaLearner(random_state=random_state)
            if task_type == "classification"
            else RegressionMetaLearner(use_non_negative=True)
        )
        self.base_models = {}
        self.oof_predictions = {}
        self.is_fitted = False

    def _get_splitter(self, groups=None):
        if groups is not None:
            return GroupKFold(n_splits=self.n_splits)
        if self.task_type == "classification":
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        return KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

    def fit_oof_and_meta(self, base_factories, X_train, y_train, train_modalities_fn=None, groups=None, task_id="task"):
        """
        Step 1 to 5: Generates OOF predictions across K folds and trains the meta-learner.
        
        Args:
            base_factories: Dict mapping model_name -> callable creating a new model instance.
            X_train: Tabular feature matrix for tree models [N, D]
            y_train: Training labels [N]
            train_modalities_fn: Optional function to slice modalities for AGRI-FUSE [fold_indices] -> modality_dict
            groups: Optional grouping array for field-grouped splitting
            task_id: Identifier for saving OOF prediction artifacts
        """
        n_samples = len(y_train)
        splitter = self._get_splitter(groups=groups)
        splits = list(splitter.split(X_train, y_train, groups=groups))
        
        # Save fold assignments for audit
        fold_manifest = {
            "task_id": task_id,
            "n_splits": self.n_splits,
            "splits": [{"fold": i, "val_size": len(val_idx)} for i, (_, val_idx) in enumerate(splits)]
        }
        with open(f"{FOLDS_DIR}/{task_id}_fold_manifest.json", "w") as f:
            json.dump(fold_manifest, f, indent=2)

        # Allocate OOF prediction buffers
        oof_dict = {}
        for m_name in base_factories.keys():
            if self.task_type == "classification":
                num_classes = len(np.unique(y_train))
                oof_dict[m_name] = np.zeros((n_samples, num_classes), dtype=np.float32)
            else:
                oof_dict[m_name] = np.zeros(n_samples, dtype=np.float32)

        # Execute Out-Of-Fold Training Loops
        print(f"Generating Out-Of-Fold predictions across {self.n_splits} folds...")
        for fold, (tr_idx, val_idx) in enumerate(splits):
            X_tr_f, y_tr_f = X_train[tr_idx], y_train[tr_idx]
            X_val_f = X_train[val_idx]
            
            for m_name, factory_fn in base_factories.items():
                model = factory_fn()
                if m_name == "AGRI_FUSE" and train_modalities_fn is not None:
                    # AGRI-FUSE receives modality sub-dictionaries
                    mod_tr = train_modalities_fn(tr_idx)
                    mod_val = train_modalities_fn(val_idx)
                    model.fit(mod_tr, y_tr_f, mod_val, y_train[val_idx])
                    if self.task_type == "classification":
                        oof_dict[m_name][val_idx] = model.predict_proba(mod_val)
                    else:
                        oof_dict[m_name][val_idx] = model.predict(mod_val)
                else:
                    model.fit(X_tr_f, y_tr_f)
                    if self.task_type == "classification":
                        oof_dict[m_name][val_idx] = model.predict_proba(X_val_f)
                    else:
                        oof_dict[m_name][val_idx] = model.predict(X_val_f)

        self.oof_predictions = oof_dict
        
        # Save OOF artifacts
        for m_name, preds in oof_dict.items():
            np.save(f"{OOF_DIR}/{task_id}_{m_name}_oof.npy", preds)

        # Step 4: Construct Meta-Feature Matrix
        if self.task_type == "classification":
            X_meta = np.hstack([oof_dict[m] for m in sorted(oof_dict.keys())])
        else:
            X_meta = np.column_stack([oof_dict[m] for m in sorted(oof_dict.keys())])

        # Step 5: Fit Meta-Learner strictly on OOF predictions
        print("Fitting Stacking Meta-Learner on Out-Of-Fold predictions...")
        self.meta_learner.fit(X_meta, y_train)

        # Step 6: Retrain Base Learners on complete training partition
        print("Retraining base learners on complete training partition...")
        self.base_models = {}
        for m_name, factory_fn in base_factories.items():
            model = factory_fn()
            if m_name == "AGRI_FUSE" and train_modalities_fn is not None:
                mod_all = train_modalities_fn(np.arange(n_samples))
                model.fit(mod_all, y_train, mod_all, y_train)
            else:
                model.fit(X_train, y_train)
            self.base_models[m_name] = model

        self.is_fitted = True
        return self

    def _construct_meta_features(self, X_test, test_modalities=None):
        preds_list = []
        for m_name in sorted(self.base_models.keys()):
            model = self.base_models[m_name]
            if m_name == "AGRI_FUSE" and test_modalities is not None:
                p = model.predict_proba(test_modalities) if self.task_type == "classification" else model.predict(test_modalities)
            else:
                p = model.predict_proba(X_test) if self.task_type == "classification" else model.predict(X_test)
            preds_list.append(p)
            
        if self.task_type == "classification":
            return np.hstack(preds_list)
        return np.column_stack(preds_list)

    def predict(self, X_test, test_modalities=None):
        if not self.is_fitted:
            raise RuntimeError("Stacking ensemble must be fitted first.")
        X_meta_test = self._construct_meta_features(X_test, test_modalities)
        return self.meta_learner.predict(X_meta_test)

    def predict_proba(self, X_test, test_modalities=None):
        if self.task_type != "classification":
            raise ValueError("predict_proba only valid for classification.")
        X_meta_test = self._construct_meta_features(X_test, test_modalities)
        return self.meta_learner.predict_proba(X_meta_test)

    def get_base_predictions(self, X_test, test_modalities=None):
        preds = {}
        for m_name, model in self.base_models.items():
            if m_name == "AGRI_FUSE" and test_modalities is not None:
                p = model.predict_proba(test_modalities) if self.task_type == "classification" else model.predict(test_modalities)
            else:
                p = model.predict_proba(X_test) if self.task_type == "classification" else model.predict(X_test)
            preds[m_name] = p
        return preds

    def save(self, dir_path):
        import joblib
        os.makedirs(dir_path, exist_ok=True)
        meta_info = {
            "task_type": self.task_type,
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "base_model_names": list(self.base_models.keys()),
            "is_fitted": self.is_fitted
        }
        with open(os.path.join(dir_path, "meta_info.json"), "w") as f:
            json.dump(meta_info, f, indent=2)
        joblib.dump(self.meta_learner, os.path.join(dir_path, "meta_learner.joblib"))
        for m_name, model in self.base_models.items():
            if m_name == "AGRI_FUSE":
                model.save(os.path.join(dir_path, "agri_fuse.pth"))
            else:
                model.save(os.path.join(dir_path, f"{m_name}.joblib"))
        print(f"AF-HE Stacked Ensemble successfully saved to {dir_path}")

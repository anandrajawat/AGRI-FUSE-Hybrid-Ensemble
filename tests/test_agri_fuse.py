"""
Automated Test Suite for AGRI-FUSE Research Platform.
Validates data splitting disjointness, encoder dimensions, availability-aware gating,
prediction heads, and metric calculation integrity.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import unittest
import numpy as np
import torch

from src.encoders.specialized_encoders import (
    SoilEncoder, EnvironmentalEncoder, IoTEncoder, AgriculturalContextEncoder, SatelliteIndexEncoder
)
from src.fusion.gated_fusion import (
    AvailabilityAwareGatedFusion, ConcatenationFusion, MeanFusion
)
from src.heads.prediction_heads import (
    CropRecommendationHead, YieldRegressionHead
)
from src.fusion.agri_fuse_model import AgriFuseModel
from src.evaluation.metrics import (
    compute_classification_metrics, compute_regression_metrics
)
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.catboost_model import CatBoostModel
from src.models.agri_fuse import AgriFuseEstimator
from src.ensemble.averaging import WeightedAveragingEnsemble
from src.ensemble.adaptive_blending import AdaptiveReliabilityBlender
from src.ensemble.stacking import AFHEStackedEnsemble
from src.calibration.calibrator import compute_brier_score, get_calibration_curve_data, ConformalPredictor

class TestAgriFusePlatform(unittest.TestCase):
    
    def test_01_split_leakage_prevention(self):
        """Verify train, val, and test splits have strictly empty intersections."""
        splits_dir = "agri-fuse/data/splits"
        tasks = ["crop_rec_seed_42", "indian_yield", "eastern_wheat", "smart_farming_iot", "soil_nutrients", "agriyield"]
        for t in tasks:
            train_f = f"{splits_dir}/{t}_train.npy"
            val_f = f"{splits_dir}/{t}_val.npy"
            test_f = f"{splits_dir}/{t}_test.npy"
            if os.path.exists(train_f) and os.path.exists(val_f) and os.path.exists(test_f):
                tr = set(np.load(train_f))
                vl = set(np.load(val_f))
                ts = set(np.load(test_f))
                self.assertEqual(len(tr.intersection(vl)), 0, f"Leakage between train and val in {t}!")
                self.assertEqual(len(tr.intersection(ts)), 0, f"Leakage between train and test in {t}!")
                self.assertEqual(len(vl.intersection(ts)), 0, f"Leakage between val and test in {t}!")
                
    def test_02_specialized_encoders_output_dimension(self):
        """Verify all specialized encoders project variable inputs into uniform 128-d latent space."""
        batch_size = 16
        latent_dim = 128
        
        soil_enc = SoilEncoder(in_features=4, latent_dim=latent_dim)
        env_enc = EnvironmentalEncoder(in_features=3, latent_dim=latent_dim)
        iot_enc = IoTEncoder(in_features=4, latent_dim=latent_dim)
        ctx_enc = AgriculturalContextEncoder(in_features=5, latent_dim=latent_dim)
        sat_enc = SatelliteIndexEncoder(in_features=1, latent_dim=latent_dim)
        
        x_soil = torch.randn(batch_size, 4)
        x_env = torch.randn(batch_size, 3)
        x_iot = torch.randn(batch_size, 4)
        x_ctx = torch.randn(batch_size, 5)
        x_sat = torch.randn(batch_size, 1)
        
        self.assertEqual(soil_enc(x_soil).shape, (batch_size, latent_dim))
        self.assertEqual(env_enc(x_env).shape, (batch_size, latent_dim))
        self.assertEqual(iot_enc(x_iot).shape, (batch_size, latent_dim))
        self.assertEqual(ctx_enc(x_ctx).shape, (batch_size, latent_dim))
        self.assertEqual(sat_enc(x_sat).shape, (batch_size, latent_dim))

    def test_03_gated_multimodal_unit_masking(self):
        """Verify availability-aware gating masks unavailable modalities and sums to 1.0."""
        batch_size = 8
        latent_dim = 128
        gmu = AvailabilityAwareGatedFusion(num_modalities=5, latent_dim=latent_dim)
        
        reps = [torch.randn(batch_size, latent_dim) for _ in range(5)]
        # Mask modalities 2, 3, 4 (only modality 0 and 1 available)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]] * batch_size)
        
        z, alpha = gmu(reps, mask)
        self.assertEqual(z.shape, (batch_size, latent_dim))
        self.assertEqual(alpha.shape, (batch_size, 5))
        
        # Verify alpha sums to 1.0
        row_sums = alpha.detach().sum(dim=1).numpy()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
        
        # Verify masked modalities have 0.0 weight
        np.testing.assert_allclose(alpha[:, 2:].detach().numpy(), 0.0, atol=1e-5)

    def test_04_end_to_end_agri_fuse_model(self):
        """Verify end-to-end forward pass for classification and regression heads."""
        batch_size = 4
        model_cls = AgriFuseModel(
            soil_dim=4, env_dim=3, iot_dim=1, context_dim=1, satellite_dim=1,
            latent_dim=128, fusion_type="gated", task_type="classification", num_classes=22
        )
        x_s = torch.randn(batch_size, 4)
        x_e = torch.randn(batch_size, 3)
        x_i = torch.zeros(batch_size, 1)
        x_c = torch.zeros(batch_size, 1)
        x_sat = torch.zeros(batch_size, 1)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]] * batch_size)
        
        out_cls, gates = model_cls(x_s, x_e, x_i, x_c, x_sat, mask)
        self.assertEqual(out_cls.shape, (batch_size, 22))
        
        model_reg = AgriFuseModel(
            soil_dim=4, env_dim=3, iot_dim=1, context_dim=1, satellite_dim=1,
            latent_dim=128, fusion_type="gated", task_type="regression"
        )
        out_reg, _ = model_reg(x_s, x_e, x_i, x_c, x_sat, mask)
        self.assertEqual(out_reg.shape, (batch_size,))

    def test_05_metrics_calculation(self):
        """Verify classification and regression metric functions compute correct bounds."""
        y_true_cls = np.array([0, 1, 2, 1, 0])
        y_pred_cls = np.array([0, 1, 2, 0, 0])
        cls_metrics = compute_classification_metrics(y_true_cls, y_pred_cls)
        self.assertAlmostEqual(cls_metrics["accuracy"], 0.8)
        self.assertIn("macro_f1", cls_metrics)
        self.assertIn("confusion_matrix", cls_metrics)
        
        y_true_reg = np.array([10.0, 20.0, 30.0])
        y_pred_reg = np.array([11.0, 19.0, 31.0])
        reg_metrics = compute_regression_metrics(y_true_reg, y_pred_reg)
        self.assertAlmostEqual(reg_metrics["mae"], 1.0)
        self.assertAlmostEqual(reg_metrics["rmse"], 1.0)
        self.assertAlmostEqual(reg_metrics["mbe"], 0.3333, places=3)

    def test_06_af_he_base_models(self):
        """Verify unified fit/predict API for RF, XGB, CatBoost, and AgriFuse wrappers."""
        np.random.seed(42)
        X = np.random.randn(20, 4).astype(np.float32)
        y_cls = np.random.randint(0, 3, size=20)
        y_reg = np.random.randn(20).astype(np.float32)
        
        # Classification
        rf = RandomForestModel(task_type="classification", n_estimators=5, max_depth=3, random_state=42).fit(X, y_cls)
        xgb_m = XGBoostModel(task_type="classification", n_estimators=5, max_depth=3, random_state=42).fit(X, y_cls)
        cb_m = CatBoostModel(task_type="classification", iterations=5, depth=3, random_seed=42).fit(X, y_cls)
        
        self.assertEqual(rf.predict_proba(X).shape, (20, 3))
        self.assertEqual(xgb_m.predict_proba(X).shape, (20, 3))
        self.assertEqual(cb_m.predict_proba(X).shape, (20, 3))
        
        # Regression
        rf_reg = RandomForestModel(task_type="regression", n_estimators=5, max_depth=3, random_state=42).fit(X, y_reg)
        self.assertEqual(rf_reg.predict(X).shape, (20,))

    def test_07_af_he_weighted_averaging(self):
        """Verify SLSQP constrained weighted averaging produces non-negative weights summing to 1.0."""
        y_val = np.array([10.0, 20.0, 30.0, 40.0])
        p1 = np.array([10.5, 19.5, 30.2, 39.8])
        p2 = np.array([12.0, 18.0, 32.0, 38.0])
        
        wa = WeightedAveragingEnsemble(task_type="regression", num_models=2)
        weights = wa.fit_weights([p1, p2], y_val)
        
        self.assertAlmostEqual(np.sum(weights), 1.0, places=4)
        self.assertTrue(all(w >= -1e-5 for w in weights))
        
        p_blend = wa.predict([p1, p2])
        self.assertEqual(p_blend.shape, (4,))

    def test_08_af_he_adaptive_blending(self):
        """Verify Adaptive Reliability Blender dynamically weights models by validation performance."""
        y_val = np.array([10.0, 20.0, 30.0, 40.0])
        # Model 1 has much lower error than Model 2
        p_good = np.array([10.0, 20.0, 30.0, 40.0])
        p_bad = np.array([20.0, 30.0, 40.0, 50.0])
        
        blender = AdaptiveReliabilityBlender(task_type="regression")
        weights = blender.fit_reliability(["Good", "Bad"], {"Good": p_good, "Bad": p_bad}, y_val)
        
        self.assertGreater(weights["Good"], weights["Bad"])
        self.assertAlmostEqual(weights["Good"] + weights["Bad"], 1.0, places=3)

    def test_09_af_he_stacked_ensemble(self):
        """Verify AFHEStackedEnsemble performs K-fold OOF cross validation and meta-learner training."""
        np.random.seed(42)
        X_tr = np.random.randn(30, 4).astype(np.float32)
        y_tr = np.random.randint(0, 2, size=30)
        X_te = np.random.randn(10, 4).astype(np.float32)
        
        base_factories = {
            "RF": lambda: RandomForestModel(task_type="classification", n_estimators=5, max_depth=3, random_state=42),
            "XGBoost": lambda: XGBoostModel(task_type="classification", n_estimators=5, max_depth=3, random_state=42)
        }
        
        stacker = AFHEStackedEnsemble(task_type="classification", n_splits=3, random_state=42)
        stacker.fit_oof_and_meta(base_factories, X_tr, y_tr, task_id="test_task")
        
        preds = stacker.predict(X_te)
        probas = stacker.predict_proba(X_te)
        
        self.assertEqual(preds.shape, (10,))
        self.assertEqual(probas.shape, (10, 2))
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    def test_10_conformal_prediction_and_brier(self):
        """Verify Brier score bounds and Inductive Conformal Prediction interval coverage."""
        y_true_cls = np.array([0, 1, 0, 1])
        y_probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        brier = compute_brier_score(y_true_cls, y_probs)
        self.assertLess(brier, 0.05)
        
        cp = ConformalPredictor(alpha=0.10)
        y_cal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred_cal = np.array([10.1, 19.9, 30.2, 39.8, 50.1])
        cp.fit_calibration(y_cal, y_pred_cal)
        
        y_te = np.array([15.0, 25.0])
        res = cp.predict_intervals(y_te)
        self.assertTrue(all(res["y_lower"] < y_te))
        self.assertTrue(all(res["y_upper"] > y_te))

if __name__ == "__main__":
    unittest.main()

"""
AGRI-FUSE Neural Estimator Wrapper for AF-HE Architecture.
Provides unified fit/predict/predict_proba interface for the PyTorch Gated Multimodal Network.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from src.fusion.agri_fuse_model import AgriFuseModel
from src.training.trainer import AgriFuseTrainer

class AgriFuseEstimator:
    def __init__(
        self,
        soil_dim=4,
        env_dim=3,
        iot_dim=1,
        context_dim=1,
        satellite_dim=1,
        latent_dim=128,
        fusion_type="gated",
        task_type="classification",
        num_classes=22,
        lr=0.0003,
        max_epochs=60,
        patience=15,
        random_state=42
    ):
        self.soil_dim = soil_dim
        self.env_dim = env_dim
        self.iot_dim = iot_dim
        self.context_dim = context_dim
        self.satellite_dim = satellite_dim
        self.latent_dim = latent_dim
        self.fusion_type = fusion_type
        self.task_type = task_type
        self.num_classes = num_classes
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        
        torch.manual_seed(random_state)
        self.model = AgriFuseModel(
            soil_dim=soil_dim,
            env_dim=env_dim,
            iot_dim=iot_dim,
            context_dim=context_dim,
            satellite_dim=satellite_dim,
            latent_dim=latent_dim,
            fusion_type=fusion_type,
            task_type=task_type,
            num_classes=num_classes,
            dropout_rate=0.2
        )
        self.trainer = AgriFuseTrainer(
            self.model,
            task_type=task_type,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience
        )
        
    def _format_modalities(self, data):
        if isinstance(data, dict):
            return data
        N = len(data)
        return {
            "soil": data.astype(np.float32),
            "env": np.zeros((N, self.env_dim), dtype=np.float32),
            "iot": np.zeros((N, self.iot_dim), dtype=np.float32),
            "context": np.zeros((N, self.context_dim), dtype=np.float32),
            "satellite": np.zeros((N, self.satellite_dim), dtype=np.float32),
            "mask": np.tile([1.0, 0.0, 0.0, 0.0, 0.0], (N, 1)).astype(np.float32)
        }

    def fit(self, train_modalities, y_train, val_modalities=None, y_val=None):
        train_modalities = self._format_modalities(train_modalities)
        if val_modalities is None or y_val is None:
            val_modalities, y_val = train_modalities, y_train
        else:
            val_modalities = self._format_modalities(val_modalities)
        return self.trainer.fit(train_modalities, y_train, val_modalities, y_val)

    def predict(self, test_modalities):
        test_modalities = self._format_modalities(test_modalities)
        metrics = self.trainer.evaluate(test_modalities, np.zeros(len(test_modalities["soil"])))
        return metrics["y_pred"]

    def predict_proba(self, test_modalities):
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification.")
        test_modalities = self._format_modalities(test_modalities)
        test_loader = self.trainer._create_loader(test_modalities, np.zeros(len(test_modalities["soil"])), shuffle=False)
        self.model.eval()
        probs_list = []
        with torch.no_grad():
            for b_soil, b_env, b_iot, b_context, b_sat, b_mask, _ in test_loader:
                b_soil = b_soil.to(self.trainer.device)
                b_env = b_env.to(self.trainer.device)
                b_iot = b_iot.to(self.trainer.device)
                b_context = b_context.to(self.trainer.device)
                b_sat = b_sat.to(self.trainer.device)
                b_mask = b_mask.to(self.trainer.device)
                
                out, _ = self.model(b_soil, b_env, b_iot, b_context, b_sat, b_mask)
                probs = F.softmax(out, dim=1).cpu().numpy()
                probs_list.append(probs)
        return np.concatenate(probs_list, axis=0)

    def get_gate_weights(self, test_modalities):
        test_loader = self.trainer._create_loader(test_modalities, np.zeros(len(test_modalities["soil"])), shuffle=False)
        self.model.eval()
        gates_list = []
        with torch.no_grad():
            for b_soil, b_env, b_iot, b_context, b_sat, b_mask, _ in test_loader:
                b_soil = b_soil.to(self.trainer.device)
                b_env = b_env.to(self.trainer.device)
                b_iot = b_iot.to(self.trainer.device)
                b_context = b_context.to(self.trainer.device)
                b_sat = b_sat.to(self.trainer.device)
                b_mask = b_mask.to(self.trainer.device)
                
                _, gates = self.model(b_soil, b_env, b_iot, b_context, b_sat, b_mask)
                gates_list.append(gates.cpu().numpy())
        return np.concatenate(gates_list, axis=0)

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.trainer.device))

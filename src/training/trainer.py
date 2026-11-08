"""
PyTorch Training and Checkpointing Engine for AGRI-FUSE Research Framework.
Implements AdamW, learning rate scheduling, gradient clipping, early stopping,
and validation evaluation on Apple MPS / CPU.
"""

import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.evaluation.metrics import (
    compute_classification_metrics, compute_regression_metrics
)

def get_device(prefer_mps=False):
    """Returns CPU by default for maximum OpenMP compatibility on macOS, or MPS if requested."""
    if prefer_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

class AgriFuseTrainer:
    """End-to-end trainer for AGRI-FUSE multimodal neural network."""
    
    def __init__(
        self,
        model,
        task_type="classification",
        lr=0.0003,
        weight_decay=0.0001,
        max_epochs=100,
        patience=15,
        grad_clip=1.0,
        batch_size=32,
        device=None
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.task_type = task_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        if task_type == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.SmoothL1Loss() # Robust Huber loss
            
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.history = {"train_loss": [], "val_loss": [], "val_metric": []}
        
    def _create_loader(self, modality_dict, y, shuffle=True):
        t_soil = torch.tensor(modality_dict["soil"], dtype=torch.float32)
        t_env = torch.tensor(modality_dict["env"], dtype=torch.float32)
        t_iot = torch.tensor(modality_dict["iot"], dtype=torch.float32)
        t_context = torch.tensor(modality_dict["context"], dtype=torch.float32)
        t_sat = torch.tensor(modality_dict["satellite"], dtype=torch.float32)
        t_mask = torch.tensor(modality_dict["mask"], dtype=torch.float32)
        
        if self.task_type == "classification":
            t_y = torch.tensor(y, dtype=torch.long)
        else:
            t_y = torch.tensor(y, dtype=torch.float32)
            
        ds = TensorDataset(t_soil, t_env, t_iot, t_context, t_sat, t_mask, t_y)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, train_modalities, y_train, val_modalities, y_val):
        train_loader = self._create_loader(train_modalities, y_train, shuffle=True)
        val_loader = self._create_loader(val_modalities, y_val, shuffle=False)
        
        start_t = time.perf_counter()
        patience_counter = 0
        
        for epoch in range(1, self.max_epochs + 1):
            # Training phase
            self.model.train()
            train_losses = []
            
            for b_soil, b_env, b_iot, b_context, b_sat, b_mask, b_y in train_loader:
                b_soil = b_soil.to(self.device)
                b_env = b_env.to(self.device)
                b_iot = b_iot.to(self.device)
                b_context = b_context.to(self.device)
                b_sat = b_sat.to(self.device)
                b_mask = b_mask.to(self.device)
                b_y = b_y.to(self.device)
                
                self.optimizer.zero_grad()
                out, _ = self.model(b_soil, b_env, b_iot, b_context, b_sat, b_mask)
                loss = self.criterion(out, b_y)
                loss.backward()
                
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                self.optimizer.step()
                train_losses.append(loss.item())
                
            mean_train_loss = float(np.mean(train_losses))
            
            # Validation phase
            self.model.eval()
            val_losses = []
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for b_soil, b_env, b_iot, b_context, b_sat, b_mask, b_y in val_loader:
                    b_soil = b_soil.to(self.device)
                    b_env = b_env.to(self.device)
                    b_iot = b_iot.to(self.device)
                    b_context = b_context.to(self.device)
                    b_sat = b_sat.to(self.device)
                    b_mask = b_mask.to(self.device)
                    b_y = b_y.to(self.device)
                    
                    out, _ = self.model(b_soil, b_env, b_iot, b_context, b_sat, b_mask)
                    v_loss = self.criterion(out, b_y)
                    val_losses.append(v_loss.item())
                    
                    if self.task_type == "classification":
                        preds = torch.argmax(out, dim=1).cpu().numpy()
                    else:
                        preds = out.cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(b_y.cpu().numpy())
                    
            mean_val_loss = float(np.mean(val_losses))
            self.scheduler.step(mean_val_loss)
            
            if self.task_type == "classification":
                val_score = float(np.mean(np.array(val_preds) == np.array(val_targets)))
            else:
                val_score = float(np.sqrt(np.mean((np.array(val_preds) - np.array(val_targets))**2)))
                
            self.history["train_loss"].append(mean_train_loss)
            self.history["val_loss"].append(mean_val_loss)
            self.history["val_metric"].append(val_score)
            
            # Early stopping check
            if mean_val_loss < self.best_val_loss:
                self.best_val_loss = mean_val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
                    
        total_train_time = round(time.perf_counter() - start_t, 3)
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        return {
            "training_time_sec": total_train_time,
            "best_val_loss": round(self.best_val_loss, 4),
            "epochs_trained": len(self.history["train_loss"])
        }

    def evaluate(self, test_modalities, y_test):
        """Evaluates on test partition and records predictions and gate weights."""
        test_loader = self._create_loader(test_modalities, y_test, shuffle=False)
        self.model.eval()
        
        all_preds = []
        all_gate_weights = []
        
        start_inf = time.perf_counter()
        with torch.no_grad():
            for b_soil, b_env, b_iot, b_context, b_sat, b_mask, _ in test_loader:
                b_soil = b_soil.to(self.device)
                b_env = b_env.to(self.device)
                b_iot = b_iot.to(self.device)
                b_context = b_context.to(self.device)
                b_sat = b_sat.to(self.device)
                b_mask = b_mask.to(self.device)
                
                out, gates = self.model(b_soil, b_env, b_iot, b_context, b_sat, b_mask)
                
                if self.task_type == "classification":
                    preds = torch.argmax(out, dim=1).cpu().numpy()
                else:
                    preds = out.cpu().numpy()
                    
                all_preds.extend(preds)
                all_gate_weights.append(gates.cpu().numpy())
                
        inf_time = time.perf_counter() - start_inf
        latency_ms = round((inf_time / len(y_test)) * 1000.0, 3)
        
        y_pred = np.array(all_preds)
        gate_matrix = np.concatenate(all_gate_weights, axis=0)
        
        if self.task_type == "classification":
            metrics = compute_classification_metrics(y_test, y_pred)
        else:
            metrics = compute_regression_metrics(y_test, y_pred)
            
        metrics["model_name"] = f"AGRI-FUSE ({self.model.fusion_type})"
        metrics["latency_ms_per_sample"] = latency_ms
        metrics["y_pred"] = y_pred
        metrics["gate_weights"] = gate_matrix
        
        return metrics

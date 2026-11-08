"""
Complete AGRI-FUSE Neural Architecture Model Definition.
Wires specialized encoders, availability-aware gated fusion unit, and task-specific heads.
Supports ablation modes (concat, mean, single-modality).
"""

import torch
import torch.nn as nn
from src.encoders.specialized_encoders import (
    SoilEncoder, EnvironmentalEncoder, IoTEncoder, AgriculturalContextEncoder, SatelliteIndexEncoder
)
from src.fusion.gated_fusion import (
    AvailabilityAwareGatedFusion, ConcatenationFusion, MeanFusion
)
from src.heads.prediction_heads import (
    CropRecommendationHead, CropTypeHead, YieldRegressionHead, SoilSuitabilityHead
)

class AgriFuseModel(nn.Module):
    """
    AGRI-FUSE: Adaptive Gated Multi-Dataset Agricultural Intelligence Framework.
    
    Architecture:
        x_soil     -> SoilEncoder             \
        x_env      -> EnvironmentalEncoder     \
        x_iot      -> IoTEncoder                -> AvailabilityAwareGatedFusion -> z (128-d) -> TaskHead
        x_context  -> AgriContextEncoder       /
        x_sat      -> SatelliteIndexEncoder   /
    """
    def __init__(
        self,
        soil_dim=4,
        env_dim=3,
        iot_dim=1,
        context_dim=1,
        satellite_dim=1,
        latent_dim=128,
        fusion_type="gated", # "gated", "concat", "mean", "soil_only", "env_only", "satellite_only"
        task_type="classification", # "classification" or "regression"
        num_classes=22,
        dropout_rate=0.2
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.task_type = task_type
        self.latent_dim = latent_dim
        
        # 1. Specialized Encoders
        self.soil_encoder = SoilEncoder(soil_dim, latent_dim, dropout_rate)
        self.env_encoder = EnvironmentalEncoder(env_dim, latent_dim, dropout_rate)
        self.iot_encoder = IoTEncoder(iot_dim, latent_dim, dropout_rate)
        self.context_encoder = AgriculturalContextEncoder(context_dim, latent_dim, dropout_rate)
        self.satellite_encoder = SatelliteIndexEncoder(satellite_dim, latent_dim, dropout_rate)
        
        # 2. Multimodal Fusion Unit
        if fusion_type == "gated":
            self.fusion_unit = AvailabilityAwareGatedFusion(num_modalities=5, latent_dim=latent_dim)
        elif fusion_type == "concat":
            self.fusion_unit = ConcatenationFusion(num_modalities=5, latent_dim=latent_dim)
        elif fusion_type == "mean":
            self.fusion_unit = MeanFusion(num_modalities=5, latent_dim=latent_dim)
        else: # Single modality ablation pass
            self.fusion_unit = None
            
        # 3. Task-Specific Prediction Head
        if task_type == "classification":
            self.head = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, num_classes)
            )
        elif task_type == "regression":
            self.head = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1)
            )
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

    def forward(self, soil_x, env_x, iot_x, context_x, sat_x, mask):
        """
        Forward pass through AGRI-FUSE architecture.
        
        Returns:
            output: Predictions (logits or continuous scalar)
            gate_weights: Availability-aware gating weights [batch, 5]
        """
        # Encode modalities
        h_soil = self.soil_encoder(soil_x)
        h_env = self.env_encoder(env_x)
        h_iot = self.iot_encoder(iot_x)
        h_context = self.context_encoder(context_x)
        h_sat = self.satellite_encoder(sat_x)
        
        reps = [h_soil, h_env, h_iot, h_context, h_sat]
        
        # Single-modality ablations
        if self.fusion_type == "soil_only":
            z = h_soil
            gate_weights = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], device=soil_x.device).repeat(len(soil_x), 1)
        elif self.fusion_type == "env_only":
            z = h_env
            gate_weights = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], device=soil_x.device).repeat(len(soil_x), 1)
        elif self.fusion_type == "satellite_only":
            z = h_sat
            gate_weights = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], device=soil_x.device).repeat(len(soil_x), 1)
        else:
            z, gate_weights = self.fusion_unit(reps, mask)
            
        out = self.head(z)
        if self.task_type == "regression":
            out = out.squeeze(-1)
            
        return out, gate_weights


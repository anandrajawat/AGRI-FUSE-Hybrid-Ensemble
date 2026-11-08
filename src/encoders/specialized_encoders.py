"""
Specialized Encoders for AGRI-FUSE Multimodal Agricultural Architecture.
Implements modular sub-networks for Soil, Environmental, IoT Sensors, Context, and Satellite Remote Sensing.
All encoders project their specific inputs into a uniform latent dimension (d_latent = 128).
"""

import torch
import torch.nn as nn

class SoilEncoder(nn.Module):
    """Encodes soil nutrient & physicochemical properties (N, P, K, pH, EC, Organic Matter, etc.)"""
    def __init__(self, in_features, latent_dim=128, dropout_rate=0.2):
        super().__init__()
        self.in_features = in_features
        hidden_dim = max(64, latent_dim // 2)
        
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.net(x)


class EnvironmentalEncoder(nn.Module):
    """Encodes meteorological & environmental variables (temperature, humidity, rainfall, sunlight)."""
    def __init__(self, in_features, latent_dim=128, dropout_rate=0.2):
        super().__init__()
        self.in_features = in_features
        hidden_dim = max(64, latent_dim // 2)
        
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.net(x)


class IoTEncoder(nn.Module):
    """Encodes telemetry from IoT sensor nodes and farm management interventions."""
    def __init__(self, in_features, latent_dim=128, dropout_rate=0.2):
        super().__init__()
        self.in_features = in_features
        hidden_dim = max(64, latent_dim // 2)
        
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.net(x)


class AgriculturalContextEncoder(nn.Module):
    """Encodes categorical regional, seasonal, crop, and variety metadata."""
    def __init__(self, in_features, latent_dim=128, dropout_rate=0.2):
        super().__init__()
        self.in_features = in_features
        hidden_dim = max(64, latent_dim // 2)
        
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.net(x)


class SatelliteIndexEncoder(nn.Module):
    """Encodes remote sensing vegetation indices (NDVI, GNDVI, NDWI, SAVI)."""
    def __init__(self, in_features, latent_dim=128, dropout_rate=0.2):
        super().__init__()
        self.in_features = in_features
        hidden_dim = max(32, latent_dim // 4)
        
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.net(x)


class TemporalEncoder(nn.Module):
    """Encodes multi-year or sequential seasonal observations."""
    def __init__(self, in_features, latent_dim=128, dropout_rate=0.2):
        super().__init__()
        self.in_features = in_features
        self.net = nn.Sequential(
            nn.Linear(in_features, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        return self.net(x)


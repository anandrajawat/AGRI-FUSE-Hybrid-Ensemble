"""
Task-Specific Prediction Heads for AGRI-FUSE Architecture.
Specialized output modules for Crop Recommendation, Crop-Type Classification,
and Field/Regional Crop Yield Regression.
"""

import torch
import torch.nn as nn

class CropRecommendationHead(nn.Module):
    """Multi-class classification head for agronomic crop recommendation."""
    def __init__(self, in_features=128, num_classes=22, dropout_rate=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, z):
        return self.classifier(z)


class CropTypeHead(nn.Module):
    """Multi-class classification head for farm crop-type identification."""
    def __init__(self, in_features=128, num_classes=5, dropout_rate=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, z):
        return self.classifier(z)


class YieldRegressionHead(nn.Module):
    """Continuous regression head for field-level or regional crop yield prediction."""
    def __init__(self, in_features=128, dropout_rate=0.2):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, z):
        return self.regressor(z).squeeze(-1)


class SoilSuitabilityHead(nn.Module):
    """Classification head for laboratory soil nutrient suitability profiling."""
    def __init__(self, in_features=128, num_classes=10, dropout_rate=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, z):
        return self.classifier(z)


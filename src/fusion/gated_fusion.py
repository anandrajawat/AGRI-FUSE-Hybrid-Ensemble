"""
Adaptive Gated Multimodal Fusion Unit for AGRI-FUSE Research Framework.
Implements availability-aware gating scores, softmax normalization across active modalities,
and provides ablation fusion mechanisms (concatenation, mean, single-modality).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AvailabilityAwareGatedFusion(nn.Module):
    """
    Availability-Aware Gated Multimodal Unit (GMU).
    Computes modality importance weights alpha_i conditioned on modality state and availability mask m_i.
    
    Formula:
        g_i = W_i * h_i + b_i
        alpha_i = (m_i * exp(g_i)) / sum_j(m_j * exp(g_j) + 1e-8)
        z = sum_i(alpha_i * h_i)
    """
    def __init__(self, num_modalities=5, latent_dim=128):
        super().__init__()
        self.num_modalities = num_modalities
        self.latent_dim = latent_dim
        
        # Scoring layers for each modality: R^latent_dim -> R^1
        self.gate_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ) for _ in range(num_modalities)
        ])
        
        # Final projection layer to preserve 128-dim representation
        self.output_projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
        # Cache for last forward pass gate weights (for interpretability & visualization)
        self.last_gate_weights = None
        
    def forward(self, modality_representations, availability_mask):
        """
        Args:
            modality_representations: List of K tensors, each shape [batch_size, latent_dim]
            availability_mask: Tensor of shape [batch_size, K], where mask[b, i] == 1 if modality i is present, 0 otherwise.
            
        Returns:
            fused_representation: Tensor of shape [batch_size, latent_dim]
            gate_weights: Tensor of shape [batch_size, K]
        """
        batch_size = modality_representations[0].shape[0]
        device = modality_representations[0].device
        
        # Compute raw gating scalar score g_i for each modality
        raw_scores = []
        for i, h_i in enumerate(modality_representations):
            score_i = self.gate_projections[i](h_i) # [batch_size, 1]
            raw_scores.append(score_i)
            
        # Shape: [batch_size, num_modalities]
        g_matrix = torch.cat(raw_scores, dim=1)
        
        # Apply availability masking: set masked modalities to very large negative number before softmax
        # If m_i == 0, score -> -1e9 so exp(score) -> 0
        mask = availability_mask.to(device)
        masked_scores = g_matrix.masked_fill(mask == 0, -1e9)
        
        # Softmax over available modalities
        alpha = F.softmax(masked_scores, dim=1) # [batch_size, num_modalities]
        self.last_gate_weights = alpha.detach().cpu()
        
        # Weighted sum of representations: z = sum_i(alpha_i * h_i)
        # Stack representations: [batch_size, num_modalities, latent_dim]
        stacked_h = torch.stack(modality_representations, dim=1)
        
        # Reshape alpha for broadcast: [batch_size, num_modalities, 1]
        alpha_expanded = alpha.unsqueeze(-1)
        
        fused = torch.sum(alpha_expanded * stacked_h, dim=1) # [batch_size, latent_dim]
        z = self.output_projection(fused)
        
        return z, alpha


class ConcatenationFusion(nn.Module):
    """Ablation Baseline: Simple concatenation followed by linear compression."""
    def __init__(self, num_modalities=5, latent_dim=128):
        super().__init__()
        self.num_modalities = num_modalities
        self.latent_dim = latent_dim
        
        self.compressor = nn.Sequential(
            nn.Linear(num_modalities * latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
    def forward(self, modality_representations, availability_mask):
        device = modality_representations[0].device
        mask = availability_mask.to(device).unsqueeze(-1) # [batch, num_modalities, 1]
        
        # Zero out unavailable representations
        stacked_h = torch.stack(modality_representations, dim=1) * mask
        cat_h = stacked_h.view(stacked_h.shape[0], -1)
        z = self.compressor(cat_h)
        
        dummy_weights = availability_mask / (availability_mask.sum(dim=1, keepdim=True) + 1e-8)
        return z, dummy_weights


class MeanFusion(nn.Module):
    """Ablation Baseline: Unweighted average across available modalities."""
    def __init__(self, num_modalities=5, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
    def forward(self, modality_representations, availability_mask):
        device = modality_representations[0].device
        mask = availability_mask.to(device).unsqueeze(-1)
        stacked_h = torch.stack(modality_representations, dim=1) * mask
        
        counts = mask.sum(dim=1).clamp(min=1.0)
        mean_h = stacked_h.sum(dim=1) / counts
        z = self.output_projection(mean_h)
        
        dummy_weights = availability_mask / (availability_mask.sum(dim=1, keepdim=True) + 1e-8)
        return z, dummy_weights


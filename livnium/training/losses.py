"""
Training Losses: Loss Functions

Loss/reward calculations live here, not in kernel.
These observe physics but do not modify it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LivniumLoss(nn.Module):
    """
    LIVNIUM loss function.
    
    Combines:
    - Classification loss (cross-entropy)
    - Negative energy term (encourages low tension)
    - Norm regularization
    
    All loss calculations happen here, not in kernel.
    """
    
    def __init__(
        self,
        d_margin: float = 0.4,
        neg_weight: float = 5.0,
        norm_reg_weight: float = 1e-4,
    ):
        """
        Initialize LIVNIUM loss.
        
        Args:
            d_margin: Margin for negative energy term
            neg_weight: Weight for negative energy term
            norm_reg_weight: Weight for norm regularization
        """
        super().__init__()
        self.d_margin = d_margin
        self.neg_weight = neg_weight
        self.norm_reg_weight = norm_reg_weight
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        divergence: Optional[torch.Tensor] = None,
        state_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute LIVNIUM loss.
        
        Args:
            logits: Model logits [B, num_classes]
            labels: Ground truth labels [B]
            divergence: Optional divergence values [B] (for negative energy term)
            state_norm: Optional state norm [B] (for regularization)
            
        Returns:
            Total loss
        """
        # Classification loss
        cls_loss = self.ce_loss(logits, labels)
        
        total_loss = cls_loss
        
        # Negative energy term (encourages low tension when divergence < 0)
        if divergence is not None:
            # Negative energy: encourage negative divergence (entailment-like)
            neg_energy = F.relu(self.d_margin - divergence)
            neg_loss = self.neg_weight * neg_energy.mean()
            total_loss = total_loss + neg_loss
        
        # Norm regularization
        if state_norm is not None:
            norm_loss = self.norm_reg_weight * state_norm.mean()
            total_loss = total_loss + norm_loss
        
        return total_loss


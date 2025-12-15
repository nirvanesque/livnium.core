"""
Ramsey Domain Head: Constraint Satisfaction Head

Takes collapsed state and outputs validity/constraint satisfaction.
Uses kernel.physics for constraint calculations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RamseyHead(nn.Module):
    """
    Ramsey constraint satisfaction head.
    
    Takes collapsed state and outputs:
    - Validity logits (valid/invalid coloring)
    - Constraint violation score
    - Clique detection signal
    
    Uses kernel.physics for constraint calculations.
    """
    
    def __init__(self, dim: int, k: int):
        """
        Initialize Ramsey head.
        
        Args:
            dim: Dimension of input state vector
            k: Clique size to avoid (monochromatic k-clique)
        """
        super().__init__()
        self.dim = dim
        self.k = k
        
        # Linear stack for validity classification
        # Features: [h_final, energy, constraint_violation, ...]
        # Total: dim + 2 features
        self.fc_validity = nn.Sequential(
            nn.Linear(dim + 2, dim + 2),
            nn.ReLU(),
            nn.Linear(dim + 2, 2)  # valid/invalid
        )
        
        # Regression head for constraint violation score
        self.fc_violation = nn.Sequential(
            nn.Linear(dim + 2, dim + 2),
            nn.ReLU(),
            nn.Linear(dim + 2, 1)  # violation score
        )
    
    def forward(
        self,
        h_final: torch.Tensor,
        v_coloring: torch.Tensor,
        v_coloring_dup: torch.Tensor  # Same as v_coloring for Ramsey
    ) -> dict:
        """
        Forward pass: state → validity and violation scores.
        
        Uses kernel.physics for constraint calculations.
        
        Args:
            h_final: Collapsed state vector [batch, dim] or [dim]
            v_coloring: Encoded coloring vector [batch, dim] or [dim]
            v_coloring_dup: Duplicate (same as v_coloring) [batch, dim] or [dim]
            
        Returns:
            Dictionary with:
            - 'validity_logits': [batch, 2] logits for valid/invalid
            - 'violation_score': [batch, 1] constraint violation score
        """
        # Ensure batch dimension
        squeeze = False
        if h_final.dim() == 1:
            h_final = h_final.unsqueeze(0)
            v_coloring = v_coloring.unsqueeze(0)
            v_coloring_dup = v_coloring_dup.unsqueeze(0)
            squeeze = True
        
        # Compute energy and constraint features
        energy = h_final.norm(p=2, dim=-1, keepdim=True)
        
        # Constraint violation: distance from valid state
        # This is a learned feature, but we can also compute geometric distance
        constraint_violation = (h_final - v_coloring).norm(p=2, dim=-1, keepdim=True)
        
        # Feature set
        features = torch.cat([
            h_final,
            energy,
            constraint_violation
        ], dim=-1)
        
        # Validity classification
        validity_logits = self.fc_validity(features)
        
        # Constraint violation score
        violation_score = self.fc_violation(features)
        
        if squeeze:
            validity_logits = validity_logits.squeeze(0)
            violation_score = violation_score.squeeze(0)
        
        return {
            'validity_logits': validity_logits,
            'violation_score': violation_score,
        }
    
    def is_valid(self, validity_logits: torch.Tensor) -> torch.Tensor:
        """
        Determine if coloring is valid from logits.
        
        Args:
            validity_logits: Logits [batch, 2] or [2]
            
        Returns:
            Boolean tensor [batch] or scalar
        """
        probs = F.softmax(validity_logits, dim=-1)
        return probs[..., 0] > 0.5  # Valid class is first


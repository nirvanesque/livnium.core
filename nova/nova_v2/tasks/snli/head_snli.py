"""
SNLI Head: Classification Head

Takes collapsed state h_final and outputs logits for E/N/C.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SNLIHead(nn.Module):
    """
    SNLI classification head.
    
    Adds explicit directional signals:
    - alignment between premise (OM) and hypothesis (LO)
    - opposition between -premise and hypothesis
    
    Final features: [h_final, alignment, opposition] → logits (E, N, C)
    """
    
    def __init__(self, dim: int):
        """
        Initialize SNLI head.
        
        Args:
            dim: Dimension of input state vector
        """
        super().__init__()
        # Linear layer: (dim + 2) → 3 (entailment, neutral, contradiction)
        self.fc = nn.Linear(dim + 2, 3)
    
    def forward(self, h_final: torch.Tensor, v_p: torch.Tensor, v_h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state → logits.
        
        Args:
            h_final: Collapsed state vector [batch, dim] or [dim]
            v_p: Premise vector (OM) [batch, dim] or [dim]
            v_h: Hypothesis vector (LO) [batch, dim] or [dim]
            
        Returns:
            Logits tensor (batch, 3) for [entailment, neutral, contradiction]
        """
        # Ensure batch dimension
        if h_final.dim() == 1:
            h_final = h_final.unsqueeze(0)
            v_p = v_p.unsqueeze(0)
            v_h = v_h.unsqueeze(0)
        
        # Normalize OM/LO
        v_p_n = F.normalize(v_p, dim=-1)
        v_h_n = F.normalize(v_h, dim=-1)
        
        # Alignment and opposition signals
        align = (v_p_n * v_h_n).sum(dim=-1, keepdim=True)  # cos(OM, LO)
        opp = (-v_p_n * v_h_n).sum(dim=-1, keepdim=True)   # cos(-OM, LO)
        
        features = torch.cat([h_final, align, opp], dim=-1)
        return self.fc(features)

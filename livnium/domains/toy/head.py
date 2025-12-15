"""
Toy Domain Head: Simple Output Head

Takes collapsed state and produces simple output (classification or regression).
"""

import torch
import torch.nn as nn


class ToyHead(nn.Module):
    """
    Simple head for toy domain.
    
    Takes collapsed state and produces output.
    """
    
    def __init__(self, dim: int = 64, num_classes: int = 3):
        """
        Initialize toy head.
        
        Args:
            dim: Dimension of input state vector
            num_classes: Number of output classes
        """
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        
        # Simple linear classifier
        self.fc = nn.Linear(dim, num_classes)
    
    def forward(
        self,
        h_final: torch.Tensor,
        v_a: torch.Tensor,
        v_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: state -> logits.
        
        Args:
            h_final: Collapsed state vector [B, dim] or [dim]
            v_a: Encoded first input [B, dim] or [dim]
            v_b: Encoded second input [B, dim] or [dim]
            
        Returns:
            Logits [B, num_classes] or [num_classes]
        """
        # Simple: just use final state
        # Real domains might use v_a, v_b for additional features
        logits = self.fc(h_final)
        return logits


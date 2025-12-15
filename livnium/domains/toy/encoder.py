"""
Toy Domain Encoder: Simple Synthetic Constraint Generator

Creates synthetic state vectors for testing kernel+engine integration.
No tokenization, no real data - just simple geometric constraints.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ToyEncoder(nn.Module):
    """
    Simple encoder for toy domain.
    
    Takes a simple input (integer or pair) and produces initial state vector.
    Uses kernel physics for constraint generation.
    """
    
    def __init__(self, dim: int = 64):
        """
        Initialize toy encoder.
        
        Args:
            dim: Dimension of state vectors
        """
        super().__init__()
        self.dim = dim
        
        # Simple learned projection
        self.proj = nn.Linear(2, dim)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to state vector.
        
        Args:
            x: Input tensor [B, 2] or [2] - simple 2D input
            
        Returns:
            State vector [B, dim] or [dim]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        h = self.proj(x)
        
        # Add small noise for symmetry breaking (use config default)
        from livnium.engine.config import defaults
        h = h + defaults.EPS_NOISE * torch.randn_like(h)
        
        if x.dim() == 1:
            h = h.squeeze(0)
        
        return h
    
    def generate_constraints(self, state: torch.Tensor) -> dict:
        """
        Generate constraints from state.
        
        For toy domain, this is minimal - just returns state info.
        Real domains would generate domain-specific constraints here.
        
        Args:
            state: State vector
            
        Returns:
            Dictionary of constraints
        """
        return {
            "state": state,
            "norm": torch.norm(state, p=2),
        }
    
    def build_initial_state(
        self,
        input_a: torch.Tensor,
        input_b: torch.Tensor,
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build initial state from two inputs.
        
        Args:
            input_a: First input [B, 2] or [2]
            input_b: Second input [B, 2] or [2]
            add_noise: Whether to add symmetry-breaking noise
            
        Returns:
            Tuple of (initial_state, encoded_a, encoded_b)
        """
        v_a = self.encode(input_a)
        v_b = self.encode(input_b)
        
        h0 = v_a + v_b
        if add_noise:
            from livnium.engine.config import defaults
            h0 = h0 + defaults.EPS_NOISE * torch.randn_like(h0)
        
        return h0, v_a, v_b


"""
Physics Embed Encoder

Simple embedding lookup that serves as the "Encoder" for this domain.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

class PhysicsEmbeddingModel(nn.Module):
    """
    Physics embedding model.
    A simple wrapper around nn.Embedding to be consistent with Domain structure.
    """
    
    def __init__(self, vocab_size: int, dim: int = 256, pad_idx: int = 0):
        from livnium.engine.config import defaults
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=pad_idx)
        nn.init.normal_(self.emb.weight, mean=0.0, std=defaults.PHYSICS_INIT_STD)
    
    def forward(self, idxs: torch.Tensor) -> torch.Tensor:
        """Forward pass (lookup)."""
        return self.emb(idxs)
    
    def build_initial_state(self, idxs: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """
        Build initial state h0.
        
        Args:
            idxs: Token indices [B] or [B, T]
            
        Returns:
            Tuple (h0, None, None) - simplified for this domain
        """
        return self.emb(idxs), None, None

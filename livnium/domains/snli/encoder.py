"""
SNLI Encoder: Premise/Hypothesis → Initial State

Builds initial state h0 from premise and hypothesis.
Can use any text encoder (quantum embeddings, BERT, etc.).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class SNLIEncoder(nn.Module):
    """
    SNLI encoder that converts premise/hypothesis to initial state.
    
    Uses kernel physics for constraint generation.
    Can work with any text encoder backend.
    """
    
    def __init__(
        self,
        text_encoder: Optional[nn.Module] = None,
        dim: int = 256,
        vocab_size: int = 2000,
        use_mlp: bool = False
    ):
        """
        Initialize SNLI encoder.
        
        Args:
            text_encoder: Optional text encoder module (if None, creates simple one)
            dim: Dimension of state vectors
            vocab_size: Vocabulary size for embedding layer
            use_mlp: Whether to use MLP transformation on encoded vectors
        """
        super().__init__()
        self.dim = dim
        
        if text_encoder is not None:
            self.text_encoder = text_encoder
            self.dim = getattr(text_encoder, 'dim', dim)
        else:
            # Simple fallback encoder (for testing)
            self.text_encoder = None
            self.embedding = nn.Embedding(vocab_size, dim)
        
        self.use_mlp = use_mlp
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(self.dim, 2 * self.dim),
                nn.GELU(),
                nn.Linear(2 * self.dim, self.dim),
            )
        else:
            self.mlp = None
    
    def encode_sentence(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode sentence token IDs to vector.
        
        Args:
            token_ids: Token IDs [B, L] or [L]
            
        Returns:
            Sentence vector [B, dim] or [dim]
        """
        if self.text_encoder is not None:
            return self.text_encoder.encode_sentence(token_ids)
        else:
            # Simple fallback: mean pooling
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)
            emb = self.embedding(token_ids)  # [B, L, dim]
            v = emb.mean(dim=1)  # [B, dim]
            if token_ids.dim() == 1:
                v = v.squeeze(0)
            return v
    
    def generate_constraints(self, state: torch.Tensor, v_p: torch.Tensor, v_h: torch.Tensor) -> dict:
        """
        Generate constraints from state and premise/hypothesis vectors.
        
        Uses kernel.physics for alignment/divergence calculations.
        
        Args:
            state: Current state vector
            v_p: Premise vector (OM)
            v_h: Hypothesis vector (LO)
            
        Returns:
            Dictionary of constraints
        """
        # Import here to avoid circular dependencies
        from livnium.kernel.physics import alignment, divergence, tension
        from livnium.engine.ops_torch import TorchOps
        
        ops = TorchOps()
        
        # Create state wrappers for kernel physics
        class StateWrapper:
            def __init__(self, vec):
                self._vec = vec
            def vector(self):
                return self._vec
            def norm(self):
                return torch.norm(self._vec, p=2)
        
        p_state = StateWrapper(v_p)
        h_state = StateWrapper(v_h)
        
        # Compute physics using kernel
        align = alignment(ops, p_state, h_state)
        div = divergence(ops, p_state, h_state)
        tens = tension(ops, div)
        
        return {
            "alignment": align,
            "divergence": div,
            "tension": tens,
            "state": state,
        }
    
    def build_initial_state(
        self,
        prem_ids: torch.Tensor,
        hyp_ids: torch.Tensor,
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build initial state from premise and hypothesis.
        
        Args:
            prem_ids: Premise token IDs [B, L] or [L]
            hyp_ids: Hypothesis token IDs [B, L] or [L]
            add_noise: Whether to add symmetry-breaking noise
            
        Returns:
            Tuple of (initial_state, premise_vector, hypothesis_vector)
        """
        v_p = self.encode_sentence(prem_ids)
        v_h = self.encode_sentence(hyp_ids)
        
        # Apply MLP if enabled
        if self.mlp is not None:
            s_p = self.mlp(v_p)
            s_h = self.mlp(v_h)
            h0 = s_p + s_h
        else:
            h0 = v_p + v_h
        
        if add_noise:
            # Use config default for noise
            from livnium.engine.config import defaults
            h0 = h0 + defaults.EPS_NOISE * torch.randn_like(h0)
        
        return h0, v_p, v_h


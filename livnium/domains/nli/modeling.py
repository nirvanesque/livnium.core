"""
Shared NLI Modeling Components

Base classes for NLI Encoder and Head using kernel physics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class NLIEncoder(nn.Module):
    """
    Base NLI encoder that converts premise/hypothesis to initial state.
    
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


class NLIHead(nn.Module):
    """
    Base NLI classification head.
    
    Adds explicit directional and radial signals using kernel.physics.
    """
    
    def __init__(self, dim: int, num_classes: int = 3):
        super().__init__()
        self.dim = dim
        
        # Learned neutral anchor to give neutral its own geometric signal
        self.neutral_dir = nn.Parameter(torch.randn(dim))
        
        # Linear stack to allow mild feature interaction
        self.fc = nn.Sequential(
            nn.Linear(dim + 10, dim + 10),
            nn.ReLU(),
            nn.Linear(dim + 10, num_classes)
        )
        
        # Learnable scale for neutral alignment contribution
        self.neutral_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(
        self,
        h_final: torch.Tensor,
        v_p: torch.Tensor,
        v_h: torch.Tensor
    ) -> torch.Tensor:
        # Ensure batch dimension
        squeeze = False
        if h_final.dim() == 1:
            h_final = h_final.unsqueeze(0)
            v_p = v_p.unsqueeze(0)
            v_h = v_h.unsqueeze(0)
            squeeze = True
        
        # Normalize OM/LO
        v_p_n = F.normalize(v_p, dim=-1)
        v_h_n = F.normalize(v_h, dim=-1)
        neutral_dir_n = F.normalize(self.neutral_dir, dim=0)
        
        # Use kernel.physics for alignment
        from livnium.kernel.physics import alignment
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
        
        # Compute alignment using kernel.physics
        align_values = []
        for i in range(v_p_n.shape[0]):
            p_state = StateWrapper(v_p_n[i])
            h_state = StateWrapper(v_h_n[i])
            align_val = alignment(ops, p_state, h_state)
            align_values.append(torch.tensor(align_val, device=v_p_n.device, dtype=v_p_n.dtype))
        
        align = torch.stack(align_values).unsqueeze(-1)  # [B, 1]
        
        # Opposition signal
        opp = (-v_p_n * v_h_n).sum(dim=-1, keepdim=True)  # cos(-OM, LO)
        
        # Neutral alignments: how close each vector is to the neutral anchor
        align_neutral_p = (v_p_n * neutral_dir_n.unsqueeze(0)).sum(dim=-1, keepdim=True)
        align_neutral_h = (v_h_n * neutral_dir_n.unsqueeze(0)).sum(dim=-1, keepdim=True)
        align_neutral_p = self.neutral_scale * align_neutral_p
        align_neutral_h = self.neutral_scale * align_neutral_h
        
        # Exposure/energy from alignment: map [-1,1] → [0,1], then scale
        # Using K_O = 9 from kernel constants
        from livnium.kernel.constants import K_O
        energy = K_O * ((1 + align) / 2)
        expose_neg = (1 - align) / 2
        
        # Radial geometry: distance and norms
        dist_p_h = (v_h - v_p).norm(p=2, dim=-1, keepdim=True)
        r_p = v_p.norm(p=2, dim=-1, keepdim=True)
        r_h = v_h.norm(p=2, dim=-1, keepdim=True)
        r_final = h_final.norm(p=2, dim=-1, keepdim=True)
        
        # Feature set
        features = torch.cat([
            h_final,
            align,
            opp,
            energy,
            expose_neg,
            dist_p_h,
            r_p,
            r_h,
            r_final,
            align_neutral_p,
            align_neutral_h
        ], dim=-1)
        
        logits = self.fc(features)
        
        if squeeze:
            logits = logits.squeeze(0)
        
        return logits

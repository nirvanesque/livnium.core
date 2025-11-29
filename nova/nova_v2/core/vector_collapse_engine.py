"""
Vector Collapse Engine: Core Collapse Dynamics

Implements the collapse process that evolves a state vector h through L steps.
Each step applies physics laws (alignment, divergence, tension) to modify the state.

This is the heart of Livnium Core v1.0.

# Short design note for Codex:
# λ-free interaction: force = (0.38 - cosθ) * normalize(vj - vi)
# 0.38 is the semantic equilibrium; geometry itself sets the strength.

# === Multi-Vector Geometry: Design Note for Codex ===================================
# The collapse engine must support REAL geometry, not just a single vector vs anchor.
#
# Required behavior:
#   • Every vector interacts with every other vector using a pure geometric force.
#   • No λ, no tunable constants. The geometry ALONE sets the strength.
#
# Pairwise interaction law (Livnium Divergence Law):
#     align_ij = cosine_similarity(v_i, v_j)
#     d_ij = 0.38 - align_ij              # 0.38 is the semantic equilibrium
#     F_ij = d_ij * normalize(v_j - v_i)  # attraction / neutral / repulsion
#
# Update rule for ALL vectors:
#     v_i ← v_i + Σ_j F_ij
#
# Requirements:
#   • Must handle a batch of B vectors: shape [B, dim]
#   • Compute pairwise forces with broadcasting (no Python loops)
#   • Preserve norms softly (keep ||v|| ≤ 10)
#   • No overlap resolution → the force law itself produces separation (kissing behavior)
#
# Summary:
#   Add pairwise collapse: each vector evolves based on the 0.38 − cosθ force from all others.
#   This creates emergent geometry: clustering, repulsion, equilibrium shells, natural basins.
# =======================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .physics_laws import alignment, divergence_from_alignment, tension


class VectorCollapseEngine(nn.Module):
    """
    Core collapse engine for Livnium.
    
    Takes an initial state h0 and evolves it through L collapse steps.
    At each step, it:
    1. Computes alignment/divergence/tension
    2. Applies state update
    3. Logs trace
    
    The trace is what watchdogs inspect.
    """
    
    def __init__(self, dim: int = 256, num_layers: int = 6):
        """
        Initialize collapse engine.
        
        Args:
            dim: Dimension of state vector
            num_layers: Number of collapse steps
        """
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # State update network
        # This learns how to evolve the state based on current configuration
        self.update = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )
        
        # Core-internal direction (OM reference axis)
        # This is a learnable "anchor" direction in the vector space
        # Think of it as the "global meaning anchor" from the laws
        self.core_dir = nn.Parameter(torch.randn(dim))
    
    def collapse(self, h0: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Collapse initial state h0 through L steps.
        
        Args:
            h0: Initial state vector (dim,) or batch of vectors [B, dim]
            
        Returns:
            Tuple of (h_final, trace)
            trace: Dict with keys 'alignment', 'divergence', 'tension'
                   Each value is a list of tensors (one per step)
        """
        h = h0.clone()
        trace = {
            "alignment": [],
            "divergence": [],
            "tension": []
        }
        
        # Normalize core direction
        core_dir_n = F.normalize(self.core_dir, dim=0)
        
        for step in range(self.num_layers):
            # Normalize current state along feature dim
            h_n = F.normalize(h, dim=-1)
            
            # Compute physics
            a = (h_n * core_dir_n).sum(dim=-1)
            d = divergence_from_alignment(a)
            t = tension(d)
            
            # Log trace
            trace["alignment"].append(a.detach())
            trace["divergence"].append(d.detach())
            trace["tension"].append(t.detach())
            
            # State update
            delta = self.update(h)
            
            # Apply divergence-based correction
            # Negative divergence → pull toward core_dir
            # Positive divergence → push away from core_dir
            h = h + delta - 0.1 * d.unsqueeze(-1) * h_n
            
            # Soft norm control (conservation-ish)
            # Keep ||h|| roughly bounded to prevent explosion
            h_norm = h.norm(p=2, dim=-1, keepdim=True)
            h = torch.where(h_norm > 10.0, h * (10.0 / (h_norm + 1e-8)), h)
        
        return h, trace
    
    def forward(self, h0: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Forward pass (alias for collapse).
        
        Args:
            h0: Initial state vector
            
        Returns:
            Tuple of (h_final, trace)
        """
        return self.collapse(h0)

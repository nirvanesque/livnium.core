"""
Collapse Engine: Runtime Dynamics Using Kernel Physics

Evolves a state vector through L collapse steps with multiple anchors.
All physics calculations use kernel.physics.* with provided Ops instance.
All constants come from kernel.constants or engine.config.defaults.
"""

from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from livnium.kernel.physics import alignment, divergence, tension
from livnium.kernel.constants import DIVERGENCE_PIVOT
from livnium.engine.config import defaults
from livnium.engine.ops_torch import TorchOps


class CollapseEngine(nn.Module):
    """
    Core collapse engine for LIVNIUM.
    
    Takes an initial state h0 and evolves it through L collapse steps with
    multiple anchors (entailment/contradiction/neutral).
    
    At each step:
    1. Computes alignment/divergence/tension using kernel.physics
    2. Applies state update + anchor forces
    3. Logs trace
    
    All physics uses kernel.physics.* with Ops instance.
    All thresholds come from engine.config.defaults.
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 6,
        strength_entail: Optional[float] = None,
        strength_contra: Optional[float] = None,
        strength_neutral: Optional[float] = None,
        max_norm: Optional[float] = None,
    ):
        """
        Initialize collapse engine.
        
        Args:
            dim: Dimension of state vector
            num_layers: Number of collapse steps
            strength_entail: Force strength for entail anchor (defaults to config)
            strength_contra: Force strength for contradiction anchor (defaults to config)
            strength_neutral: Force strength for neutral anchor (defaults to config)
            max_norm: Maximum norm for clipping (defaults to config)
        """
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # Use config defaults if not specified
        self.strength_entail = strength_entail if strength_entail is not None else defaults.STRENGTH_ENTAIL
        self.strength_contra = strength_contra if strength_contra is not None else defaults.STRENGTH_CONTRA
        self.strength_neutral = strength_neutral if strength_neutral is not None else defaults.STRENGTH_NEUTRAL
        self.max_norm = max_norm if max_norm is not None else defaults.MAX_NORM
        
        # Ops instance for kernel physics
        self.ops = TorchOps()
        
        # State update network (learnable component)
        self.update = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )
        
        # Three anchors to create multi-basin geometry
        self.anchor_entail = nn.Parameter(torch.randn(dim))
        self.anchor_contra = nn.Parameter(torch.randn(dim))
        self.anchor_neutral = nn.Parameter(torch.randn(dim))
    
    def step(
        self,
        h: torch.Tensor,
        anchors: List[torch.Tensor],
        strengths: List[float]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Single collapse step.
        
        Args:
            h: Current state vector [B, dim] or [dim]
            anchors: List of anchor vectors (normalized)
            strengths: List of force strengths for each anchor
            
        Returns:
            Tuple of (updated_state, trace_dict)
        """
        squeeze = False
        if h.dim() == 1:
            h = h.unsqueeze(0)
            squeeze = True
        
        # Normalize current state
        h_n = F.normalize(h, dim=-1, p=2)
        
        trace = {
            "alignment": [],
            "divergence": [],
            "tension": [],
        }
        
        # Compute physics to each anchor using kernel
        total_force = torch.zeros_like(h)
        
        for anchor, strength in zip(anchors, strengths):
            # Create State-like objects for kernel physics
            # For now, we'll use a simple wrapper
            class StateWrapper:
                def __init__(self, vec):
                    self._vec = vec
                def vector(self):
                    return self._vec
                def norm(self):
                    return torch.norm(self._vec, p=2)
            
            h_state = StateWrapper(h_n)
            anchor_state = StateWrapper(anchor.unsqueeze(0) if anchor.dim() == 1 else anchor)
            
            # Use kernel physics
            align = alignment(self.ops, h_state, anchor_state)
            div = divergence(self.ops, h_state, anchor_state)
            tens = tension(self.ops, div)
            
            trace["alignment"].append(align)
            trace["divergence"].append(div)
            trace["tension"].append(tens)
            
            # Compute force direction
            anchor_expanded = anchor.unsqueeze(0) if anchor.dim() == 1 else anchor
            direction = F.normalize(h - anchor_expanded, dim=-1, p=2)
            
            # Apply force: -strength * divergence * direction
            force = -strength * div * direction
            total_force = total_force + force
        
        # State update
        delta = self.update(h)
        
        # Apply update and forces
        h_new = h + delta + total_force
        
        # Norm clipping (using config max_norm)
        h_norm = torch.norm(h_new, p=2, dim=-1, keepdim=True)
        h_new = torch.where(
            h_norm > self.max_norm,
            h_new * (self.max_norm / (h_norm + self.ops.eps())),
            h_new
        )
        
        if squeeze:
            h_new = h_new.squeeze(0)
        
        return h_new, trace
    
    def collapse(
        self,
        h0: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Collapse initial state h0 through L steps.
        
        Args:
            h0: Initial state vector [B, dim] or [dim]
            
        Returns:
            Tuple of (h_final, trace_dict)
        """
        h = h0.clone()
        
        # Normalize anchor directions
        anchors = [
            F.normalize(self.anchor_entail, dim=0),
            F.normalize(self.anchor_contra, dim=0),
            F.normalize(self.anchor_neutral, dim=0),
        ]
        
        strengths = [
            self.strength_entail,
            self.strength_contra,
            self.strength_neutral,
        ]
        
        full_trace = {
            "alignment_entail": [],
            "alignment_contra": [],
            "alignment_neutral": [],
            "divergence_entail": [],
            "divergence_contra": [],
            "divergence_neutral": [],
            "tension_entail": [],
            "tension_contra": [],
            "tension_neutral": [],
        }
        
        for step in range(self.num_layers):
            h, step_trace = self.step(h, anchors, strengths)
            
            # Store per-anchor traces
            if len(step_trace["alignment"]) == 3:
                full_trace["alignment_entail"].append(step_trace["alignment"][0])
                full_trace["alignment_contra"].append(step_trace["alignment"][1])
                full_trace["alignment_neutral"].append(step_trace["alignment"][2])
                full_trace["divergence_entail"].append(step_trace["divergence"][0])
                full_trace["divergence_contra"].append(step_trace["divergence"][1])
                full_trace["divergence_neutral"].append(step_trace["divergence"][2])
                full_trace["tension_entail"].append(step_trace["tension"][0])
                full_trace["tension_contra"].append(step_trace["tension"][1])
                full_trace["tension_neutral"].append(step_trace["tension"][2])
        
        return h, full_trace


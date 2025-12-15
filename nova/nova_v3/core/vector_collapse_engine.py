"""
Vector Collapse Engine: Multi-Basin Collapse Dynamics

Evolves a state vector h through L steps with multiple anchors (E/C/N) to
encourage three basins. Each anchor uses the Livnium divergence law.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .physics_laws import divergence_from_alignment, tension
from .basin_field import (
    BasinField,
    route_to_basin,
    update_basin_center,
    maybe_spawn_basin,
    prune_and_merge,
)


class VectorCollapseEngine(nn.Module):
    """
    Core collapse engine for Livnium.
    
    Takes an initial state h0 and evolves it through L collapse steps with
    multiple anchors (entailment/contradiction/neutral).
    At each step, it:
    1. Computes alignment/divergence/tension to each anchor
    2. Applies state update + anchor forces
    3. Logs trace
    
    The trace is what watchdogs inspect.
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 6,
        strength_entail: float = 0.1,
        strength_contra: float = 0.1,
        strength_neutral: float = 0.05,
        # Dynamic basin defaults
        basin_tension_threshold: float = 0.15,
        basin_align_threshold: float = 0.6,
        basin_anchor_lr: float = 0.05,
        basin_prune_min_count: int = 10,
        basin_prune_merge_cos: float = 0.97,
    ):
        """
        Initialize collapse engine.
        
        Args:
            dim: Dimension of state vector
            num_layers: Number of collapse steps
            strength_entail: Force strength for entail anchor
            strength_contra: Force strength for contradiction anchor
            strength_neutral: Force strength for neutral anchor
            basin_*: Defaults for dynamic basin behavior (spawn/update/prune)
        """
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.strength_entail = strength_entail
        self.strength_contra = strength_contra
        self.strength_neutral = strength_neutral
        self.basin_tension_threshold = basin_tension_threshold
        self.basin_align_threshold = basin_align_threshold
        self.basin_anchor_lr = basin_anchor_lr
        self.basin_prune_min_count = basin_prune_min_count
        self.basin_prune_merge_cos = basin_prune_merge_cos

        # Three anchors to create multi-basin geometry
        self.anchor_entail = nn.Parameter(torch.randn(dim))
        self.anchor_contra = nn.Parameter(torch.randn(dim))
        self.anchor_neutral = nn.Parameter(torch.randn(dim))
    
    def collapse(self, h0: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Backward-compatible static collapse (three fixed anchors).
        """
        return self._collapse_static(h0)

    def collapse_dynamic(
        self,
        h0: torch.Tensor,
        labels: torch.Tensor,
        basin_field: BasinField,
        global_step: int = 0,
        spawn_new: bool = True,
        prune_every: int = 0,
        update_anchors: bool = True,
        entropy_pressure: float = 0.0,
        entropy_budget: Optional[float] = None,
        deletion_log: Optional[List[Dict]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Collapse with dynamic per-label basins.
        
        Args:
            h0: Initial state vector(s) [B, dim] or [dim]
            labels: Ground-truth labels as integers (0=E,1=C,2=N)
            basin_field: Shared BasinField instance
            global_step: Training step for bookkeeping
            spawn_new: Whether to allow spawning new basins
            prune_every: If >0, prune/merge every N steps
            update_anchors: Whether to adapt basin centers after collapse
            entropy_pressure: Scalar pressure to decay/prune low-utility basins (model-controlled knob)
            entropy_budget: Optional cap on total utility removed per call
            deletion_log: Optional list to append deletion events for auditing
        """
        squeeze = False
        h = h0
        if h.dim() == 1:
            h = h.unsqueeze(0)
            labels = labels.unsqueeze(0)
            squeeze = True
        h = h.clone()
        labels = labels.to(h.device)
        basin_field.to(h.device)

        label_to_char = {0: "E", 1: "C", 2: "N"}
        label_strength = {
            0: self.strength_entail,
            1: self.strength_contra,
            2: self.strength_neutral,
        }

        # Apply entropy pressure before routing to drop weak basins under physics control
        basin_field.apply_entropy_pressure(
            pressure=float(entropy_pressure),
            budget=entropy_budget,
            step=global_step,
            log=deletion_log,
        )

        anchors = []

        # Route each sample to its label-specific basin (and possibly spawn)
        for i in range(h.size(0)):
            y_char = label_to_char.get(int(labels[i].item()))
            anchor, align_val, div_val, tens_val = route_to_basin(
                basin_field, h[i], y_char, step=global_step
            )
            anchors.append(anchor)
            if spawn_new:
                maybe_spawn_basin(
                    basin_field,
                    h[i],
                    y_char,
                    tens_val,
                    align_val,
                    step=global_step,
                    tension_threshold=self.basin_tension_threshold,
                    align_threshold=self.basin_align_threshold,
                )

        anchor_dirs = torch.stack([a.center for a in anchors]).to(h.device)
        strengths = torch.tensor([label_strength[int(l.item())] for l in labels], device=h.device)

        trace = {
            "alignment_local": [],
            "divergence_local": [],
            "tension_local": [],
        }

        for step in range(self.num_layers):
            h_n = F.normalize(h, dim=-1)
            align = (h_n * anchor_dirs).sum(dim=-1)
            div = divergence_from_alignment(align)
            tens = tension(div)

            trace["alignment_local"].append(align.detach())
            trace["divergence_local"].append(div.detach())
            trace["tension_local"].append(tens.detach())

            delta = torch.zeros_like(h)
            anchor_vec = F.normalize(h - anchor_dirs, dim=-1)
            h = h + delta - strengths.unsqueeze(-1) * div.unsqueeze(-1) * anchor_vec

            h_norm = h.norm(p=2, dim=-1, keepdim=True)
            h = torch.where(h_norm > 10.0, h * (10.0 / (h_norm + 1e-8)), h)

        # Update anchors post-collapse
        if update_anchors:
            for i, anchor in enumerate(anchors):
                update_basin_center(anchor, h[i], lr=self.basin_anchor_lr)
                anchor.last_used_step = global_step

        if prune_every and global_step > 0 and global_step % prune_every == 0:
            prune_and_merge(
                basin_field,
                min_count=self.basin_prune_min_count,
                merge_cos_threshold=self.basin_prune_merge_cos,
            )

        if squeeze:
            h = h.squeeze(0)
        return h, trace

    def _collapse_static(self, h0: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Collapse initial state h0 through L steps.
        
        Args:
            h0: Initial state vector (dim,) or batch of vectors [B, dim]
            
        Returns:
            Tuple of (h_final, trace)
            trace: Dict with per-anchor align/div/tension lists
        """
        squeeze = False
        h = h0
        if h.dim() == 1:
            h = h.unsqueeze(0)
            squeeze = True
        h = h.clone()
        trace = {
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
        
        # Normalize anchor directions
        e_dir = F.normalize(self.anchor_entail, dim=0)
        c_dir = F.normalize(self.anchor_contra, dim=0)
        n_dir = F.normalize(self.anchor_neutral, dim=0)
        
        for step in range(self.num_layers):
            # Normalize current state along feature dim
            h_n = F.normalize(h, dim=-1)
            
            # Compute physics to each anchor
            a_e = (h_n * e_dir).sum(dim=-1)
            a_c = (h_n * c_dir).sum(dim=-1)
            a_n = (h_n * n_dir).sum(dim=-1)
            d_e = divergence_from_alignment(a_e)
            d_c = divergence_from_alignment(a_c)
            d_n = divergence_from_alignment(a_n)
            t_e = tension(d_e)
            t_c = tension(d_c)
            t_n = tension(d_n)
            
            # Log trace
            trace["alignment_entail"].append(a_e.detach())
            trace["alignment_contra"].append(a_c.detach())
            trace["alignment_neutral"].append(a_n.detach())
            trace["divergence_entail"].append(d_e.detach())
            trace["divergence_contra"].append(d_c.detach())
            trace["divergence_neutral"].append(d_n.detach())
            trace["tension_entail"].append(t_e.detach())
            trace["tension_contra"].append(t_c.detach())
            trace["tension_neutral"].append(t_n.detach())

            # State update is physics-only (no learned MLP)
            delta = torch.zeros_like(h)
            # Anchor forces: move toward/away each anchor along their difference vector
            e_vec = F.normalize(h - e_dir.unsqueeze(0), dim=-1)
            c_vec = F.normalize(h - c_dir.unsqueeze(0), dim=-1)
            n_vec = F.normalize(h - n_dir.unsqueeze(0), dim=-1)
            h = (
                h
                + delta
                - self.strength_entail * d_e.unsqueeze(-1) * e_vec
                - self.strength_contra * d_c.unsqueeze(-1) * c_vec
                - self.strength_neutral * d_n.unsqueeze(-1) * n_vec
            )
            
            # Soft norm control (conservation-ish)
            # Keep ||h|| roughly bounded to prevent explosion
            h_norm = h.norm(p=2, dim=-1, keepdim=True)
            h = torch.where(h_norm > 10.0, h * (10.0 / (h_norm + 1e-8)), h)
        
        if squeeze:
            h = h.squeeze(0)
        return h, trace
    
    def forward(self, h0: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Forward pass (alias for collapse).
        
        Args:
            h0: Initial state vector
            
        Returns:
            Tuple of (h_final, trace)
        """
        return self._collapse_static(h0)

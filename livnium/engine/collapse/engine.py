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
from livnium.engine.fields.basin_field import BasinField


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
        enable_basins: bool = False,
        basin_threshold: str = "v4",
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
            enable_basins: Whether to enable dynamic basin memory
            basin_threshold: Basin threshold version ("v3" or "v4")
        """
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.enable_basins = enable_basins
        
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
        
        # Basin field for dynamic memory (if enabled)
        if enable_basins:
            tension_threshold = (
                defaults.BASIN_TENSION_THRESHOLD_V4 if basin_threshold == "v4"
                else defaults.BASIN_TENSION_THRESHOLD_V3
            )
            self.basin_field = BasinField(
                max_basins_per_label=64,
                device=None,  # Will be set when model moves to device
                tension_threshold=tension_threshold,
            )
        else:
            self.basin_field = None
    
    def step(
        self,
        h: torch.Tensor,
        anchors: List[torch.Tensor],
        strengths: List[float],
        step_idx: int = 0,
        label_map: Optional[Dict[int, str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Single collapse step.
        
        Args:
            h: Current state vector [B, dim] or [dim]
            anchors: List of anchor vectors (normalized)
            strengths: List of force strengths for each anchor
            step_idx: Current step index (for basin tracking)
            label_map: Optional mapping from batch index to label ("E", "N", "C")
            
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
        # Handle batching: compute per batch element
        batch_size = h_n.shape[0]
        total_force = torch.zeros_like(h)
        
        # Expand anchor to batch dimension if needed
        anchor_expanded_list = []
        for anchor in anchors:
            if anchor.dim() == 1:
                anchor_expanded = anchor.unsqueeze(0).expand(batch_size, -1)  # [B, dim]
            else:
                anchor_expanded = anchor
            anchor_expanded_list.append(anchor_expanded)
        
        for anchor, anchor_expanded, strength in zip(anchors, anchor_expanded_list, strengths):
            # Vectorized batched physics computation (MPS-optimized)
            # Compute alignment (cosine similarity) for entire batch at once
            # h_n: [B, dim], anchor_expanded: [B, dim]
            # Normalize (already normalized, but ensure)
            h_normalized = F.normalize(h_n, dim=-1, p=2)
            anchor_normalized = F.normalize(anchor_expanded, dim=-1, p=2)
            
            # Batched dot product: sum(h_n * anchor_n, dim=-1) -> [B]
            align_tensor = (h_normalized * anchor_normalized).sum(dim=-1)  # [B]
            
            # Divergence: DIVERGENCE_PIVOT - alignment
            div_tensor = DIVERGENCE_PIVOT - align_tensor  # [B]
            
            # Tension: |divergence|
            tens_tensor = torch.abs(div_tensor)  # [B]
            
            trace["alignment"].append(align_tensor)
            trace["divergence"].append(div_tensor)
            trace["tension"].append(tens_tensor)
            
            # Compute force direction
            direction = F.normalize(h - anchor_expanded, dim=-1, p=2)
            
            # Apply force: -strength * divergence * direction
            # Expand divergence to match dimensions
            div_expanded = div_tensor.unsqueeze(-1)  # [B, 1]
            force = -strength * div_expanded * direction
            total_force = total_force + force
        
        # Basin forces (if enabled) - PHYSICS-BASED ROUTING (no label leakage)
        # Optimized: Skip if no basins exist yet, and reduce routing frequency
        basin_force = torch.zeros_like(h)
        if self.enable_basins and self.basin_field is not None:
            # Early exit if no basins exist yet (common in early training)
            total_basins = sum(len(self.basin_field.anchors.get(l, [])) for l in ["E", "N", "C"])
            if total_basins == 0:
                # No basins yet, skip routing (will be created during spawn)
                pass
            else:
                # Compute additional forces from basin anchors
                # Use average strength as baseline for basin forces
                avg_strength = sum(strengths) / len(strengths) if strengths else 0.1
                
                # OPTIMIZATION: Only route every N steps to reduce overhead
                # Route every step for first 100 steps (basin formation), then every 3 steps
                route_this_step = (step_idx < 100) or (step_idx % 3 == 0)
                
                if route_this_step:
                    for i in range(batch_size):
                        h_i = h[i] if h.dim() > 1 else h
                        
                        # PHYSICS-BASED: Route to best basin across ALL labels based on alignment
                        # This avoids label leakage - we use geometry, not ground truth
                        best_basin_anchor = None
                        best_basin_align = -1.0
                        best_basin_label = None
                        best_basin_div = 0.0
                        best_basin_tens = 0.0
                        
                        # Check all label sets and find best match by physics
                        for label in ["E", "N", "C"]:
                            try:
                                basin_anchor, basin_align, basin_div, basin_tens = self.basin_field.route(
                                    h_i, label, step_idx
                                )
                                
                                # Keep track of best alignment (physics-based selection)
                                if basin_align > best_basin_align:
                                    best_basin_align = basin_align
                                    best_basin_anchor = basin_anchor
                                    best_basin_label = label
                                    best_basin_div = basin_div
                                    best_basin_tens = basin_tens
                            except Exception:
                                continue
                        
                        # Apply force from best-matching basin (if found)
                        if best_basin_anchor is not None and best_basin_align > 0.0:
                            basin_strength = avg_strength * 0.5  # Basins contribute 50% of static anchor strength
                            basin_direction = F.normalize(h_i - best_basin_anchor.center, dim=0)
                            basin_force_i = -basin_strength * best_basin_div * basin_direction
                            
                            if h.dim() > 1:
                                basin_force[i] = basin_force_i
                            else:
                                basin_force = basin_force_i
                            
                            # Update basin anchor if alignment is good (physics-gated)
                            if best_basin_align > self.basin_field.align_threshold:
                                self.basin_field.update(best_basin_anchor, h_i)
                            
                            # Spawn new basin if tension is high and alignment is low (physics-gated)
                            # Only spawn if we have a label (for training updates, not routing)
                            if label_map is not None and best_basin_label is not None:
                                self.basin_field.spawn(h_i, best_basin_label, best_basin_tens, best_basin_align, step_idx)
        
        # State update
        delta = self.update(h)
        
        # Apply update and forces (static + basin)
        h_new = h + delta + total_force + basin_force
        
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
        h0: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Collapse initial state h0 through L steps.
        
        Args:
            h0: Initial state vector [B, dim] or [dim]
            labels: Optional label tensor [B] with values 0=E, 1=N, 2=C (for basin routing)
            
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
        
        # Map labels to basin labels ("E", "N", "C") - ONLY for basin updates, NOT routing
        # Labels are used for basin maintenance (spawning/updating), not for force computation
        # This prevents label leakage - routing is physics-based
        # Optimized: avoid .item() calls to prevent CPU sync on MPS
        label_map = None
        if labels is not None and self.enable_basins:
            label_map = {}
            label_to_basin = {0: "E", 1: "N", 2: "C"}
            # Vectorized label mapping (avoid per-element .item() calls)
            if labels.dim() == 0:
                label_idx = int(labels.cpu().numpy()) if labels.device.type == "mps" else labels.item()
                label_map[0] = label_to_basin.get(label_idx, "N")
            else:
                # Batch: convert to CPU numpy once, then map
                labels_cpu = labels.cpu().numpy() if labels.device.type == "mps" else labels.numpy()
                for i, label_idx in enumerate(labels_cpu):
                    label_map[i] = label_to_basin.get(int(label_idx), "N")
        
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
            h, step_trace = self.step(h, anchors, strengths, step_idx=step, label_map=label_map)
            
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
        
        # Periodic basin maintenance (prune and merge)
        if self.enable_basins and self.basin_field is not None and step % 10 == 9:
            self.basin_field.prune_and_merge()
        
        return h, full_trace
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Custom state_dict that includes basin_field state.
        """
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        
        # Add basin_field state if it exists
        if self.enable_basins and self.basin_field is not None:
            basin_state = self.basin_field.state_dict()
            for key, value in basin_state.items():
                state[f"{prefix}basin_field.{key}"] = value
        
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Custom load_state_dict that restores basin_field state.
        """
        # Extract basin_field state if present
        basin_state = {}
        basin_keys = []
        for key in list(state_dict.keys()):
            if key.startswith("basin_field."):
                basin_key = key[len("basin_field."):]
                basin_state[basin_key] = state_dict.pop(key)
                basin_keys.append(key)
        
        # Load standard parameters
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)
        
        # Restore basin_field if it exists and basins are enabled
        if self.enable_basins and self.basin_field is not None and basin_state:
            try:
                self.basin_field.load_state_dict(basin_state)
            except Exception as e:
                if strict:
                    raise RuntimeError(f"Failed to load basin_field state: {e}")
                else:
                    print(f"Warning: Could not load basin_field state: {e}")
        
        # Handle strict mode
        if strict:
            if missing_keys:
                raise RuntimeError(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                raise RuntimeError(f"Unexpected keys: {unexpected_keys}")
        
        return missing_keys, unexpected_keys


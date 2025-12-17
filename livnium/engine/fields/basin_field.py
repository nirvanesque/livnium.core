"""
Basin Field: Dynamic Basin Management Using Kernel Physics

Maintains per-label micro-basins that can be routed to, updated, spawned,
and pruned during training. All physics calculations use kernel.physics.*
with provided Ops instance. All thresholds come from engine.config.defaults.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F

from livnium.kernel.physics import alignment, divergence, tension
from livnium.engine.config import defaults
from livnium.engine.ops_torch import TorchOps


LABELS = ("E", "N", "C")


class _TensorState:
    """Adapter for torch.Tensor to State protocol (same pattern as CollapseEngine)."""
    def __init__(self, t: torch.Tensor):
        self.t = t
    
    def vector(self):
        return self.t
    
    def norm(self):
        return torch.norm(self.t, p=2, dim=-1, keepdim=True)


class BasinAnchor:
    """
    Single micro-basin anchor.
    """
    
    def __init__(
        self,
        center: torch.Tensor,
        label: str,
        step: int = 0,
        decay_rate: float = 1.0,
        utility: float = 1.0
    ):
        if label not in LABELS:
            raise ValueError(f"Invalid label {label}; expected one of {LABELS}")
        # Store as detached, unit vector on creation
        self.center = F.normalize(center.detach(), dim=0)
        self.label = label
        self.count: int = 0
        self.last_used_step: int = step
        self.decay_rate: float = decay_rate
        self.utility: float = utility


class BasinField:
    """
    Collection of anchors per label with routing, spawning, merging, pruning.
    
    All physics calculations use kernel.physics.* with Ops instance.
    All thresholds come from engine.config.defaults.
    """
    
    def __init__(
        self,
        max_basins_per_label: int = 64,
        device: torch.device = None,
        tension_threshold: Optional[float] = None,
        align_threshold: Optional[float] = None,
        anchor_lr: Optional[float] = None,
        prune_min_count: Optional[int] = None,
        merge_cos_threshold: Optional[float] = None,
    ):
        """
        Initialize basin field.
        
        Args:
            max_basins_per_label: Maximum number of basins per label
            device: Device for tensors
            tension_threshold: Tension threshold for spawning (defaults to config)
            align_threshold: Alignment threshold for spawning (defaults to config)
            anchor_lr: Learning rate for anchor updates (defaults to config)
            prune_min_count: Minimum count to keep anchor (defaults to config)
            merge_cos_threshold: Cosine threshold for merging (defaults to config)
        """
        self.anchors: Dict[str, List[BasinAnchor]] = {l: [] for l in LABELS}
        self.max_basins_per_label = max_basins_per_label
        self.device = device
        
        # Use config defaults if not specified
        self.tension_threshold = tension_threshold if tension_threshold is not None else defaults.BASIN_TENSION_THRESHOLD_V4
        self.align_threshold = align_threshold if align_threshold is not None else defaults.BASIN_ALIGN_THRESHOLD
        self.anchor_lr = anchor_lr if anchor_lr is not None else defaults.BASIN_ANCHOR_LR
        self.prune_min_count = prune_min_count if prune_min_count is not None else defaults.BASIN_PRUNE_MIN_COUNT
        self.merge_cos_threshold = merge_cos_threshold if merge_cos_threshold is not None else defaults.BASIN_PRUNE_MERGE_COS
        
        # Ops instance for kernel physics
        self.ops = TorchOps()
    
    def to(self, device: torch.device):
        """Move all anchor centers to the target device."""
        self.device = device
        for anchors in self.anchors.values():
            for a in anchors:
                a.center = a.center.to(device)
        return self
    
    def route(
        self,
        h: torch.Tensor,
        label: str,
        step: int,
        allow_create: bool = False,
        allow_usage_update: bool = False
    ) -> Tuple[BasinAnchor, float, float, float]:
        """
        Route state h to best basin for label.
        
        Vectorized for MPS efficiency - computes all alignments at once.
        Final physics uses canonical kernel.physics.* calls.
        
        Args:
            h: State vector
            label: Label ("E", "N", or "C")
            step: Current step
            allow_create: If True, create first anchor if none exist (training-only)
            allow_usage_update: If True, update count/last_used_step (training-only)
            
        Returns:
            Tuple of (anchor, alignment, divergence, tension)
        """
        if label not in LABELS:
            raise ValueError(f"Invalid label {label}; expected one of {LABELS}")
        
        anchors = self.anchors[label]
        h_n = F.normalize(h, dim=0)
        
        # If no anchors, create first one (only during training)
        if not anchors:
            if not allow_create:
                raise RuntimeError(
                    f"No basins for label '{label}' and allow_create=False. "
                    f"Cannot route during inference without existing basins."
                )
            anchor = BasinAnchor(h_n, label, step=step)
            self.anchors[label].append(anchor)
            return anchor, 1.0, 0.0, 0.0
        
        # VECTORIZED: Compute all alignments at once (MPS-optimized)
        # Stack all anchor centers: [num_anchors, dim]
        # Note: centers are already normalized, no need to normalize again
        anchor_centers = torch.stack([a.center for a in anchors])  # [N, dim]
        h_n_expanded = h_n.unsqueeze(0)  # [1, dim]
        
        # Batched dot product: [N, dim] @ [1, dim]^T -> [N]
        alignments = (anchor_centers * h_n_expanded).sum(dim=-1)  # [N]
        
        # Find best anchor (vectorized)
        best_idx = int(alignments.argmax())
        best_anchor = anchors[best_idx]
        
        # CANONICAL PHYSICS: Use kernel.physics.* for final computation
        # This ensures consistency with CollapseEngine and prevents drift
        state_obj = _TensorState(h_n)
        anchor_obj = _TensorState(best_anchor.center)
        
        align_t = alignment(self.ops, state_obj, anchor_obj)  # tensor scalar
        div_t = divergence(self.ops, state_obj, anchor_obj)    # tensor scalar
        tens_t = tension(self.ops, div_t)                      # tensor scalar
        
        # Update anchor usage (only during training for deterministic inference)
        if allow_usage_update:
            best_anchor.count += 1
            best_anchor.last_used_step = step
        
        return best_anchor, float(align_t.item()), float(div_t.item()), float(tens_t.item())
    
    def update(self, anchor: BasinAnchor, h: torch.Tensor, lr: Optional[float] = None):
        """
        Update basin anchor center using EMA.
        
        Args:
            anchor: Anchor to update
            h: New state vector
            lr: Learning rate (defaults to config)
        """
        if lr is None:
            lr = self.anchor_lr
        
        h_n = F.normalize(h, dim=0)
        new_center = (1 - lr) * anchor.center + lr * h_n.detach()
        anchor.center = F.normalize(new_center, dim=0)
    
    def spawn(
        self,
        h: torch.Tensor,
        label: str,
        tension_value: float,
        align_value: float,
        step: int
    ) -> bool:
        """
        Spawn a new basin when tension is high and alignment is low.
        
        Uses config thresholds.
        
        Args:
            h: State vector
            label: Label ("E", "N", or "C")
            tension_value: Current tension
            align_value: Current alignment
            step: Current step
            
        Returns:
            True if basin was spawned, False otherwise
        """
        anchors = self.anchors[label]
        
        # Check capacity
        if len(anchors) >= self.max_basins_per_label:
            return False
        
        # Check spawn conditions (using config thresholds)
        if tension_value > self.tension_threshold and align_value < self.align_threshold:
            h_n = F.normalize(h, dim=0)
            new_anchor = BasinAnchor(h_n, label, step=step)
            self.anchors[label].append(new_anchor)
            return True
        
        return False
    
    def merge(self, merge_cos_threshold: Optional[float] = None):
        """
        Merge similar basin anchors.
        
        Args:
            merge_cos_threshold: Cosine threshold for merging (defaults to config)
        """
        if merge_cos_threshold is None:
            merge_cos_threshold = self.merge_cos_threshold
        
        for label, anchors in self.anchors.items():
            kept: List[BasinAnchor] = []
            
            while anchors:
                a = anchors.pop()
                merged = False
                
                for b in kept:
                    cos_sim = float(torch.dot(a.center, b.center))
                    if cos_sim > merge_cos_threshold:
                        # Merge: weighted average by count
                        total = a.count + b.count
                        merged_center = (a.center * a.count + b.center * b.count) / max(total, 1)
                        b.center = F.normalize(merged_center, dim=0)
                        b.count = total
                        merged = True
                        break
                
                if not merged:
                    kept.append(a)
            
            self.anchors[label] = kept
    
    def prune(self, min_count: Optional[int] = None):
        """
        Prune weak anchors (below minimum count).
        
        Args:
            min_count: Minimum count to keep (defaults to config)
        """
        if min_count is None:
            min_count = self.prune_min_count
        
        for label in LABELS:
            self.anchors[label] = [
                a for a in self.anchors[label]
                if a.count >= min_count
            ]
    
    def prune_and_merge(
        self,
        min_count: Optional[int] = None,
        merge_cos_threshold: Optional[float] = None
    ):
        """
        Prune weak anchors and merge similar ones.
        
        Args:
            min_count: Minimum count to keep (defaults to config)
            merge_cos_threshold: Cosine threshold for merging (defaults to config)
        """
        self.prune(min_count)
        self.merge(merge_cos_threshold)
    
    def state_dict(self) -> Dict:
        """Lightweight serialization for checkpoints."""
        return {
            "max_basins_per_label": self.max_basins_per_label,
            "anchors": {
                label: [
                    {
                        "center": anchor.center,
                        "label": anchor.label,
                        "count": anchor.count,
                        "last_used_step": anchor.last_used_step,
                        "decay_rate": anchor.decay_rate,
                        "utility": anchor.utility,
                    }
                    for anchor in anchors
                ]
                for label, anchors in self.anchors.items()
            },
        }
    
    def load_state_dict(self, state: Dict):
        """Restore anchors from serialized form."""
        self.max_basins_per_label = state.get("max_basins_per_label", self.max_basins_per_label)
        anchors_state = state.get("anchors", {})
        self.anchors = {l: [] for l in LABELS}
        
        for label, anchors in anchors_state.items():
            if label not in LABELS:
                continue
            for anchor_state in anchors:
                center = anchor_state["center"]
                if self.device is not None:
                    center = center.to(self.device)
                anchor = BasinAnchor(
                    center=center,
                    label=anchor_state["label"],
                    decay_rate=anchor_state.get("decay_rate", 1.0),
                    utility=anchor_state.get("utility", 1.0),
                )
                anchor.count = anchor_state.get("count", 0)
                anchor.last_used_step = anchor_state.get("last_used_step", 0)
                self.anchors[label].append(anchor)


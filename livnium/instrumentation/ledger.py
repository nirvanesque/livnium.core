"""
Instrumentation Ledger: Auditing the Collapse

Provides tools to record, track, and analyze the dynamics of the 
reconciliation process.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class StepRecord:
    """Snapshot of a single reconciliation step."""
    step: int
    global_tension: float
    delta_tension: float
    # forces: Optional[torch.Tensor] = None # Memory intensive, enable if needed
    max_repulsion: float = 0.0
    max_attraction: float = 0.0

class TensionLedger:
    """
    A persistent record of the path to Moksha.
    """
    def __init__(self, claim_ids: List[str]):
        self.claim_ids = claim_ids
        self.history: List[StepRecord] = []
        self._prev_tension: Optional[float] = None

    def record(self, 
               step: int, 
               tension: float, 
               forces: Optional[torch.Tensor] = None):
        """Record a single step of physics."""
        delta = (tension - self._prev_tension) if self._prev_tension is not None else 0.0
        
        # Extract extreme forces if provided
        max_rep = 0.0
        max_att = 0.0
        if forces is not None:
            # We assume forces is [N, dim]
            # This is a simplification; we'd ideally look at pairwise force magnitudes
            # But let's look at the norm of the force applied to each claim
            force_norms = torch.norm(forces, dim=-1)
            max_att = torch.max(force_norms).item() 
            # Note: identifying attraction vs repulsion requires pairwise inspection
            
        record = StepRecord(
            step=step,
            global_tension=tension,
            delta_tension=delta,
            max_attraction=max_att
        )
        
        self.history.append(record)
        self._prev_tension = tension

    def get_summary(self) -> Dict[str, Any]:
        """Returns a high-level audit summary."""
        if not self.history:
            return {}
            
        start_tension = self.history[0].global_tension
        end_tension = self.history[-1].global_tension
        reduction = (start_tension - end_tension) / start_tension if start_tension > 0 else 0
        
        return {
            "initial_tension": start_tension,
            "final_tension": end_tension,
            "reduction_pct": reduction * 100,
            "steps": len(self.history),
            "is_stable": abs(self.history[-1].delta_tension) < 1e-4
        }

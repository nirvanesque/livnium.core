"""
Energy/Tension Computation for Stability Experiment

Global τ / conflict / φ-tension. This is your E(t).

You'll plug in your actual energy/tension logic. Placeholders and structure provided.
"""

import numpy as np
from typing import Any, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from core.classical.livnium_core_system import LivniumCoreSystem

# Types: can be LivniumCoreSystem or numpy array
LatticeState = Any


def compute_energy(state: LatticeState, method: str = "symbolic_weight_variance") -> float:
    """
    Compute global energy/tension E(t) for the given state.
    
    This should reflect τ (local conflicts) summed or averaged.
    
    Replace the body with:
      - sum of violated constraints
      - sum of K4 tensions
      - φ-based inconsistency, etc.
    
    Args:
        state: Lattice state (LivniumCoreSystem or array)
        method: Energy computation method
        
    Returns:
        Energy value (lower = more stable)
    """
    # If it's a LivniumCoreSystem, use symbolic weight variance
    if isinstance(state, LivniumCoreSystem):
        if method == "symbolic_weight_variance":
            return _compute_sw_variance(state)
        elif method == "local_tension":
            return _compute_local_tension(state)
        elif method == "conflict_count":
            return _compute_conflict_count(state)
        else:
            return _compute_sw_variance(state)  # Default
    
    # For generic arrays, use variance as placeholder
    arr = np.asarray(state, dtype=float)
    return float(arr.var())


def _compute_sw_variance(system: LivniumCoreSystem) -> float:
    """Compute variance in symbolic weights as energy metric."""
    weights = []
    for cell in system.lattice.values():
        if cell.symbolic_weight is not None:
            weights.append(cell.symbolic_weight)
    
    if len(weights) == 0:
        return 0.0
    
    return float(np.var(weights))


def _compute_local_tension(system: LivniumCoreSystem) -> float:
    """Compute local tension based on neighbor conflicts (K₄-like)."""
    total_tension = 0.0
    
    for coords, cell in system.lattice.items():
        x, y, z = coords
        cell_tension = 0.0
        
        # Check neighbors (6-connected)
        neighbors = [
            (x+1, y, z), (x-1, y, z),
            (x, y+1, z), (x, y-1, z),
            (x, y, z+1), (x, y, z-1)
        ]
        
        for ncoords in neighbors:
            if ncoords in system.lattice:
                neighbor = system.lattice[ncoords]
                
                if cell.symbolic_weight is not None and neighbor.symbolic_weight is not None:
                    diff = abs(cell.symbolic_weight - neighbor.symbolic_weight)
                    cell_tension += diff
        
        total_tension += cell_tension
    
    return total_tension / len(system.lattice) if len(system.lattice) > 0 else 0.0


def _compute_conflict_count(system: LivniumCoreSystem) -> float:
    """Count violated constraints as energy."""
    conflicts = 0
    expected_counts = system.get_expected_class_counts()
    actual_counts = system.get_class_counts()
    
    for cell_class, expected_count in expected_counts.items():
        actual_count = actual_counts.get(cell_class, 0)
        conflicts += abs(expected_count - actual_count)
    
    return float(conflicts)

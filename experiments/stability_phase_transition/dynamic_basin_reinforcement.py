"""
Dynamic Basin Reinforcement: Geometry-Driven Self-Tuning

Instead of static parameters (alpha, beta, noise), this uses dynamic values
computed from the geometry itself:
- alpha = f(curvature) - how deep the basin is becoming
- beta = g(tension) - internal contradictions in symbolic weight
- noise = h(entropy) - how noisy the state is

This creates a self-regulating system that adapts to the geometry.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import Tuple, List, Dict, Any
import random

from core.classical.livnium_core_system import LivniumCoreSystem, RotationAxis


def compute_local_curvature(
    system: LivniumCoreSystem,
    task,
    active_coords: List[Tuple[int, int, int]]
) -> float:
    """
    Compute local curvature: how deep the basin is becoming.
    
    Curvature = variance in symbolic weights around active cells.
    Higher variance = deeper basin = stronger attractor.
    
    Returns:
        Curvature value (0.0 to 1.0+)
    """
    if not active_coords:
        return 0.0
    
    # Get SW values for active cells
    sw_values = []
    for coords in active_coords:
        cell = system.get_cell(coords)
        if cell:
            sw_values.append(cell.symbolic_weight)
    
    if len(sw_values) < 2:
        return 0.0
    
    # Curvature = normalized variance
    sw_array = np.array(sw_values)
    mean_sw = np.mean(sw_array)
    if mean_sw == 0:
        return 0.0
    
    variance = np.var(sw_array)
    # Normalize: variance relative to mean
    curvature = variance / (mean_sw + 1.0)  # +1 to avoid division by zero
    
    # Scale to reasonable range (0 to ~2)
    curvature = min(curvature, 2.0)
    
    return float(curvature)


def compute_symbolic_tension(
    system: LivniumCoreSystem,
    task,
    active_coords: List[Tuple[int, int, int]]
) -> float:
    """
    Compute symbolic tension: internal contradictions in SW.
    
    Tension = how much SW values conflict with each other.
    Higher tension = more contradictions = need more decay.
    
    Returns:
        Tension value (0.0 to 1.0+)
    """
    if not active_coords:
        return 0.0
    
    # Get SW values
    sw_values = []
    for coords in active_coords:
        cell = system.get_cell(coords)
        if cell:
            sw_values.append(cell.symbolic_weight)
    
    if len(sw_values) < 2:
        return 0.0
    
    sw_array = np.array(sw_values)
    
    # Tension = how spread out the values are
    # If all values are similar, tension is low
    # If values are very different, tension is high
    range_sw = np.max(sw_array) - np.min(sw_array)
    mean_sw = np.mean(sw_array)
    
    if mean_sw == 0:
        return 0.0
    
    # Normalized range (relative to mean)
    tension = range_sw / (mean_sw + 1.0)
    
    # Also check for extreme values (sign of tension)
    max_sw = np.max(sw_array)
    if max_sw > 50.0:  # Very high SW = tension
        tension += 0.5
    
    # Scale to reasonable range
    tension = min(tension, 2.0)
    
    return float(tension)


def compute_noise_entropy(
    system: LivniumCoreSystem,
    task,
    active_coords: List[Tuple[int, int, int]]
) -> float:
    """
    Compute noise entropy: how noisy/disordered the state is.
    
    Entropy = randomness in the system state.
    Higher entropy = more noise needed to decorrelate.
    
    Returns:
        Entropy value (0.0 to 1.0)
    """
    if not active_coords:
        return 0.0
    
    # Entropy based on face exposure distribution
    # More varied face exposures = higher entropy
    face_exposures = []
    for coords in active_coords:
        cell = system.get_cell(coords)
        if cell:
            face_exposures.append(cell.face_exposure)
    
    if len(face_exposures) < 2:
        return 0.0
    
    # Entropy = normalized variance in face exposures
    fe_array = np.array(face_exposures)
    variance = np.var(fe_array)
    
    # Normalize: max variance for 6 faces is ~9 (3^2)
    entropy = variance / 9.0
    
    # Also consider SW distribution
    sw_values = [system.get_cell(c).symbolic_weight for c in active_coords if system.get_cell(c)]
    if sw_values:
        sw_variance = np.var(sw_values)
        sw_entropy = min(sw_variance / 100.0, 1.0)  # Normalize
        entropy = (entropy + sw_entropy) / 2.0
    
    return float(min(entropy, 1.0))


def update_basin_dynamic(
    system: LivniumCoreSystem,
    task,
    is_correct: bool,
    base_alpha: float = 0.10,   # Base strength multiplier
    base_beta: float = 0.15,    # Base decay multiplier
    base_noise: float = 0.03     # Base noise multiplier
):
    """
    Dynamic basin shaping: parameters adapt to geometry.
    
    Instead of fixed alpha/beta/noise, these are computed from:
    - alpha = base_alpha * curvature (deeper basin = more reinforcement)
    - beta = base_beta * tension (more tension = more decay)
    - noise = base_noise * entropy (more entropy = more decorrelation)
    
    This creates a self-regulating system that adapts to the geometry.
    """
    # Get active cells (input + output)
    active_coords = []
    if hasattr(task, 'input_coords'):
        active_coords.extend(task.input_coords)
    if hasattr(task, 'output_coord'):
        active_coords.append(task.output_coord)
    
    if not active_coords:
        return
    
    # Compute geometry signals
    curvature = compute_local_curvature(system, task, active_coords)
    tension = compute_symbolic_tension(system, task, active_coords)
    entropy = compute_noise_entropy(system, task, active_coords)
    
    # Dynamic parameters
    alpha = base_alpha * (1.0 + curvature)  # Curvature amplifies reinforcement
    beta = base_beta * (1.0 + tension)     # Tension amplifies decay
    noise = base_noise * (1.0 + entropy)   # Entropy amplifies decorrelation
    
    if is_correct:
        # Strengthen the attractor: deepen the well
        for coords in active_coords:
            cell = system.get_cell(coords)
            if cell:
                # Deepen well: increase SW proportional to curvature
                cell.symbolic_weight += alpha
                
        # Smooth small contradictions (enforce local equilibrium)
        # This happens naturally through the system's conservation rules
        
    else:
        # Add decoherence noise: flatten wrong basin
        for coords in active_coords:
            cell = system.get_cell(coords)
            if cell:
                # Decay SW proportional to tension
                cell.symbolic_weight *= (1.0 - beta)
                # Ensure SW doesn't go negative
                if cell.symbolic_weight < 0:
                    cell.symbolic_weight = 0.0
        
        # Inject noise proportional to entropy
        if random.random() < noise:  # Scale noise probability
            axis = random.choice(list(RotationAxis))
            system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
    
    # Re-normalize: Keep global conservation intact
    # Ensure SW stays in reasonable bounds
    for coords, cell in system.lattice.items():
        if cell.symbolic_weight < 0:
            cell.symbolic_weight = 0.0
        # Cap at reasonable maximum (prevent explosion)
        if cell.symbolic_weight > 100.0:
            cell.symbolic_weight = 100.0


def get_geometry_signals(
    system: LivniumCoreSystem,
    task
) -> Dict[str, float]:
    """
    Get current geometry signals for monitoring.
    
    Returns:
        Dictionary with curvature, tension, entropy values
    """
    active_coords = []
    if hasattr(task, 'input_coords'):
        active_coords.extend(task.input_coords)
    if hasattr(task, 'output_coord'):
        active_coords.append(task.output_coord)
    
    return {
        'curvature': compute_local_curvature(system, task, active_coords),
        'tension': compute_symbolic_tension(system, task, active_coords),
        'entropy': compute_noise_entropy(system, task, active_coords)
    }


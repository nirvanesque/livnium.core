"""
Native Dynamic Basin Search: Geometry-Driven Self-Tuning Basin Reinforcement

This is the core implementation of dynamic basin reinforcement that uses
geometry signals (curvature, tension, entropy) to self-tune basin shaping
parameters instead of static hyperparameters.

Key Principle: Geometry decides the physics, not a parameter list.

Instead of:
    alpha = 0.10  # static
    beta = 0.15   # static
    noise = 0.03  # static

We use:
    alpha = f(curvature)  # geometry-driven
    beta = g(tension)     # geometry-driven
    noise = h(entropy)    # geometry-driven

This creates a self-regulating system that adapts to the geometry itself.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import random

from core.classical.livnium_core_system import LivniumCoreSystem, RotationAxis


def compute_local_curvature(
    system: LivniumCoreSystem,
    active_coords: List[Tuple[int, int, int]]
) -> float:
    """
    Compute local curvature: how deep the basin is becoming.
    
    Curvature = variance in symbolic weights around active cells.
    Higher variance = deeper basin = stronger attractor.
    
    Args:
        system: LivniumCoreSystem
        active_coords: List of active cell coordinates (input/output cells)
        
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
    active_coords: List[Tuple[int, int, int]]
) -> float:
    """
    Compute symbolic tension: internal contradictions in SW.
    
    Tension = how much SW values conflict with each other.
    Higher tension = more contradictions = need more decay.
    
    Args:
        system: LivniumCoreSystem
        active_coords: List of active cell coordinates
        
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
    
    # Normalized tension: range relative to mean
    tension = range_sw / (mean_sw + 1.0)
    
    # Scale to reasonable range (0 to ~2)
    tension = min(tension, 2.0)
    
    return float(tension)


def compute_noise_entropy(
    system: LivniumCoreSystem,
    active_coords: List[Tuple[int, int, int]]
) -> float:
    """
    Compute noise entropy: how noisy/disordered the state is.
    
    Entropy = variance in SW values relative to expected values.
    Higher entropy = more disorder = need more decorrelation.
    
    Args:
        system: LivniumCoreSystem
        active_coords: List of active cell coordinates
        
    Returns:
        Entropy value (0.0 to 1.0+)
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
    
    # Entropy = variance in SW values
    # High variance = high entropy = disordered state
    variance = np.var(sw_array)
    mean_sw = np.mean(sw_array)
    
    if mean_sw == 0:
        return 0.0
    
    # Normalized entropy: variance relative to mean
    entropy = variance / (mean_sw + 1.0)
    
    # Scale to reasonable range (0 to ~2)
    entropy = min(entropy, 2.0)
    
    return float(entropy)


def get_geometry_signals(
    system: LivniumCoreSystem,
    active_coords: List[Tuple[int, int, int]]
) -> Tuple[float, float, float]:
    """
    Get all geometry signals at once.
    
    Args:
        system: LivniumCoreSystem
        active_coords: List of active cell coordinates
        
    Returns:
        (curvature, tension, entropy) tuple
    """
    curvature = compute_local_curvature(system, active_coords)
    tension = compute_symbolic_tension(system, active_coords)
    entropy = compute_noise_entropy(system, active_coords)
    
    return (curvature, tension, entropy)


def update_basin_dynamic(
    system: LivniumCoreSystem,
    task: Any,
    is_correct: bool,
    base_alpha: float = 0.10,
    base_beta: float = 0.15,
    base_noise: float = 0.03
) -> None:
    """
    Update basin using dynamic (geometry-driven) parameters.
    
    This is the core dynamic basin reinforcement function.
    
    Instead of static parameters, it computes:
    - alpha = base_alpha * (1.0 + curvature)  # Curvature amplifies reinforcement
    - beta = base_beta * (1.0 + tension)     # Tension amplifies decay
    - noise = base_noise * (1.0 + entropy)   # Entropy amplifies decorrelation
    
    Args:
        system: LivniumCoreSystem
        task: Task object (must have input_coords and output_coord attributes)
        is_correct: Whether the task was solved correctly
        base_alpha: Base reinforcement strength (default 0.10)
        base_beta: Base decay strength (default 0.15)
        base_noise: Base decorrelation strength (default 0.03)
    """
    # Get active coordinates (input + output cells)
    active_coords = []
    if hasattr(task, 'input_coords'):
        active_coords.extend(task.input_coords)
    if hasattr(task, 'output_coord'):
        active_coords.append(task.output_coord)
    
    if not active_coords:
        return
    
    # Compute geometry signals
    curvature, tension, entropy = get_geometry_signals(system, active_coords)
    
    # Dynamic parameters: geometry-driven
    alpha = base_alpha * (1.0 + curvature)  # Curvature amplifies reinforcement
    beta = base_beta * (1.0 + tension)     # Tension amplifies decay
    noise = base_noise * (1.0 + entropy)   # Entropy amplifies decorrelation
    
    if is_correct:
        # Strengthen the attractor: deepen the well
        for coords in active_coords:
            cell = system.get_cell(coords)
            if cell:
                # Deepen well: increase SW (stronger attractor)
                cell.symbolic_weight += alpha
                # Clamp to reasonable bounds
                cell.symbolic_weight = min(cell.symbolic_weight, 200.0)
    else:
        # Add decoherence noise: flatten wrong basin
        for coords in active_coords:
            cell = system.get_cell(coords)
            if cell:
                # Decay SW (flatten well)
                cell.symbolic_weight *= (1.0 - beta)
                # Ensure SW doesn't go negative
                if cell.symbolic_weight < 0:
                    cell.symbolic_weight = 0.0
        
        # Inject small random drift so wrong basin can't re-form
        # Apply random rotation to decorrelate
        if random.random() < noise * 10:  # Scale noise probability
            axis = random.choice(list(RotationAxis))
            system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
    
    # Re-normalize: Keep global conservation intact
    # The system's conservation rules handle this automatically
    # But we ensure SW stays in reasonable bounds
    for coords, cell in system.lattice.items():
        if cell.symbolic_weight < 0:
            cell.symbolic_weight = 0.0
        # Cap at reasonable maximum (prevent explosion)
        if cell.symbolic_weight > 200.0:
            cell.symbolic_weight = 200.0


class DynamicBasinSearch:
    """
    High-level interface for dynamic basin search.
    
    This class provides a clean interface for using dynamic basin reinforcement
    in search and problem-solving contexts.
    """
    
    def __init__(
        self,
        base_alpha: float = 0.10,
        base_beta: float = 0.15,
        base_noise: float = 0.03
    ):
        """
        Initialize dynamic basin search.
        
        Args:
            base_alpha: Base reinforcement strength
            base_beta: Base decay strength
            base_noise: Base decorrelation strength
        """
        self.base_alpha = base_alpha
        self.base_beta = base_beta
        self.base_noise = base_noise
    
    def update(
        self,
        system: LivniumCoreSystem,
        task: Any,
        is_correct: bool
    ) -> None:
        """
        Update basin using dynamic parameters.
        
        Args:
            system: LivniumCoreSystem
            task: Task object
            is_correct: Whether task was solved correctly
        """
        update_basin_dynamic(
            system, task, is_correct,
            self.base_alpha, self.base_beta, self.base_noise
        )
    
    def get_signals(
        self,
        system: LivniumCoreSystem,
        active_coords: List[Tuple[int, int, int]]
    ) -> Dict[str, float]:
        """
        Get geometry signals for monitoring.
        
        Args:
            system: LivniumCoreSystem
            active_coords: List of active cell coordinates
            
        Returns:
            Dictionary with 'curvature', 'tension', 'entropy'
        """
        curvature, tension, entropy = get_geometry_signals(system, active_coords)
        return {
            'curvature': curvature,
            'tension': tension,
            'entropy': entropy
        }


# Convenience function for direct use
def apply_dynamic_basin(
    system: LivniumCoreSystem,
    task: Any,
    is_correct: bool,
    base_alpha: float = 0.10,
    base_beta: float = 0.15,
    base_noise: float = 0.03
) -> None:
    """
    Convenience function for applying dynamic basin reinforcement.
    
    This is an alias for update_basin_dynamic() for cleaner imports.
    
    Args:
        system: LivniumCoreSystem
        task: Task object
        is_correct: Whether task was solved correctly
        base_alpha: Base reinforcement strength
        base_beta: Base decay strength
        base_noise: Base decorrelation strength
    """
    update_basin_dynamic(system, task, is_correct, base_alpha, base_beta, base_noise)


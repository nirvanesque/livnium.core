"""
Local Dynamics / Update Rules for Stability Experiment

This is where you hook into your Livnium universe.

For now we wrap/normalize Livnium core system operations.
Later you can replace with your actual omcube step functions.
"""

import numpy as np
import random
from typing import Any, Optional
import copy

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from core.classical.livnium_core_system import LivniumCoreSystem, RotationAxis
from core.config import LivniumCoreConfig

# Types: can be LivniumCoreSystem or numpy array or your actual lattice class
LatticeState = Any


def random_init_lattice(N: int, num_symbols: Optional[int] = None) -> LatticeState:
    """
    Create a random N×N×N lattice.
    
    Replace this with your actual Livnium core initialization.
    For now, uses LivniumCoreSystem with random rotations.
    
    Args:
        N: Lattice size (must be odd, >= 3)
        num_symbols: Number of symbols (for generic arrays, not used for Livnium)
        
    Returns:
        Initialized LivniumCoreSystem or lattice state
    """
    # Use Livnium core system
    config = LivniumCoreConfig(
        lattice_size=N,
        enable_symbol_alphabet=True,
        enable_symbolic_weight=True,
        enable_90_degree_rotations=True
    )
    system = LivniumCoreSystem(config)
    
    # Randomize initial state via random rotations
    num_random_rotations = random.randint(5, 20)
    for _ in range(num_random_rotations):
        axis = random.choice(list(RotationAxis))
        system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
    
    return system


def apply_local_update_rules(state: LatticeState) -> LatticeState:
    """
    Apply one full 'time step' of your conflict-healing dynamics.
    
    Replace this body with your actual update function.
    For now, applies random rotations periodically.
    
    Args:
        state: Current lattice state (LivniumCoreSystem or array)
        
    Returns:
        Updated state (may modify in place)
    """
    # If it's a LivniumCoreSystem, apply rotation-based update
    if isinstance(state, LivniumCoreSystem):
        # Simple: rotate on random axis periodically
        step = getattr(state, '_step_counter', 0)
        state._step_counter = step + 1
        
        if step % 3 == 0:
            axis = random.choice(list(RotationAxis))
            state.rotate(axis, quarter_turns=1)
        
        return state
    
    # For generic arrays, return as-is (placeholder)
    # Replace with: from growth.lattice import step_omcube or similar
    return state


def apply_small_random_flips(
    state: LatticeState,
    fraction: float,
    num_symbols: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> LatticeState:
    """
    Flip a small random subset of lattice cells.
    
    Args:
        state: Lattice state to perturb
        fraction: Fraction of cells to flip (0.0 to 1.0)
        num_symbols: Number of symbols (for generic arrays)
        rng: Random number generator (optional)
        
    Returns:
        Perturbed state
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # If it's a LivniumCoreSystem, perturb symbols
    if isinstance(state, LivniumCoreSystem):
        perturbed = copy.deepcopy(state)
        
        if not perturbed.config.enable_symbol_alphabet:
            # Can't perturb symbols, so apply random rotation instead
            axis = random.choice(list(RotationAxis))
            perturbed.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
            return perturbed
        
        # Select random cells to perturb
        num_cells = len(perturbed.lattice)
        num_flips = max(1, int(num_cells * fraction))
        
        cells_to_perturb = random.sample(list(perturbed.lattice.keys()), num_flips)
        alphabet = perturbed.generate_alphabet(perturbed.lattice_size)
        
        for coords in cells_to_perturb:
            new_symbol = random.choice(alphabet)
            perturbed.set_symbol(coords, new_symbol)
        
        return perturbed
    
    # For generic numpy arrays
    perturbed = state.copy()
    N = state.shape[0]
    total_cells = N ** 3
    num_flips = max(1, int(total_cells * fraction))
    
    flat_indices = rng.choice(total_cells, size=num_flips, replace=False)
    coords = np.unravel_index(flat_indices, (N, N, N))
    
    if num_symbols is None:
        num_symbols = int(state.max()) + 1
    
    new_values = rng.integers(0, num_symbols, size=num_flips)
    perturbed[coords] = new_values
    
    return perturbed

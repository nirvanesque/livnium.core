"""
Stability Detection for Phase Transition Experiment

Encapsulates the definitions: energy settling, fixed point detection, and self-healing.
"""

import numpy as np
from typing import Any, List, Tuple, Optional

# Handle both relative and absolute imports
try:
    from .config import StabilityConfig
    from .energy_computer import compute_energy
    from .local_dynamics import (
        apply_local_update_rules,
        apply_small_random_flips,
    )
except ImportError:
    from config import StabilityConfig
    from energy_computer import compute_energy
    from local_dynamics import (
        apply_local_update_rules,
        apply_small_random_flips,
    )

# Types: can be LivniumCoreSystem or numpy array
LatticeState = Any


def hash_state(state: LatticeState) -> int:
    """
    Cheap hash of the lattice contents.
    
    You can swap in a better hash if needed.
    
    Args:
        state: Lattice state
        
    Returns:
        Integer hash
    """
    # If it's a LivniumCoreSystem, hash based on symbolic weights and classes
    if hasattr(state, 'lattice'):
        # LivniumCoreSystem
        state_parts = []
        sorted_coords = sorted(state.lattice.keys())
        for coords in sorted_coords:
            cell = state.lattice[coords]
            sw = cell.symbolic_weight if cell.symbolic_weight is not None else 0.0
            cls = cell.cell_class.value if cell.cell_class is not None else -1
            state_parts.append(f"{coords}:{sw:.2f}:{cls}")
        state_str = "|".join(state_parts)
        return hash(state_str.encode())
    
    # For numpy arrays
    arr = np.asarray(state, dtype=np.int64)
    return hash(arr.tobytes())


def energy_settled(energies: List[float], cfg: StabilityConfig) -> bool:
    """
    Check if |E(t+1) - E(t)| < ε_E over the last W steps.
    
    Args:
        energies: List of energy values over time
        cfg: Configuration
        
    Returns:
        True if energy has settled
    """
    if len(energies) < cfg.window_E + 1:
        return False
    
    tail = energies[-(cfg.window_E + 1):]
    diffs = [abs(tail[i+1] - tail[i]) for i in range(len(tail) - 1)]
    return all(d < cfg.epsilon_E for d in diffs)


def fixed_point_reached(hashes: List[int], cfg: StabilityConfig) -> bool:
    """
    Simple fixed point check: last W hashes all equal.
    
    Args:
        hashes: List of state hashes over time
        cfg: Configuration
        
    Returns:
        True if fixed point reached
    """
    if len(hashes) < cfg.window_H:
        return False
    
    tail = hashes[-cfg.window_H:]
    return all(h == tail[0] for h in tail)


def run_until_candidate_stable(
    state: LatticeState,
    cfg: StabilityConfig
) -> Tuple[bool, LatticeState, List[float]]:
    """
    Run dynamics up to T_max, return whether we hit a candidate stable state.
    
    Args:
        state: Initial lattice state
        cfg: Configuration
        
    Returns:
        Tuple of (is_stable, final_state, energy_curve)
    """
    energies: List[float] = []
    hashes: List[int] = []
    
    current = state
    
    for t in range(cfg.t_max):
        E_t = compute_energy(current, method=cfg.energy_method)
        H_t = hash_state(current)
        
        energies.append(E_t)
        hashes.append(H_t)
        
        if energy_settled(energies, cfg) and fixed_point_reached(hashes, cfg):
            return True, current, energies
        
        current = apply_local_update_rules(current)
    
    return False, current, energies


def passes_self_healing_test(
    stable_state: LatticeState,
    cfg: StabilityConfig,
    num_symbols: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> bool:
    """
    Perturb the state slightly and see if it returns to the original basin.
    
    Args:
        stable_state: System in stable state
        cfg: Configuration
        num_symbols: Number of symbols (for generic arrays)
        rng: Random number generator
        
    Returns:
        True if system returns to original basin
    """
    if rng is None:
        rng = np.random.default_rng(cfg.random_seed)
    
    original_E = compute_energy(stable_state, method=cfg.energy_method)
    original_hash = hash_state(stable_state)
    
    perturbed = apply_small_random_flips(
        state=stable_state,
        fraction=cfg.perturb_fraction,
        num_symbols=num_symbols,
        rng=rng,
    )
    
    energies: List[float] = []
    hashes: List[int] = []
    
    current = perturbed
    for t in range(cfg.t_perturb):
        E_t = compute_energy(current, method=cfg.energy_method)
        H_t = hash_state(current)
        
        energies.append(E_t)
        hashes.append(H_t)
        
        current = apply_local_update_rules(current)
    
    final_E = energies[-1]
    final_hash = hashes[-1]
    
    energy_close = abs(final_E - original_E) < cfg.epsilon_E
    pattern_same = final_hash == original_hash
    
    return energy_close or pattern_same

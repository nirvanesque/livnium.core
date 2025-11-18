"""
Task-Driven Stability Detection

Stability is now defined relative to a task:
1. System produces correct answer
2. Internals stop changing while working
3. Correct answer survives internal perturbation
"""

import numpy as np
from typing import Any, List, Tuple, Optional

# Handle imports (works as script or module)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

try:
    from .config import StabilityConfig
    from .tasks import Task
    from .energy_computer import compute_energy
    from .task_dynamics import apply_task_driven_update, perturb_task_state
except ImportError:
    from config import StabilityConfig
    from tasks import Task
    from energy_computer import compute_energy
    from task_dynamics import apply_task_driven_update, perturb_task_state

# Import LivniumCoreSystem for type checking
try:
    from core.classical.livnium_core_system import LivniumCoreSystem
except ImportError:
    # Fallback if not available
    LivniumCoreSystem = None

# Types
LatticeState = Any


def hash_state(state: LatticeState) -> int:
    """Hash lattice state."""
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
    """Check if energy (task loss) has settled."""
    if len(energies) < cfg.window_E + 1:
        return False
    
    tail = energies[-(cfg.window_E + 1):]
    diffs = [abs(tail[i+1] - tail[i]) for i in range(len(tail) - 1)]
    return all(d < cfg.epsilon_E for d in diffs)


def fixed_point_reached(hashes: List[int], cfg: StabilityConfig) -> bool:
    """Check if pattern has reached fixed point."""
    if len(hashes) < cfg.window_H:
        return False
    
    tail = hashes[-cfg.window_H:]
    return all(h == tail[0] for h in tail)


def run_until_task_stable(
    state: LatticeState,
    task: Task,
    cfg: StabilityConfig,
    use_moksha: bool = True
) -> Tuple[bool, LatticeState, List[float], List[bool], List[int]]:
    """
    Run task-driven dynamics until candidate stable state.
    
    OPTIMIZED: Can use MokshaEngine for fast fixed-point detection.
    
    Args:
        state: Initial state
        task: Task to solve
        cfg: Configuration
        use_moksha: Use recursive MokshaEngine for fast convergence (default: True)
        
    Returns:
        (is_stable, final_state, loss_curve, correctness_curve, hash_history)
    """
    # Try using MokshaEngine if available and requested
    if use_moksha and LivniumCoreSystem is not None and isinstance(state, LivniumCoreSystem):
        try:
            from .recursive_stability import run_until_moksha
            reached_moksha, final_state, losses, correctness = run_until_moksha(
                state, task, cfg
            )
            # Convert to expected format (add hash_history)
            hashes = [hash_state(final_state)] * len(losses)  # Simplified
            return reached_moksha, final_state, losses, correctness, hashes
        except (ImportError, AttributeError):
            # Fall back to manual detection if moksha not available
            pass
    
    # Manual stability detection (fallback)
    losses: List[float] = []
    correctness: List[bool] = []
    hashes: List[int] = []
    
    current = state
    
    for t in range(cfg.t_max):
        # Compute task loss (energy = wrongness)
        loss = task.compute_loss(current)
        answer = task.decode_answer(current)
        is_correct = task.is_correct(answer)
        H_t = hash_state(current)
        
        losses.append(loss)
        correctness.append(is_correct)
        hashes.append(H_t)
        
        # Check for task stability:
        # 1. Answer is correct
        # 2. Loss has settled (energy converged)
        # 3. Pattern is stable (fixed point)
        if (is_correct and 
            energy_settled(losses, cfg) and 
            fixed_point_reached(hashes, cfg)):
            return True, current, losses, correctness, hashes
        
        # Apply task-driven update
        current = apply_task_driven_update(
            current, task, t, method=cfg.update_rule, cfg=cfg
        )
    
    return False, current, losses, correctness, hashes


def passes_task_self_healing_test(
    stable_state: LatticeState,
    task: Task,
    cfg: StabilityConfig,
    rng: Optional[np.random.Generator] = None
) -> bool:
    """
    Test if system self-heals: correct answer survives internal perturbation.
    
    OPTIMIZED: Uses in-place perturbation to avoid deep copying.
    
    Key: task input stays fixed, only internal state is perturbed.
    System must restore the same correct answer.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.random_seed)
    
    # Get original answer
    original_answer = task.decode_answer(stable_state)
    original_correct = task.is_correct(original_answer)
    original_hash = hash_state(stable_state)
    
    if not original_correct:
        # Can't test self-healing if original wasn't correct
        return False
    
    # OPTIMIZATION: Create a copy only once for perturbation test
    # (We need to preserve original state for comparison)
    import copy
    perturbed = copy.deepcopy(stable_state)
    
    # Perturb internal state (task input stays fixed)
    perturb_task_state(
        perturbed, task, fraction=cfg.perturb_fraction, rng=rng
    )
    
    # Run dynamics to see if it restores correct answer
    losses: List[float] = []
    correctness: List[bool] = []
    
    current = perturbed
    for t in range(cfg.t_perturb):
        loss = task.compute_loss(current)
        answer = task.decode_answer(current)
        is_correct = task.is_correct(answer)
        
        losses.append(loss)
        correctness.append(is_correct)
        
        # Check if restored to correct answer
        if is_correct:
            # Also check if pattern is similar (optional)
            current_hash = hash_state(current)
            if current_hash == original_hash:
                return True  # Perfect restoration
            # Or just check if answer is correct
            return True
        
        # Apply update
        current = apply_task_driven_update(
            current, task, t, method=cfg.update_rule
        )
    
    # Check final state
    final_answer = task.decode_answer(current)
    final_correct = task.is_correct(final_answer)
    
    return final_correct and (final_answer == original_answer)


"""
Recursive Stability Detection using Moksha Engine

Uses the recursive layer's MokshaEngine for fast fixed-point convergence detection.
This is much faster than manual stability checking.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from typing import Any, List, Tuple, Optional, Dict
from core.recursive import RecursiveGeometryEngine, MokshaEngine, ConvergenceState
from core.classical.livnium_core_system import LivniumCoreSystem
from .config import StabilityConfig
from .tasks import Task

# Handle imports
try:
    from .task_dynamics import apply_task_driven_update
except ImportError:
    from task_dynamics import apply_task_driven_update


# Cache recursive engines for moksha (per system size)
_moksha_cache: Dict[int, RecursiveGeometryEngine] = {}

def run_until_moksha(
    system: LivniumCoreSystem,
    task: Task,
    cfg: StabilityConfig
) -> Tuple[bool, LivniumCoreSystem, List[float], List[bool]]:
    """
    Run task-driven dynamics until moksha (fixed point) is reached.
    
    OPTIMIZED: Reuses cached recursive engine to avoid rebuilding.
    
    Uses MokshaEngine for fast convergence detection instead of manual checking.
    
    Args:
        system: Initial lattice state
        task: Task to solve
        cfg: Configuration
        
    Returns:
        (reached_moksha, final_state, loss_curve, correctness_curve)
    """
    # OPTIMIZATION: Cache recursive engine per system size
    # For small N, use fewer layers to avoid overhead
    cache_key = system.lattice_size
    
    # Adaptive depth: more layers for larger N
    if system.lattice_size >= 7:
        max_depth = 3  # Full 3 layers for large N
    elif system.lattice_size >= 5:
        max_depth = 2  # 2 layers for medium N
    else:
        max_depth = 1  # 1 layer for small N (just base + one level)
    
    if cache_key not in _moksha_cache:
        # Initialize recursive engine with moksha (once per system size)
        # Use adaptive depth based on system size
        _moksha_cache[cache_key] = RecursiveGeometryEngine(
            base_geometry=system,
            max_depth=max_depth
        )
    
    # Reuse cached engine
    recursive_engine = _moksha_cache[cache_key]
    recursive_engine.base_geometry = system  # Update reference (no rebuild)
    
    moksha = recursive_engine.moksha
    moksha.convergence_threshold = cfg.epsilon_E  # Use config threshold
    moksha.stability_window = cfg.window_H  # Use config window
    
    losses: List[float] = []
    correctness: List[bool] = []
    
    current = system
    
    for t in range(cfg.t_max):
        # Compute task metrics
        loss = task.compute_loss(current)
        answer = task.decode_answer(current)
        is_correct = task.is_correct(answer)
        
        losses.append(loss)
        correctness.append(is_correct)
        
        # Update recursive engine's base geometry (just reference, no rebuild)
        recursive_engine.base_geometry = current
        
        # Check for moksha (fixed point) using recursive engine
        convergence = moksha.check_convergence()
        
        # If moksha reached and answer is correct, we're done
        if convergence == ConvergenceState.MOKSHA and is_correct:
            return True, current, losses, correctness
        
        # Apply task-driven update
        current = apply_task_driven_update(
            current, task, t, method=cfg.update_rule
        )
    
    return False, current, losses, correctness


def check_moksha_fast(
    system: LivniumCoreSystem,
    recursive_engine: Optional[RecursiveGeometryEngine] = None
) -> bool:
    """
    Fast moksha check using recursive engine.
    
    Args:
        system: System to check
        recursive_engine: Optional pre-initialized engine
        
    Returns:
        True if moksha (fixed point) reached
    """
    if recursive_engine is None:
        recursive_engine = RecursiveGeometryEngine(
            base_geometry=system,
            max_depth=1
        )
    else:
        recursive_engine.base_geometry = system
    
    convergence = recursive_engine.moksha.check_convergence()
    return convergence == ConvergenceState.MOKSHA


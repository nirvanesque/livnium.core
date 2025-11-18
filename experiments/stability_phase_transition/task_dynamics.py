"""
Task-Driven Dynamics for Stability Experiment

The physics only emerges when there's a task to solve.
This module implements task-driven update rules that minimize task loss.
"""

import numpy as np
import random
from typing import Any, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from core.classical.livnium_core_system import LivniumCoreSystem, RotationAxis
from core.config import LivniumCoreConfig

# Handle imports (works as script or module)
try:
    from .tasks import Task
except ImportError:
    from tasks import Task

# Types
LatticeState = Any


def apply_task_driven_update(
    state: LatticeState,
    task: Task,
    step: int,
    method: str = "loss_minimization",
    cfg: Optional[Any] = None
) -> LatticeState:
    """
    Apply task-driven update rules.
    
    The physics emerges from trying to solve the task.
    Energy = task loss, dynamics = minimize loss.
    
    Now supports Reasoning Layer (Layer 4) for intelligent search.
    
    Args:
        state: Lattice state (LivniumCoreSystem)
        task: Task to solve
        step: Current timestep
        method: Update method:
            - "loss_minimization": Manual rotation testing (fast)
            - "random_search": Random rotations
            - "gradient_like": Gradient-like updates
            - "reasoning": Use Reasoning Layer (Layer 4) - intelligent search
            - "hybrid_reasoning": Combine reasoning + simple updates
            - "recursive": Use Layer 0 recursive problem solving (for large N)
            - "hybrid_recursive": Combine recursive + simple updates (best for N>=5)
        cfg: Optional config (for hybrid_reasoning)
        
    Returns:
        Updated state
    """
    if not isinstance(state, LivniumCoreSystem):
        return state
    
    if method == "reasoning":
        # Use Reasoning Layer (Layer 4)
        try:
            from .reasoning_dynamics import apply_reasoning_constraint_satisfaction
            return apply_reasoning_constraint_satisfaction(state, task, step)
        except (ImportError, AttributeError):
            # Fallback to loss minimization if reasoning not available
            return _loss_minimization_update(state, task, step)
    
    elif method == "hybrid_reasoning":
        # Hybrid: reasoning + simple updates
        try:
            from .reasoning_dynamics import apply_hybrid_reasoning_update
            return apply_hybrid_reasoning_update(state, task, step, cfg)
        except (ImportError, AttributeError):
            # Fallback to loss minimization
            return _loss_minimization_update(state, task, step)
    
    elif method == "recursive":
        # Use Layer 0 recursive problem solving (3 layers like quantum)
        try:
            from .recursive_problem_solving import apply_recursive_problem_solving
            return apply_recursive_problem_solving(state, task, step, max_depth=3)
        except (ImportError, AttributeError):
            # Fallback to loss minimization
            return _loss_minimization_update(state, task, step)
    
    elif method == "hybrid_recursive":
        # Hybrid: recursive + simple updates (best for large systems)
        # Uses 3 layers like quantum structure
        try:
            from .recursive_problem_solving import apply_hybrid_recursive_update
            return apply_hybrid_recursive_update(state, task, step, cfg, max_depth=3)
        except (ImportError, AttributeError):
            # Fallback to loss minimization
            return _loss_minimization_update(state, task, step)
    
    elif method == "loss_minimization":
        return _loss_minimization_update(state, task, step)
    elif method == "random_search":
        return _random_search_update(state, task, step)
    elif method == "gradient_like":
        return _gradient_like_update(state, task, step)
    else:
        return _random_search_update(state, task, step)


def _loss_minimization_update(
    system: LivniumCoreSystem,
    task: Task,
    step: int
) -> LivniumCoreSystem:
    """
    Try rotations that reduce task loss.
    
    OPTIMIZED: Instead of deep copying, we rotate, test, then rotate back.
    This is much faster than copying the entire system.
    """
    current_loss = task.compute_loss(system)
    best_loss = current_loss
    best_axis = None
    best_turns = 1
    
    # Try all rotation options (3 axes × 3 turns = 9 options)
    # OPTIMIZATION: Rotate, test, then rotate back (no deep copy!)
    found_perfect = False
    for axis in RotationAxis:
        for turns in [1, 2, 3]:
            # Apply rotation
            system.rotate(axis, quarter_turns=turns)
            task.encode_into_lattice(system)
            
            # Test loss
            test_loss = task.compute_loss(system)
            
            if test_loss < best_loss:
                best_loss = test_loss
                best_axis = axis
                best_turns = turns
                
                # EARLY STOP: If we found perfect solution (loss=0), stop testing
                if best_loss == 0.0:
                    found_perfect = True
                    break
            
            # Rotate back to original state
            system.rotate(axis, quarter_turns=4 - turns)
            task.encode_into_lattice(system)
        
        # Early exit if perfect solution found
        if found_perfect:
            # Re-apply the perfect rotation (we rotated back above)
            system.rotate(best_axis, quarter_turns=best_turns)
            task.encode_into_lattice(system)
            return system
    
    # Apply best rotation if it improves
    if best_axis is not None and best_loss < current_loss:
        system.rotate(best_axis, quarter_turns=best_turns)
        task.encode_into_lattice(system)
    else:
        # No improvement, try random rotation occasionally
        if step % 5 == 0:
            axis = random.choice(list(RotationAxis))
            system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
            task.encode_into_lattice(system)
    
    return system


def _random_search_update(
    system: LivniumCoreSystem,
    task: Task,
    step: int
) -> LivniumCoreSystem:
    """
    Random search: try random rotations, keep if loss improves.
    """
    current_loss = task.compute_loss(system)
    
    # Try random rotation
    axis = random.choice(list(RotationAxis))
    turns = random.choice([1, 2, 3])
    
    system.rotate(axis, quarter_turns=turns)
    task.encode_into_lattice(system)
    
    new_loss = task.compute_loss(system)
    
    # If worse, rotate back (with some probability to allow exploration)
    if new_loss > current_loss and random.random() > 0.1:  # 10% exploration
        system.rotate(axis, quarter_turns=4 - turns)
        task.encode_into_lattice(system)
    
    return system


def _gradient_like_update(
    system: LivniumCoreSystem,
    task: Task,
    step: int
) -> LivniumCoreSystem:
    """
    Gradient-like update: try small changes, move in direction of lower loss.
    """
    current_loss = task.compute_loss(system)
    
    # Try small rotation
    axis = random.choice(list(RotationAxis))
    turns = 1  # Small step
    
    system.rotate(axis, quarter_turns=turns)
    task.encode_into_lattice(system)
    
    new_loss = task.compute_loss(system)
    
    # If worse, reverse
    if new_loss > current_loss:
        system.rotate(axis, quarter_turns=3)  # Reverse
        task.encode_into_lattice(system)
    
    return system


def initialize_lattice_with_task(N: int, task: Task) -> LivniumCoreSystem:
    """
    Initialize lattice and encode task into it.
    
    Args:
        N: Lattice size
        task: Task to encode
        
    Returns:
        Initialized system with task encoded
    """
    config = LivniumCoreConfig(
        lattice_size=N,
        enable_symbol_alphabet=True,
        enable_symbolic_weight=True,
        enable_90_degree_rotations=True
    )
    system = LivniumCoreSystem(config)
    
    # Encode task into lattice
    task.encode_into_lattice(system)
    
    return system


def perturb_task_state(
    system: LivniumCoreSystem,
    task: Task,
    fraction: float = 0.02,
    rng: Optional[np.random.Generator] = None
) -> LivniumCoreSystem:
    """
    Perturb internal state while keeping task input fixed.
    
    OPTIMIZED: In-place perturbation instead of deep copy.
    
    This tests self-healing: can the system restore the correct answer
    after internal noise, while the task remains the same?
    
    Args:
        system: System in stable state (will be modified in place)
        task: Task (input stays fixed)
        fraction: Fraction of cells to perturb
        rng: Random number generator
        
    Returns:
        Perturbed system (same object, modified in place)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # OPTIMIZATION: In-place perturbation (no deep copy)
    # Perturb internal cells (not the input region)
    num_cells = len(system.lattice)
    num_perturb = max(1, int(num_cells * fraction))
    
    cells_to_perturb = random.sample(list(system.lattice.keys()), num_perturb)
    alphabet = system.generate_alphabet(system.lattice_size)
    
    for coords in cells_to_perturb:
        new_symbol = random.choice(alphabet)
        system.set_symbol(coords, new_symbol)
    
    # Re-encode task (input stays the same)
    task.encode_into_lattice(system)
    
    return system


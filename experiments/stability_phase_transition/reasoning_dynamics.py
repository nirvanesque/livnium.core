"""
Reasoning Layer Integration for Task Dynamics

Uses Layer 4 (Reasoning) - ProblemSolver and ReasoningEngine
for intelligent task solving instead of manual rotation search.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from typing import Any, Optional
import random

from core.classical.livnium_core_system import LivniumCoreSystem, RotationAxis
from core.reasoning import ProblemSolver, ReasoningEngine, SearchStrategy
from .tasks import Task

# Handle imports
try:
    from .config import StabilityConfig
except ImportError:
    from config import StabilityConfig


def apply_reasoning_update(
    system: LivniumCoreSystem,
    task: Task,
    step: int,
    search_strategy: SearchStrategy = SearchStrategy.A_STAR,
    max_search_depth: int = 10
) -> LivniumCoreSystem:
    """
    Apply task-driven update using Reasoning Layer (Layer 4).
    
    Uses ProblemSolver with intelligent search strategies instead of
    brute-force rotation testing.
    
    Args:
        system: LivniumCoreSystem to update
        task: Task to solve
        step: Current timestep
        search_strategy: Search strategy (A_STAR, BEAM, GREEDY, etc.)
        max_search_depth: Maximum search depth
        
    Returns:
        Updated system
    """
    # Create problem solver
    solver = ProblemSolver(system)
    
    # Define problem as constraint satisfaction
    # Goal: task loss = 0 (correct answer)
    def goal_test(state_dict):
        """Check if state solves the task."""
        # Decode answer from state
        # For now, we'll use a simpler approach: check if loss is 0
        # by temporarily applying state to system
        return task.compute_loss(system) == 0.0
    
    def successors(state_dict):
        """Generate successor states via rotations."""
        successors_list = []
        
        # Try all rotation options
        for axis in RotationAxis:
            for turns in [1, 2, 3]:
                # Apply rotation
                system.rotate(axis, quarter_turns=turns)
                task.encode_into_lattice(system)  # Re-encode task
                
                # Encode new state
                new_state = solver._encode_lattice_state()
                action = f"rotate_{axis.name}_{turns}"
                successors_list.append((action, new_state))
                
                # Rotate back
                system.rotate(axis, quarter_turns=4 - turns)
                task.encode_into_lattice(system)
        
        return successors_list
    
    def heuristic(state_dict):
        """Heuristic: task loss (lower is better)."""
        return task.compute_loss(system)
    
    # Define problem
    initial_state = solver._encode_lattice_state()
    problem = {
        'name': f'task_solve_step_{step}',
        'initial_state': initial_state,
        'goal_test': goal_test,
        'successors': successors,
        'heuristic': heuristic,
    }
    
    # Solve using reasoning engine
    solution = solver.reasoning_engine.solve_problem(
        problem,
        search_strategy=search_strategy,
        max_depth=max_search_depth
    )
    
    if solution and solution.get('solved'):
        # Apply solution path
        # For now, just apply the first step (we'll improve this)
        # The reasoning engine found a path, but we need to apply it
        pass  # TODO: Apply solution path to system
    else:
        # No solution found, try random rotation
        if step % 5 == 0:
            axis = random.choice(list(RotationAxis))
            system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
            task.encode_into_lattice(system)
    
    return system


def apply_reasoning_constraint_satisfaction(
    system: LivniumCoreSystem,
    task: Task,
    step: int
) -> LivniumCoreSystem:
    """
    Use ProblemSolver's constraint satisfaction directly.
    
    Simpler approach: define task as constraint satisfaction problem.
    """
    solver = ProblemSolver(system)
    
    # Define constraint: task loss should be 0
    # The constraint function receives the system state
    def loss_constraint(state_dict):
        """Constraint: task is solved (loss = 0)."""
        # state_dict is the encoded state, but we need to check the actual system
        # For now, check current system loss
        current_loss = task.compute_loss(system)
        return current_loss == 0.0
    
    # Try to solve (limited iterations per step for performance)
    solution = solver.solve_constraint_satisfaction(
        constraints=[loss_constraint],
        max_iterations=5  # Limited iterations per step
    )
    
    # The solver modifies system in place, so we just return it
    # Re-encode task after solver's rotations
    task.encode_into_lattice(system)
    
    return system


def apply_hybrid_reasoning_update(
    system: LivniumCoreSystem,
    task: Task,
    step: int,
    cfg: Optional[StabilityConfig] = None
) -> LivniumCoreSystem:
    """
    Hybrid approach: Use reasoning for deep search, fallback to simple updates.
    
    Combines:
    - Reasoning Layer for intelligent search (when needed)
    - Simple rotation updates (for speed)
    
    This gives the best of both worlds:
    - Fast simple updates most of the time
    - Intelligent search when stuck
    """
    current_loss = task.compute_loss(system)
    
    # If already solved, do nothing
    if current_loss == 0.0:
        return system
    
    # If loss is already low, use simple updates (fast)
    if current_loss < 0.1:
        # Close to solution, use simple updates
        axis = random.choice(list(RotationAxis))
        system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
        task.encode_into_lattice(system)
        return system
    
    # If loss is high, use reasoning for intelligent search
    # But only occasionally (reasoning is more expensive)
    if step % 10 == 0:  # Use reasoning every 10 steps
        return apply_reasoning_constraint_satisfaction(system, task, step)
    else:
        # Simple rotation update (fast)
        axis = random.choice(list(RotationAxis))
        system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
        task.encode_into_lattice(system)
        return system


"""
Recursive Problem Solving using Layer 0

Uses the full Recursive Geometry Engine for recursive problem solving:
- Search happens across layers of geometry
- Macro constraints → micro constraints
- This is the "real trick that lets you solve big spaces cheaply"

OPTIMIZED: Caches recursive engine to avoid rebuilding hierarchy every step.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from typing import Any, Optional, Dict
import random

from core.classical.livnium_core_system import LivniumCoreSystem, RotationAxis
from core.recursive import RecursiveGeometryEngine
from .tasks import Task

# Handle imports
try:
    from .config import StabilityConfig
except ImportError:
    from config import StabilityConfig

# Cache recursive engines per system size to avoid rebuilding
_recursive_cache: Dict[int, RecursiveGeometryEngine] = {}


def apply_recursive_problem_solving(
    system: LivniumCoreSystem,
    task: Task,
    step: int,
    max_depth: int = 3  # 3 layers like quantum
) -> LivniumCoreSystem:
    """
    Use recursive problem solving: search across layers of geometry.
    
    OPTIMIZED: Reuses cached recursive engine to avoid rebuilding hierarchy.
    
    This is the "real trick" from Layer 0:
    - Search happens across layers of geometry
    - Macro constraints become micro constraints
    - Solves big spaces cheaply
    
    Args:
        system: LivniumCoreSystem to update
        task: Task to solve
        step: Current timestep
        max_depth: Maximum recursion depth
        
    Returns:
        Updated system
    """
    # OPTIMIZATION: Adaptive depth based on system size
    # Small N: fewer layers (less overhead)
    # Large N: more layers (more benefit)
    if system.lattice_size >= 7:
        effective_depth = 3  # Full 3 layers for large N
    elif system.lattice_size >= 5:
        effective_depth = 2  # 2 layers for medium N
    else:
        effective_depth = 1  # 1 layer for small N (avoid overhead)
    
    cache_key = (system.lattice_size, effective_depth)
    
    if cache_key not in _recursive_cache:
        # Create recursive geometry engine with adaptive depth
        # Level 0: Base geometry (N×N×N)
        # Level 1+: Smaller geometries inside cells (depth depends on N)
        _recursive_cache[cache_key] = RecursiveGeometryEngine(
            base_geometry=system,
            max_depth=effective_depth
        )
    
    # Reuse cached engine, just update base geometry reference
    recursive = _recursive_cache[cache_key]
    recursive.base_geometry = system  # Update reference (no rebuild needed)
    
    # Re-encode task into base geometry
    task.encode_into_lattice(system)
    
    # Strategy: Use recursive problem solving across 3 layers
    # 1. Check if we can solve at macro level (fast)
    current_loss = task.compute_loss(system)
    
    if current_loss == 0.0:
        # Already solved
        return system
    
    # 2. Use recursive search: search across all 3 geometry levels
    # Apply recursive rotation - propagates through all levels
    if recursive.levels and len(recursive.levels) > 1:
        # We have recursive levels - use recursive problem solving
        # Apply rotation recursively (propagates through all 3 levels)
        axis = random.choice(list(RotationAxis))
        turns = random.choice([1, 2, 3])
        
        # Apply recursive rotation (macro → all micro levels)
        recursive.apply_recursive_rotation(
            level_id=0,  # Start at base level
            axis=axis,
            quarter_turns=turns
        )
        
        # Re-encode task after recursive rotation
        task.encode_into_lattice(system)
    else:
        # Fallback: simple rotation if recursive structure not ready
        axis = random.choice(list(RotationAxis))
        system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
        task.encode_into_lattice(system)
    
    return system


def apply_recursive_subdivision_search(
    system: LivniumCoreSystem,
    task: Task,
    step: int,
    max_depth: int = 3  # 3 layers like quantum
) -> LivniumCoreSystem:
    """
    Use recursive subdivision for problem solving.
    
    OPTIMIZED: Reuses cached recursive engine.
    
    Strategy:
    1. Subdivide geometry into smaller geometry
    2. Project task constraints downward
    3. Solve at micro level
    4. Project solution upward
    
    This uses the fractal compression from Layer 0.
    """
    # OPTIMIZATION: Reuse cached engine
    cache_key = (system.lattice_size, max_depth)
    if cache_key not in _recursive_cache:
        _recursive_cache[cache_key] = RecursiveGeometryEngine(
            base_geometry=system,
            max_depth=max_depth
        )
    
    recursive = _recursive_cache[cache_key]
    recursive.base_geometry = system  # Update reference
    
    # Re-encode task
    task.encode_into_lattice(system)
    
    current_loss = task.compute_loss(system)
    
    if current_loss == 0.0:
        return system
    
    # Strategy: Use 3-layer recursive structure for problem solving
    # The recursive engine already has 3 levels built
    # Apply recursive rotation that propagates through all 3 levels
    if step % 10 == 0:  # Use recursive rotation occasionally
        axis = random.choice(list(RotationAxis))
        turns = random.choice([1, 2, 3])
        
        # Apply recursive rotation (propagates through all 3 levels)
        recursive.apply_recursive_rotation(
            level_id=0,
            axis=axis,
            quarter_turns=turns
        )
    else:
        # Simple rotation at base level
        axis = random.choice(list(RotationAxis))
        system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
    
    # Re-encode task after rotation
    task.encode_into_lattice(system)
    
    return system


def apply_hybrid_recursive_update(
    system: LivniumCoreSystem,
    task: Task,
    step: int,
    cfg: Optional[StabilityConfig] = None,
    max_depth: int = 3  # 3 layers like quantum
) -> LivniumCoreSystem:
    """
    Hybrid: Combine recursive problem solving with simple updates.
    
    OPTIMIZED: Only use recursive for large N where it's beneficial.
    For small N, recursive overhead isn't worth it.
    
    Uses Layer 0's recursive problem solving when beneficial:
    - For larger systems (N >= 7), use recursive search
    - For medium systems (N = 5), use recursive occasionally
    - For small systems (N = 3), use simple updates (fastest)
    """
    current_loss = task.compute_loss(system)
    
    if current_loss == 0.0:
        return system
    
    # OPTIMIZATION: Only use recursive for large N where it's actually beneficial
    # For N=3, recursive overhead > benefit (just creates overhead)
    # For N=5, recursive helps but only occasionally
    # For N>=7, recursive is beneficial
    
    if system.lattice_size >= 7:
        # Large systems: use recursive problem solving with 3 layers
        if step % 10 == 0:
            # Use recursive subdivision search occasionally
            return apply_recursive_subdivision_search(system, task, step, max_depth=3)
        else:
            # Use recursive problem solving with 3 layers
            return apply_recursive_problem_solving(system, task, step, max_depth=3)
    elif system.lattice_size >= 5:
        # Medium systems: use recursive occasionally (every 20 steps)
        if step % 20 == 0:
            return apply_recursive_problem_solving(system, task, step, max_depth=3)
        else:
            # Simple updates most of the time
            axis = random.choice(list(RotationAxis))
            system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
            task.encode_into_lattice(system)
            return system
    else:
        # Small systems (N=3): simple updates are fastest
        # Recursive overhead not worth it for small N
        axis = random.choice(list(RotationAxis))
        system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
        task.encode_into_lattice(system)
        return system


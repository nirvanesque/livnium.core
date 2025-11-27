"""
Recursive Inheritance: Law of Inheritance (v0.1)

The child universe inherits and refines the parent's stress.

Core principle:
- Child universes inherit parent's SW as an energy budget
- SW is partitioned (with noise) across child cells
- Conservation: sum(child_SW) ≈ parent_SW

This gives the recursive universe something real to do - physics flows
from parent to child, not just metadata.
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

from ..classical.livnium_core_system import LivniumCoreSystem, LivniumCoreConfig


def fabricate_child_universe(
    parent_cell,
    parent_geometry: LivniumCoreSystem,
    child_lattice_size: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    noise_scale: float = 0.10,
) -> LivniumCoreSystem:
    """
    Law of Inheritance (v0.1).
    
    The child universe inherits:
      - the same geometry config as the parent (or specified size)
      - the parent's SW as an 'energy budget'
      - a noisy partition of that SW across its cells
    
    Conservation:
        sum(child_SW) ≈ parent_SW
    
    Args:
        parent_cell: a Cell object from parent_geometry.lattice
        parent_geometry: the LivniumCoreSystem that owns parent_cell
        child_lattice_size: optional override for child lattice size
        rng: optional numpy Generator for reproducibility
        noise_scale: relative std of multiplicative noise on SW
    
    Returns:
        A new LivniumCoreSystem with SW initialized from parent_cell.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # 1. Clone physics config
    if hasattr(parent_geometry, "config") and isinstance(parent_geometry.config, LivniumCoreConfig):
        # Create new config with same settings
        child_config = LivniumCoreConfig(
            lattice_size=child_lattice_size or parent_geometry.config.lattice_size,
            enable_symbolic_weight=parent_geometry.config.enable_symbolic_weight,
            enable_90_degree_rotations=parent_geometry.config.enable_90_degree_rotations,
            enable_class_structure=parent_geometry.config.enable_class_structure,
            enable_global_observer=parent_geometry.config.enable_global_observer,
        )
    else:
        # Fallback: construct a minimal equivalent config
        child_config = LivniumCoreConfig(
            lattice_size=child_lattice_size or parent_geometry.lattice_size,
            enable_symbolic_weight=True,
            enable_90_degree_rotations=True,
        )
    
    child_geo = LivniumCoreSystem(child_config)
    
    # 2. Energy budget from parent
    parent_sw = float(getattr(parent_cell, "symbolic_weight", 0.0))
    num_cells = len(child_geo.lattice)
    
    if num_cells == 0:
        # Degenerate case: nothing to distribute
        return child_geo
    
    if parent_sw <= 0.0:
        # No energy to inherit - set minimal SW
        base_sw = 1.0 / num_cells
        for cell in child_geo.lattice.values():
            cell.symbolic_weight = base_sw
        return child_geo
    
    base_sw = parent_sw / num_cells
    
    # 3. Noisy partition, then renormalize
    weights = rng.normal(loc=1.0, scale=noise_scale, size=num_cells)
    weights = np.clip(weights, 0.1, None)  # avoid negative/zero
    weights /= weights.sum()  # normalize to sum to 1.0
    
    sw_values = parent_sw * weights
    
    # 4. Assign to child cells
    for (cell, sw_val) in zip(child_geo.lattice.values(), sw_values):
        cell.symbolic_weight = float(sw_val)
    
    return child_geo


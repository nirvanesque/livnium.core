"""
Rule 30 Cellular Automaton Integration with Livnium Geometry

This module integrates Rule 30 CA into the Livnium geometric engine.
"""

from experiments.rule30.rule30_core import generate_rule30
from experiments.rule30.center_column import extract_center_column
from experiments.rule30.geometry_embed import embed_into_cube, Rule30Path
from experiments.rule30.diagnostics import (
    compute_divergence_path,
    compute_tension_curve,
    compute_basin_depth,
    compute_all_diagnostics
)
from experiments.rule30.recursive_embed import (
    embed_recursive,
    analyze_recursive_patterns
)

__all__ = [
    'generate_rule30',
    'extract_center_column',
    'embed_into_cube',
    'Rule30Path',
    'compute_divergence_path',
    'compute_tension_curve',
    'compute_basin_depth',
    'compute_all_diagnostics',
    'embed_recursive',
    'analyze_recursive_patterns'
]


# """
# Livnium Core v1.0 - Vector-Based Physics Engine

# Layer 0: Pure physics. No tokens, no labels, no tasks.
# """

# from .vector_state import VectorState
# from .physics_laws import alignment, divergence_from_alignment, tension
# from .vector_collapse_engine import VectorCollapseEngine
# from .basin_field import (
#     BasinField,
#     BasinAnchor,
#     route_to_basin,
#     update_basin_center,
#     maybe_spawn_basin,
#     prune_and_merge,
# )
# from .intension import LessonLogger, IntensionNet, apply_intension

# __all__ = [
#     'VectorState',
#     'alignment',
#     'divergence_from_alignment',
#     'tension',
#     'VectorCollapseEngine',
#     'BasinField',
#     'BasinAnchor',
#     'route_to_basin',
#     'update_basin_center',
#     'maybe_spawn_basin',
#     'prune_and_merge',
#     'LessonLogger',
#     'IntensionNet',
#     'apply_intension',
# ]


"""
Core Physics Engine (Livnium v4)

Exports the main physics components for external use.
"""

from .vector_state import VectorState
from .physics_laws import alignment, divergence_from_alignment, tension, compute_om_lo_physics
from .vector_collapse_engine import VectorCollapseEngine
from .basin_field import (
    BasinField,
    route_to_basin,
    route_to_basin_vectorized,
    maybe_spawn_vectorized,
    prune_and_merge_vectorized
)
from .intension import IntensionNet, LessonLogger, apply_intension

__all__ = [
    'VectorState',
    'alignment',
    'divergence_from_alignment',
    'tension',
    'compute_om_lo_physics',
    'VectorCollapseEngine',
    'BasinField',
    'route_to_basin',
    'route_to_basin_vectorized',
    'maybe_spawn_vectorized',
    'prune_and_merge_vectorized',
    'IntensionNet',
    'LessonLogger',
    'apply_intension'
]
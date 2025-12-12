"""
Livnium-O Flow Engine

Implements O-A7: The Flow Law - Continuous Tangential Dynamics

This module provides the dynamics layer that transforms Livnium-O from
static geometry into a living, computational universe.

Without this, Livnium-O is frozen.
With this, Livnium-O becomes alive.
"""

from .flow_engine import (
    FlowEngine,
    move_neighbor,
    evolve_system,
    compute_tangential_velocity,
    create_velocity_field,
)

__all__ = [
    'FlowEngine',
    'move_neighbor',
    'evolve_system',
    'compute_tangential_velocity',
    'create_velocity_field',
]


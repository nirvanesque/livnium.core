"""
Stability Phase Transition Experiment

Finds the smallest N×N×N lattice where self-healing stability emerges.

This is a modular, pluggable experiment framework that you can drop into
your Livnium universe. Once you plug in your real:
  - lattice initialization (Livnium core)
  - local update rules (τ-reduction)
  - energy function (τ / φ-tension)

...this experiment will show:
  - for which N the lattice acquires memory of its own shape
  - where p_stable(N) first becomes non-zero → your N*_crit
  - how quickly the universe finds its attractors as a function of size

That's your first empirical map of the truth manifold's birth point.
"""

# Main exports
from .config import StabilityConfig, StabilityExperimentConfig
from .experiment import run_experiment

# Core functions
from .local_dynamics import (
    random_init_lattice,
    apply_local_update_rules,
    apply_small_random_flips,
)
from .energy_computer import compute_energy
from .stability_detector import (
    hash_state,
    energy_settled,
    fixed_point_reached,
    run_until_candidate_stable,
    passes_self_healing_test,
)

__all__ = [
    # Config
    "StabilityConfig",
    "StabilityExperimentConfig",  # Alias for backward compatibility
    
    # Main runner
    "run_experiment",
    
    # Core functions
    "random_init_lattice",
    "apply_local_update_rules",
    "apply_small_random_flips",
    "compute_energy",
    "hash_state",
    "energy_settled",
    "fixed_point_reached",
    "run_until_candidate_stable",
    "passes_self_healing_test",
]


"""
Configuration for Stability Phase Transition Experiment

Central place to tune everything.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


@dataclass
class StabilityConfig:
    """Configuration for the stability phase transition experiment."""
    
    # Lattice sizes to test (must be odd, >= 3)
    lattice_sizes: List[int] = field(default_factory=lambda: [3, 5, 7, 9, 11])
    
    # Number of random initial configurations per size
    runs_per_size: int = 200
    
    # Dynamics
    t_max: int = 2000          # Maximum timesteps per run
    t_perturb: int = 500        # Maximum timesteps for perturbation test
    
    # Stability thresholds
    epsilon_E: float = 1e-3     # Energy convergence threshold
    window_E: int = 20          # W for energy settling window
    window_H: int = 10          # Window to confirm fixed point
    
    # Perturbation
    perturb_fraction: float = 0.02  # 2% of cells
    
    # Randomness
    random_seed: int = 42
    
    # Output
    results_dir: str = "results/stability_phase_transition"
    run_tag: str = "v1_stability_scan"
    
    # Task configuration
    task_type: str = "parity_3bit"  # or "classification", "constraint"
    task_params: Dict[str, Any] = field(default_factory=dict)  # Task-specific parameters
    
    # Update rule (for task-driven dynamics)
    update_rule: str = "loss_minimization"  # Options:
    #   - "loss_minimization": Manual rotation testing (fast, default)
    #   - "random_search": Random rotations
    #   - "gradient_like": Gradient-like updates
    #   - "reasoning": Use Reasoning Layer (Layer 4) - intelligent search
    #   - "hybrid_reasoning": Combine reasoning + simple updates
    #   - "recursive": Use Layer 0 recursive problem solving (for large N)
    #   - "hybrid_recursive": Combine recursive + simple (best for N>=5)
    
    # Recursive layer optimization
    use_moksha: bool = True  # Use MokshaEngine for fast fixed-point detection
    
    # Energy method (legacy, for non-task experiments)
    energy_method: str = "symbolic_weight_variance"  # or "local_tension", "conflict_count"
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate all sizes are odd and >= 3
        for n in self.lattice_sizes:
            if n < 3:
                raise ValueError(f"Lattice size must be >= 3, got {n}")
            if n % 2 == 0:
                raise ValueError(f"Lattice size must be odd, got {n}")
        
        if self.perturb_fraction <= 0 or self.perturb_fraction > 0.1:
            raise ValueError(f"Perturbation fraction should be in (0, 0.1], got {self.perturb_fraction}")
        
        if self.epsilon_E <= 0:
            raise ValueError(f"Energy epsilon must be positive, got {self.epsilon_E}")


# Alias for backward compatibility
StabilityExperimentConfig = StabilityConfig


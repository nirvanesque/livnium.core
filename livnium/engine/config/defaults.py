"""
Engine Config Defaults: Hyperparameters

All hyperparameters (versioned thresholds, learning rates, force strengths, etc.)
belong here, NOT in kernel/constants.py.

These can be overridden per domain/experiment.
"""

# Divergence Shaping
ALIGN_SHAPE: float = 2.0
"""
Divergence shaping factor for tanh saturation.
Higher values saturate sooner.
"""

# Anchor Force Strengths
STRENGTH_ENTAIL: float = 0.1
"""Force strength for entailment anchor."""

STRENGTH_CONTRA: float = 0.1
"""Force strength for contradiction anchor."""

STRENGTH_NEUTRAL: float = 0.05
"""Force strength for neutral anchor."""

# Basin Thresholds (Versioned)
BASIN_TENSION_THRESHOLD_V3: float = 0.15
"""Basin tension threshold for v3 (tension level above which new basin is spawned)."""

BASIN_TENSION_THRESHOLD_V4: float = 0.20
"""Basin tension threshold for v4 (tension level above which new basin is spawned)."""

BASIN_ALIGN_THRESHOLD: float = 0.6
"""Basin alignment threshold (alignment level below which new basin may be spawned)."""

BASIN_ANCHOR_LR: float = 0.05
"""Learning rate for basin anchor center EMA update."""

BASIN_PRUNE_MIN_COUNT: int = 10
"""Minimum count to keep basin anchor (prune if below this)."""

BASIN_PRUNE_MERGE_COS: float = 0.97
"""Cosine similarity threshold for merging basins."""

# Geometric Constraints
MAX_NORM: float = 10.0
"""Maximum vector norm allowed during collapse (norm clipping)."""

EPS_NOISE: float = 0.01
"""Symmetry-breaking noise for initial state (std dev)."""

# Training Constants
ALIGN_SHAPE_TRAINING: float = 2.0
"""Divergence shaping factor for training (may differ from inference)."""

D_MARGIN: float = 0.4
"""Margin for negative energy in Livnium loss."""

NEG_WEIGHT: float = 5.0
"""Weight for negative energy term in loss."""

LAMBDA_ENERGY: float = 0.1
"""Weight for energy term in LivniumLoss."""

LAMBDA_TENSION: float = 0.1
"""Weight for tension term in LivniumLoss."""

NORM_REG_WEIGHT: float = 1e-4
"""Weight for norm regularization."""

EXPLORATION_COST: float = 0.1
"""Cost per new basin spawned."""

COST_COEFFICIENT: float = 0.05
"""Movement cost coefficient in reward function."""

GHOST_MASS: float = 0.1
"""Mass for ghost basin (unknown words)."""


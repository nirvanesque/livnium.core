"""
Livnium Quantum Layer - Configuration Constants

This file contains constants for the quantum layer, following Livnium's
architectural principle of centralizing configurable values.

NOTE: This quantum layer is from the archived quantum system and is included
as an experimental feature. Some constants reference the archived classical
Livnium system and may need adaptation.
"""

# Precision thresholds
QUANTUM_PROBABILITY_EPSILON = 1e-9
"""Minimum probability threshold for quantum state normalization."""

# Meta-interference parameters
META_INTERFERENCE_BIAS_FACTOR = 0.1
"""Bias factor for meta-interference amplitude modulation."""

# Geometry-quantum coupling (from archived classical system)
MAX_SYMBOLIC_WEIGHT = 27
"""Maximum symbolic weight value (corners in 3x3x3 lattice)."""

INVARIANT_TOLERANCE = 0.1
"""Tolerance for invariance checks in geometry coupling."""

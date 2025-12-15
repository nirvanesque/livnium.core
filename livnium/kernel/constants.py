"""
Kernel Constants: Axiom-Level Constants Only

ONLY law-level constants belong here. All hyperparameters go to engine/config/defaults.py.

These constants are:
- DIVERGENCE_PIVOT: Core physics law (0.38)
- K_O, K_T, K_C: Equilibrium constants (axioms)

NOT here (these are hyperparameters):
- Learning rates
- Thresholds
- Force strengths
- Versioned parameters
"""

# Core Physics Constant
DIVERGENCE_PIVOT: float = 0.38
"""
Divergence pivot/barrier constant.

This is the core LIVNIUM law: divergence = DIVERGENCE_PIVOT - alignment
Empirically anchors neutral equilibrium in SNLI space.

This is a LAW, not a hyperparameter.
"""

# Equilibrium Constants (Axioms)
K_O: int = 9
"""
Livnium-O (Spherical System) equilibrium constant.
Normalizes energy across exposed classes.
"""

K_T: int = 27
"""
Livnium-T (Tetrahedral System) equilibrium constant.
Normalization constant for the tetrahedral universe.
"""

K_C: int = 9
"""
Livnium-C (Circular System) equilibrium constant.
Normalization constant for the circular universe.
"""


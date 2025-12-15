"""
Kernel Physics: Pure Formulas Using Ops Protocol

CRITICAL SCOPE LIMIT: Kernel physics = measurement + invariance, NOT motion.

✅ Allowed:
- alignment() - measurement
- divergence() - measurement + invariance (uses pivot constant)
- tension() - measurement

❌ NOT allowed (belongs in engine):
- Forces
- Gradients
- Dynamics
- Attraction laws
- Motion/update rules

Kernel imports nothing except typing. All tensor operations go through Ops protocol.
"""

from typing import Tuple
from .ops import Ops
from .types import State
from .constants import DIVERGENCE_PIVOT


def alignment(ops: Ops, a: State, b: State) -> float:
    """
    Compute alignment (cosine similarity) between two state vectors.
    
    Alignment is the core geometric measure:
    - alignment = 1.0 → vectors point in same direction
    - alignment = -1.0 → vectors point in opposite directions
    - alignment = 0.0 → vectors are orthogonal
    
    This is pure measurement - no forces, no dynamics.
    
    Args:
        ops: Ops instance for tensor operations
        a: First state vector
        b: Second state vector
        
    Returns:
        Scalar alignment value (cosine similarity)
    """
    a_vec = a.vector()
    b_vec = b.vector()
    
    # Normalize both vectors
    a_n = ops.normalize(a_vec, dim=-1)
    b_n = ops.normalize(b_vec, dim=-1)
    
    # Compute dot product (cosine similarity)
    return ops.dot(a_n, b_n)


def divergence(ops: Ops, a: State, b: State) -> float:
    """
    Compute divergence from alignment using the core LIVNIUM law.
    
    Law: divergence = DIVERGENCE_PIVOT - alignment
    
    Physical meaning:
    - divergence < 0 → vectors pull inward (entailment)
    - divergence ≈ 0 → forces balance (neutral)
    - divergence > 0 → vectors push apart (contradiction)
    
    This is measurement + invariance (uses pivot constant).
    The actual force application belongs in the engine.
    
    Args:
        ops: Ops instance for tensor operations
        a: First state vector
        b: Second state vector
        
    Returns:
        Divergence value
    """
    align = alignment(ops, a, b)
    return DIVERGENCE_PIVOT - align


def tension(ops: Ops, divergence: float) -> float:
    """
    Compute tension from divergence.
    
    Tension measures the "stress" in the geometric field.
    High tension = system is under stress (high divergence magnitude).
    
    This is pure measurement - the magnitude of divergence.
    
    Args:
        ops: Ops instance (for consistency, though not strictly needed here)
        divergence: Divergence value
        
    Returns:
        Tension (absolute value of divergence)
    """
    return abs(divergence)


def compute_om_lo_physics(ops: Ops, om: State, lo: State) -> Tuple[float, float, float]:
    """
    Compute full physics from OM and LO state vectors.
    
    This is the complete physics computation:
    1. Alignment (cosine similarity)
    2. Divergence (DIVERGENCE_PIVOT - alignment)
    3. Tension (|divergence|)
    
    All three are measurements/invariants. No forces or dynamics.
    
    Args:
        ops: Ops instance for tensor operations
        om: OM state (premise direction)
        lo: LO state (hypothesis direction)
        
    Returns:
        Tuple of (alignment, divergence, tension)
    """
    align = alignment(ops, om, lo)
    div = divergence(ops, om, lo)
    tens = tension(ops, div)
    
    return align, div, tens


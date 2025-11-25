"""
Invariant Solver V3: Linear Algebra Approach with Exact Rationals

Builds a linear system encoding D3(row) = D3(rule30_step(row)) for many rows,
then finds the nullspace (candidate invariants) using exact rational arithmetic.
"""

import random
from typing import Dict, List, Tuple
from fractions import Fraction
import math

import numpy as np

try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("Warning: sympy not available, using numpy (approximate)")

from experiments.rule30.divergence_v3 import enumerate_patterns, pattern_frequencies_3_rational
from experiments.rule30.rule30_algebra import rule30_step


Pattern = Tuple[int, int, int]


def random_row(n: int) -> List[int]:
    """Generate a random binary row."""
    return [random.randint(0, 1) for _ in range(n)]


def build_invariance_system_rational(
    num_rows: int = 200,
    row_length: int = 200,
    cyclic: bool = True
):
    """
    Build a system using exact rationals.
    
    Returns either sympy.Matrix (if sympy available) or numpy array.
    """
    patterns = enumerate_patterns()
    P = len(patterns)  # Should be 8
    
    if SYMPY_AVAILABLE:
        rows = []
        for _ in range(num_rows):
            row = random_row(row_length)
            nxt = rule30_step(row, cyclic=cyclic)
            
            freq_before = pattern_frequencies_3_rational(row, cyclic=cyclic)
            freq_after = pattern_frequencies_3_rational(nxt, cyclic=cyclic)
            
            vec = []
            for p in patterns:
                fb = freq_before[p]
                fa = freq_after[p]
                vec.append(sympy.Rational(fb - fa))
            
            rows.append(vec)
        
        return sympy.Matrix(rows)
    else:
        # Fallback to numpy with floats
        rows = []
        for _ in range(num_rows):
            row = random_row(row_length)
            nxt = rule30_step(row, cyclic=cyclic)
            
            freq_before = pattern_frequencies_3_rational(row, cyclic=cyclic)
            freq_after = pattern_frequencies_3_rational(nxt, cyclic=cyclic)
            
            vec = []
            for p in patterns:
                fb = freq_before[p]
                fa = freq_after[p]
                vec.append(float(fb - fa))
            
            rows.append(vec)
        
        return np.array(rows, dtype=float)


def build_invariance_system(
    num_rows: int = 200,
    row_length: int = 200,
    cyclic: bool = True
) -> np.ndarray:
    """
    Build a matrix A such that A @ w = 0 encodes D3(row) = D3(rule30_step(row))
    for all sampled rows.
    
    Each row of A is (freq_before - freq_after) for one random row.
    
    Args:
        num_rows: Number of random rows to sample
        row_length: Length of each row
        cyclic: Use cyclic boundary conditions
        
    Returns:
        Matrix A of shape (num_rows, 8) where columns correspond to 8 patterns
    """
    patterns = enumerate_patterns()
    P = len(patterns)  # Should be 8
    
    rows = []
    
    for _ in range(num_rows):
        row = random_row(row_length)
        nxt = rule30_step(row, cyclic=cyclic)
        
        freq_before = pattern_frequencies_3_rational(row, cyclic=cyclic)
        freq_after = pattern_frequencies_3_rational(nxt, cyclic=cyclic)
        
        # Build constraint vector: freq_before - freq_after
        vec = []
        for p in patterns:
            fb = freq_before[p]
            fa = freq_after[p]
            vec.append(float(fb - fa))
        
        rows.append(vec)
    
    A = np.array(rows, dtype=float)
    return A


def find_nullspace(A: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Compute an approximate basis for the nullspace of A.
    
    Uses SVD: nullspace vectors correspond to small singular values.
    
    Args:
        A: Matrix of shape (m, n)
        tol: Tolerance for considering singular values as zero
        
    Returns:
        Matrix of shape (n, k) whose columns form a basis for the nullspace
        (k = dimension of nullspace)
    """
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    
    # Small singular values correspond to nullspace
    null_mask = (s < tol)
    
    if not null_mask.any():
        # No nullspace found
        return np.zeros((A.shape[1], 0))
    
    # Columns of vh corresponding to small singular values
    null_space = vh[null_mask, :].T
    
    return null_space


def find_nullspace_exact(A_rational):
    """
    Find exact nullspace using sympy.
    
    Args:
        A_rational: sympy.Matrix
        
    Returns:
        List of sympy vectors (basis for nullspace)
    """
    if not SYMPY_AVAILABLE:
        raise RuntimeError("sympy not available for exact computation")
    
    nullspace = A_rational.nullspace()
    return nullspace


def analyze_nullspace(nullspace: np.ndarray) -> Dict:
    """
    Analyze the nullspace to understand invariant structure.
    
    Args:
        nullspace: Nullspace basis matrix
        
    Returns:
        Dict with analysis results
    """
    if nullspace.shape[1] == 0:
        return {
            'dimension': 0,
            'has_invariants': False,
            'message': 'No non-trivial nullspace found (no invariant of this form)'
        }
    
    dimension = nullspace.shape[1]
    
    # Check if any basis vectors are non-zero
    norms = np.linalg.norm(nullspace, axis=0)
    non_zero = np.sum(norms > 1e-6)
    
    return {
        'dimension': dimension,
        'has_invariants': dimension > 0,
        'non_zero_vectors': non_zero,
        'message': f'Found {dimension}-dimensional nullspace ({non_zero} non-zero vectors)'
    }


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize vector so max absolute value is 1."""
    max_abs = np.max(np.abs(v))
    if max_abs > 1e-10:
        return v / max_abs
    return v


def gcd_normalize_rational(weights: Dict[Pattern, Fraction]) -> Dict[Pattern, Fraction]:
    """
    Normalize rational weights by dividing by GCD to get smallest integer coefficients.
    
    Args:
        weights: Dict mapping patterns to Fraction weights
        
    Returns:
        Normalized weights with integer coefficients
    """
    # Get all numerators and denominators
    numerators = []
    denominators = []
    
    for w in weights.values():
        if w != 0:
            numerators.append(abs(w.numerator))
            denominators.append(w.denominator)
    
    if not numerators:
        return weights
    
    # Find GCD of numerators and LCM of denominators
    from math import gcd
    from functools import reduce
    
    num_gcd = reduce(gcd, numerators) if numerators else 1
    
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    den_lcm = reduce(lcm, denominators) if denominators else 1
    
    # Scale all weights
    scale = Fraction(num_gcd, den_lcm)
    normalized = {p: w / scale for p, w in weights.items()}
    
    return normalized


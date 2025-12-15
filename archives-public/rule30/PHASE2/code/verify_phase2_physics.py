#!/usr/bin/env python3
"""
Phase 2 Physical Validation

Tests whether the 4-bit constraint system accurately describes REAL Rule 30 grid physics.

This is a "reality check" - does A*x = b hold for actual CA evolution?

This complements verify_phase2_integrity.py which tests algebraic consistency.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from typing import Dict, Tuple

try:
    import sympy
    from sympy import expand, Eq
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("ERROR: sympy required")
    sys.exit(1)

# Add project root to path (go up: code -> PHASE2 -> rule30 -> experiments -> root)
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from PHASE2/code (same directory)
# Files in PHASE2/code can import from each other directly
from four_bit_system import (
    build_4bit_constraint_system,
    enumerate_4bit_patterns,
    center_value_4bit
)


def build_numerical_matrix(system) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Convert symbolic system to numerical A, b.
    
    Returns:
        (A, b, variables) where A is (num_eqs, num_vars) and b is (num_eqs,)
    """
    all_vars = system['variables']
    rows = []
    rhs = []
    
    for eq in system['equations']:
        expr = eq.lhs - eq.rhs if isinstance(eq, Eq) else eq
        expr = expand(expr)
        coeff_dict = expr.as_coefficients_dict()
        
        coeffs = [float(coeff_dict.get(v, 0.0)) for v in all_vars]
        constant = float(coeff_dict.get(1, 0.0))
        
        rows.append(coeffs)
        rhs.append(-constant)
    
    return np.array(rows, dtype=float), np.array(rhs, dtype=float), all_vars


def verify_system_physics(verbose: bool = True, width: int = 10000, steps: int = 50) -> Tuple[bool, Dict]:
    """
    Check #1: The Reality Check.
    
    Run a real Rule 30 grid simulation and verify the data satisfies A*x = b.
    
    Args:
        verbose: Print detailed output
        width: Width of CA grid
        steps: Number of evolution steps to test
        
    Returns:
        (is_valid, details) where details contains error statistics
    """
    if verbose:
        print("="*70)
        print("CHECK #1: Physical Validity (Grid vs Equations)")
        print("="*70)
        print()
    
    # 1. Build the System
    system = build_4bit_constraint_system(remove_flow=True)
    A, b, variables = build_numerical_matrix(system)
    
    if verbose:
        print(f"Constraint system built:")
        print(f"  - Matrix shape: {A.shape}")
        print(f"  - Variables: {len(variables)}")
        print()
    
    # 2. Build variable name mapping
    # Variables are like: f_0000_t, f_0000_{t+1}, c_t, c_{t+1}
    var_map = {}
    patterns = enumerate_4bit_patterns()
    pattern_str = {p: ''.join(str(b) for b in p) for p in patterns}
    
    for i, var in enumerate(variables):
        var_str = str(var)
        var_map[var_str] = i
    
    # 3. Run Real Simulation
    if verbose:
        print(f"Running {steps} steps on grid width {width}...")
        print()
    
    np.random.seed(42)
    cells = np.random.randint(0, 2, width, dtype=np.uint8)
    
    errors = []
    max_error = 0.0
    min_error = float('inf')
    
    # Helper to count 4-bit pattern frequencies
    def get_freqs(c):
        """
        Efficient 4-bit pattern frequency counting.
        Patterns are encoded as: 8*a + 4*b + 2*c + 1*d
        """
        patterns_encoded = (8 * c + 
                           4 * np.roll(c, -1) + 
                           2 * np.roll(c, -2) + 
                           1 * np.roll(c, -3))
        counts = np.bincount(patterns_encoded, minlength=16)
        return counts.astype(float) / len(c)
    
    for t in range(steps):
        # State t
        f_t = get_freqs(cells)
        
        # Compute center column value (sum of frequencies where second bit = 1)
        c_t = 0.0
        for i, p in enumerate(patterns):
            a, b, c, d = p
            if b == 1:  # Second bit is center
                c_t += f_t[i]
        
        # Evolve grid (Rule 30)
        l = np.roll(cells, 1)
        c = cells
        r = np.roll(cells, -1)
        cells_next = np.bitwise_xor(l, np.bitwise_or(c, r))
        
        # State t+1
        f_tp1 = get_freqs(cells_next)
        
        # Compute center column value at t+1
        c_tp1 = 0.0
        for i, p in enumerate(patterns):
            a, b, c, d = p
            if b == 1:
                c_tp1 += f_tp1[i]
        
        # Construct State Vector matching 'variables' order
        x = np.zeros(len(variables))
        
        # Map data to x
        # Variables are: [f_0000_t, ..., f_1111_t, f_0000_{t+1}, ..., f_1111_{t+1}, c_t, c_{t+1}]
        for i, p in enumerate(patterns):
            p_str = pattern_str[p]
            
            # Map freq_t - search for variable containing f_{p_str}_t
            for var_name, var_idx in var_map.items():
                if f'f_{p_str}_t' in var_name and '{t+1}' not in var_name:
                    x[var_idx] = f_t[i]
                    break
            
            # Map freq_tp1 - search for variable containing f_{p_str} and {t+1}
            for var_name, var_idx in var_map.items():
                if f'f_{p_str}' in var_name and '{t+1}' in var_name:
                    x[var_idx] = f_tp1[i]
                    break
        
        # Map center values
        # Handle sympy's variable naming (may have special formatting)
        for var_name, var_idx in var_map.items():
            if 'c_t' in var_name and '{t+1}' not in var_name:
                x[var_idx] = c_t
            elif 'c_' in var_name and '{t+1}' in var_name:
                x[var_idx] = c_tp1
        
        # Check Error: ||Ax - b||
        residual = A @ x - b
        error_norm = np.linalg.norm(residual)
        errors.append(error_norm)
        max_error = max(max_error, error_norm)
        min_error = min(min_error, error_norm)
        
        cells = cells_next
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    # Tolerance for floating point
    is_valid = max_error < 1e-10
    
    if verbose:
        print(f"Error Statistics:")
        print(f"  Max constraint violation: {max_error:.2e}")
        print(f"  Mean constraint violation: {mean_error:.2e}")
        print(f"  Std constraint violation: {std_error:.2e}")
        print(f"  Min constraint violation: {min_error:.2e}")
        print()
        
        if is_valid:
            print("✓ PASS: The equations accurately describe the physics.")
            print("  The constraint system matches real Rule 30 grid evolution.")
        else:
            print("✗ FAIL: The equations do not match the grid simulation.")
            print(f"  Max error {max_error:.2e} exceeds tolerance 1e-10")
        print()
    
    return is_valid, {
        'max_error': max_error,
        'mean_error': mean_error,
        'std_error': std_error,
        'min_error': min_error,
        'errors': errors,
        'is_valid': is_valid
    }


def verify_constraint_matrix_rank(verbose: bool = True) -> Tuple[bool, Dict]:
    """
    Check #2: Rank & Nullity
    
    Verifies the constraint matrix has the expected rank.
    """
    if verbose:
        print("="*70)
        print("CHECK #2: Matrix Rank & Nullity")
        print("="*70)
        print()
    
    system = build_4bit_constraint_system(remove_flow=True)
    A, b, _ = build_numerical_matrix(system)
    
    rank = np.linalg.matrix_rank(A, tol=1e-10)
    nullity = A.shape[1] - rank
    
    expected_rank = 19
    expected_nullity = 15
    
    rank_correct = rank == expected_rank
    nullity_correct = nullity == expected_nullity
    all_correct = rank_correct and nullity_correct
    
    if verbose:
        print(f"Constraint matrix shape: {A.shape}")
        print(f"Matrix rank: {rank} (expected: {expected_rank})")
        print(f"Nullity: {nullity} (expected: {expected_nullity})")
        print()
        
        if rank_correct:
            print("✓ Rank is correct!")
        else:
            print(f"✗ Rank mismatch: got {rank}, expected {expected_rank}")
        
        if nullity_correct:
            print("✓ Nullity is correct!")
        else:
            print(f"✗ Nullity mismatch: got {nullity}, expected {expected_nullity}")
        print()
    
    return all_correct, {
        'rank': rank,
        'nullity': nullity,
        'expected_rank': expected_rank,
        'expected_nullity': expected_nullity
    }


def verify_null_space_stability(verbose: bool = True) -> Tuple[bool, Dict]:
    """
    Check #3: Basis Stability
    
    Verifies the null space basis is orthonormal and stable.
    """
    if verbose:
        print("="*70)
        print("CHECK #3: Null Space Stability")
        print("="*70)
        print()
    
    system = build_4bit_constraint_system(remove_flow=True)
    A, b, _ = build_numerical_matrix(system)
    
    N = null_space(A)
    
    # 1. Check dimension
    dimension = N.shape[1]
    expected_dimension = 15
    
    # 2. Orthonormality: N^T @ N should be identity
    NtN = N.T @ N
    identity = np.eye(dimension)
    orthonormality_error = np.linalg.norm(NtN - identity, ord='fro')
    is_orthonormal = orthonormality_error < 1e-10
    
    # 3. Projection Error: A @ N should be zero (N is in null space)
    proj_err = np.linalg.norm(A @ N, ord='fro')
    projection_correct = proj_err < 1e-10
    
    all_stable = (dimension == expected_dimension and 
                  is_orthonormal and 
                  projection_correct)
    
    if verbose:
        print(f"Null space dimension: {dimension} (expected: {expected_dimension})")
        print(f"Orthonormality error: {orthonormality_error:.2e} (threshold: 1e-10)")
        print(f"Projection error (A @ N): {proj_err:.2e} (threshold: 1e-10)")
        print()
        
        if dimension == expected_dimension:
            print("✓ Dimension is correct!")
        else:
            print(f"✗ Dimension mismatch: got {dimension}, expected {expected_dimension}")
        
        if is_orthonormal:
            print("✓ Basis is orthonormal!")
        else:
            print(f"✗ Basis not orthonormal: error = {orthonormality_error:.2e}")
        
        if projection_correct:
            print("✓ Projection is correct (A @ N ≈ 0)!")
        else:
            print(f"✗ Projection error too large: {proj_err:.2e}")
        print()
    
    return all_stable, {
        'dimension': dimension,
        'expected_dimension': expected_dimension,
        'orthonormality_error': orthonormality_error,
        'projection_error': proj_err
    }


def main():
    """Run all three physical validation checks."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase 2 Physical Validation (Reality Check)"
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=10000,
        help='Width of CA grid (default: 10000)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='Number of evolution steps (default: 50)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    
    args = parser.parse_args()
    
    print()
    print("="*70)
    print("PHASE 2 PHYSICAL VALIDATION")
    print("="*70)
    print()
    print("Testing whether the constraint system matches REAL Rule 30 physics:")
    print("  1. Physical validity (grid simulation satisfies A*x = b)")
    print("  2. Matrix rank & nullity (algebraic structure)")
    print("  3. Null space stability (geometric structure)")
    print()
    print("Note: This complements verify_phase2_integrity.py which tests")
    print("      algebraic consistency of the model itself.")
    print()
    
    results = {}
    
    # Check #1: Physical validity
    check1_pass, check1_details = verify_system_physics(
        verbose=args.verbose,
        width=args.width,
        steps=args.steps
    )
    results['physical_validity'] = {
        'pass': check1_pass,
        'details': check1_details
    }
    
    # Check #2: Constraint matrix rank
    check2_pass, check2_details = verify_constraint_matrix_rank(verbose=args.verbose)
    results['constraint_rank'] = {
        'pass': check2_pass,
        'details': check2_details
    }
    
    # Check #3: Null space stability
    check3_pass, check3_details = verify_null_space_stability(verbose=args.verbose)
    results['null_space_stability'] = {
        'pass': check3_pass,
        'details': check3_details
    }
    
    # Final summary
    print("="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print()
    
    all_pass = check1_pass and check2_pass and check3_pass
    
    print(f"Check #1 (Physical Validity): {'✓ PASS' if check1_pass else '✗ FAIL'}")
    print(f"Check #2 (Matrix Rank): {'✓ PASS' if check2_pass else '✗ FAIL'}")
    print(f"Check #3 (Null Space Stability): {'✓ PASS' if check3_pass else '✗ FAIL'}")
    print()
    
    if all_pass:
        print("="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print()
        print("The 4-bit constraint system accurately describes real Rule 30 physics.")
        print("The 15-D chaos tracker is built on a physically correct foundation.")
        print()
    else:
        print("="*70)
        print("✗ SOME TESTS FAILED")
        print("="*70)
        print()
        print("The constraint system may not accurately match real CA physics.")
        print("Review the detailed output above to identify issues.")
        print()
        sys.exit(1)
    
    return results


if __name__ == "__main__":
    main()


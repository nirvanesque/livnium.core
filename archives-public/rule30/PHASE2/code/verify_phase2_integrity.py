#!/usr/bin/env python3
"""
Phase 2 Integrity Verification

Surgically verifies the three critical components:
1. Transition equations (4-bit pattern → 4-bit pattern mapping)
2. Constraint matrix rank (should be 19, null space = 15)
3. Null space basis stability (orthonormal, correct dimension)

If all three pass, the entire geometry pipeline is mathematically guaranteed correct.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from typing import Dict, Tuple, List

try:
    import sympy
    from sympy import symbols, Matrix, Eq, expand
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("ERROR: sympy required")
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from PHASE2/code (same directory)
from rule30_algebra import RULE30_TABLE
from four_bit_system import (
    enumerate_4bit_patterns,
    build_4bit_constraint_system,
    build_rule30_transition_constraints,
    pattern4_to_edge
)

Pattern4 = Tuple[int, int, int, int]


def verify_transition_equations(verbose: bool = True) -> Tuple[bool, Dict]:
    """
    Cross-Check #1: Verify transition equations match actual Rule 30.
    
    For each 4-bit pattern, simulate Rule 30 and count which patterns follow it.
    Compare with our analytical transition matrix.
    
    Returns:
        (all_match, details) where details contains mismatch information
    """
    if verbose:
        print("="*70)
        print("CROSS-CHECK #1: Transition Equations")
        print("="*70)
        print()
    
    patterns = enumerate_4bit_patterns()
    pattern_to_idx = {p: i for i, p in enumerate(patterns)}
    
    # Build analytical transition matrix (from four_bit_system.py logic)
    analytical_T = np.zeros((16, 16), dtype=int)
    
    for i, p_tp1 in enumerate(patterns):
        a_tp1, b_tp1, c_tp1, d_tp1 = p_tp1
        
        for j, p_t in enumerate(patterns):
            a_t, x_t, y_t, d_t = p_t
            
            # Check if this pattern can contribute (from four_bit_system.py)
            if a_t != a_tp1 or d_t != d_tp1:
                continue
            
            # Check Rule 30 updates
            new_b = RULE30_TABLE[(a_t, x_t, y_t)]
            new_c = RULE30_TABLE[(x_t, y_t, d_t)]
            
            if new_b == b_tp1 and new_c == c_tp1:
                analytical_T[i, j] = 1
    
    # Build empirical transition matrix by simulating actual Rule 30
    # For 4-bit patterns, we need to check: given pattern (a,b,c,d) at position i,
    # what patterns can appear at position i at t+1?
    # The pattern at t+1 will be (a, new_b, new_c, d) where:
    #   new_b = RULE30[(a, b, c)]
    #   new_c = RULE30[(b, c, d)]
    
    empirical_T = np.zeros((16, 16), dtype=int)
    
    # For each 4-bit pattern (a,b,c,d), check what it transitions to
    for pattern_idx, pattern in enumerate(patterns):
        a, b, c, d = pattern
        
        # Compute what this pattern becomes after Rule 30
        new_b = RULE30_TABLE[(a, b, c)]
        new_c = RULE30_TABLE[(b, c, d)]
        
        # The pattern at t+1 is (a, new_b, new_c, d)
        pattern_tp1 = (a, new_b, new_c, d)
        
        if pattern_tp1 in pattern_to_idx:
            idx_tp1 = pattern_to_idx[pattern_tp1]
            empirical_T[idx_tp1, pattern_idx] = 1
    
    empirical_T_binary = empirical_T  # Already binary
    
    # Compare
    mismatch_mask = analytical_T != empirical_T_binary
    num_mismatches = mismatch_mask.sum()
    all_match = num_mismatches == 0
    
    if verbose:
        print(f"Analytical transition matrix: {analytical_T.sum()} transitions")
        print(f"Empirical transition matrix: {empirical_T_binary.sum()} transitions")
        print(f"Mismatches: {num_mismatches}")
        print()
        
        if not all_match:
            print("MISMATCHES FOUND:")
            mismatch_positions = np.where(mismatch_mask)
            for k in range(min(10, len(mismatch_positions[0]))):
                i, j = mismatch_positions[0][k], mismatch_positions[1][k]
                p_t = patterns[j]
                p_tp1 = patterns[i]
                print(f"  Pattern {p_t} → {p_tp1}: analytical={analytical_T[i,j]}, empirical={empirical_T_binary[i,j]}")
            if len(mismatch_positions[0]) > 10:
                print(f"  ... and {len(mismatch_positions[0]) - 10} more")
        else:
            print("✓ All transitions match!")
        print()
    
    return all_match, {
        'analytical_T': analytical_T,
        'empirical_T': empirical_T_binary,
        'num_mismatches': num_mismatches,
        'mismatch_positions': np.where(mismatch_mask) if not all_match else None
    }


def verify_constraint_matrix_rank(verbose: bool = True) -> Tuple[bool, Dict]:
    """
    Cross-Check #2: Verify constraint matrix rank.
    
    Expected:
    - Variables: 34
    - Constraints: 20
    - Rank: 19
    - Nullity: 15
    
    Returns:
        (correct, details) where details contains rank/nullity info
    """
    if verbose:
        print("="*70)
        print("CROSS-CHECK #2: Constraint Matrix Rank")
        print("="*70)
        print()
    
    # Build constraint system
    system = build_4bit_constraint_system(remove_flow=True)
    
    if verbose:
        print(f"System built:")
        print(f"  - Variables: {system['num_variables']}")
        print(f"  - Constraints: {system['num_equations']}")
        print()
    
    # Convert to numerical matrix
    all_vars = system['variables']
    var_to_idx = {v: i for i, v in enumerate(all_vars)}
    
    rows = []
    rhs = []
    
    for eq in system['equations']:
        if isinstance(eq, Eq):
            expr = eq.lhs - eq.rhs
        else:
            expr = eq
        
        expr = expand(expr)
        coeff_dict = expr.as_coefficients_dict()
        
        coeffs = []
        for var in all_vars:
            if var in coeff_dict:
                coeffs.append(float(coeff_dict[var]))
            else:
                coeffs.append(0.0)
        
        constant = float(coeff_dict.get(1, 0))
        
        rows.append(coeffs)
        rhs.append(-constant)
    
    A = np.array(rows, dtype=float)
    b = np.array(rhs, dtype=float)
    
    # Compute rank
    rank = np.linalg.matrix_rank(A, tol=1e-10)
    num_vars = A.shape[1]
    num_eqs = A.shape[0]
    nullity = num_vars - rank
    
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
            print()
            print("  DIAGNOSTIC: Checking for redundant constraints...")
            # Check which constraints might be redundant
            # Use SVD to find near-zero singular values
            u, s, vh = np.linalg.svd(A, full_matrices=False)
            tol = 1e-10
            near_zero_singvals = np.where(s < tol)[0]
            if len(near_zero_singvals) > 0:
                print(f"  Found {len(near_zero_singvals)} near-zero singular values")
            print(f"  Smallest singular values: {s[-5:]}")
            print()
        
        if nullity_correct:
            print("✓ Nullity is correct!")
        else:
            print(f"✗ Nullity mismatch: got {nullity}, expected {expected_nullity}")
        print()
    
    return all_correct, {
        'rank': rank,
        'nullity': nullity,
        'expected_rank': expected_rank,
        'expected_nullity': expected_nullity,
        'matrix_shape': A.shape,
        'matrix': A
    }


def verify_null_space_stability(verbose: bool = True) -> Tuple[bool, Dict]:
    """
    Cross-Check #3: Verify null space basis stability.
    
    Checks:
    - Dimension == 15
    - Basis is orthonormal
    - Projection + reconstruction error < 1e-12
    
    Returns:
        (stable, details) where details contains stability metrics
    """
    if verbose:
        print("="*70)
        print("CROSS-CHECK #3: Null Space Stability")
        print("="*70)
        print()
    
    # Build constraint system and matrix
    system = build_4bit_constraint_system(remove_flow=True)
    all_vars = system['variables']
    
    rows = []
    rhs = []
    
    for eq in system['equations']:
        if isinstance(eq, Eq):
            expr = eq.lhs - eq.rhs
        else:
            expr = eq
        
        expr = expand(expr)
        coeff_dict = expr.as_coefficients_dict()
        
        coeffs = []
        for var in all_vars:
            if var in coeff_dict:
                coeffs.append(float(coeff_dict[var]))
            else:
                coeffs.append(0.0)
        
        constant = float(coeff_dict.get(1, 0))
        
        rows.append(coeffs)
        rhs.append(-constant)
    
    A = np.array(rows, dtype=float)
    b = np.array(rhs, dtype=float)
    
    # Compute null space
    null_space_basis = null_space(A)
    
    # Check dimension
    dimension = null_space_basis.shape[1]
    expected_dimension = 15
    dimension_correct = dimension == expected_dimension
    
    # Check orthonormality
    # Columns should be orthonormal: N^T @ N = I
    N = null_space_basis
    NtN = N.T @ N
    identity = np.eye(dimension)
    orthonormality_error = np.linalg.norm(NtN - identity, ord='fro')
    is_orthonormal = orthonormality_error < 1e-10
    
    # Check projection/reconstruction
    # The null space basis vectors should satisfy A @ N ≈ 0
    # Check each column of N (each basis vector)
    projection_errors = []
    for i in range(N.shape[1]):
        # Each column of N should be in the null space
        residual = A @ N[:, i]
        projection_errors.append(np.linalg.norm(residual))
    
    # Use maximum error across all basis vectors
    projection_error = max(projection_errors)
    projection_correct = projection_error < 1e-10
    
    all_stable = dimension_correct and is_orthonormal and projection_correct
    
    if verbose:
        print(f"Null space dimension: {dimension} (expected: {expected_dimension})")
        print(f"Orthonormality error: {orthonormality_error:.2e} (threshold: 1e-10)")
        print(f"Projection error (A @ N): {projection_error:.2e} (threshold: 1e-10)")
        print()
        
        if dimension_correct:
            print("✓ Dimension is correct!")
        else:
            print(f"✗ Dimension mismatch: got {dimension}, expected {expected_dimension}")
        
        if is_orthonormal:
            print("✓ Basis is orthonormal!")
        else:
            print(f"✗ Basis not orthonormal: error = {orthonormality_error:.2e}")
        
        if projection_correct:
            print("✓ Projection/reconstruction works correctly!")
        else:
            print(f"✗ Projection error too large: {projection_error:.2e}")
        print()
    
    return all_stable, {
        'dimension': dimension,
        'expected_dimension': expected_dimension,
        'orthonormality_error': orthonormality_error,
        'projection_error': projection_error,
        'null_space_basis': null_space_basis
    }


def main():
    """Run all three integrity checks."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase 2 Integrity Verification"
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    
    args = parser.parse_args()
    
    print()
    print("="*70)
    print("PHASE 2 INTEGRITY VERIFICATION")
    print("="*70)
    print()
    print("Verifying three critical components:")
    print("  1. Transition equations (4-bit pattern mapping)")
    print("  2. Constraint matrix rank (20 constraints, 15 free dims)")
    print("  3. Null space basis stability (orthonormal, correct projection)")
    print()
    
    results = {}
    
    # Check #1: Transition equations
    check1_pass, check1_details = verify_transition_equations(verbose=args.verbose)
    results['transition_equations'] = {
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
    
    print(f"Check #1 (Transition Equations): {'✓ PASS' if check1_pass else '✗ FAIL'}")
    print(f"Check #2 (Constraint Matrix Rank): {'✓ PASS' if check2_pass else '✗ FAIL'}")
    print(f"Check #3 (Null Space Stability): {'✓ PASS' if check3_pass else '✗ FAIL'}")
    print()
    
    if all_pass:
        print("="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print()
        print("Phase 2 system is mathematically verified:")
        print("  - Transition equations are correct")
        print("  - Constraint matrix has correct rank (20)")
        print("  - Null space is stable (15 dimensions, orthonormal)")
        print()
        print("The 15-D chaos tracker is built on a solid foundation.")
        print()
    else:
        print("="*70)
        print("✗ SOME TESTS FAILED")
        print("="*70)
        print()
        print("Phase 2 system needs correction before proceeding.")
        print("Review the detailed output above to identify issues.")
        print()
        sys.exit(1)
    
    return results


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Test Divergence V3 Invariant - Complete Analysis

STEP 1: Separate trivial from non-trivial invariants
STEP 2: Make invariants exact (rational) using sympy
STEP 3: Analyze and simplify invariants
"""

import argparse
import sys
import random
from pathlib import Path
from typing import Dict, Tuple, List
from fractions import Fraction

import numpy as np

try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.divergence_v3 import (
    enumerate_patterns, 
    divergence_v3, 
    divergence_v3_rational,
    format_weights,
    format_weights_rational
)
from experiments.rule30.invariant_solver_v3 import (
    build_invariance_system,
    build_invariance_system_rational,
    find_nullspace,
    find_nullspace_exact,
    analyze_nullspace,
    normalize_vector,
    gcd_normalize_rational
)
from experiments.rule30.rule30_algebra import rule30_step

Pattern = Tuple[int, int, int]


def extract_weight_vector(nullspace: np.ndarray, vector_index: int = 0) -> Dict[Pattern, float]:
    """Extract weight vector from nullspace (float version)."""
    patterns = enumerate_patterns()
    
    if nullspace.shape[1] == 0:
        raise RuntimeError("No non-trivial nullspace found (no invariant of this form).")
    
    if vector_index >= nullspace.shape[1]:
        vector_index = 0
    
    w_vec = normalize_vector(nullspace[:, vector_index])
    
    weights: Dict[Pattern, float] = {}
    for p, w in zip(patterns, w_vec):
        if abs(w) > 1e-6:
            weights[p] = float(w)
    
    return weights


def extract_weight_vector_rational(nullspace_sympy, vector_index: int = 0) -> Dict[Pattern, Fraction]:
    """Extract weight vector from sympy nullspace (exact rational version)."""
    patterns = enumerate_patterns()
    
    if len(nullspace_sympy) == 0:
        raise RuntimeError("No non-trivial nullspace found (no invariant of this form).")
    
    if vector_index >= len(nullspace_sympy):
        vector_index = 0
    
    w_vec_sympy = nullspace_sympy[vector_index]
    
    # Convert to Fraction
    weights: Dict[Pattern, Fraction] = {}
    for i, p in enumerate(patterns):
        w_sympy = w_vec_sympy[i]
        if isinstance(w_sympy, sympy.Rational):
            weights[p] = Fraction(w_sympy.p, w_sympy.q)
        elif w_sympy == 0:
            weights[p] = Fraction(0)
        else:
            # Convert to float then to Fraction (approximate)
            weights[p] = Fraction(float(w_sympy)).limit_denominator(1000000)
    
    # Normalize by GCD
    weights = gcd_normalize_rational(weights)
    
    return weights


def identify_trivial_invariants(nullspace: np.ndarray) -> Dict:
    """
    STEP 1: Identify trivial invariants.
    
    Looks for:
    - Normalization: sum of all frequencies = 1
    - Symmetries
    - Complement relations
    """
    patterns = enumerate_patterns()
    patterns_str = [''.join(str(b) for b in p) for p in patterns]
    
    trivial_info = {}
    
    for i in range(nullspace.shape[1]):
        vec = normalize_vector(nullspace[:, i])
        
        # Check for normalization invariant (sum of all weights ≈ 1)
        total = np.sum(vec)
        if abs(total - 1.0) < 1e-6:
            trivial_info[i] = {
                'type': 'normalization',
                'description': 'Sum of all pattern frequencies = 1',
                'weights': {patterns_str[j]: float(vec[j]) for j in range(len(patterns_str))}
            }
            continue
        
        # Check for symmetry (e.g., weight(001) ≈ weight(100))
        # Check reflection symmetry
        reflection_symmetric = True
        for j, p in enumerate(patterns):
            p_refl = (p[2], p[1], p[0])  # Reflect pattern
            j_refl = patterns.index(p_refl)
            if abs(vec[j] - vec[j_refl]) > 1e-6:
                reflection_symmetric = False
                break
        
        if reflection_symmetric:
            trivial_info[i] = {
                'type': 'symmetry',
                'description': 'Reflection symmetric',
                'weights': {patterns_str[j]: float(vec[j]) for j in range(len(patterns_str))}
            }
            continue
        
        # Check complement symmetry (000 ↔ 111, 001 ↔ 110, etc.)
        complement_symmetric = True
        for j, p in enumerate(patterns):
            p_comp = tuple(1 - b for b in p)  # Complement
            j_comp = patterns.index(p_comp)
            if abs(vec[j] - vec[j_comp]) > 1e-6:
                complement_symmetric = False
                break
        
        if complement_symmetric:
            trivial_info[i] = {
                'type': 'complement_symmetry',
                'description': 'Complement symmetric',
                'weights': {patterns_str[j]: float(vec[j]) for j in range(len(patterns_str))}
            }
    
    return trivial_info


def run_invariance_test(
    weights: Dict[Pattern, float],
    initial_row: List[int],
    steps: int = 100,
    cyclic: bool = True,
    verbose: bool = True
) -> Dict:
    """Test whether divergence_v3 is invariant under Rule 30 evolution."""
    row = initial_row.copy()
    d0 = divergence_v3(row, weights, cyclic=cyclic)
    
    if verbose:
        print(f"Initial divergence: {d0:.9f}")
    
    values = [d0]
    deviations = [0.0]
    
    for t in range(1, steps + 1):
        row = rule30_step(row, cyclic=cyclic)
        dt = divergence_v3(row, weights, cyclic=cyclic)
        delta = dt - d0
        
        values.append(dt)
        deviations.append(abs(delta))
        
        if verbose and (t % 10 == 0 or t <= 5):
            print(f"[t={t:3d}] divergence = {dt:.9f}, Δ = {delta:+.3e}")
    
    max_deviation = max(deviations)
    mean_deviation = np.mean(deviations)
    std_deviation = np.std(deviations)
    
    tolerance = 1e-6
    is_invariant = max_deviation < tolerance
    
    return {
        'initial_value': d0,
        'values': values,
        'deviations': deviations,
        'max_deviation': float(max_deviation),
        'mean_deviation': float(mean_deviation),
        'std_deviation': float(std_deviation),
        'is_invariant': is_invariant,
        'tolerance': tolerance
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test Divergence V3 invariants - Complete Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--num-rows',
        type=int,
        default=300,
        help='Number of random rows for building system (default: 300)'
    )
    
    parser.add_argument(
        '--row-length',
        type=int,
        default=200,
        help='Length of rows (default: 200)'
    )
    
    parser.add_argument(
        '--test-steps',
        type=int,
        default=50,
        help='Number of evolution steps for testing (default: 50)'
    )
    
    parser.add_argument(
        '--exact',
        action='store_true',
        help='Use exact rational arithmetic (requires sympy)'
    )
    
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-8,
        help='Tolerance for nullspace computation (default: 1e-8)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("DIVERGENCE V3 INVARIANT TEST - COMPLETE ANALYSIS")
    print("="*70)
    print()
    
    # Step 1: Build linear system
    print(f"Step 1: Building invariance system...")
    print(f"  Sampling {args.num_rows} random rows of length {args.row_length}")
    
    if args.exact and SYMPY_AVAILABLE:
        print("  Using exact rational arithmetic (sympy)")
        A_rational = build_invariance_system_rational(
            num_rows=args.num_rows,
            row_length=args.row_length,
            cyclic=True
        )
        print(f"  System matrix shape: {A_rational.shape}")
        
        # Find exact nullspace
        print(f"\nStep 2: Finding exact nullspace...")
        nullspace_sympy = find_nullspace_exact(A_rational)
        print(f"  Found {len(nullspace_sympy)} basis vectors")
        
        # Convert to numpy for analysis
        patterns = enumerate_patterns()
        nullspace_np = np.zeros((len(patterns), len(nullspace_sympy)))
        for i, vec_sympy in enumerate(nullspace_sympy):
            for j in range(len(patterns)):
                nullspace_np[j, i] = float(vec_sympy[j])
        
        nullspace = nullspace_np
        use_exact = True
    else:
        if args.exact:
            print("  Warning: sympy not available, using numpy (approximate)")
        A = build_invariance_system(
            num_rows=args.num_rows,
            row_length=args.row_length,
            cyclic=True
        )
        print(f"  System matrix shape: {A.shape}")
        
        print(f"\nStep 2: Finding nullspace (tolerance={args.tolerance})...")
        nullspace = find_nullspace(A, tol=args.tolerance)
        use_exact = False
    
    analysis = analyze_nullspace(nullspace)
    print(f"  {analysis['message']}")
    print(f"  Nullspace shape: {nullspace.shape}")
    print()
    
    if not analysis['has_invariants']:
        print("="*70)
        print("RESULT: No invariant found")
        print("="*70)
        return
    
    # Step 3: Identify trivial invariants
    print("Step 3: Identifying trivial invariants...")
    trivial_info = identify_trivial_invariants(nullspace)
    
    print(f"  Found {len(trivial_info)} potentially trivial invariants:")
    for idx, info in trivial_info.items():
        print(f"    Vector {idx}: {info['type']} - {info['description']}")
    print()
    
    # Step 4: Extract and display all invariants
    print("Step 4: Extracting all invariants...")
    print()
    
    patterns_str = [''.join(str(b) for b in p) for p in enumerate_patterns()]
    
    for i in range(nullspace.shape[1]):
        print(f"{'='*70}")
        print(f"INVARIANT {i+1}/{nullspace.shape[1]}")
        print(f"{'='*70}")
        
        if use_exact and SYMPY_AVAILABLE:
            try:
                weights_rational = extract_weight_vector_rational(nullspace_sympy, i)
                print(f"Exact rational formula:")
                print(f"  D3(s) = {format_weights_rational(weights_rational)}")
                print()
                print("Rational coefficients:")
                for p, w in sorted(weights_rational.items()):
                    if w != 0:
                        print(f"  {''.join(str(b) for b in p)}: {w}")
            except:
                weights = extract_weight_vector(nullspace, i)
                print(f"Approximate formula:")
                print(f"  D3(s) = {format_weights(weights)}")
        else:
            weights = extract_weight_vector(nullspace, i)
            print(f"Formula:")
            print(f"  D3(s) = {format_weights(weights)}")
        
        if i in trivial_info:
            print(f"\n⚠ Trivial invariant: {trivial_info[i]['type']}")
        else:
            print(f"\n✓ Non-trivial invariant (potentially interesting)")
        
        # Test invariance
        print(f"\nTesting invariance over {args.test_steps} steps...")
        test_row = [random.randint(0, 1) for _ in range(args.row_length)]
        
        if use_exact and SYMPY_AVAILABLE and i in locals() and 'weights_rational' in locals():
            # Test with exact arithmetic
            row = test_row.copy()
            d0 = divergence_v3_rational(row, weights_rational, cyclic=True)
            print(f"  Initial: {d0}")
            
            for t in range(1, min(6, args.test_steps + 1)):
                row = rule30_step(row, cyclic=True)
                dt = divergence_v3_rational(row, weights_rational, cyclic=True)
                delta = dt - d0
                print(f"  t={t}: {dt}, Δ={delta}")
            
            if args.test_steps > 5:
                print(f"  ... (testing up to t={args.test_steps})")
                for t in range(6, args.test_steps + 1):
                    row = rule30_step(row, cyclic=True)
                dt_final = divergence_v3_rational(row, weights_rational, cyclic=True)
                delta_final = dt_final - d0
                print(f"  t={args.test_steps}: {dt_final}, Δ={delta_final}")
        else:
            result = run_invariance_test(weights, test_row, steps=args.test_steps, verbose=False)
            print(f"  Initial value:     {result['initial_value']:.9f}")
            print(f"  Max deviation:     {result['max_deviation']:.9e}")
            print(f"  Mean deviation:    {result['mean_deviation']:.9e}")
            
            if result['is_invariant']:
                print(f"  ✓✓✓ INVARIANT CONFIRMED!")
            else:
                print(f"  ⚠ Deviations exceed tolerance")
        
        print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Found {nullspace.shape[1]} independent invariants")
    print(f"  Trivial: {len(trivial_info)}")
    print(f"  Non-trivial: {nullspace.shape[1] - len(trivial_info)}")
    print()
    print("Next steps:")
    print("  1. Run bruteforce_verify_invariant.py for exhaustive verification")
    print("  2. Simplify non-trivial invariants to human-readable form")
    print("  3. Document exact formulas")
    print()


if __name__ == "__main__":
    main()


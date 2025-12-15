#!/usr/bin/env python3
"""
Center Column Analysis: Combining Invariants with Center-Column Dynamics

This module combines the 4 exact invariants with the symbolic center-column
update rule to derive reduced recurrence relations and dimensional collapses.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from fractions import Fraction
import numpy as np

try:
    import sympy
    from sympy import symbols, Matrix, Eq, solve, simplify, expand
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("Warning: sympy required for symbolic analysis")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.divergence_v3 import enumerate_patterns, pattern_frequencies_3_rational
from experiments.rule30.rule30_algebra import rule30_step

Pattern = Tuple[int, int, int]


def get_center_column_update_rule(N: int, center_idx: int = None) -> Dict:
    """
    Express the center column update rule symbolically.
    
    For a row of length N with periodic boundary conditions, the center column
    is at index N//2. The update rule for Rule 30 is:
    
        new[i] = (left[i] + center[i] + right[i] + center[i]*right[i]) % 2
    
    But we need to express this in terms of pattern frequencies and how they
    constrain the center column value.
    
    Args:
        N: Row length
        center_idx: Center index (default: N//2)
        
    Returns:
        Dict with symbolic representation of center column update
    """
    if not SYMPY_AVAILABLE:
        raise RuntimeError("sympy required for symbolic analysis")
    
    if center_idx is None:
        center_idx = N // 2
    
    # Create symbolic variables for the row state
    # We'll use pattern frequencies as the state variables
    patterns = enumerate_patterns()
    pattern_str = {p: ''.join(str(b) for b in p) for p in patterns}
    
    # Symbolic pattern frequencies
    freq_vars = {}
    for p in patterns:
        p_str = pattern_str[p]
        freq_vars[p] = symbols(f'f_{p_str}', real=True, nonnegative=True)
    
    # The center column value depends on the patterns around it
    # For Rule 30: new_center = (left + center + right + center*right) % 2
    # where left, center, right are the three bits around position center_idx
    
    # We need to express this in terms of pattern frequencies
    # The center column is part of patterns: (left, center, right)
    
    # Create symbolic center column value
    center_curr = symbols('c_t', integer=True, domain=sympy.Integers.mod(2))
    center_next = symbols('c_{t+1}', integer=True, domain=sympy.Integers.mod(2))
    
    # The update rule: c_{t+1} = (left + c_t + right + c_t*right) mod 2
    # We need to express left and right in terms of pattern frequencies
    
    return {
        'N': N,
        'center_idx': center_idx,
        'freq_vars': freq_vars,
        'center_curr': center_curr,
        'center_next': center_next,
        'patterns': patterns,
        'pattern_str': pattern_str
    }


def row_to_pattern_frequency_vector(row: List[int], cyclic: bool = True) -> np.ndarray:
    """
    Map a row into the 8-dimensional pattern-frequency vector.
    
    Args:
        row: Binary row
        cyclic: Use cyclic boundary conditions
        
    Returns:
        numpy array of shape (8,) with pattern frequencies
    """
    patterns = enumerate_patterns()
    freq_rational = pattern_frequencies_3_rational(row, cyclic=cyclic)
    
    freq_vector = np.array([float(freq_rational[p]) for p in patterns])
    return freq_vector


def get_invariant_equations() -> List:
    """
    Get the 4 invariant equations as symbolic constraints.
    
    Returns:
        List of sympy equations representing the invariants
    """
    if not SYMPY_AVAILABLE:
        raise RuntimeError("sympy required")
    
    patterns = enumerate_patterns()
    pattern_str = {p: ''.join(str(b) for b in p) for p in patterns}
    
    # Create symbolic frequency variables
    freq_vars = {}
    for p in patterns:
        p_str = pattern_str[p]
        freq_vars[p] = symbols(f'f_{p_str}', real=True, nonnegative=True)
    
    # The 4 invariants
    invariants = [
        # I1: freq(100) - freq(001) = constant
        freq_vars[patterns[4]] - freq_vars[patterns[1]],  # 100 - 001
        
        # I2: freq(001) - freq(010) - freq(011) + freq(101) = constant
        freq_vars[patterns[1]] - freq_vars[patterns[2]] - freq_vars[patterns[3]] + freq_vars[patterns[5]],
        
        # I3: freq(110) - freq(011) = constant
        freq_vars[patterns[6]] - freq_vars[patterns[3]],  # 110 - 011
        
        # I4: freq(000) + freq(001) + 2*freq(010) + 3*freq(011) + freq(111) = 1
        freq_vars[patterns[0]] + freq_vars[patterns[1]] + 2*freq_vars[patterns[2]] + 
        3*freq_vars[patterns[3]] + freq_vars[patterns[7]] - 1
    ]
    
    return invariants, freq_vars


def build_center_column_constraint_system(N: int) -> Dict:
    """
    Build a system combining center-column update rule with invariant constraints.
    
    Args:
        N: Row length
        
    Returns:
        Dict with symbolic system
    """
    if not SYMPY_AVAILABLE:
        raise RuntimeError("sympy required")
    
    # Get invariant equations
    invariant_eqs, freq_vars = get_invariant_equations()
    
    # Get center column update rule structure
    center_info = get_center_column_update_rule(N)
    
    # The key insight: pattern frequencies constrain what values the center
    # column can take. We need to express the center column update in terms
    # of pattern frequencies and then apply the invariant constraints.
    
    # For a given center column value c_t, what patterns can exist?
    # Patterns involving center column: (left, c_t, right)
    # The center column appears in positions: (c-1, c, c+1) mod N
    
    # Express center column value in terms of pattern frequencies
    # c_t = sum of frequencies of patterns where middle bit = 1
    patterns = enumerate_patterns()
    pattern_str = {p: ''.join(str(b) for b in p) for p in patterns}
    
    # Patterns with center bit = 1: 001, 011, 101, 111 (indices 1, 3, 5, 7)
    center_patterns = [patterns[i] for i in [1, 3, 5, 7]]
    
    # Symbolic expression for center column value
    center_expr = sum(freq_vars[p] for p in center_patterns)
    
    # But wait - this is a frequency, not the actual value
    # We need to think differently: the center column value is determined by
    # the actual bit at that position, not the frequency
    
    # Actually, let's think about this more carefully:
    # The center column value c_t is a single bit (0 or 1)
    # Pattern frequencies tell us the distribution of patterns
    # We need to relate the center column bit to the pattern frequencies
    
    # For now, let's build the constraint system structure
    return {
        'N': N,
        'invariant_equations': invariant_eqs,
        'freq_vars': freq_vars,
        'center_info': center_info,
        'patterns': patterns,
        'pattern_str': pattern_str
    }


def solve_reduced_system(system: Dict, verbose: bool = True) -> Dict:
    """
    Solve the reduced system to find recurrence relations.
    
    Args:
        system: System dict from build_center_column_constraint_system
        verbose: Print progress
        
    Returns:
        Dict with solutions, recurrence relations, dimensional analysis
    """
    if not SYMPY_AVAILABLE:
        raise RuntimeError("sympy required")
    
    if verbose:
        print("Solving reduced constraint system...")
        print(f"  N = {system['N']}")
        print(f"  Invariant constraints: {len(system['invariant_equations'])}")
    
    # The system has:
    # - 8 frequency variables (f_000, f_001, ..., f_111)
    # - 4 invariant constraints
    # - 1 normalization constraint (sum of frequencies = 1)
    # - Center column update rule constraints
    
    # This gives us 8 variables - 4 invariants - 1 normalization = 3 free dimensions
    # But we also have the center column update rule which adds more constraints
    
    # For now, let's analyze the dimensional reduction
    num_vars = 8  # pattern frequencies
    num_constraints = len(system['invariant_equations']) + 1  # +1 for normalization
    
    free_dimensions = num_vars - num_constraints
    
    if verbose:
        print(f"  Free dimensions: {free_dimensions}")
    
    # Try to solve symbolically
    freq_vars = system['freq_vars']
    invariant_eqs = system['invariant_equations']
    
    # Add normalization constraint
    patterns = system['patterns']
    normalization = sum(freq_vars[p] for p in patterns) - 1
    all_eqs = list(invariant_eqs) + [normalization]
    
    # Try to solve for some variables in terms of others
    # Pick some variables to solve for
    vars_to_solve = list(freq_vars.values())[:4]  # First 4 variables
    vars_to_keep = list(freq_vars.values())[4:]   # Last 4 variables
    
    try:
        solutions = solve(all_eqs, vars_to_solve, dict=True)
        
        if verbose:
            print(f"  Found {len(solutions)} solution(s)")
        
        return {
            'solutions': solutions,
            'free_dimensions': free_dimensions,
            'reduced_vars': vars_to_keep,
            'solved_vars': vars_to_solve,
            'status': 'solved'
        }
    except Exception as e:
        if verbose:
            print(f"  Symbolic solve failed: {e}")
            print("  Attempting dimensional analysis...")
        
        # Fall back to dimensional analysis
        return {
            'solutions': None,
            'free_dimensions': free_dimensions,
            'status': 'dimensional_analysis_only',
            'message': f'System has {free_dimensions} free dimensions after applying {num_constraints} constraints'
        }


def analyze_center_column_dynamics(N: int, verbose: bool = True) -> Dict:
    """
    Complete analysis: combine invariants with center-column update rule.
    
    Args:
        N: Row length
        verbose: Print detailed output
        
    Returns:
        Dict with complete analysis results
    """
    if not SYMPY_AVAILABLE:
        raise RuntimeError("sympy required for symbolic analysis")
    
    print("="*70)
    print("CENTER COLUMN ANALYSIS")
    print("="*70)
    print()
    
    # Step 1: Build constraint system
    if verbose:
        print("Step 1: Building constraint system...")
    system = build_center_column_constraint_system(N)
    
    # Step 2: Solve reduced system
    if verbose:
        print("\nStep 2: Solving reduced system...")
    solution = solve_reduced_system(system, verbose=verbose)
    
    # Step 3: Analyze dimensional collapse
    if verbose:
        print("\nStep 3: Analyzing dimensional reduction...")
    
    num_vars = 8
    num_invariants = 4
    num_normalization = 1
    total_constraints = num_invariants + num_normalization
    
    dimensional_reduction = {
        'original_dimensions': num_vars,
        'invariant_constraints': num_invariants,
        'normalization_constraints': num_normalization,
        'total_constraints': total_constraints,
        'reduced_dimensions': num_vars - total_constraints,
        'reduction_ratio': (num_vars - total_constraints) / num_vars
    }
    
    if verbose:
        print(f"  Original space: {dimensional_reduction['original_dimensions']}D")
        print(f"  Constraints: {dimensional_reduction['total_constraints']}")
        print(f"  Reduced space: {dimensional_reduction['reduced_dimensions']}D")
        print(f"  Reduction: {dimensional_reduction['reduction_ratio']:.1%}")
    
    return {
        'system': system,
        'solution': solution,
        'dimensional_reduction': dimensional_reduction,
        'N': N
    }


def main():
    parser = argparse.ArgumentParser(
        description="Center column analysis combining invariants with update rule"
    )
    
    parser.add_argument(
        '--N',
        type=int,
        default=10,
        help='Row length (default: 10)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    
    args = parser.parse_args()
    
    if not SYMPY_AVAILABLE:
        print("Error: sympy required. Install with: pip install sympy")
        return
    
    results = analyze_center_column_dynamics(args.N, verbose=args.verbose)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Refine center-column update rule expression")
    print("  2. Derive explicit recurrence relations")
    print("  3. Analyze allowed state transitions")
    print()


if __name__ == "__main__":
    import argparse
    main()


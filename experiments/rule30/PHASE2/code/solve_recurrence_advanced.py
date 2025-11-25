#!/usr/bin/env python3
"""
Advanced Recurrence Solver

Uses Gaussian elimination and Groebner basis methods to derive
c_{t+1} = f(c_t) from the constraint system.
"""

import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

try:
    import sympy
    from sympy import symbols, Matrix, Eq, solve, simplify, expand, groebner, linsolve
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.divergence_v3 import enumerate_patterns
from experiments.rule30.center_column_symbolic import (
    center_value,
    next_center_value,
    build_pattern_transition_matrix
)


def eliminate_variables_using_invariants(verbose: bool = True) -> Dict:
    """
    Use invariants to express some pattern frequencies in terms of others.
    
    This reduces the number of free variables before solving.
    """
    if not SYMPY_AVAILABLE:
        raise RuntimeError("sympy required")
    
    patterns = enumerate_patterns()
    pattern_str = {p: ''.join(str(b) for b in p) for p in patterns}
    
    # Create symbolic variables
    freq_t = {}
    for p in patterns:
        p_str = pattern_str[p]
        freq_t[p] = symbols(f'f_{p_str}_t', real=True)
    
    # Get invariant values (constants)
    I1 = symbols('I1', real=True)  # freq(100) - freq(001)
    I2 = symbols('I2', real=True)  # freq(001) - freq(010) - freq(011) + freq(101)
    I3 = symbols('I3', real=True)  # freq(110) - freq(011)
    # I4 = 1 always
    
    # Express some frequencies in terms of others using invariants
    substitutions = {}
    
    # From I1: f_100_t = f_001_t + I1
    substitutions[freq_t[patterns[4]]] = freq_t[patterns[1]] + I1
    
    # From I3: f_110_t = f_011_t + I3
    substitutions[freq_t[patterns[6]]] = freq_t[patterns[3]] + I3
    
    # From I2: f_101_t = f_010_t + f_011_t - f_001_t + I2
    substitutions[freq_t[patterns[5]]] = freq_t[patterns[2]] + freq_t[patterns[3]] - freq_t[patterns[1]] + I2
    
    # From I4: f_000_t = 1 - f_001_t - 2*f_010_t - 3*f_011_t - f_111_t
    substitutions[freq_t[patterns[0]]] = 1 - freq_t[patterns[1]] - 2*freq_t[patterns[2]] - 3*freq_t[patterns[3]] - freq_t[patterns[7]]
    
    # Now we have reduced from 8 variables to 4 free variables:
    # f_001_t, f_010_t, f_011_t, f_111_t
    # Plus the 3 invariant constants I1, I2, I3
    
    if verbose:
        print("Variable elimination using invariants:")
        print(f"  Original variables: 8")
        print(f"  Reduced to: 4 free variables + 3 invariant constants")
        print(f"  Substitutions:")
        for var, expr in substitutions.items():
            print(f"    {var} = {expr}")
    
    return {
        'substitutions': substitutions,
        'free_vars': [freq_t[patterns[i]] for i in [1, 2, 3, 7]],  # 001, 010, 011, 111
        'invariant_constants': [I1, I2, I3],
        'freq_t': freq_t
    }


def express_center_in_reduced_vars(reduced_system: Dict) -> sympy.Expr:
    """
    Express c_t in terms of the reduced variable set.
    """
    if not SYMPY_AVAILABLE:
        raise RuntimeError("sympy required")
    
    freq_t = reduced_system['freq_t']
    substitutions = reduced_system['substitutions']
    patterns = enumerate_patterns()
    
    # c_t = freq(001) + freq(011) + freq(101) + freq(111)
    c_t_full = freq_t[patterns[1]] + freq_t[patterns[3]] + freq_t[patterns[5]] + freq_t[patterns[7]]
    
    # Apply substitutions
    c_t_reduced = c_t_full.subs(substitutions)
    c_t_reduced = simplify(c_t_reduced)
    
    return c_t_reduced


def build_linear_system_for_recurrence(N: int = 10, verbose: bool = True) -> Dict:
    """
    Build a linear system to solve for recurrence relation.
    
    Uses the reduced variable set and transition constraints.
    """
    if not SYMPY_AVAILABLE:
        raise RuntimeError("sympy required")
    
    if verbose:
        print("="*70)
        print("ADVANCED RECURRENCE SOLVING")
        print("="*70)
        print()
    
    # Step 1: Eliminate variables using invariants
    reduced = eliminate_variables_using_invariants(verbose=verbose)
    
    # Step 2: Express c_t in reduced variables
    c_t_reduced = express_center_in_reduced_vars(reduced)
    
    if verbose:
        print()
        print(f"c_t (reduced) = {c_t_reduced}")
        print()
    
    # Step 3: Do the same for t+1
    # This requires applying transition constraints
    
    # For now, return what we have
    return {
        'reduced_system': reduced,
        'c_t_reduced': c_t_reduced,
        'status': 'partial_reduction',
        'message': 'Reduced system built. Next: apply transition constraints to get c_{t+1}.'
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced recurrence solver using variable elimination"
    )
    
    parser.add_argument(
        '--N',
        type=int,
        default=10,
        help='Row length'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    
    args = parser.parse_args()
    
    if not SYMPY_AVAILABLE:
        print("Error: sympy required")
        return
    
    results = build_linear_system_for_recurrence(N=args.N, verbose=args.verbose)
    
    print("\n" + "="*70)
    print("REDUCTION COMPLETE")
    print("="*70)
    print()
    print(f"Status: {results['status']}")
    print(f"Message: {results['message']}")
    print()


if __name__ == "__main__":
    main()


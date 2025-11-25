#!/usr/bin/env python3
"""
Groebner Basis Solver for Center Column Recurrence

Uses Groebner basis computation to eliminate all variables except c_t and c_{t+1},
deriving the recurrence relation R(c_t, c_{t+1}) = 0.
"""

import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

try:
    import sympy
    from sympy import symbols, Matrix, Eq, solve, simplify, expand, groebner, Poly
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.divergence_v3 import enumerate_patterns
from experiments.rule30.center_column_symbolic import (
    center_value,
    next_center_value
)
from experiments.rule30.debruijn_transitions import (
    build_correct_transition_constraints
)


def solve_center_groebner(N: int = 10, verbose: bool = True) -> Dict:
    """
    Use Groebner basis to eliminate all variables except c_t and c_{t+1}.
    
    Steps:
    1. Build polynomial equations for:
       - invariants (4 equations)
       - normalization at t and t+1 (2 equations)
       - transition constraints (8 equations)
       - c_t and c_tp1 definitions
    
    2. Use sympy.groebner with lex order to eliminate:
       all variables except c_t and c_tp1.
    
    3. Return the reduced polynomial relation R(c_t, c_tp1) = 0
    
    Args:
        N: Row length
        verbose: Print detailed output
        
    Returns:
        Dict with Groebner basis, recurrence relation, and analysis
    """
    if not SYMPY_AVAILABLE:
        raise RuntimeError("sympy required")
    
    if verbose:
        print("="*70)
        print("GROEBNER BASIS SOLVER FOR CENTER COLUMN RECURRENCE")
        print("="*70)
        print()
    
    patterns = enumerate_patterns()
    pattern_str = {p: ''.join(str(b) for b in p) for p in patterns}
    
    # Step 1: Build all polynomial equations
    
    # 1a. Pattern frequency variables at t and t+1
    freq_t = {}
    freq_tp1 = {}
    for p in patterns:
        p_str = pattern_str[p]
        freq_t[p] = symbols(f'f_{p_str}_t', real=True, nonnegative=True)
        freq_tp1[p] = symbols(f'f_{p_str}_{{t+1}}', real=True, nonnegative=True)
    
    # 1b. Center column variables
    c_t = symbols('c_t', real=True)
    c_tp1 = symbols('c_{t+1}', real=True)
    
    if verbose:
        print("Step 1: Building polynomial equations...")
        print(f"  Variables at t: {len(freq_t)}")
        print(f"  Variables at t+1: {len(freq_tp1)}")
        print(f"  Center variables: c_t, c_{{t+1}}")
        print()
    
    # 1c. Invariant equations (at t)
    inv_t = [
        freq_t[patterns[4]] - freq_t[patterns[1]],  # I1: freq(100) - freq(001)
        freq_t[patterns[1]] - freq_t[patterns[2]] - freq_t[patterns[3]] + freq_t[patterns[5]],  # I2
        freq_t[patterns[6]] - freq_t[patterns[3]],  # I3: freq(110) - freq(011)
        freq_t[patterns[0]] + freq_t[patterns[1]] + 2*freq_t[patterns[2]] + 
        3*freq_t[patterns[3]] + freq_t[patterns[7]] - 1  # I4: weighted sum = 1
    ]
    
    # Invariant equations (at t+1) - same structure
    inv_tp1 = [
        freq_tp1[patterns[4]] - freq_tp1[patterns[1]],  # I1
        freq_tp1[patterns[1]] - freq_tp1[patterns[2]] - freq_tp1[patterns[3]] + freq_tp1[patterns[5]],  # I2
        freq_tp1[patterns[6]] - freq_tp1[patterns[3]],  # I3
        freq_tp1[patterns[0]] + freq_tp1[patterns[1]] + 2*freq_tp1[patterns[2]] + 
        3*freq_tp1[patterns[3]] + freq_tp1[patterns[7]] - 1  # I4
    ]
    
    # Invariance: I_t = I_{t+1} (4 equations)
    invariance_eqs = [inv_t[i] - inv_tp1[i] for i in range(4)]
    
    # 1d. Normalization constraints
    norm_t = sum(freq_t[p] for p in patterns) - 1
    norm_tp1 = sum(freq_tp1[p] for p in patterns) - 1
    
    # 1e. Transition constraints using de Bruijn graph model
    transition_eqs = build_correct_transition_constraints(freq_t, freq_tp1)
    # Convert Eq objects to polynomials: expr = 0
    transition_polys = [eq.lhs - eq.rhs for eq in transition_eqs]
    
    # 1f. Center column definitions
    c_t_expr = center_value(freq_t)
    c_tp1_expr = next_center_value(freq_tp1)
    
    center_defs = [
        c_t - c_t_expr,  # c_t = sum of patterns with center=1
        c_tp1 - c_tp1_expr  # c_{t+1} = sum of patterns with center=1 at t+1
    ]
    
    # Combine all equations
    # Note: transition_polys includes flow constraints, so we have:
    # - Flow conservation at t (4 equations)
    # - Flow conservation at t+1 (4 equations)  
    # - Transition equations (8 equations)
    all_equations = (
        invariance_eqs +
        [norm_t, norm_tp1] +
        transition_polys +
        center_defs
    )
    
    if verbose:
        print(f"  Total equations: {len(all_equations)}")
        print(f"    - Invariance: {len(invariance_eqs)}")
        print(f"    - Normalization: 2")
        print(f"    - De Bruijn transitions: {len(transition_polys)}")
        print(f"      (includes flow conservation at t and t+1)")
        print(f"    - Center definitions: 2")
        print()
    
    # Step 2: Collect all variables
    all_vars = list(freq_t.values()) + list(freq_tp1.values()) + [c_t, c_tp1]
    
    # Variables to eliminate (everything except c_t and c_tp1)
    vars_to_eliminate = [v for v in all_vars if v != c_t and v != c_tp1]
    
    if verbose:
        print("Step 2: Setting up Groebner basis elimination...")
        print(f"  Total variables: {len(all_vars)}")
        print(f"  Variables to eliminate: {len(vars_to_eliminate)}")
        print(f"  Target variables: c_t, c_{{t+1}}")
        print()
    
    # Step 3: Compute Groebner basis with lexicographic ordering
    # Order: eliminate all freq variables first, then c_t, then c_tp1
    # This will give us relations involving only c_t and c_tp1
    
    try:
        if verbose:
            print("Step 3: Computing Groebner basis...")
            print("  This may take a moment for large systems...")
            print()
        
        # Create polynomial ring ordering: eliminate freq vars, keep c_t and c_tp1
        # Order: freq_t vars > freq_tp1 vars > c_t > c_tp1
        ordering = vars_to_eliminate + [c_t, c_tp1]
        
        # Convert equations to polynomials
        polys = [Poly(eq, *ordering) for eq in all_equations if eq != 0]
        
        # Compute Groebner basis
        G = groebner(polys, *ordering, order='lex')
        
        if verbose:
            print(f"  Groebner basis computed: {len(G)} polynomials")
            print()
        
        # Step 4: Extract relations involving only c_t and c_tp1
        if verbose:
            print("Step 4: Extracting recurrence relation...")
            print()
        
        # Look for polynomials that only involve c_t and c_tp1
        recurrence_polys = []
        for poly in G:
            # Get variables in this polynomial
            poly_vars = poly.free_symbols
            # Check if it only involves c_t and c_tp1 (or is a constant)
            if poly_vars.issubset({c_t, c_tp1}):
                recurrence_polys.append(poly)
        
        if recurrence_polys:
            if verbose:
                print(f"  Found {len(recurrence_polys)} recurrence relation(s):")
                print()
                for i, poly in enumerate(recurrence_polys, 1):
                    print(f"  Relation {i}:")
                    print(f"    {poly.as_expr()} = 0")
                    print()
        else:
            if verbose:
                print("  No pure recurrence found in Groebner basis")
                print("  Checking for partially reduced relations...")
                print()
            
            # Look for polynomials with minimal variables
            # Sort by number of variables
            sorted_polys = sorted(G, key=lambda p: len(p.free_symbols))
            if sorted_polys:
                min_vars = len(sorted_polys[0].free_symbols)
                recurrence_polys = [p for p in sorted_polys if len(p.free_symbols) == min_vars]
                
                if verbose:
                    print(f"  Found {len(recurrence_polys)} relation(s) with {min_vars} variables:")
                    print()
                    for i, poly in enumerate(recurrence_polys[:3], 1):  # Show first 3
                        print(f"  Relation {i}:")
                        print(f"    {poly.as_expr()} = 0")
                        print()
        
        return {
            'groebner_basis': G,
            'recurrence_polynomials': recurrence_polys,
            'num_basis_polys': len(G),
            'c_t': c_t,
            'c_tp1': c_tp1,
            'status': 'success',
            'message': f'Groebner basis computed. Found {len(recurrence_polys)} recurrence relation(s).'
        }
        
    except Exception as e:
        if verbose:
            print(f"  Error computing Groebner basis: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            'status': 'error',
            'error': str(e),
            'message': 'Groebner basis computation failed. System may be too complex or inconsistent.'
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Groebner basis solver for center column recurrence"
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
    
    results = solve_center_groebner(N=args.N, verbose=args.verbose)
    
    print("="*70)
    print("GROEBNER BASIS SOLVING COMPLETE")
    print("="*70)
    print()
    print(f"Status: {results['status']}")
    if 'message' in results:
        print(f"Message: {results['message']}")
    print()
    
    if results['status'] == 'success' and results['recurrence_polynomials']:
        print("="*70)
        print("RECURRENCE RELATION(S) FOUND")
        print("="*70)
        print()
        for i, poly in enumerate(results['recurrence_polynomials'], 1):
            print(f"Relation {i}:")
            print(f"  {poly.as_expr()} = 0")
            print()
            # Try to solve for c_{t+1} in terms of c_t
            try:
                solved = solve(poly.as_expr(), results['c_tp1'])
                if solved:
                    print(f"  Solved for c_{{t+1}}:")
                    for sol in solved:
                        print(f"    c_{{t+1}} = {simplify(sol)}")
                    print()
            except:
                pass


if __name__ == "__main__":
    main()


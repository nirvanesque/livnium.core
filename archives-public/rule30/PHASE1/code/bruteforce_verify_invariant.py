#!/usr/bin/env python3
"""
Bruteforce Verification of Invariants

STEP 3: Exhaustive verification for all binary rows of length N.
Turns "looks right" into "I checked everything up to N."
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from fractions import Fraction
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.divergence_v3 import divergence_v3_rational, enumerate_patterns
from experiments.rule30.rule30_algebra import rule30_step

Pattern = Tuple[int, int, int]


def verify_invariant_exhaustive(
    weights: Dict[Pattern, Fraction],
    N: int,
    max_steps: int = 20,
    cyclic: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Verify invariant for ALL possible binary rows of length N.
    
    Args:
        weights: Pattern weights (rational)
        N: Row length
        max_steps: Maximum evolution steps to test
        cyclic: Use cyclic boundary conditions
        verbose: Print progress
        
    Returns:
        Dict with verification results
    """
    total_rows = 2 ** N
    
    if verbose:
        print(f"Verifying invariant for all {total_rows:,} rows of length {N}...")
        print(f"Testing up to {max_steps} evolution steps")
        print()
    
    counterexamples = []
    verified_count = 0
    
    for i, row_bits in enumerate(product([0, 1], repeat=N)):
        if verbose and (i + 1) % 1000 == 0:
            progress = ((i + 1) / total_rows) * 100
            print(f"  Progress: {i+1:,}/{total_rows:,} ({progress:.1f}%)", end='\r')
        
        row = list(row_bits)
        d0 = divergence_v3_rational(row, weights, cyclic=cyclic)
        
        current_row = row.copy()
        
        for step in range(1, max_steps + 1):
            current_row = rule30_step(current_row, cyclic=cyclic)
            dt = divergence_v3_rational(current_row, weights, cyclic=cyclic)
            
            if dt != d0:
                counterexamples.append({
                    'initial_row': row.copy(),
                    'step': step,
                    'initial_divergence': d0,
                    'divergence_at_step': dt,
                    'difference': dt - d0
                })
                break
        
        if len(counterexamples) == 0 or counterexamples[-1]['initial_row'] != row:
            verified_count += 1
    
    if verbose:
        print()  # New line after progress
    
    return {
        'N': N,
        'total_rows': total_rows,
        'verified_count': verified_count,
        'counterexamples': counterexamples,
        'all_verified': len(counterexamples) == 0,
        'max_steps': max_steps
    }


def main():
    parser = argparse.ArgumentParser(
        description="Bruteforce verification of invariants",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--N',
        type=int,
        nargs='+',
        default=[8, 10, 12],
        help='Row lengths to test (default: 8 10 12, warning: 2^N rows each)'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        default=20,
        help='Maximum evolution steps to test (default: 20)'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        help='Weights as comma-separated list: w000,w001,w010,w011,w100,w101,w110,w111'
    )
    
    parser.add_argument(
        '--from-nullspace',
        type=int,
        default=0,
        help='Use invariant from nullspace (vector index, requires running test_divergence_v3_invariant.py first)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("BRUTEFORCE INVARIANT VERIFICATION")
    print("="*70)
    print()
    
    # Get weights
    if args.weights:
        # Parse weights from command line
        weight_values = [Fraction(x) for x in args.weights.split(',')]
        if len(weight_values) != 8:
            print("Error: Must provide exactly 8 weights")
            return
        
        patterns = enumerate_patterns()
        weights = {patterns[i]: weight_values[i] for i in range(8)}
    else:
        # For now, use a default or require --from-nullspace
        print("Error: Must provide --weights or implement --from-nullspace")
        print("Example: --weights '0,1,-1,0,0,-1,1,0'")
        return
    
    print("Testing invariant:")
    from experiments.rule30.divergence_v3 import format_weights_rational
    print(f"  D3(s) = {format_weights_rational(weights)}")
    print()
    
    # Verify for each N
    all_results = {}
    
    for N in args.N:
        print(f"{'='*70}")
        print(f"Testing N = {N}")
        print(f"{'='*70}")
        
        if 2 ** N > 100000:
            print(f"Warning: {2**N:,} rows to test - this will take a while")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                continue
        
        result = verify_invariant_exhaustive(
            weights,
            N,
            max_steps=args.max_steps,
            cyclic=True,
            verbose=True
        )
        
        all_results[N] = result
        
        print()
        print(f"Results for N = {N}:")
        print(f"  Total rows tested: {result['total_rows']:,}")
        print(f"  Verified: {result['verified_count']:,}")
        
        if result['all_verified']:
            print(f"  ✓✓✓ ALL ROWS VERIFIED!")
            print(f"  No counterexamples found for N={N} up to {args.max_steps} steps")
        else:
            print(f"  ✗ COUNTEREXAMPLES FOUND: {len(result['counterexamples'])}")
            for i, cex in enumerate(result['counterexamples'][:5], 1):
                print(f"    Counterexample {i}:")
                print(f"      Initial row: {cex['initial_row']}")
                print(f"      Fails at step: {cex['step']}")
                print(f"      D0 = {cex['initial_divergence']}")
                print(f"      D{cex['step']} = {cex['divergence_at_step']}")
                print(f"      Difference: {cex['difference']}")
            if len(result['counterexamples']) > 5:
                print(f"    ... and {len(result['counterexamples']) - 5} more")
        print()
    
    # Final summary
    print("="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_verified = all(r['all_verified'] for r in all_results.values())
    
    for N, result in all_results.items():
        status = "✓ VERIFIED" if result['all_verified'] else "✗ FAILED"
        print(f"N = {N:2d}: {status} ({result['verified_count']:,}/{result['total_rows']:,} rows)")
    
    print()
    
    if all_verified:
        print("✓✓✓ INVARIANT VERIFIED FOR ALL TESTED N")
        print()
        print("This invariant holds exactly for:")
        for N in args.N:
            print(f"  - All {2**N:,} rows of length {N} up to {args.max_steps} steps")
        print()
        print("This is a STRONG computational confirmation.")
    else:
        print("⚠ Some counterexamples found")
        print("The invariant may not hold exactly, or there may be numerical issues")
    
    print()


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Extended Verification for Large N

Verifies invariants on random samples for larger N values (N=14, 16, etc.)
where exhaustive verification is computationally infeasible.
"""

import argparse
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple
from fractions import Fraction

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.divergence_v3 import divergence_v3_rational, enumerate_patterns
from experiments.rule30.rule30_algebra import rule30_step

Pattern = Tuple[int, int, int]


def get_invariant_weights(invariant_num: int) -> Dict[Pattern, Fraction]:
    """Get weights for a specific invariant."""
    patterns = enumerate_patterns()
    pattern_str = {p: ''.join(str(b) for b in p) for p in patterns}
    
    if invariant_num == 1:
        # I1: freq(100) - freq(001)
        return {p: Fraction(1 if pattern_str[p]=='100' else -1 if pattern_str[p]=='001' else 0) for p in patterns}
    elif invariant_num == 2:
        # I2: freq(001) - freq(010) - freq(011) + freq(101)
        return {p: Fraction(1 if pattern_str[p]=='001' else -1 if pattern_str[p] in ['010','011'] else 1 if pattern_str[p]=='101' else 0) for p in patterns}
    elif invariant_num == 3:
        # I3: freq(110) - freq(011)
        return {p: Fraction(1 if pattern_str[p]=='110' else -1 if pattern_str[p]=='011' else 0) for p in patterns}
    elif invariant_num == 4:
        # I4: freq(000) + freq(001) + 2*freq(010) + 3*freq(011) + freq(111)
        return {p: Fraction(1 if pattern_str[p] in ['000','001','111'] else 2 if pattern_str[p]=='010' else 3 if pattern_str[p]=='011' else 0) for p in patterns}
    else:
        raise ValueError(f"Invalid invariant number: {invariant_num}")


def verify_invariant_random_sample(
    weights: Dict[Pattern, Fraction],
    N: int,
    num_samples: int,
    max_steps: int = 20,
    cyclic: bool = True
) -> Dict:
    """
    Verify invariant on random sample of rows.
    
    Args:
        weights: Pattern weights
        N: Row length
        num_samples: Number of random rows to test
        max_steps: Maximum evolution steps
        cyclic: Use cyclic boundary conditions
        
    Returns:
        Dict with verification results
    """
    print(f"  Testing {num_samples:,} random rows of length {N}...")
    
    counterexamples = []
    verified_count = 0
    
    for i in range(num_samples):
        if (i + 1) % max(1, num_samples // 10) == 0:
            progress = ((i + 1) / num_samples) * 100
            print(f"    Progress: {i+1:,}/{num_samples:,} ({progress:.1f}%)", end='\r')
        
        row = [random.randint(0, 1) for _ in range(N)]
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
    
    print()  # New line
    
    return {
        'N': N,
        'num_samples': num_samples,
        'verified_count': verified_count,
        'counterexamples': counterexamples,
        'all_verified': len(counterexamples) == 0,
        'max_steps': max_steps
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extended verification for large N using random sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--N',
        type=int,
        nargs='+',
        default=[14, 16],
        help='Row lengths to test (default: 14 16)'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10000,
        help='Number of random samples per N (default: 10000)'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        default=20,
        help='Maximum evolution steps (default: 20)'
    )
    
    parser.add_argument(
        '--invariant',
        type=int,
        choices=[1, 2, 3, 4],
        help='Test specific invariant (1-4), or test all if not specified'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("EXTENDED VERIFICATION - LARGE N (RANDOM SAMPLING)")
    print("="*70)
    print()
    print(f"Testing {args.num_samples:,} random samples per N")
    print(f"Evolution steps: {args.max_steps}")
    print()
    
    invariants_to_test = [args.invariant] if args.invariant else [1, 2, 3, 4]
    
    all_results = {}
    
    for inv_num in invariants_to_test:
        print(f"{'='*70}")
        print(f"INVARIANT {inv_num}/4")
        print(f"{'='*70}")
        
        weights = get_invariant_weights(inv_num)
        
        # Print formula
        from experiments.rule30.divergence_v3 import format_weights_rational
        print(f"Formula: D3(s) = {format_weights_rational(weights)}")
        print()
        
        inv_results = {}
        
        for N in args.N:
            print(f"Testing N = {N}")
            print(f"  Total possible rows: 2^{N} = {2**N:,}")
            print(f"  Sampling: {args.num_samples:,} rows ({100 * args.num_samples / (2**N):.4f}% of space)")
            
            result = verify_invariant_random_sample(
                weights,
                N,
                num_samples=args.num_samples,
                max_steps=args.max_steps,
                cyclic=True
            )
            
            inv_results[N] = result
            
            print(f"  Results:")
            print(f"    Verified: {result['verified_count']:,}/{result['num_samples']:,}")
            
            if result['all_verified']:
                print(f"    ✓✓✓ NO COUNTEREXAMPLES FOUND")
                print(f"    (Strong evidence for N={N} with {args.num_samples:,} samples)")
            else:
                print(f"    ✗ COUNTEREXAMPLES: {len(result['counterexamples'])}")
                for i, cex in enumerate(result['counterexamples'][:3], 1):
                    print(f"      Counterexample {i}: step {cex['step']}, Δ={cex['difference']}")
            
            print()
        
        all_results[inv_num] = inv_results
    
    # Summary
    print("="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print()
    
    for inv_num in invariants_to_test:
        print(f"Invariant {inv_num}:")
        for N in args.N:
            result = all_results[inv_num][N]
            status = "✓ VERIFIED" if result['all_verified'] else "✗ FAILED"
            print(f"  N={N:2d}: {status} ({result['verified_count']:,}/{result['num_samples']:,} samples)")
        print()
    
    print("Note: This is statistical verification, not exhaustive proof.")
    print("For exhaustive verification up to N=12, use bruteforce_verify_invariant.py")
    print()


if __name__ == '__main__':
    main()


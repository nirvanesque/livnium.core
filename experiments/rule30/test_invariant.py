#!/usr/bin/env python3
"""
Comprehensive Rule 30 Invariant Test

Consolidated test suite for the divergence invariant -0.572222233.
Tests across sequence lengths, cube sizes, and recursive scales.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.rule30_optimized import generate_center_column_direct
from experiments.rule30.geometry_embed import embed_into_cube
from experiments.rule30.diagnostics import compute_divergence_path
from experiments.rule30.diagnostics import create_sequence_vectors, _compute_field_divergence

def compute_direct_divergence(sequence: list, window_size: int = 5) -> float:
    """Compute divergence directly from sequence."""
    vectors = create_sequence_vectors(sequence)
    if len(vectors) < window_size:
        return 0.0
    divergence_values = []
    for i in range(len(vectors) - window_size + 1):
        window_vecs = vectors[i:i + window_size]
        divergence = _compute_field_divergence(window_vecs, window_vecs)
        divergence_values.append(divergence)
    return float(np.mean(divergence_values)) if divergence_values else 0.0
from experiments.rule30.recursive_embed import embed_recursive, analyze_recursive_patterns


EXPECTED_INVARIANT = -0.572222233
TOLERANCE = 1e-6


def test_sequence_lengths(lengths: list) -> bool:
    """Test invariant across sequence lengths."""
    print(f"\n{'='*60}")
    print("TEST: Sequence Length Independence")
    print(f"{'='*60}")
    
    all_pass = True
    for n in lengths:
        seq = generate_center_column_direct(n, show_progress=False)
        div = compute_direct_divergence(seq)
        dev = abs(div - EXPECTED_INVARIANT)
        status = "✓" if dev < TOLERANCE else "✗"
        print(f"{status} n={n:6,}: {div:.9f} (dev: {dev:.2e})")
        if dev >= TOLERANCE:
            all_pass = False
    return all_pass


def test_cube_sizes(n_steps: int, sizes: list) -> bool:
    """Test invariant across cube sizes."""
    print(f"\n{'='*60}")
    print("TEST: Cube Size Independence")
    print(f"{'='*60}")
    
    seq = generate_center_column_direct(n_steps, show_progress=False)
    all_pass = True
    
    for size in sizes:
        _, path = embed_into_cube(seq, cube_size=size)
        div_path = compute_divergence_path(path, show_progress=False)
        div = float(np.mean(div_path))
        dev = abs(div - EXPECTED_INVARIANT)
        status = "✓" if dev < TOLERANCE else "✗"
        print(f"{status} {size}×{size}×{size}: {div:.9f} (dev: {dev:.2e})")
        if dev >= TOLERANCE:
            all_pass = False
    return all_pass


def test_recursive(n_steps: int, max_depth: int = 3) -> bool:
    """Test invariant across recursive scales."""
    print(f"\n{'='*60}")
    print("TEST: Recursive Scale Independence")
    print(f"{'='*60}")
    
    seq = generate_center_column_direct(n_steps, show_progress=False)
    engine, paths = embed_recursive(seq, base_cube_size=3, max_depth=max_depth)
    results = analyze_recursive_patterns(engine, paths)
    
    all_pass = True
    for level_id in sorted(results.keys()):
        div = results[level_id]['divergence_mean']
        dev = abs(div - EXPECTED_INVARIANT)
        status = "✓" if dev < TOLERANCE else "✗"
        print(f"{status} Level {level_id}: {div:.9f} (dev: {dev:.2e})")
        if dev >= TOLERANCE:
            all_pass = False
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Rule 30 invariant test")
    parser.add_argument('--quick', action='store_true', help='Quick test (smaller sequences)')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    args = parser.parse_args()
    
    print("="*60)
    print("RULE 30 DIVERGENCE INVARIANT TEST")
    print("="*60)
    print(f"Expected: {EXPECTED_INVARIANT:.9f}")
    print(f"Tolerance: ±{TOLERANCE}")
    
    if args.quick:
        lengths = [1000, 10000]
        sizes = [3, 5]
        n_steps = 5000
        max_depth = 2
    else:
        lengths = [1000, 10000, 100000]
        sizes = [3, 5, 7]
        n_steps = 10000
        max_depth = 3
    
    results = {}
    
    if args.all or True:  # Always run sequence test
        results['sequence'] = test_sequence_lengths(lengths)
    
    if args.all:
        results['cubes'] = test_cube_sizes(n_steps, sizes)
        results['recursive'] = test_recursive(n_steps, max_depth)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for test, passed in results.items():
        print(f"{test:15s}: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    all_passed = all(results.values())
    print(f"\n{'✓✓✓ ALL TESTS PASSED' if all_passed else '✗✗✗ SOME TESTS FAILED'}")
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())


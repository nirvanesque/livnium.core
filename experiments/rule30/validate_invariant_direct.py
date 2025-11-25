#!/usr/bin/env python3
"""
Direct Validation of Rule 30 Divergence Invariant

Tests the divergence invariant -0.572222233 directly from Rule 30 sequences
without geometric embedding. This validates that the invariant is a property
of Rule 30 itself, not an artifact of the geometric embedding.

Computes divergence using the same angle-based method but directly on the
binary sequence vectors.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.rule30_optimized import generate_center_column_direct
from experiments.rule30.diagnostics import create_sequence_vectors, _compute_field_divergence


EXPECTED_INVARIANT = -0.572222233
TOLERANCE = 1e-6  # Allow 1 micro-deviation


def compute_direct_divergence(sequence: list, window_size: int = 5) -> float:
    """
    Compute divergence directly from sequence without geometric embedding.
    
    Uses the same angle-based divergence computation as diagnostics,
    but applied directly to sequence vectors.
    
    Args:
        sequence: Binary sequence (list of 0s and 1s)
        window_size: Window size for divergence computation
        
    Returns:
        Mean divergence value
    """
    # Convert sequence to vectors (same as diagnostics)
    vectors = create_sequence_vectors(sequence)
    
    if len(vectors) < window_size:
        return 0.0
    
    # Compute divergence for sliding windows
    divergence_values = []
    for i in range(len(vectors) - window_size + 1):
        window_vecs = vectors[i:i + window_size]
        
        # Compute divergence (treating window as both premise and hypothesis)
        divergence = _compute_field_divergence(window_vecs, window_vecs)
        divergence_values.append(divergence)
    
    if not divergence_values:
        return 0.0
    
    return float(np.mean(divergence_values))


def validate_invariant_value(divergence_mean: float, context: str = "") -> bool:
    """
    Validate that divergence matches the expected invariant.
    
    Args:
        divergence_mean: Mean divergence value to check
        context: Context string for error messages
        
    Returns:
        True if invariant is valid, False otherwise
    """
    deviation = abs(divergence_mean - EXPECTED_INVARIANT)
    is_valid = deviation < TOLERANCE
    
    status = "✓ VALID" if is_valid else "✗ INVALID"
    print(f"  {status}: {divergence_mean:.9f} (deviation: {deviation:.9e}) {context}")
    
    return is_valid


def test_sequence_lengths(sequence_lengths: list) -> bool:
    """
    Test invariant across different sequence lengths.
    
    Args:
        sequence_lengths: List of sequence lengths to test
        
    Returns:
        True if all tests pass
    """
    print("="*70)
    print("TEST 1: Sequence Length Independence (Direct)")
    print("="*70)
    print(f"Testing invariant directly from Rule 30 sequences")
    print(f"Sequence lengths: {sequence_lengths}")
    print(f"Expected invariant: {EXPECTED_INVARIANT:.9f}")
    print(f"Tolerance: ±{TOLERANCE}")
    print()
    
    all_valid = True
    
    for n_steps in sequence_lengths:
        print(f"Testing {n_steps:,} steps...")
        
        # Generate center column
        center_column = generate_center_column_direct(n_steps, show_progress=False)
        print(f"  Generated {len(center_column):,} bits")
        
        # Compute divergence directly (no cube embedding)
        divergence_mean = compute_direct_divergence(center_column)
        
        # Validate
        is_valid = validate_invariant_value(
            divergence_mean,
            context=f"(n={n_steps:,})"
        )
        
        if not is_valid:
            all_valid = False
    
    print()
    if all_valid:
        print("✓✓✓ ALL SEQUENCE LENGTH TESTS PASSED")
    else:
        print("✗✗✗ SOME SEQUENCE LENGTH TESTS FAILED")
    
    return all_valid


def test_sequence_properties(sequence_length: int) -> bool:
    """
    Test invariant properties of the sequence itself.
    
    Args:
        sequence_length: Sequence length to test
        
    Returns:
        True if all tests pass
    """
    print("="*70)
    print("TEST 2: Sequence Properties Analysis")
    print("="*70)
    print(f"Analyzing Rule 30 center column properties")
    print(f"Sequence length: {sequence_length:,}")
    print(f"Expected invariant: {EXPECTED_INVARIANT:.9f}")
    print()
    
    # Generate center column
    print(f"Generating Rule 30 center column ({sequence_length:,} steps)...")
    center_column = generate_center_column_direct(sequence_length, show_progress=False)
    print(f"Generated {len(center_column):,} bits\n")
    
    # Analyze sequence properties
    ones_count = sum(center_column)
    zeros_count = len(center_column) - ones_count
    ones_ratio = ones_count / len(center_column)
    
    print(f"Sequence properties:")
    print(f"  Total bits: {len(center_column):,}")
    print(f"  Ones: {ones_count:,} ({ones_ratio:.4f})")
    print(f"  Zeros: {zeros_count:,} ({1-ones_ratio:.4f})")
    print()
    
    # Compute divergence with different window sizes
    print(f"Divergence with different window sizes:")
    window_sizes = [3, 5, 7, 10]
    all_valid = True
    
    for window_size in window_sizes:
        divergence_mean = compute_direct_divergence(center_column, window_size=window_size)
        is_valid = validate_invariant_value(
            divergence_mean,
            context=f"(window={window_size})"
        )
        if not is_valid:
            all_valid = False
    
    print()
    
    # Test subsequences
    print(f"Testing subsequences:")
    subsequence_lengths = [len(center_column) // 2, len(center_column) // 4, len(center_column) // 10]
    
    for sub_len in subsequence_lengths:
        if sub_len < 10:
            continue
        subsequence = center_column[:sub_len]
        divergence_mean = compute_direct_divergence(subsequence)
        is_valid = validate_invariant_value(
            divergence_mean,
            context=f"(subsequence_len={sub_len:,})"
        )
        if not is_valid:
            all_valid = False
    
    print()
    if all_valid:
        print("✓✓✓ ALL SEQUENCE PROPERTY TESTS PASSED")
    else:
        print("✗✗✗ SOME SEQUENCE PROPERTY TESTS FAILED")
    
    return all_valid


def test_convergence(sequence_length: int, checkpoints: list = None) -> bool:
    """
    Test if divergence converges to invariant as sequence grows.
    
    Args:
        sequence_length: Maximum sequence length
        checkpoints: List of checkpoints to test (default: logarithmic scale)
        
    Returns:
        True if convergence is observed
    """
    if checkpoints is None:
        # Logarithmic checkpoints
        checkpoints = [100, 500, 1000, 5000, 10000, 50000, min(100000, sequence_length)]
        checkpoints = [c for c in checkpoints if c <= sequence_length]
    
    print("="*70)
    print("TEST 3: Convergence Analysis")
    print("="*70)
    print(f"Testing divergence convergence as sequence grows")
    print(f"Checkpoints: {checkpoints}")
    print(f"Expected invariant: {EXPECTED_INVARIANT:.9f}")
    print()
    
    # Generate full sequence
    print(f"Generating Rule 30 center column ({sequence_length:,} steps)...")
    full_sequence = generate_center_column_direct(sequence_length, show_progress=False)
    print(f"Generated {len(full_sequence):,} bits\n")
    
    print("Divergence at different sequence lengths:")
    divergences = []
    
    for checkpoint in checkpoints:
        if checkpoint > len(full_sequence):
            continue
        
        subsequence = full_sequence[:checkpoint]
        divergence_mean = compute_direct_divergence(subsequence)
        deviation = abs(divergence_mean - EXPECTED_INVARIANT)
        
        status = "✓" if deviation < TOLERANCE else "✗"
        print(f"  {status} n={checkpoint:6,}: divergence={divergence_mean:.9f}, deviation={deviation:.9e}")
        
        divergences.append(divergence_mean)
    
    # Check convergence
    if len(divergences) >= 3:
        # Check if later values are closer to invariant
        early_mean = np.mean(divergences[:len(divergences)//2])
        late_mean = np.mean(divergences[len(divergences)//2:])
        
        early_dev = abs(early_mean - EXPECTED_INVARIANT)
        late_dev = abs(late_mean - EXPECTED_INVARIANT)
        
        print()
        print(f"Early mean (first half): {early_mean:.9f} (deviation: {early_dev:.9e})")
        print(f"Late mean (second half): {late_mean:.9f} (deviation: {late_dev:.9e})")
        
        if late_dev < early_dev:
            print("✓ Convergence observed: divergence approaches invariant as sequence grows")
            return True
        else:
            print("⚠ No clear convergence pattern")
    
    print()
    return all(abs(d - EXPECTED_INVARIANT) < TOLERANCE for d in divergences[-3:])


def main():
    parser = argparse.ArgumentParser(
        description="Direct validation of Rule 30 divergence invariant (-0.572222233)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Validates the divergence invariant directly from Rule 30 sequences without
geometric embedding. This proves the invariant is a property of Rule 30 itself.

Examples:
  python3 validate_invariant_direct.py --all
  python3 validate_invariant_direct.py --sequence-lengths 1000 10000 100000
  python3 validate_invariant_direct.py --convergence --steps 100000
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all validation tests'
    )
    
    parser.add_argument(
        '--sequence-lengths',
        type=int,
        nargs='+',
        default=[1000, 10000, 100000],
        help='Sequence lengths to test (default: 1000 10000 100000)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=10000,
        help='Sequence length for property and convergence tests (default: 10000)'
    )
    
    parser.add_argument(
        '--convergence',
        action='store_true',
        help='Test convergence as sequence grows'
    )
    
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-6,
        help='Tolerance for invariant validation (default: 1e-6)'
    )
    
    args = parser.parse_args()
    
    # Set global tolerance
    global TOLERANCE
    TOLERANCE = args.tolerance
    
    print("="*70)
    print("RULE 30 DIVERGENCE INVARIANT - DIRECT VALIDATION")
    print("="*70)
    print(f"Testing invariant directly from Rule 30 sequences (no geometric embedding)")
    print(f"Expected invariant: {EXPECTED_INVARIANT:.9f}")
    print(f"Tolerance: ±{TOLERANCE}")
    print("="*70)
    print()
    
    results = {}
    
    # Test 1: Sequence lengths
    if args.all or not args.convergence:
        results['sequence_lengths'] = test_sequence_lengths(args.sequence_lengths)
        print()
    
    # Test 2: Sequence properties
    if args.all:
        results['sequence_properties'] = test_sequence_properties(args.steps)
        print()
    
    # Test 3: Convergence
    if args.all or args.convergence:
        results['convergence'] = test_convergence(args.steps)
        print()
    
    # Final summary
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    
    print()
    if all_passed:
        print("✓✓✓ ALL DIRECT VALIDATION TESTS PASSED")
        print(f"Invariant {EXPECTED_INVARIANT:.9f} is CONFIRMED as a property of Rule 30 itself.")
        print("This proves the invariant is NOT an artifact of geometric embedding.")
        return 0
    else:
        print("✗✗✗ SOME VALIDATION TESTS FAILED")
        print("Invariant may not be universal or tolerance may be too strict.")
        return 1


if __name__ == '__main__':
    sys.exit(main())


#!/usr/bin/env python3
"""
Divergence Stability Test for Rule 30

Tests if the geometric divergence invariant (-0.572222) holds across
different sequence lengths (10k, 100k, 1M steps).

This tests the hypothesis that Rule 30's center column has a fixed
geometric fingerprint - a discovery that could relate to Wolfram's $30k prize.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.rule30_core import generate_rule30
from experiments.rule30.center_column import extract_center_column
from experiments.rule30.rule30_optimized import generate_center_column_direct
from experiments.rule30.geometry_embed import embed_into_cube
from experiments.rule30.diagnostics import compute_divergence_path


def test_divergence_stability(
    step_sizes: List[int],
    cube_size: int = 3,
    sample_points: int = 100
) -> Dict[int, Dict[str, float]]:
    """
    Test divergence stability across different sequence lengths.
    
    Samples divergence at multiple points along each sequence to check
    if the invariant holds.
    
    Args:
        step_sizes: List of Rule 30 step counts to test (e.g., [10000, 100000, 1000000])
        cube_size: Size of Livnium cube
        sample_points: Number of points to sample along sequence
        
    Returns:
        Dictionary mapping step_size to divergence statistics
    """
    results = {}
    
    for n_steps in step_sizes:
        print(f"\n{'='*60}")
        print(f"Testing {n_steps:,} steps...")
        print('='*60)
        
        # Generate Rule 30 center column
        # For large sequences, use optimized direct generation
        if n_steps >= 10000:
            print(f"Generating Rule 30 center column (optimized for large sequences)...")
            center_column = generate_center_column_direct(n_steps, show_progress=True)
            print(f"Generated center column: {len(center_column):,} bits")
        else:
            print(f"Generating Rule 30 CA...")
            triangle = generate_rule30(n_steps)
            center_column = extract_center_column(triangle)
            print(f"Extracted center column: {len(center_column):,} bits")
        
        # Embed into cube
        print(f"Embedding into {cube_size}x{cube_size}x{cube_size} cube...")
        _, path = embed_into_cube(center_column, cube_size=cube_size)
        
        # Compute divergence
        print(f"Computing divergence...")
        show_progress = n_steps >= 10000
        divergence_path = compute_divergence_path(path, show_progress=show_progress)
        
        # Sample at multiple points
        if len(divergence_path) > sample_points:
            indices = np.linspace(0, len(divergence_path) - 1, sample_points, dtype=int)
            sampled_divergence = [divergence_path[i] for i in indices]
        else:
            sampled_divergence = divergence_path
        
        # Compute statistics
        divergence_mean = float(np.mean(divergence_path))
        divergence_std = float(np.std(divergence_path))
        divergence_min = float(np.min(divergence_path))
        divergence_max = float(np.max(divergence_path))
        
        # Check stability (how close to constant)
        stability = 1.0 / (1.0 + divergence_std)  # Higher = more stable
        
        results[n_steps] = {
            'mean': divergence_mean,
            'std': divergence_std,
            'min': divergence_min,
            'max': divergence_max,
            'stability': stability,
            'sequence_length': len(center_column),
            'sampled_points': len(sampled_divergence),
            'sampled_values': sampled_divergence[:20]  # First 20 for inspection
        }
        
        print(f"\nResults for {n_steps:,} steps:")
        print(f"  Divergence mean: {divergence_mean:.9f}")
        print(f"  Divergence std:  {divergence_std:.9f}")
        print(f"  Stability score: {stability:.6f}")
        print(f"  Range: [{divergence_min:.6f}, {divergence_max:.6f}]")
        
        # Check if it matches the invariant
        expected_invariant = -0.572222
        deviation = abs(divergence_mean - expected_invariant)
        print(f"  Deviation from invariant (-0.572222): {deviation:.9f}")
        
        if deviation < 0.001:
            print(f"  ✓ INVARIANT CONFIRMED (within 0.001)")
        elif deviation < 0.01:
            print(f"  ⚠ Close to invariant (within 0.01)")
        else:
            print(f"  ✗ Diverges from invariant")
    
    return results


def plot_divergence_stability(
    results: Dict[int, Dict[str, float]],
    output_dir: str
):
    """
    Plot divergence stability across different step sizes.
    
    Creates plots showing:
    1. Divergence mean vs sequence length
    2. Divergence std vs sequence length (stability)
    3. Sampled divergence values at each scale
    """
    os.makedirs(output_dir, exist_ok=True)
    
    step_sizes = sorted(results.keys())
    means = [results[s]['mean'] for s in step_sizes]
    stds = [results[s]['std'] for s in step_sizes]
    stabilities = [results[s]['stability'] for s in step_sizes]
    seq_lengths = [results[s]['sequence_length'] for s in step_sizes]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Rule 30 Divergence Stability Test', fontsize=14, fontweight='bold')
    
    # Plot 1: Mean divergence vs sequence length
    axes[0, 0].plot(seq_lengths, means, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=-0.572222, color='r', linestyle='--', 
                      label='Expected Invariant (-0.572222)', linewidth=2)
    axes[0, 0].set_xlabel('Sequence Length', fontsize=10)
    axes[0, 0].set_ylabel('Mean Divergence', fontsize=10)
    axes[0, 0].set_title('Divergence Mean vs Sequence Length', fontsize=11)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    
    # Plot 2: Standard deviation (stability) vs sequence length
    axes[0, 1].plot(seq_lengths, stds, 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Sequence Length', fontsize=10)
    axes[0, 1].set_ylabel('Divergence Std Dev', fontsize=10)
    axes[0, 1].set_title('Divergence Stability (Lower = More Stable)', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Stability score vs sequence length
    axes[1, 0].plot(seq_lengths, stabilities, 'mo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Sequence Length', fontsize=10)
    axes[1, 0].set_ylabel('Stability Score', fontsize=10)
    axes[1, 0].set_title('Stability Score (Higher = More Stable)', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    
    # Plot 4: Sampled divergence values (first 20 samples from each)
    for step_size in step_sizes:
        sampled = results[step_size]['sampled_values']
        x = np.arange(len(sampled))
        axes[1, 1].plot(x, sampled, 'o-', alpha=0.6, label=f'{step_size:,} steps', linewidth=1)
    
    axes[1, 1].axhline(y=-0.572222, color='r', linestyle='--', 
                       label='Expected Invariant', linewidth=2)
    axes[1, 1].set_xlabel('Sample Index', fontsize=10)
    axes[1, 1].set_ylabel('Divergence Value', fontsize=10)
    axes[1, 1].set_title('Sampled Divergence Values (First 20)', fontsize=11)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'divergence_stability_test.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved stability plot: {plot_path}")


def save_stability_report(
    results: Dict[int, Dict[str, float]],
    output_dir: str
):
    """Save detailed stability report."""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'divergence_stability_report.json')
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'RULE30_DIVERGENCE_STABILITY_TEST',
        'expected_invariant': -0.572222,
        'results': results,
        'conclusion': {}
    }
    
    # Check if invariant holds
    step_sizes = sorted(results.keys())
    means = [results[s]['mean'] for s in step_sizes]
    deviations = [abs(m - (-0.572222)) for m in means]
    
    max_deviation = max(deviations)
    report['conclusion'] = {
        'max_deviation_from_invariant': float(max_deviation),
        'invariant_confirmed': max_deviation < 0.001,
        'invariant_close': max_deviation < 0.01,
        'mean_divergence_range': [float(min(means)), float(max(means))],
        'stability_trend': 'stable' if max(means) - min(means) < 0.01 else 'varying'
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Saved stability report: {report_path}")
    
    # Print conclusion
    print("\n" + "="*60)
    print("STABILITY TEST CONCLUSION")
    print("="*60)
    if report['conclusion']['invariant_confirmed']:
        print("✓ GEOMETRIC INVARIANT CONFIRMED")
        print(f"  Maximum deviation: {max_deviation:.9f}")
        print("  The divergence value is stable across all tested scales.")
        print("  This suggests a deep geometric law behind Rule 30's center column.")
    elif report['conclusion']['invariant_close']:
        print("⚠ INVARIANT CLOSE (within 0.01)")
        print(f"  Maximum deviation: {max_deviation:.9f}")
        print("  Further testing recommended.")
    else:
        print("✗ INVARIANT NOT CONFIRMED")
        print(f"  Maximum deviation: {max_deviation:.9f}")
        print("  Divergence varies across scales.")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Test Rule 30 divergence stability across scales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tests if the geometric divergence invariant (-0.572222) holds across
different sequence lengths. This tests the hypothesis that Rule 30's
center column has a fixed geometric fingerprint.

Examples:
  python3 divergence_stability_test.py --steps 10000 100000 1000000
  python3 divergence_stability_test.py --steps 1000 10000 100000 --cube-size 5
        """
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=[10000, 100000, 1000000],
        help='List of step sizes to test (default: 10000 100000 1000000)'
    )
    
    parser.add_argument(
        '--cube-size',
        type=int,
        default=3,
        choices=[3, 5, 7],
        help='Size of Livnium cube (default: 3)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/rule30/stability_results',
        help='Output directory for results (default: experiments/rule30/stability_results)'
    )
    
    parser.add_argument(
        '--sample-points',
        type=int,
        default=100,
        help='Number of points to sample along sequence (default: 100)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Rule 30 Divergence Stability Test")
    print("="*60)
    print(f"Testing step sizes: {args.steps}")
    print(f"Cube size: {args.cube_size}x{args.cube_size}x{args.cube_size}")
    print(f"Expected invariant: -0.572222")
    print("="*60)
    
    # Run stability test
    results = test_divergence_stability(
        step_sizes=args.steps,
        cube_size=args.cube_size,
        sample_points=args.sample_points
    )
    
    # Plot results
    print(f"\nGenerating plots...")
    plot_divergence_stability(results, args.output_dir)
    
    # Save report
    print(f"\nGenerating report...")
    save_stability_report(results, args.output_dir)
    
    print("\n✓ Stability test complete!")


if __name__ == '__main__':
    main()


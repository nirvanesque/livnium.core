#!/usr/bin/env python3
"""
Rule 30 Geometric Analysis Runner

Generates Rule 30 CA, extracts center column, embeds into Livnium cube,
and computes geometric diagnostics (divergence, tension, basin depth).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.rule30_core import generate_rule30
from experiments.rule30.center_column import extract_center_column
from experiments.rule30.rule30_optimized import generate_center_column_direct
from experiments.rule30.geometry_embed import embed_into_cube
from experiments.rule30.diagnostics import compute_all_diagnostics
from experiments.rule30.recursive_embed import embed_recursive, analyze_recursive_patterns


def save_plots(
    diagnostics: Dict[str, List[float]],
    output_dir: str,
    prefix: str = "rule30"
):
    """
    Save diagnostic plots.
    
    Args:
        diagnostics: Dictionary with 'divergence', 'tension', 'basin_depth'
        output_dir: Output directory for plots
        prefix: Filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Rule 30 Geometric Diagnostics', fontsize=14, fontweight='bold')
    
    x = np.arange(len(diagnostics['divergence']))
    
    # Divergence plot
    axes[0].plot(x, diagnostics['divergence'], 'b-', linewidth=1, alpha=0.7)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Divergence', fontsize=10)
    axes[0].set_title('Divergence Curve', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Tension plot
    axes[1].plot(x, diagnostics['tension'], 'g-', linewidth=1, alpha=0.7)
    axes[1].set_ylabel('Tension', fontsize=10)
    axes[1].set_title('Tension Curve', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Basin depth plot
    axes[2].plot(x, diagnostics['basin_depth'], 'm-', linewidth=1, alpha=0.7)
    axes[2].set_xlabel('Step', fontsize=10)
    axes[2].set_ylabel('Basin Depth', fontsize=10)
    axes[2].set_title('Basin Depth Trajectory', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{prefix}_diagnostics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {plot_path}")


def log_to_growth_journal(
    diagnostics: Dict[str, List[float]],
    n_steps: int,
    journal_path: str = "growth_journal.jsonl"
):
    """
    Append metrics to growth_journal.jsonl.
    
    Args:
        diagnostics: Dictionary with diagnostic metrics
        n_steps: Number of Rule 30 steps
        journal_path: Path to journal file
    """
    # Compute summary statistics
    divergence_mean = float(np.mean(diagnostics['divergence']))
    divergence_std = float(np.std(diagnostics['divergence']))
    tension_mean = float(np.mean(diagnostics['tension']))
    tension_std = float(np.std(diagnostics['tension']))
    basin_depth_mean = float(np.mean(diagnostics['basin_depth']))
    basin_depth_std = float(np.std(diagnostics['basin_depth']))
    
    # Create journal entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "run_type": "RULE30_GEOMETRIC_TEST",
        "n_steps": n_steps,
        "sequence_length": len(diagnostics['divergence']),
        "metrics": {
            "divergence": {
                "mean": divergence_mean,
                "std": divergence_std,
                "min": float(np.min(diagnostics['divergence'])),
                "max": float(np.max(diagnostics['divergence']))
            },
            "tension": {
                "mean": tension_mean,
                "std": tension_std,
                "min": float(np.min(diagnostics['tension'])),
                "max": float(np.max(diagnostics['tension']))
            },
            "basin_depth": {
                "mean": basin_depth_mean,
                "std": basin_depth_std,
                "min": float(np.min(diagnostics['basin_depth'])),
                "max": float(np.max(diagnostics['basin_depth']))
            }
        }
    }
    
    # Append to journal file
    os.makedirs(os.path.dirname(journal_path) if os.path.dirname(journal_path) else '.', exist_ok=True)
    with open(journal_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')
    
    print(f"Logged to journal: {journal_path}")


def log_recursive_to_journal(
    recursive_results: Dict[int, Dict[str, float]],
    n_steps: int,
    max_depth: int,
    journal_path: str = "growth_journal.jsonl"
):
    """
    Append recursive metrics to growth_journal.jsonl.
    
    Args:
        recursive_results: Dictionary mapping level_id to diagnostic summary
        n_steps: Number of Rule 30 steps
        max_depth: Maximum recursion depth
        journal_path: Path to journal file
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "run_type": "RULE30_RECURSIVE_GEOMETRIC_TEST",
        "n_steps": n_steps,
        "max_depth": max_depth,
        "levels": len(recursive_results),
        "metrics_by_level": recursive_results
    }
    
    # Append to journal file
    os.makedirs(os.path.dirname(journal_path) if os.path.dirname(journal_path) else '.', exist_ok=True)
    with open(journal_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')
    
    print(f"Logged recursive analysis to journal: {journal_path}")


def save_recursive_plots(
    recursive_results: Dict[int, Dict[str, float]],
    diagnostics: Dict[str, List[float]],
    output_dir: str,
    prefix: str = "rule30_recursive"
):
    """
    Save recursive multi-scale diagnostic plots.
    
    Args:
        recursive_results: Dictionary mapping level_id to diagnostic summary
        diagnostics: Level 0 diagnostics (for comparison)
        output_dir: Output directory for plots
        prefix: Filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots for each level
    n_levels = len(recursive_results)
    fig, axes = plt.subplots(n_levels, 3, figsize=(15, 5 * n_levels))
    if n_levels == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Rule 30 Recursive Multi-Scale Diagnostics', fontsize=14, fontweight='bold')
    
    levels = sorted(recursive_results.keys())
    for idx, level_id in enumerate(levels):
        result = recursive_results[level_id]
        scale_factor = 2 ** level_id if level_id > 0 else 1
        
        # Divergence
        axes[idx, 0].bar(['Mean'], [result['divergence_mean']], 
                        yerr=[result['divergence_std']], capsize=5)
        axes[idx, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[idx, 0].set_ylabel('Divergence', fontsize=10)
        axes[idx, 0].set_title(f'Level {level_id} (1/{scale_factor}) - Divergence', fontsize=11)
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Tension
        axes[idx, 1].bar(['Mean'], [result['tension_mean']], 
                        yerr=[result['tension_std']], capsize=5)
        axes[idx, 1].set_ylabel('Tension', fontsize=10)
        axes[idx, 1].set_title(f'Level {level_id} (1/{scale_factor}) - Tension', fontsize=11)
        axes[idx, 1].grid(True, alpha=0.3)
        
        # Basin depth
        axes[idx, 2].bar(['Mean'], [result['basin_depth_mean']], 
                         yerr=[result['basin_depth_std']], capsize=5)
        axes[idx, 2].set_ylabel('Basin Depth', fontsize=10)
        axes[idx, 2].set_title(f'Level {level_id} (1/{scale_factor}) - Basin Depth', fontsize=11)
        axes[idx, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{prefix}_multiscale.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save level 0 detailed plot
    save_plots(diagnostics, output_dir, prefix=f"{prefix}_level0")
    
    print(f"Saved recursive plots: {plot_path}")


def print_summary(diagnostics: Dict[str, List[float]], n_steps: int):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("Rule 30 Geometric Analysis Summary")
    print("="*60)
    print(f"Steps: {n_steps}")
    print(f"Sequence length: {len(diagnostics['divergence'])}")
    print("\nDivergence:")
    print(f"  Mean: {np.mean(diagnostics['divergence']):.6f}")
    print(f"  Std:  {np.std(diagnostics['divergence']):.6f}")
    print(f"  Min:   {np.min(diagnostics['divergence']):.6f}")
    print(f"  Max:   {np.max(diagnostics['divergence']):.6f}")
    print("\nTension:")
    print(f"  Mean: {np.mean(diagnostics['tension']):.6f}")
    print(f"  Std:  {np.std(diagnostics['tension']):.6f}")
    print(f"  Min:   {np.min(diagnostics['tension']):.6f}")
    print(f"  Max:   {np.max(diagnostics['tension']):.6f}")
    print("\nBasin Depth:")
    print(f"  Mean: {np.mean(diagnostics['basin_depth']):.6f}")
    print(f"  Std:  {np.std(diagnostics['basin_depth']):.6f}")
    print(f"  Min:   {np.min(diagnostics['basin_depth']):.6f}")
    print(f"  Max:   {np.max(diagnostics['basin_depth']):.6f}")
    print("="*60 + "\n")


def print_recursive_summary(recursive_results: Dict[int, Dict[str, float]], n_steps: int):
    """Print recursive multi-scale summary."""
    print("\n" + "="*60)
    print("Rule 30 Recursive Multi-Scale Analysis")
    print("="*60)
    print(f"Steps: {n_steps}")
    print(f"Levels: {len(recursive_results)}")
    
    for level_id in sorted(recursive_results.keys()):
        result = recursive_results[level_id]
        scale_factor = 2 ** level_id if level_id > 0 else 1
        print(f"\nLevel {level_id} (scale: 1/{scale_factor}):")
        print(f"  Sequence length: {result['sequence_length']}")
        print(f"  Divergence: {result['divergence_mean']:.6f} ± {result['divergence_std']:.6f}")
        print(f"  Tension:    {result['tension_mean']:.6f} ± {result['tension_std']:.6f}")
        print(f"  Basin depth: {result['basin_depth_mean']:.6f} ± {result['basin_depth_std']:.6f}")
    
    # Check for self-similarity (similar patterns across scales)
    if len(recursive_results) > 1:
        print("\nSelf-Similarity Analysis:")
        levels = sorted(recursive_results.keys())
        for i in range(len(levels) - 1):
            level1 = recursive_results[levels[i]]
            level2 = recursive_results[levels[i + 1]]
            
            # Compare divergence patterns
            div_diff = abs(level1['divergence_mean'] - level2['divergence_mean'])
            tension_diff = abs(level1['tension_mean'] - level2['tension_mean'])
            
            similarity_score = 1.0 / (1.0 + div_diff + tension_diff)
            print(f"  Level {levels[i]} ↔ Level {levels[i+1]}: similarity = {similarity_score:.4f}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Rule 30 Geometric Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_rule30_analysis.py --steps 1000
  python3 run_rule30_analysis.py --steps 200000 --cube-size 5 --output-dir results
        """
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=1000,
        help='Number of Rule 30 steps to generate (default: 1000)'
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
        default='experiments/rule30/results',
        help='Output directory for plots and logs (default: experiments/rule30/results)'
    )
    
    parser.add_argument(
        '--journal',
        type=str,
        default='growth_journal.jsonl',
        help='Path to growth journal file (default: growth_journal.jsonl)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    
    parser.add_argument(
        '--no-journal',
        action='store_true',
        help='Skip logging to journal'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Enable recursive multi-scale analysis'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        default=3,
        help='Maximum recursion depth for recursive mode (default: 3)'
    )
    
    parser.add_argument(
        '--stability-test',
        action='store_true',
        help='Run divergence stability test across multiple scales'
    )
    
    parser.add_argument(
        '--stability-steps',
        type=int,
        nargs='+',
        default=[10000, 100000, 1000000],
        help='Step sizes for stability test (default: 10000 100000 1000000)'
    )
    
    args = parser.parse_args()
    
    # Check if running stability test
    if args.stability_test:
        from experiments.rule30.divergence_stability_test import (
            test_divergence_stability,
            plot_divergence_stability,
            save_stability_report
        )
        
        print("Running divergence stability test...")
        results = test_divergence_stability(
            step_sizes=args.stability_steps,
            cube_size=args.cube_size,
            sample_points=100
        )
        
        if not args.no_plots:
            plot_divergence_stability(results, args.output_dir)
        
        save_stability_report(results, args.output_dir)
        return
    
    # Step 1: Generate Rule 30 center column
    # Use optimized generator for large sequences (>= 10000 steps)
    if args.steps >= 10000:
        print(f"Generating Rule 30 center column (optimized for large sequences)...")
        center_column = generate_center_column_direct(args.steps, show_progress=True)
        print(f"Generated {len(center_column):,} bits")
    else:
        print(f"Generating Rule 30 CA with {args.steps} steps...")
        triangle = generate_rule30(args.steps)
        print(f"Generated triangle with {len(triangle)} rows")
        center_column = extract_center_column(triangle)
        print(f"Extracted center column: {len(center_column)} bits")
    
    print(f"  First 20 bits: {center_column[:20]}")
    print(f"  Sum (ones): {sum(center_column)}")
    
    # Step 3: Embed into cube (recursive or single-level)
    if args.recursive:
        print(f"\nEmbedding recursively (max_depth={args.max_depth})...")
        engine, paths_by_level = embed_recursive(
            center_column,
            base_cube_size=args.cube_size,
            max_depth=args.max_depth
        )
        print(f"Created recursive hierarchy with {len(paths_by_level)} levels")
        for level_id, path in paths_by_level.items():
            print(f"  Level {level_id}: {path.get_path_length()} steps")
        
        # Compute recursive diagnostics
        print("\nComputing recursive geometric diagnostics...")
        recursive_results = analyze_recursive_patterns(engine, paths_by_level)
        
        # Use level 0 for main diagnostics (backward compatibility)
        path = paths_by_level[0]
        diagnostics = compute_all_diagnostics(path)
        
        # Print recursive summary
        print_recursive_summary(recursive_results, args.steps)
    else:
        print(f"\nEmbedding into {args.cube_size}x{args.cube_size}x{args.cube_size} cube...")
        system, path = embed_into_cube(center_column, cube_size=args.cube_size)
        print(f"Created path with {path.get_path_length()} steps")
        
        # Step 4: Compute diagnostics
        print("\nComputing geometric diagnostics...")
        diagnostics = compute_all_diagnostics(path)
    
    # Step 5: Print summary
    print_summary(diagnostics, args.steps)
    
    # Step 6: Save plots
    if not args.no_plots:
        print(f"\nSaving plots to {args.output_dir}...")
        if args.recursive:
            save_recursive_plots(recursive_results, diagnostics, args.output_dir, prefix=f"rule30_{args.steps}_recursive")
        else:
            save_plots(diagnostics, args.output_dir, prefix=f"rule30_{args.steps}")
    
    # Step 7: Log to journal
    if not args.no_journal:
        print(f"\nLogging to journal: {args.journal}...")
        if args.recursive:
            log_recursive_to_journal(recursive_results, args.steps, args.max_depth, args.journal)
        else:
            log_to_growth_journal(diagnostics, args.steps, args.journal)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()


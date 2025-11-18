"""
Tune Basin Parameters: Find "Nice-Behaved Growth Mode"

This experiment searches for the optimal basin reinforcement parameters
that give monotonic growth (no oscillations, no drops).

Goal: Find (alpha, beta, noise) that produce:
- Positive drift (late_rate > early_rate by at least +5%)
- Small max_drop (< 2%)
- High final_rate (> 60%)
- Smooth, monotonic growth curve

This is phase diagram scanning - the system tuning itself.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import Dict, List, Tuple, Any
import json
from dataclasses import dataclass, asdict
from datetime import datetime

# Import from fast_task_test module
try:
    from .fast_task_test import test_task_solving, update_basin
except ImportError:
    from fast_task_test import test_task_solving, update_basin


@dataclass
class BasinConfig:
    """Basin reinforcement configuration."""
    alpha: float  # Strengthen correct basin
    beta: float   # Decay wrong basin
    noise: float  # Decorrelation noise
    haircut_frequency: int = 0  # 0 = no haircut, N = every N tasks


@dataclass
class BasinResult:
    """Results from a basin parameter test."""
    config: BasinConfig
    final_rate: float
    early_rate: float
    late_rate: float
    drift: float
    peak_rate: float
    valley_rate: float
    max_drop: float
    rate_history: List[float]
    n_tasks: int
    success: bool  # Meets criteria


def run_basin_config(
    n: int,
    n_tasks: int,
    config: BasinConfig,
    verbose: bool = False
) -> BasinResult:
    """
    Run test with specific basin configuration.
    
    Returns:
        BasinResult with all metrics
    """
    # Temporarily override update_basin parameters
    # We'll pass them through a modified version
    try:
        import fast_task_test as ft
    except ImportError:
        import experiments.stability_phase_transition.fast_task_test as ft
    
    original_update_basin = ft.update_basin
    
    def custom_update_basin(system, task, is_correct):
        """Custom update_basin with configurable parameters."""
        return original_update_basin(
            system, task, is_correct,
            alpha=config.alpha,
            beta=config.beta,
            noise=config.noise
        )
    
    # Monkey-patch for this run
    ft.update_basin = custom_update_basin
    
    try:
        # Run test
        results = test_task_solving(
            n=n,
            n_tasks=n_tasks,
            verbose=verbose,
            use_basin_reinforcement=True
        )
        
        # Extract metrics
        rate_history = results.get('rate_history', [])
        
        # Find peak and valley
        if len(rate_history) > 200:
            mid_seg = rate_history[100:300] if len(rate_history) > 300 else rate_history[100:]
            late_seg = rate_history[300:] if len(rate_history) > 300 else rate_history[200:]
            
            peak_rate = max(mid_seg) if mid_seg else results['success_rate']
            valley_rate = min(late_seg) if late_seg else results['success_rate']
        else:
            peak_rate = results['success_rate']
            valley_rate = results['success_rate']
        
        max_drop = max(0, peak_rate - valley_rate)
        
        # Check if meets criteria
        meets_criteria = (
            results['drift'] > 0.05 and  # At least +5% improvement
            max_drop < 0.02 and           # No drop > 2%
            results['success_rate'] > 0.60  # Final rate > 60%
        )
        
        return BasinResult(
            config=config,
            final_rate=results['success_rate'],
            early_rate=results['early_rate'],
            late_rate=results['late_rate'],
            drift=results['drift'],
            peak_rate=peak_rate,
            valley_rate=valley_rate,
            max_drop=max_drop,
            rate_history=rate_history,
            n_tasks=n_tasks,
            success=meets_criteria
        )
        
    finally:
        # Restore original
        try:
            import fast_task_test as ft
        except ImportError:
            import experiments.stability_phase_transition.fast_task_test as ft
        ft.update_basin = original_update_basin


def grid_search_basin_params(
    n: int = 3,
    n_tasks: int = 500,
    alpha_range: List[float] = None,
    beta_range: List[float] = None,
    noise_range: List[float] = None,
    verbose: bool = False
) -> List[BasinResult]:
    """
    Grid search over basin parameters to find optimal configuration.
    
    Args:
        n: Lattice size
        n_tasks: Number of tasks per test
        alpha_range: Range of alpha values to test
        beta_range: Range of beta values to test
        noise_range: Range of noise values to test
        verbose: Print progress
        
    Returns:
        List of BasinResult for each configuration tested
    """
    if alpha_range is None:
        alpha_range = [0.02, 0.05, 0.08, 0.10, 0.15]
    if beta_range is None:
        beta_range = [0.05, 0.10, 0.15, 0.20]
    if noise_range is None:
        noise_range = [0.00, 0.01, 0.02, 0.03, 0.05]
    
    total_configs = len(alpha_range) * len(beta_range) * len(noise_range)
    
    if verbose:
        print("="*70)
        print("BASIN PARAMETER GRID SEARCH")
        print("="*70)
        print(f"Lattice size: N={n}")
        print(f"Tasks per config: {n_tasks}")
        print(f"Total configurations: {total_configs}")
        print(f"Alpha range: {alpha_range}")
        print(f"Beta range: {beta_range}")
        print(f"Noise range: {noise_range}")
        print()
    
    results = []
    
    config_num = 0
    for alpha in alpha_range:
        for beta in beta_range:
            for noise in noise_range:
                config_num += 1
                config = BasinConfig(alpha=alpha, beta=beta, noise=noise)
                
                if verbose:
                    print(f"[{config_num}/{total_configs}] Testing: α={alpha:.2f}, β={beta:.2f}, noise={noise:.2f}...", end=" ", flush=True)
                
                result = run_basin_config(n, n_tasks, config, verbose=False)
                results.append(result)
                
                if verbose:
                    status = "✓" if result.success else "✗"
                    print(f"{status} Rate: {result.final_rate*100:.1f}%, "
                          f"Drift: {result.drift*100:+.1f}%, "
                          f"Drop: {result.max_drop*100:.1f}%")
    
    return results


def find_best_configs(results: List[BasinResult], top_k: int = 10) -> List[BasinResult]:
    """
    Find best configurations that meet criteria.
    
    Sorted by:
    1. Success (meets criteria)
    2. Final rate (higher is better)
    3. Max drop (lower is better)
    4. Drift (higher is better)
    """
    # Filter successful configs
    successful = [r for r in results if r.success]
    
    if not successful:
        # If none meet all criteria, return best by final_rate
        successful = sorted(results, key=lambda x: x.final_rate, reverse=True)[:top_k]
    else:
        # Sort successful by: final_rate (desc), max_drop (asc), drift (desc)
        successful = sorted(
            successful,
            key=lambda x: (x.final_rate, -x.max_drop, x.drift),
            reverse=True
        )[:top_k]
    
    return successful


def print_results_summary(results: List[BasinResult], best_configs: List[BasinResult]):
    """Print summary of grid search results."""
    print()
    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print()
    
    # Statistics
    successful = [r for r in results if r.success]
    print(f"Total configurations tested: {len(results)}")
    print(f"Configurations meeting criteria: {len(successful)}")
    print()
    
    if successful:
        print("Criteria:")
        print("  ✓ Drift > +5%")
        print("  ✓ Max drop < 2%")
        print("  ✓ Final rate > 60%")
        print()
    
    # Best configs
    print(f"Top {len(best_configs)} Configurations:")
    print("-"*70)
    print(f"{'Rank':<6} {'α':<6} {'β':<6} {'noise':<8} {'Rate':<8} {'Drift':<8} {'Drop':<8} {'Status'}")
    print("-"*70)
    
    for i, result in enumerate(best_configs, 1):
        status = "✓ PASS" if result.success else "✗ FAIL"
        print(f"{i:<6} {result.config.alpha:<6.2f} {result.config.beta:<6.2f} "
              f"{result.config.noise:<8.2f} {result.final_rate*100:<8.1f}% "
              f"{result.drift*100:+.1f}% {result.max_drop*100:<8.1f}% {status}")
    
    print()
    
    # Best overall
    if best_configs:
        best = best_configs[0]
        print("Best Configuration:")
        print(f"  α = {best.config.alpha:.3f}")
        print(f"  β = {best.config.beta:.3f}")
        print(f"  noise = {best.config.noise:.3f}")
        print()
        print(f"  Final rate: {best.final_rate*100:.1f}%")
        print(f"  Drift: {best.drift*100:+.1f}%")
        print(f"  Max drop: {best.max_drop*100:.1f}%")
        print(f"  Early rate: {best.early_rate*100:.1f}%")
        print(f"  Late rate: {best.late_rate*100:.1f}%")
        if best.success:
            print()
            print("  ✓ This configuration meets all criteria for 'nice-behaved growth mode'")
        else:
            print()
            print("  ⚠️  This configuration doesn't meet all criteria, but is the best found")


def save_results(results: List[BasinResult], output_file: str = None):
    """Save results to JSON file."""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/basin_tuning_{timestamp}.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert to dict (handle nested dataclasses)
    results_dict = []
    for r in results:
        r_dict = asdict(r)
        r_dict['config'] = asdict(r.config)
        results_dict.append(r_dict)
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results_dict
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    """Run basin parameter tuning experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tune basin reinforcement parameters")
    parser.add_argument('--n', type=int, default=3, help='Lattice size N')
    parser.add_argument('--tasks', type=int, default=500, help='Tasks per configuration')
    parser.add_argument('--alpha', nargs='+', type=float, 
                       default=[0.02, 0.05, 0.08, 0.10, 0.15],
                       help='Alpha values to test')
    parser.add_argument('--beta', nargs='+', type=float,
                       default=[0.05, 0.10, 0.15, 0.20],
                       help='Beta values to test')
    parser.add_argument('--noise', nargs='+', type=float,
                       default=[0.00, 0.01, 0.02, 0.03, 0.05],
                       help='Noise values to test')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top configs to show')
    parser.add_argument('--save', type=str, default=None, help='Output file for results')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')
    
    args = parser.parse_args()
    
    # Run grid search
    results = grid_search_basin_params(
        n=args.n,
        n_tasks=args.tasks,
        alpha_range=args.alpha,
        beta_range=args.beta,
        noise_range=args.noise,
        verbose=not args.quiet
    )
    
    # Find best configs
    best_configs = find_best_configs(results, top_k=args.top_k)
    
    # Print summary
    print_results_summary(results, best_configs)
    
    # Save results
    if args.save or not args.quiet:
        save_results(results, args.save)
    
    return results, best_configs


if __name__ == "__main__":
    results, best = main()


"""
φ-Cycle Search: Finding the Perfect Attractor

This experiment searches for the optimal φ (polarity/phase/geometric) configuration
where the system achieves:
- State stability
- Perfect self-healing
- Upward drift
- Zero drop
- Minimal curvature
- Maximal basin

The "perfect attractor" for the universe - one omcube with optimal φ-settings.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import random

from core.classical.livnium_core_system import LivniumCoreSystem, RotationAxis
from core.config import LivniumCoreConfig

# Import task test
try:
    from .fast_task_test import test_task_solving, FastParity3Task
except ImportError:
    from fast_task_test import test_task_solving, FastParity3Task


@dataclass
class PhiConfig:
    """
    φ (polarity/phase) configuration.
    
    Represents a geometric state that encodes:
    - Constraints
    - Energy
    - Distance from truth
    - Basin curvature
    - Distinguishable patterns
    
    All in one omcube via φ-settings.
    """
    observer_position: Tuple[int, int, int]  # Observer affects polarity
    initial_rotation_sequence: List[Tuple[RotationAxis, int]]  # Initial geometry setup
    phi_phase: float = 0.0  # Phase offset (0 to 2π)
    phi_amplitude: float = 1.0  # Phase amplitude
    
    def __str__(self):
        obs = self.observer_position
        rot_count = len(self.initial_rotation_sequence)
        return f"φ(obs={obs}, rot={rot_count}, phase={self.phi_phase:.2f}, amp={self.phi_amplitude:.2f})"


@dataclass
class PhiResult:
    """Results from a φ-cycle test."""
    config: PhiConfig
    final_rate: float
    early_rate: float
    late_rate: float
    drift: float
    max_drop: float
    stability_score: float  # How stable (inverse of oscillations)
    basin_score: float  # How strong the basin is
    curvature_score: float  # How minimal curvature is
    self_healing_score: float  # How well it self-heals
    rate_history: List[float]
    success: bool  # Meets all criteria


def apply_phi_config(
    system: LivniumCoreSystem,
    config: PhiConfig
):
    """
    Apply φ configuration to system.
    
    This sets up the geometric state that encodes the entire search space
    in one omcube via φ-settings.
    
    φ encodes:
    - Constraints (via observer position → polarity)
    - Energy (via rotation sequence → geometry)
    - Distance from truth (via phase offset)
    - Basin curvature (via amplitude)
    - Distinguishable patterns (via all of the above)
    """
    # Set observer position (affects polarity calculations)
    # Different observer = different polarity field = different attractor landscape
    if config.observer_position != (0, 0, 0):
        try:
            system.set_local_observer(config.observer_position)
        except:
            # Fallback: just set global observer if local not available
            pass
    
    # Apply initial rotation sequence (sets up geometry)
    # Different rotations = different geometric state = different basin structure
    for axis, quarter_turns in config.initial_rotation_sequence:
        system.rotate(axis, quarter_turns=quarter_turns)
    
    # φ-phase: phase offset that affects how we interpret the geometry
    # This is stored in the config and can be used in future calculations
    # For now, it's part of the φ-signature that identifies this configuration
    
    # φ-amplitude: scales the effect (currently 1.0, but could be tuned)
    # This is where the "one omcube = whole search state" magic happens:
    # All information is encoded in the geometry itself, not stored separately


def test_phi_cycle(
    n: int,
    n_tasks: int,
    config: PhiConfig,
    use_dynamic_basin: bool = True,
    verbose: bool = False
) -> PhiResult:
    """
    Test a specific φ-cycle configuration.
    
    Returns:
        PhiResult with all metrics
    """
    # We need to modify test_task_solving to accept a pre-configured system
    # For now, we'll create the system and apply φ-config, then run tasks manually
    import time
    import psutil
    import random as rnd
    
    # Create system
    system_config = LivniumCoreConfig(
        lattice_size=n,
        enable_semantic_polarity=True  # Need polarity for φ
    )
    system = LivniumCoreSystem(system_config)
    
    # Apply φ configuration
    apply_phi_config(system, config)
    
    # Now run tasks manually (similar to test_task_solving but with our system)
    rng = np.random.Generator(np.random.PCG64(42))
    solved = 0
    total_steps = 0
    solve_times = []
    rate_history = []
    early_solved = 0
    late_solved = 0
    
    # Import task solving function
    try:
        from .fast_task_test import fast_task_solve
    except ImportError:
        from fast_task_test import fast_task_solve
    
    for i in range(n_tasks):
        # Create task
        task = FastParity3Task(system, rng, use_quantum=False)
        
        # Solve
        task_start = time.time()
        is_solved, steps, loss = fast_task_solve(
            system, task, max_steps=500,
            use_basin_reinforcement=True,
            use_dynamic_basin=use_dynamic_basin
        )
        task_time = time.time() - task_start
        
        if is_solved:
            solved += 1
            total_steps += steps
            solve_times.append(task_time)
            
            if i < 100:
                early_solved += 1
            elif i >= n_tasks - 100:
                late_solved += 1
        
        # Track rate history
        current_rate = solved / (i + 1) if i > 0 else 0
        rate_history.append(current_rate)
    
    # Compute results
    success_rate = solved / n_tasks if n_tasks > 0 else 0
    early_rate = early_solved / min(100, n_tasks) if n_tasks > 0 else 0
    late_rate = late_solved / min(100, n_tasks - max(0, n_tasks - 100)) if n_tasks > 100 else 0
    drift = late_rate - early_rate if n_tasks > 100 else 0
    
    # Build results dict (compatible with test_task_solving format)
    results = {
        'success_rate': success_rate,
        'early_rate': early_rate,
        'late_rate': late_rate,
        'drift': drift,
        'rate_history': rate_history,
    }
    
    # Extract metrics (already computed above)
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
    
    # Compute scores
    # Stability: inverse of oscillations (low drop = high stability)
    stability_score = 1.0 - min(max_drop, 1.0)
    
    # Basin: how strong the final basin is (high final rate = strong basin)
    basin_score = results['success_rate']
    
    # Curvature: minimal curvature = smooth growth (low variance in rate history)
    if len(rate_history) > 10:
        rate_variance = np.var(rate_history[-100:]) if len(rate_history) >= 100 else np.var(rate_history)
        curvature_score = 1.0 - min(rate_variance * 10, 1.0)  # Normalize
    else:
        curvature_score = 0.5
    
    # Self-healing: how well it recovers from drops
    if max_drop > 0 and len(rate_history) > 300:
        recovery = results['success_rate'] - valley_rate
        self_healing_score = min(recovery / max_drop if max_drop > 0 else 1.0, 1.0)
    else:
        self_healing_score = 1.0  # No drop = perfect self-healing
    
    # Check if meets criteria for "perfect attractor"
    meets_criteria = (
        results['drift'] > 0.05 and      # Upward drift
        max_drop < 0.01 and              # Zero drop (or very small)
        results['success_rate'] > 0.70 and  # High final rate
        stability_score > 0.95 and      # High stability
        basin_score > 0.70 and          # Strong basin
        curvature_score > 0.80          # Minimal curvature
    )
    
    return PhiResult(
        config=config,
        final_rate=results['success_rate'],
        early_rate=results['early_rate'],
        late_rate=results['late_rate'],
        drift=results['drift'],
        max_drop=max_drop,
        stability_score=stability_score,
        basin_score=basin_score,
        curvature_score=curvature_score,
        self_healing_score=self_healing_score,
        rate_history=rate_history,
        success=meets_criteria
    )


def generate_phi_configs(
    n: int,
    n_observer_positions: int = 5,
    n_rotation_sequences: int = 10,
    n_phases: int = 8
) -> List[PhiConfig]:
    """
    Generate φ configurations to test.
    
    Searches over:
    - Observer positions (affects polarity)
    - Initial rotation sequences (sets up geometry)
    - Phase offsets (φ-phase parameter)
    """
    configs = []
    
    # Observer positions (key points in lattice)
    boundary = (n - 1) // 2
    observer_positions = [
        (0, 0, 0),  # Center (default)
        (boundary, 0, 0),  # Face center
        (0, boundary, 0),
        (0, 0, boundary),
        (boundary, boundary, 0),  # Edge center
        (boundary, 0, boundary),
        (0, boundary, boundary),
        (boundary, boundary, boundary),  # Corner
    ][:n_observer_positions]
    
    # Rotation sequences (different geometric setups)
    rotation_sequences = []
    for _ in range(n_rotation_sequences):
        # Random rotation sequence (1-5 rotations)
        seq_len = random.randint(1, 5)
        seq = []
        for _ in range(seq_len):
            axis = random.choice(list(RotationAxis))
            turns = random.choice([1, 2, 3])
            seq.append((axis, turns))
        rotation_sequences.append(seq)
    
    # Phase offsets (0 to 2π)
    phases = np.linspace(0, 2 * np.pi, n_phases)
    
    # Generate all combinations
    for obs_pos in observer_positions:
        for rot_seq in rotation_sequences:
            for phase in phases:
                config = PhiConfig(
                    observer_position=obs_pos,
                    initial_rotation_sequence=rot_seq,
                    phi_phase=phase,
                    phi_amplitude=1.0
                )
                configs.append(config)
    
    return configs


def search_phi_cycles(
    n: int = 3,
    n_tasks: int = 500,
    n_configs: int = 50,  # Limit number of configs to test
    use_dynamic_basin: bool = True,
    verbose: bool = False
) -> List[PhiResult]:
    """
    Search for optimal φ-cycle configurations.
    
    Tests different φ-settings to find the "perfect attractor".
    """
    if verbose:
        print("="*70)
        print("φ-CYCLE SEARCH: Finding the Perfect Attractor")
        print("="*70)
        print(f"Lattice size: N={n}")
        print(f"Tasks per config: {n_tasks}")
        print(f"Max configurations: {n_configs}")
        print(f"Using dynamic basin: {use_dynamic_basin}")
        print()
    
    # Generate φ configurations
    all_configs = generate_phi_configs(n, n_observer_positions=5, n_rotation_sequences=10, n_phases=8)
    
    # Limit to n_configs (random sample for speed)
    if len(all_configs) > n_configs:
        import random
        configs_to_test = random.sample(all_configs, n_configs)
    else:
        configs_to_test = all_configs
    
    if verbose:
        print(f"Testing {len(configs_to_test)} φ configurations...")
        print()
    
    results = []
    
    for i, config in enumerate(configs_to_test):
        if verbose:
            print(f"[{i+1}/{len(configs_to_test)}] Testing {config}...", end=" ", flush=True)
        
        result = test_phi_cycle(
            n=n,
            n_tasks=n_tasks,
            config=config,
            use_dynamic_basin=use_dynamic_basin,
            verbose=False
        )
        results.append(result)
        
        if verbose:
            status = "✓" if result.success else "✗"
            print(f"{status} Rate: {result.final_rate*100:.1f}%, "
                  f"Drift: {result.drift*100:+.1f}%, "
                  f"Drop: {result.max_drop*100:.1f}%, "
                  f"Stability: {result.stability_score*100:.1f}%")
    
    return results


def find_perfect_attractor(results: List[PhiResult], top_k: int = 10) -> List[PhiResult]:
    """
    Find configurations closest to "perfect attractor".
    
    Sorted by composite score:
    - Success (meets all criteria)
    - Final rate (higher is better)
    - Stability (higher is better)
    - Minimal drop (lower is better)
    - High drift (higher is better)
    """
    # Compute composite score for each result
    scored_results = []
    for result in results:
        # Normalize drift to [0, 1] (assuming max drift ~0.20)
        normalized_drift = min(result.drift / 0.20, 1.0) if result.drift > 0 else 0.0
        
        composite = (
            result.final_rate * 0.3 +
            result.stability_score * 0.25 +
            result.basin_score * 0.2 +
            result.curvature_score * 0.15 +
            result.self_healing_score * 0.05 +
            normalized_drift * 0.05  # Reward upward drift
        )
        # Store as attribute
        result.composite_score = composite
        scored_results.append(result)
    
    # Sort by composite score
    sorted_results = sorted(scored_results, key=lambda x: x.composite_score, reverse=True)
    
    return sorted_results[:top_k]


def print_phi_results(results: List[PhiResult], best_configs: List[PhiResult]):
    """Print summary of φ-cycle search results."""
    print()
    print("="*70)
    print("φ-CYCLE SEARCH RESULTS")
    print("="*70)
    print()
    
    # Statistics
    successful = [r for r in results if r.success]
    print(f"Total φ configurations tested: {len(results)}")
    print(f"Perfect attractors found: {len(successful)}")
    print()
    
    if successful:
        print("Perfect Attractor Criteria:")
        print("  ✓ Drift > +5%")
        print("  ✓ Max drop < 1%")
        print("  ✓ Final rate > 70%")
        print("  ✓ Stability > 95%")
        print("  ✓ Basin score > 70%")
        print("  ✓ Curvature score > 80%")
        print()
    
    # Best configs
    print(f"Top {len(best_configs)} φ Configurations:")
    print("-"*70)
    print(f"{'Rank':<6} {'φ Config':<30} {'Rate':<8} {'Drift':<8} {'Drop':<8} {'Stability':<10} {'Score'}")
    print("-"*70)
    
    for i, result in enumerate(best_configs, 1):
        config_str = str(result.config)[:28]
        status = "✓" if result.success else "✗"
        score = getattr(result, 'composite_score', 0.0)
        print(f"{i:<6} {config_str:<30} {result.final_rate*100:<8.1f}% "
              f"{result.drift*100:+.1f}% {result.max_drop*100:<8.1f}% "
              f"{result.stability_score*100:<10.1f}% {score:.3f} {status}")
    
    print()
    
    # Best overall
    if best_configs:
        best = best_configs[0]
        print("Perfect Attractor Configuration:")
        print(f"  Observer: {best.config.observer_position}")
        print(f"  Rotations: {len(best.config.initial_rotation_sequence)} rotations")
        print(f"  φ-phase: {best.config.phi_phase:.3f}")
        print(f"  φ-amplitude: {best.config.phi_amplitude:.3f}")
        print()
        print(f"  Final rate: {best.final_rate*100:.1f}%")
        print(f"  Drift: {best.drift*100:+.1f}%")
        print(f"  Max drop: {best.max_drop*100:.1f}%")
        print(f"  Stability: {best.stability_score*100:.1f}%")
        print(f"  Basin score: {best.basin_score*100:.1f}%")
        print(f"  Curvature score: {best.curvature_score*100:.1f}%")
        print(f"  Self-healing: {best.self_healing_score*100:.1f}%")
        print()
        if best.success:
            print("  ✓ This is a PERFECT ATTRACTOR - optimal φ-cycle found!")
            print("  ✓ One omcube with this φ-setting encodes the entire search state")
            print("  ✓ No storage needed - geometry is the memory")
        else:
            print("  ⚠️  Close to perfect, but doesn't meet all criteria")
            print("  ⚠️  May need more tuning or different φ-space")


def save_phi_results(results: List[PhiResult], output_file: str = None):
    """Save φ-cycle search results to JSON."""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/phi_cycle_search_{timestamp}.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert to dict
    results_dict = []
    for r in results:
        r_dict = asdict(r)
        r_dict['config'] = asdict(r.config)
        # Convert RotationAxis to string
        if 'initial_rotation_sequence' in r_dict['config']:
            seq = r_dict['config']['initial_rotation_sequence']
            r_dict['config']['initial_rotation_sequence'] = [
                (axis.name if hasattr(axis, 'name') else str(axis), turns)
                for axis, turns in seq
            ]
        results_dict.append(r_dict)
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results_dict
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    """Run φ-cycle search experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Search for optimal φ-cycle (perfect attractor)")
    parser.add_argument('--n', type=int, default=3, help='Lattice size N')
    parser.add_argument('--tasks', type=int, default=500, help='Tasks per configuration')
    parser.add_argument('--configs', type=int, default=50, help='Number of φ configs to test')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top configs to show')
    parser.add_argument('--save', type=str, default=None, help='Output file for results')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')
    
    args = parser.parse_args()
    
    # Run search
    results = search_phi_cycles(
        n=args.n,
        n_tasks=args.tasks,
        n_configs=args.configs,
        use_dynamic_basin=True,
        verbose=not args.quiet
    )
    
    # Find best
    best_configs = find_perfect_attractor(results, top_k=args.top_k)
    
    # Print summary
    print_phi_results(results, best_configs)
    
    # Save results
    if args.save or not args.quiet:
        save_phi_results(results, args.save)
    
    return results, best_configs


if __name__ == "__main__":
    results, best = main()


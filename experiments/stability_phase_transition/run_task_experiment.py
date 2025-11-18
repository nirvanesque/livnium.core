#!/usr/bin/env python3
"""
Run Task-Driven Stability Phase Transition Experiment

The physics only emerges when there's a task to solve.
This measures stability relative to task correctness, not pattern stability in vacuum.
"""

# Handle imports (works as script or module)
try:
    from .config import StabilityConfig
    from .task_experiment import run_task_experiment
except ImportError:
    from config import StabilityConfig
    from task_experiment import run_task_experiment


def main():
    """Run the task-driven experiment."""
    
    # Configure experiment
    config = StabilityConfig(
        # Task configuration
        task_type="parity_3bit",  # Start with simple 3-bit parity
        task_params={},  # Task-specific parameters
        
        # Lattice sizes to test
        lattice_sizes=[3, 5, 7, 9],
        
        # Number of runs per size
        runs_per_size=100,  # Increase for more statistical power
        
        # Dynamics
        t_max=2000,          # Maximum timesteps per run
        t_perturb=500,       # Maximum timesteps for perturbation test
        
        # Stability thresholds
        epsilon_E=1e-3,      # Energy (loss) convergence threshold
        window_E=20,         # Energy settling window
        window_H=10,         # Fixed point confirmation window
        
        # Perturbation
        perturb_fraction=0.02,  # 2% of cells
        
        # Update rule
        # For small N (3, 5): use hybrid_reasoning
        # For large N (7, 9): use hybrid_recursive (Layer 0 recursive problem solving)
        update_rule="hybrid_reasoning",  # Use Reasoning Layer (Layer 4) + simple updates
        # Options: 
        #   - "loss_minimization": Fast manual testing
        #   - "hybrid_reasoning": Reasoning Layer (good for all N)
        #   - "hybrid_recursive": Layer 0 recursive (best for N>=5)
        
        # Random seed
        random_seed=42,
        
        # Output
        results_dir="results/stability_phase_transition",
        run_tag="v1_task_driven"
    )
    
    # Run experiment
    results = run_task_experiment(config)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TASK-DRIVEN STABILITY EXPERIMENT SUMMARY")
    print("=" * 80)
    
    critical_sizes = [N for N, r in results.items() if r.critical_size]
    
    if critical_sizes:
        n_crit = min(critical_sizes)
        print(f"\n🎯 CRITICAL SIZE: N* = {n_crit}")
        print(f"   This is the smallest N×N×N lattice where:")
        print(f"   - System can solve the task")
        print(f"   - Solution is stable")
        print(f"   - Solution survives internal perturbation")
        print(f"\n   Phase transition:")
        for N in sorted(results.keys()):
            r = results[N]
            marker = "✓" if N >= n_crit else "✗"
            print(f"   {marker} N={N}: p_correct={r.p_correct:.2%}, "
                  f"p_stable={r.p_stable:.2%}, p_self_healing={r.p_self_healing:.2%}")
    else:
        print("\n⚠️  No critical size found in tested range.")
        print("   Try:")
        print("   - Increasing t_max")
        print("   - Testing larger sizes")
        print("   - Adjusting epsilon_E or window_E/window_H")
        print("   - Trying different update rules")
        print("   - Using a simpler task")
    
    print(f"\n📊 Results saved to: {config.results_dir}/")
    print(f"   - task_stability_{config.task_type}_{config.run_tag}.json")


if __name__ == "__main__":
    main()


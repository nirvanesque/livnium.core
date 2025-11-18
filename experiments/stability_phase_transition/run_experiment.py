#!/usr/bin/env python3
"""
Quick-start script to run the stability phase transition experiment.

Usage:
    python run_experiment.py
"""

from config import StabilityConfig
from experiment import run_experiment


def main():
    """Run the experiment with customizable parameters."""
    
    # Customize these parameters as needed
    config = StabilityConfig(
        # Test these lattice sizes (must be odd, >= 3)
        lattice_sizes=[3, 5, 7, 9],
        
        # Number of random initial configurations per size
        runs_per_size=100,  # Increase for more statistical power
        
        # Dynamics
        t_max=1000,          # Maximum timesteps per run
        t_perturb=500,       # Maximum timesteps for perturbation test
        
        # Stability thresholds
        epsilon_E=1e-3,      # Energy convergence threshold
        window_E=20,         # Energy settling window
        window_H=10,         # Fixed point confirmation window
        
        # Perturbation
        perturb_fraction=0.02,  # 2% of cells
        
        # Random seed
        random_seed=42,
        
        # Output
        results_dir="results/stability_phase_transition",
        run_tag="v1_stability_scan",
        
        # Optional: energy method and update rule
        energy_method="symbolic_weight_variance",
        update_rule="rotation_based"
    )
    
    # Run experiment
    results = run_experiment(config)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    critical_sizes = [N for N, r in results.items() if r.get("critical_size", False)]
    
    if critical_sizes:
        n_crit = min(critical_sizes)
        print(f"\n🎯 CRITICAL SIZE: N* = {n_crit}")
        print(f"   This is the smallest N×N×N lattice where self-healing stability appears.")
        print(f"\n   Phase transition:")
        for N in sorted(results.keys()):
            r = results[N]
            marker = "✓" if N >= n_crit else "✗"
            print(f"   {marker} N={N}: p_stable={r['p_stable']:.2%}, p_fixed={r['p_fixed']:.2%}")
    else:
        print("\n⚠️  No critical size found in tested range.")
        print("   Try:")
        print("   - Increasing t_max")
        print("   - Testing larger sizes")
        print("   - Adjusting epsilon_E or window_E/window_H")
        print("   - Trying different update rules")
    
    print(f"\n📊 Results saved to: {config.results_dir}/")
    print(f"   - stability_scan_{config.run_tag}.json: Summary statistics")


if __name__ == "__main__":
    main()


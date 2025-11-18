"""
Main Stability Phase Transition Experiment Runner

Ties it all together, loops over N, and spits JSON/CSV into results/.
"""

import os
import json
from typing import Dict, Any, Optional

import numpy as np

# Handle both relative and absolute imports
try:
    from .config import StabilityConfig
    from .local_dynamics import random_init_lattice
    from .stability_detector import (
        run_until_candidate_stable,
        passes_self_healing_test,
    )
except ImportError:
    from config import StabilityConfig
    from local_dynamics import random_init_lattice
    from stability_detector import (
        run_until_candidate_stable,
        passes_self_healing_test,
    )


def run_experiment(cfg: StabilityConfig) -> Dict[int, Dict[str, Any]]:
    """
    Run the full stability phase transition experiment.
    
    Args:
        cfg: Experiment configuration
        
    Returns:
        Dictionary mapping N -> results dict
    """
    os.makedirs(cfg.results_dir, exist_ok=True)
    
    rng = np.random.default_rng(cfg.random_seed)
    results: Dict[int, Dict[str, Any]] = {}
    
    # You can parameterize num_symbols based on N / Livnium core
    # For Livnium, num_symbols = N³ (alphabet size)
    # For now, we'll use None and let the functions handle it
    
    print("=" * 80)
    print("STABILITY PHASE TRANSITION EXPERIMENT")
    print("=" * 80)
    print(f"Testing sizes: {cfg.lattice_sizes}")
    print(f"Runs per size: {cfg.runs_per_size}")
    print(f"Max timesteps: {cfg.t_max}")
    print()
    
    for N in cfg.lattice_sizes:
        fixed_count = 0
        stable_count = 0
        steps_to_stable = []
        
        print(f"\n=== Scanning N={N} ===")
        
        for run_id in range(cfg.runs_per_size):
            if (run_id + 1) % 10 == 0:
                print(f"  Run {run_id + 1}/{cfg.runs_per_size}...")
            
            state = random_init_lattice(N)
            
            candidate, stable_state, energies = run_until_candidate_stable(
                state=state,
                cfg=cfg,
            )
            
            if not candidate:
                continue  # never reached candidate stability
            
            fixed_count += 1
            steps_to_stable.append(len(energies))
            
            # Determine num_symbols for perturbation
            num_symbols = None
            if hasattr(stable_state, 'lattice_size'):
                # LivniumCoreSystem: alphabet size is N³
                num_symbols = stable_state.lattice_size ** 3
            
            if passes_self_healing_test(
                stable_state=stable_state,
                cfg=cfg,
                num_symbols=num_symbols,
                rng=rng,
            ):
                stable_count += 1
        
        p_fixed = fixed_count / cfg.runs_per_size
        p_stable = stable_count / cfg.runs_per_size
        t_avg = float(np.mean(steps_to_stable)) if steps_to_stable else None
        
        print(f"N={N}: p_fixed={p_fixed:.3f}, p_stable={p_stable:.3f}, t_avg={t_avg}")
        
        results[N] = {
            "N": N,
            "runs": cfg.runs_per_size,
            "p_fixed": p_fixed,
            "p_stable": p_stable,
            "t_avg": t_avg,
            "fixed_count": fixed_count,
            "stable_count": stable_count,
        }
    
    # Find critical size
    critical_sizes = [N for N, r in results.items() if r["p_stable"] > 0]
    if critical_sizes:
        n_crit = min(critical_sizes)
        print(f"\n{'='*80}")
        print(f"🎯 CRITICAL SIZE: N* = {n_crit}")
        print(f"{'='*80}")
        results[n_crit]["critical_size"] = True
    
    # Save results
    out_path = os.path.join(cfg.results_dir, f"stability_scan_{cfg.run_tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {out_path}")
    return results


if __name__ == "__main__":
    cfg = StabilityConfig()
    run_experiment(cfg)

"""
Task-Driven Stability Phase Transition Experiment

Finds the smallest N×N×N lattice where the system can:
1. Solve a task (produce correct answer)
2. Maintain the solution stably
3. Self-heal after internal perturbation

The physics only emerges when there's a task to solve.
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

# Handle imports (works as script or module)
try:
    from .config import StabilityConfig
    from .tasks import Task, create_task
    from .task_dynamics import initialize_lattice_with_task
    from .task_stability_detector import (
        run_until_task_stable,
        passes_task_self_healing_test,
    )
except ImportError:
    from config import StabilityConfig
    from tasks import Task, create_task
    from task_dynamics import initialize_lattice_with_task
    from task_stability_detector import (
        run_until_task_stable,
        passes_task_self_healing_test,
    )


@dataclass
class TaskRunResult:
    """Result of a single task-driven run."""
    run_id: int
    task_input: Any
    correct_answer: Any
    reached_correct: bool
    reached_stable: bool
    self_healing: bool
    convergence_step: int
    final_loss: float
    loss_curve: List[float]
    correctness_curve: List[bool]


@dataclass
class TaskSizeResult:
    """Results for a specific lattice size on a task."""
    size: int
    task_type: str
    total_runs: int
    correct_count: int  # Solved the task
    stable_count: int  # Solved + stable
    self_healing_count: int  # Solved + stable + self-healing
    avg_convergence_steps: float
    p_correct: float
    p_stable: float
    p_self_healing: float
    critical_size: bool = False


def run_task_experiment(cfg: StabilityConfig) -> Dict[int, TaskSizeResult]:
    """
    Run task-driven stability phase transition experiment.
    
    Args:
        cfg: Experiment configuration
        
    Returns:
        Dictionary mapping N -> TaskSizeResult
    """
    os.makedirs(cfg.results_dir, exist_ok=True)
    
    rng = np.random.default_rng(cfg.random_seed)
    results: Dict[int, TaskSizeResult] = {}
    
    print("=" * 80)
    print("TASK-DRIVEN STABILITY PHASE TRANSITION EXPERIMENT")
    print("=" * 80)
    print(f"Task: {cfg.task_type}")
    print(f"Testing sizes: {cfg.lattice_sizes}")
    print(f"Runs per size: {cfg.runs_per_size}")
    print(f"Max timesteps: {cfg.t_max}")
    print()
    
    for N in cfg.lattice_sizes:
        correct_count = 0
        stable_count = 0
        self_healing_count = 0
        convergence_steps = []
        
        print(f"\n=== Testing N={N} on {cfg.task_type} ===")
        
        for run_id in range(cfg.runs_per_size):
            if (run_id + 1) % 10 == 0:
                print(f"  Run {run_id + 1}/{cfg.runs_per_size}...")
            
            # Create task instance
            task = _create_task_instance(cfg, run_id, rng)
            
            # Initialize lattice with task
            system = initialize_lattice_with_task(N, task)
            
            # Run until stable
            # OPTIMIZATION: Uses MokshaEngine for fast fixed-point detection
            is_stable, final_state, losses, correctness, hashes = run_until_task_stable(
                system, task, cfg, use_moksha=cfg.use_moksha
            )
            
            # Note: system and final_state are the same object (in-place updates)
            
            # Check if reached correct answer
            final_answer = task.decode_answer(final_state)
            is_correct = task.is_correct(final_answer)
            
            if is_correct:
                correct_count += 1
                
                if is_stable:
                    stable_count += 1
                    convergence_steps.append(len(losses))
                    
                    # Test self-healing
                    if passes_task_self_healing_test(
                        final_state, task, cfg, rng
                    ):
                        self_healing_count += 1
        
        p_correct = correct_count / cfg.runs_per_size
        p_stable = stable_count / cfg.runs_per_size
        p_self_healing = self_healing_count / cfg.runs_per_size
        t_avg = float(np.mean(convergence_steps)) if convergence_steps else None
        
        print(f"N={N}: p_correct={p_correct:.3f}, p_stable={p_stable:.3f}, "
              f"p_self_healing={p_self_healing:.3f}, t_avg={t_avg}")
        
        results[N] = TaskSizeResult(
            size=N,
            task_type=cfg.task_type,
            total_runs=cfg.runs_per_size,
            correct_count=correct_count,
            stable_count=stable_count,
            self_healing_count=self_healing_count,
            avg_convergence_steps=t_avg or 0.0,
            p_correct=p_correct,
            p_stable=p_stable,
            p_self_healing=p_self_healing
        )
    
    # Find critical size
    critical_sizes = [N for N, r in results.items() if r.p_self_healing > 0]
    if critical_sizes:
        n_crit = min(critical_sizes)
        print(f"\n{'='*80}")
        print(f"🎯 CRITICAL SIZE: N* = {n_crit}")
        print(f"   Smallest N where task-stable self-healing appears")
        print(f"{'='*80}")
        results[n_crit].critical_size = True
    
    # Save results
    _save_task_results(results, cfg)
    
    return results


def _create_task_instance(
    cfg: StabilityConfig,
    run_id: int,
    rng: np.random.Generator
) -> Task:
    """Create a task instance for this run."""
    if cfg.task_type == "parity_3bit":
        # Generate random 3-bit input
        bits = tuple(rng.integers(0, 2, size=3))
        return create_task("parity_3bit", bits=bits)
    
    elif cfg.task_type == "classification":
        # Generate random 2D point
        point = tuple(rng.uniform(-1, 1, size=2))
        threshold = cfg.task_params.get("threshold", 0.0)
        return create_task("classification", point=point, threshold=threshold)
    
    elif cfg.task_type == "constraint":
        # Use constraints from params
        constraints = cfg.task_params.get("constraints", [])
        return create_task("constraint", constraints=constraints)
    
    else:
        # Default: parity
        bits = tuple(rng.integers(0, 2, size=3))
        return create_task("parity_3bit", bits=bits)


def _save_task_results(results: Dict[int, TaskSizeResult], cfg: StabilityConfig):
    """Save task experiment results."""
    results_dict = {}
    for N, result in results.items():
        results_dict[N] = {
            "size": result.size,
            "task_type": result.task_type,
            "total_runs": result.total_runs,
            "correct_count": result.correct_count,
            "stable_count": result.stable_count,
            "self_healing_count": result.self_healing_count,
            "avg_convergence_steps": result.avg_convergence_steps,
            "p_correct": result.p_correct,
            "p_stable": result.p_stable,
            "p_self_healing": result.p_self_healing,
            "critical_size": result.critical_size,
        }
    
    out_path = os.path.join(
        cfg.results_dir,
        f"task_stability_{cfg.task_type}_{cfg.run_tag}.json"
    )
    
    with open(out_path, "w") as f:
        json.dump({
            "experiment_config": {
                "task_type": cfg.task_type,
                "task_params": cfg.task_params,
                "lattice_sizes": cfg.lattice_sizes,
                "runs_per_size": cfg.runs_per_size,
                "t_max": cfg.t_max,
                "update_rule": cfg.update_rule,
            },
            "results": results_dict
        }, f, indent=2)
    
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    cfg = StabilityConfig(
        task_type="parity_3bit",
        lattice_sizes=[3, 5, 7],
        runs_per_size=50,
        t_max=1000
    )
    run_task_experiment(cfg)


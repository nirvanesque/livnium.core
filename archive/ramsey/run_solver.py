#!/usr/bin/env python3
"""
Ramsey Number Solver - Main Entry Point

(Parallel Classical Search with Geometric and Quantum Guidance)

This script initializes and runs the RamseySolver.
"""

import sys
import argparse
import json
import time
from pathlib import Path

# Add project root to path to find 'core' and 'experiments'
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now, we can import the solver components
try:
    from ramsey_solver.solver import RamseySolver
    from ramsey_solver.config import (
        DEFAULT_NUM_OMCUBES,
        DEFAULT_N,
        DEFAULT_K,
    )
except ImportError as e:
    print(
        f"Error: Could not import solver components. Make sure 'ramsey_solver' directory is present."
    )
    print(f"Details: {e}")
    sys.exit(1)


def solve_ramsey_problem(
    n: int,
    k: int,
    num_omcubes: int = DEFAULT_NUM_OMCUBES,
    max_iterations: int | None = None,
    enable_dual_monitor: bool = False,
):
    """
    Solve Ramsey number problem using omcubes.

    Args:
        n: Number of vertices
        k: Clique size to avoid
        num_omcubes: Number of omcubes to use
        max_iterations: Maximum iterations (None = unlimited)
        enable_dual_monitor: If True, enable dual cube monitoring
    Returns:
        Valid coloring if found, None otherwise
    """
    print("=" * 70)
    print("RAMSEY NUMBER SOLVER - Break the 61% Plateau")
    print("=" * 70)
    print(f"\nProblem: Find 2-coloring of K_{n} avoiding monochromatic K_{k}")
    print(f"If found, this proves: R({k},{k}) > {n}")
    print(f"\nUsing {num_omcubes:,} omcubes for parallel search...")
    print(
        f"Features: Hybrid Quantum Layer | Pattern Library | Geometric Guidance | Symmetry Breaking"
    )

    solver = RamseySolver(
        n, k, num_omcubes, enable_dual_monitor=enable_dual_monitor
    )

    # Initialize omcubes
    solver.initialize_omcubes()

    # Search for valid coloring
    start_time = time.time()
    valid_coloring = solver.search_for_valid_coloring(
        max_iterations=max_iterations
    )
    elapsed_time = time.time() - start_time

    # Write metrics log to JSONL file if monitor was enabled
    if solver.dual_monitor is not None and solver.metrics_log:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        log_file = logs_dir / f"ramsey_dual_cube_metrics_R{k}_{k}_n{n}.jsonl"

        with open(log_file, "w") as f:
            for entry in solver.metrics_log:
                f.write(json.dumps(entry) + "\n")
        print(f"\n  💾 Saved metrics log to {log_file}")
        print(f"     ({len(solver.metrics_log)} entries)")

    print(f"\nTotal search time: {elapsed_time:.2f} seconds")

    if valid_coloring:
        # Verify
        if solver.verify_coloring(valid_coloring):
            print(f"\n" + "=" * 70)
            print(f"✅ SUCCESS: R({k},{k}) > {n}")
            print("=" * 70)
            # Save result
            filename = f"ramsey_R{k}_{k}_n{n}.txt"
            solver.save_coloring(valid_coloring, filename)

            return valid_coloring
    else:
        print(f"\n" + "=" * 70)
        print(f"⚠️  No complete valid coloring found")
        print(
            f"This does NOT prove R({k},{k}) <= {n} (may need more search)"
        )
        print("=" * 70)

    return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ramsey Number Solver - Break the 61% Plateau"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help=f"Number of vertices (default: {DEFAULT_N} for R(4,4))",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Clique size to avoid (default: {DEFAULT_K} for R(4,4))",
    )
    parser.add_argument(
        "--omcubes",
        type=int,
        default=DEFAULT_NUM_OMCUBES,
        help=f"Number of omcubes to use (default: {DEFAULT_NUM_OMCUBES})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Max iterations (default: None = unlimited)",
    )
    parser.add_argument(
        "--dual-monitor",
        action="store_true",
        help="Enable dual cube monitoring for semantic diagnostics",
    )
    args = parser.parse_args()

    solve_ramsey_problem(
        args.n,
        args.k,
        args.omcubes,
        max_iterations=args.iterations,
        enable_dual_monitor=args.dual_monitor,
    )


if __name__ == "__main__":
    main()


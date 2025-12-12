"""
Test Basin Independence: Does the 99.12% basin exist without checkpoint memory?

This test:
1. Deletes any existing checkpoint
2. Starts fresh with random colors
3. Runs without checkpoint saving
4. Checks if system still converges to ~99.0-99.2%

If YES → Basin is real geometric property (independent of SW memory)
If NO → Basin was checkpoint artifact
"""

import os
import sys
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.classical.livnium_core_system import LivniumCoreSystem, LivniumCoreConfig
from experiments.ramsey.ramsey_encoder import RamseyEncoder
from experiments.ramsey.ramsey_dynamic_search import solve_ramsey_dynamic
from experiments.ramsey.ramsey_tension import get_all_k4_subsets


def test_basin_independence(n_vertices: int = 17, max_steps: int = 5000, verbose: bool = True):
    """
    Test if 99.12% basin exists independently of checkpoint memory.
    
    Args:
        n_vertices: Number of vertices (default 17 for K₁₇)
        max_steps: Maximum steps to run
        verbose: Print progress
    """
    print("="*70)
    print("BASIN INDEPENDENCE TEST")
    print("="*70)
    print()
    print("Hypothesis: The 99.12% basin is a REAL geometric property,")
    print("            not just checkpoint memory.")
    print()
    print("Test: Delete checkpoint, start fresh, see if we converge to ~99%")
    print()
    
    # Determine constraint type
    constraint_type = "k3" if n_vertices <= 6 else "k4"
    
    # Delete checkpoint if it exists
    checkpoint_dir = Path("experiments/ramsey/checkpoints")
    checkpoint_file = checkpoint_dir / f"ramsey_k{n_vertices}_{constraint_type}.ckpt"
    checkpoint_json = checkpoint_dir / f"ramsey_k{n_vertices}_{constraint_type}.json"
    
    deleted_checkpoint = False
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        deleted_checkpoint = True
        if verbose:
            print(f"  🗑️  Deleted checkpoint: {checkpoint_file}")
    
    if checkpoint_json.exists():
        checkpoint_json.unlink()
        if verbose:
            print(f"  🗑️  Deleted checkpoint summary: {checkpoint_json}")
    
    if deleted_checkpoint:
        print("  ✅ Checkpoint deleted - starting fresh")
    else:
        print("  ✅ No checkpoint found - starting fresh")
    
    print()
    
    # Create fresh system
    n_edges = n_vertices * (n_vertices - 1) // 2
    # Need at least n_edges cells, so calculate lattice size properly
    n_lattice = max(3, int((n_edges) ** (1/3)) + 1)  # +1 to ensure enough cells
    if n_lattice % 2 == 0:
        n_lattice += 1  # Must be odd
    
    # Verify we have enough cells
    if n_lattice ** 3 < n_edges:
        # Increase lattice size until we have enough
        while n_lattice ** 3 < n_edges:
            n_lattice += 2  # Keep odd
    
    config = LivniumCoreConfig(lattice_size=n_lattice, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    # Create encoder
    encoder = RamseyEncoder(system, n_vertices)
    
    # Encode constraints
    if constraint_type == "k3":
        encoder.encode_k3_constraints()
    else:
        encoder.encode_k4_constraints()
    
    # Calculate total constraints for percentage
    if constraint_type == "k3":
        from experiments.ramsey.ramsey_tension import get_all_k3_subsets
        total_constraints = len(get_all_k3_subsets(encoder.vertices))
    else:
        total_constraints = len(get_all_k4_subsets(encoder.vertices))
    
    print(f"  Starting fresh run:")
    print(f"    Vertices: {n_vertices}")
    print(f"    Edges: {n_edges}")
    print(f"    Total constraints: {total_constraints}")
    print(f"    Max steps: {max_steps}")
    print(f"    Checkpoint: DISABLED (no memory)")
    print()
    
    # Run WITHOUT checkpoint (no memory, pure geometry)
    result = solve_ramsey_dynamic(
        system=system,
        encoder=encoder,
        max_steps=max_steps,
        verbose=verbose,
        constraint_type=constraint_type,
        initialize_random=True,  # Start with random colors
        checkpoint_path=None,  # NO CHECKPOINT - no SW memory
        save_checkpoint_interval=0,  # Don't save
        visualize=False
    )
    
    # Analyze results
    final_violations = result['violations']
    final_steps = result['steps']
    
    if total_constraints > 0:
        final_satisfied = total_constraints - final_violations
        final_percent = (final_satisfied / total_constraints) * 100.0
    else:
        final_percent = 0.0
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"  Final violations: {final_violations}")
    print(f"  Final satisfaction: {final_percent:.2f}%")
    print(f"  Steps: {final_steps}")
    print()
    
    # Test conclusion
    if final_percent >= 99.0:
        print("  ✅ BASIN IS REAL!")
        print()
        print("  The system converged to ~99% WITHOUT checkpoint memory.")
        print("  This proves the 99.12% basin is a genuine geometric property")
        print("  of the K₁₇ constraint landscape, not just SW memory.")
        print()
        print("  The basin exists independently of resume/checkpoint.")
        return True
    elif final_percent >= 98.0:
        print("  ⚠️  PARTIAL CONVERGENCE")
        print()
        print(f"  System reached {final_percent:.2f}% without checkpoint.")
        print("  This suggests the basin exists but may require more steps")
        print("  or the basin is weaker without SW memory.")
        return None
    else:
        print("  ❌ BASIN NOT FOUND")
        print()
        print(f"  System only reached {final_percent:.2f}% without checkpoint.")
        print("  This suggests the 99.12% basin may have been checkpoint-dependent,")
        print("  OR the system needs more steps to discover it.")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test if 99.12% basin exists without checkpoint")
    parser.add_argument('--n', type=int, default=17, help='Number of vertices (default: 17 for K₁₇)')
    parser.add_argument('--steps', type=int, default=5000, help='Maximum steps to run')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')
    
    args = parser.parse_args()
    
    test_basin_independence(
        n_vertices=args.n,
        max_steps=args.steps,
        verbose=not args.quiet
    )


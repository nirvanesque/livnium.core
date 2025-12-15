#!/usr/bin/env python3
"""
Analyze Invariant Geometry

Studies how Rule 30 dynamics live within the 4D invariant subspace.
Projects pattern-frequency vectors into invariant subspace + orthogonal complement.
"""

import argparse
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple
from fractions import Fraction
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, skipping plots")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.divergence_v3 import (
    enumerate_patterns, 
    pattern_frequencies_3_rational,
    divergence_v3_rational
)
from experiments.rule30.invariant_solver_v3 import (
    build_invariance_system_rational,
    find_nullspace_exact
)
from experiments.rule30.rule30_algebra import rule30_step, rule30_evolve

Pattern = Tuple[int, int, int]


def get_invariant_basis():
    """
    Get the 4 invariant basis vectors as numpy arrays.
    
    Returns:
        Matrix of shape (8, 4) where columns are invariant basis vectors
    """
    patterns = enumerate_patterns()
    pattern_str = {p: ''.join(str(b) for b in p) for p in patterns}
    
    # The 4 invariants (normalized)
    invariants = [
        # I1: freq(100) - freq(001)
        {p: 1.0 if pattern_str[p]=='100' else -1.0 if pattern_str[p]=='001' else 0.0 for p in patterns},
        # I2: freq(001) - freq(010) - freq(011) + freq(101)
        {p: 1.0 if pattern_str[p]=='001' else -1.0 if pattern_str[p] in ['010','011'] else 1.0 if pattern_str[p]=='101' else 0.0 for p in patterns},
        # I3: freq(110) - freq(011)
        {p: 1.0 if pattern_str[p]=='110' else -1.0 if pattern_str[p]=='011' else 0.0 for p in patterns},
        # I4: freq(000) + freq(001) + 2*freq(010) + 3*freq(011) + freq(111)
        {p: 1.0 if pattern_str[p] in ['000','001','111'] else 2.0 if pattern_str[p]=='010' else 3.0 if pattern_str[p]=='011' else 0.0 for p in patterns},
    ]
    
    # Convert to numpy matrix
    basis = np.zeros((len(patterns), 4))
    for i, inv in enumerate(invariants):
        for j, p in enumerate(patterns):
            basis[j, i] = inv[p]
    
    # Normalize each column
    for i in range(4):
        norm = np.linalg.norm(basis[:, i])
        if norm > 1e-10:
            basis[:, i] /= norm
    
    return basis


def project_to_invariant_subspace(freq_vector: np.ndarray, basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project a frequency vector into invariant subspace and orthogonal complement.
    
    Args:
        freq_vector: Vector of shape (8,) with pattern frequencies
        basis: Invariant basis matrix of shape (8, 4)
        
    Returns:
        (invariant_component, orthogonal_component)
    """
    # Project onto invariant subspace
    invariant_proj = basis @ (basis.T @ freq_vector)
    
    # Orthogonal complement
    orthogonal = freq_vector - invariant_proj
    
    return invariant_proj, orthogonal


def analyze_row_evolution(
    row: List[int],
    steps: int = 50,
    cyclic: bool = True
) -> Dict:
    """
    Analyze how a row's pattern frequencies evolve in invariant space.
    
    Returns:
        Dict with evolution data
    """
    patterns = enumerate_patterns()
    basis = get_invariant_basis()
    
    # Evolve row
    evolution = rule30_evolve(row, steps, cyclic=cyclic)
    
    # Compute frequencies at each step
    freq_vectors = []
    invariant_components = []
    orthogonal_components = []
    invariant_values = []
    
    for step_row in evolution:
        freq_rational = pattern_frequencies_3_rational(step_row, cyclic=cyclic)
        freq_vec = np.array([float(freq_rational[p]) for p in patterns])
        
        inv_proj, orth_proj = project_to_invariant_subspace(freq_vec, basis)
        
        freq_vectors.append(freq_vec)
        invariant_components.append(inv_proj)
        orthogonal_components.append(orth_proj)
        
        # Compute invariant values
        inv_vals = []
        for i in range(4):
            val = basis[:, i] @ freq_vec
            inv_vals.append(val)
        invariant_values.append(inv_vals)
    
    return {
        'freq_vectors': freq_vectors,
        'invariant_components': invariant_components,
        'orthogonal_components': orthogonal_components,
        'invariant_values': invariant_values,
        'evolution': evolution
    }


def compute_invariant_statistics(num_samples: int = 100, row_length: int = 200, steps: int = 50):
    """
    Compute statistics about how invariants constrain the pattern space.
    """
    patterns = enumerate_patterns()
    basis = get_invariant_basis()
    
    print(f"Analyzing {num_samples} random rows over {steps} steps...")
    print()
    
    invariant_variances = []
    orthogonal_norms = []
    
    for i in range(num_samples):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_samples}", end='\r')
        
        row = [random.randint(0, 1) for _ in range(row_length)]
        data = analyze_row_evolution(row, steps=steps, cyclic=True)
        
        # Check variance of invariant values (should be ~0 if truly invariant)
        inv_vals = np.array(data['invariant_values'])
        inv_vars = np.var(inv_vals, axis=0)
        invariant_variances.append(inv_vars)
        
        # Check norm of orthogonal component
        orth_norms = [np.linalg.norm(orth) for orth in data['orthogonal_components']]
        orthogonal_norms.append(orth_norms)
    
    print()  # New line
    
    # Statistics
    inv_vars_array = np.array(invariant_variances)
    print("Invariant Variance Statistics (should be ~0):")
    for i in range(4):
        mean_var = np.mean(inv_vars_array[:, i])
        max_var = np.max(inv_vars_array[:, i])
        print(f"  I{i+1}: mean={mean_var:.2e}, max={max_var:.2e}")
    
    print()
    print("Orthogonal Component Norm Statistics:")
    all_orth_norms = [n for norms in orthogonal_norms for n in norms]
    print(f"  Mean: {np.mean(all_orth_norms):.6f}")
    print(f"  Std:  {np.std(all_orth_norms):.6f}")
    print(f"  Max:  {np.max(all_orth_norms):.6f}")
    
    return {
        'invariant_variances': invariant_variances,
        'orthogonal_norms': orthogonal_norms
    }


def plot_evolution(data: Dict, save_path: str = None):
    """
    Plot evolution of pattern frequencies in invariant space.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plots")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Rule 30 Evolution in Invariant Subspace', fontsize=14)
    
    # Plot 1: Invariant values over time (should be constant)
    ax = axes[0, 0]
    inv_vals = np.array(data['invariant_values'])
    for i in range(4):
        ax.plot(inv_vals[:, i], label=f'I{i+1}', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Invariant Value')
    ax.set_title('Invariant Values (should be constant)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Norm of orthogonal component
    ax = axes[0, 1]
    orth_norms = [np.linalg.norm(orth) for orth in data['orthogonal_components']]
    ax.plot(orth_norms)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('||Orthogonal Component||')
    ax.set_title('Free Dynamics (orthogonal to invariants)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Pattern frequencies over time
    ax = axes[1, 0]
    patterns = enumerate_patterns()
    pattern_str = [''.join(str(b) for b in p) for p in patterns]
    freq_vectors = np.array(data['freq_vectors'])
    for i, p_str in enumerate(pattern_str):
        ax.plot(freq_vectors[:, i], label=p_str, alpha=0.6)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Frequency')
    ax.set_title('Pattern Frequencies Over Time')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: 2D projection of invariant subspace
    ax = axes[1, 1]
    inv_comps = np.array(data['invariant_components'])
    # Project onto first two invariant dimensions
    ax.scatter(inv_comps[:, 0], inv_comps[:, 1], c=range(len(inv_comps)), cmap='viridis', alpha=0.6)
    ax.set_xlabel('I1 Component')
    ax.set_ylabel('I2 Component')
    ax.set_title('Trajectory in Invariant Subspace (2D projection)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Rule 30 invariant geometry",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of random rows to analyze (default: 100)'
    )
    
    parser.add_argument(
        '--row-length',
        type=int,
        default=200,
        help='Length of rows (default: 200)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='Number of evolution steps (default: 50)'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots (requires matplotlib)'
    )
    
    parser.add_argument(
        '--plot-file',
        type=str,
        help='Save plot to file instead of displaying'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("RULE 30 INVARIANT GEOMETRY ANALYSIS")
    print("="*70)
    print()
    
    # Compute statistics
    stats = compute_invariant_statistics(
        num_samples=args.num_samples,
        row_length=args.row_length,
        steps=args.steps
    )
    
    print()
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    print()
    print("The invariants constrain Rule 30 dynamics to a 4D subspace")
    print("in the 8D pattern-frequency space. The 'free' dynamics (orthogonal")
    print("to the invariants) is where the chaos lives.")
    print()
    
    # Plot if requested
    if args.plot or args.plot_file:
        print("Generating plot for sample evolution...")
        sample_row = [random.randint(0, 1) for _ in range(args.row_length)]
        data = analyze_row_evolution(sample_row, steps=args.steps, cyclic=True)
        plot_evolution(data, save_path=args.plot_file)
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Manifold Decoder for Rule 30

Decodes the abstract 15D geometry back into physical 4-bit patterns.

1. Reconstructs the full 34-dimensional state from the 15D PCA components.
2. Identifies the physical meaning of PC1, PC2, PC3 (e.g., "PC1 is the density of 0000").
3. Checks for "Separability": Does the center column value (c_t) cluster in the 3D manifold?
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Add project root and code directory to path
project_root = Path(__file__).parent.parent.parent.parent.parent
code_dir = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(code_dir))

# Import from same directory (PHASE2/code)
from four_bit_chaos_tracker import ChaosTracker15D
from four_bit_system import enumerate_4bit_patterns


def interpret_pc_component(pc_vector_full: np.ndarray, patterns: list) -> list:
    """
    Analyzes which patterns contribute most to a Principal Component.
    Returns sorted list of (contribution, pattern_string).
    """
    # The first 16 elements correspond to freq_t patterns
    contributions = []
    pattern_strs = [''.join(map(str, p)) for p in patterns]
    
    for i, p_str in enumerate(pattern_strs):
        # We look at the first 16 variables (frequencies at time t)
        weight = pc_vector_full[i]
        contributions.append((weight, p_str))
        
    # Sort by absolute magnitude
    contributions.sort(key=lambda x: abs(x[0]), reverse=True)
    return contributions


def main():
    print("="*70)
    print("RULE 30 MANIFOLD DECODER")
    print("="*70)
    
    # 1. Load Data
    # Tracker saves to 'results/chaos15' relative to experiments/rule30 directory
    # Check multiple possible locations
    possible_dirs = [
        project_root / "experiments" / "rule30" / "results" / "chaos15",  # Actual save location
        project_root / "experiments" / "rule30" / "PHASE2" / "results" / "chaos15",  # Alternative
        project_root / "results" / "chaos15",  # Project root fallback
    ]
    
    data_dir = None
    for dir_path in possible_dirs:
        if (dir_path / "trajectory_15d.npy").exists():
            data_dir = dir_path
            break
    
    if data_dir is None:
        print("Error: trajectory files not found. Run four_bit_chaos_tracker.py first.")
        print(f"Checked locations:")
        for dir_path in possible_dirs:
            print(f"  - {dir_path}")
        return
    
    try:
        traj_15d = np.load(data_dir / "trajectory_15d.npy")
        traj_full = np.load(data_dir / "trajectory_full.npy")
        print(f"Loaded trajectory: {traj_15d.shape} steps from {data_dir}")
    except FileNotFoundError as e:
        print(f"Error loading trajectory files: {e}")
        return

    # 2. Re-instantiate Tracker to get Null Space Matrix
    print("Re-computing Null Space Basis...")
    tracker = ChaosTracker15D(verbose=False)
    null_space_matrix = tracker.null_space # Shape (34, 15)
    
    # 3. Run PCA on the 15D Free Space
    print("Running PCA on 15D subspace...")
    pca = PCA(n_components=3)
    pca.fit(traj_15d)
    
    print(f"Explained Variance: {pca.explained_variance_ratio_}")
    
    # 4. Decode Principal Components
    patterns = enumerate_4bit_patterns()
    components_15d = pca.components_ # Shape (3, 15)
    
    print("\n" + "-"*50)
    print("PHYSICAL INTERPRETATION OF CHAOS AXES")
    print("-" * 50)

    for i in range(3):
        pc_id = i + 1
        print(f"\n🔍 DECODING PC{pc_id} ({pca.explained_variance_ratio_[i]*100:.1f}% Variance)")
        
        # Project 15D eigenvector back to 34D full space
        # Full Vector = NullSpace @ PC_15d
        pc_vector_full = null_space_matrix @ components_15d[i]
        
        # Analyze contributions
        contribs = interpret_pc_component(pc_vector_full, patterns)
        
        print("   Dominant Patterns driving this axis:")
        # Print top 3 positive and top 3 negative contributors
        
        # Sort by signed value to separate positive/negative
        sorted_signed = sorted(contribs, key=lambda x: x[0], reverse=True)
        
        print("   (+) Positive drivers:")
        for val, p in sorted_signed[:4]:
            if val > 1e-4:
                print(f"       + {p}: {val:.4f}")
        
        print("   (-) Negative drivers:")
        for val, p in sorted_signed[-4:]:
            if val < -1e-4:
                print(f"       - {p}: {val:.4f}")

    # 5. The "Separability" Test (Color by Center Column)
    print("\n" + "-"*50)
    print("PERFORMING SEPARABILITY TEST")
    print("-" * 50)
    
    # Transform trajectory to PCA space
    traj_pca = pca.transform(traj_15d)
    
    # Extract Center Column Values (c_t)
    # In trajectory_full (34 dims), c_t is index 32
    c_t_values = traj_full[:, 32]
    
    # Create the plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot colored by c_t value
    # c_t is continuous (average over grid), but effectively bimodal or clustered for Rule 30?
    # Actually c_t is a float (density of center=1 patterns).
    
    sc = ax.scatter(traj_pca[:, 0], traj_pca[:, 1], traj_pca[:, 2],
                    c=c_t_values, cmap='coolwarm', s=1, alpha=0.8)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
    ax.set_title('Rule 30 Attractor\nColored by Center Column Density (c_t)')
    
    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Center Column Density (c_t)')
    
    save_path = data_dir / "attractor_colored_by_ct.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n📸 Separability Plot saved to: {save_path}")
    
    # 6. Correlation Analysis
    # Check correlation between PC coordinates and c_t
    corrs = []
    for i in range(3):
        corr = np.corrcoef(traj_pca[:, i], c_t_values)[0, 1]
        corrs.append(corr)
        print(f"   Correlation(PC{i+1}, c_t): {corr:.4f}")
        
    print("\nInterpetation:")
    if abs(corrs[0]) > 0.5:
        print("   ★ PC1 is strongly correlated with the Center Column value!")
    else:
        print("   PC1 is structural, distinct from the center column value.")


if __name__ == "__main__":
    main()


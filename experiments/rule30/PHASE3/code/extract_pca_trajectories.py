#!/usr/bin/env python3
"""
Extract PCA Trajectories from Phase 2 Data

This script:
1. Loads the 15D trajectory from Phase 2
2. Fits PCA on the full trajectory
3. Extracts top k PCA components (default: 8)
4. Saves PCA trajectories and fitted PCA model
5. Computes correlation with center column
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from PHASE2
phase2_code = project_root / "experiments" / "rule30" / "PHASE2" / "code"
sys.path.insert(0, str(phase2_code))

from four_bit_chaos_tracker import ChaosTracker15D


def find_phase2_data():
    """Find Phase 2 trajectory data files."""
    possible_dirs = [
        project_root / "experiments" / "rule30" / "PHASE2" / "results" / "chaos14",
        project_root / "experiments" / "rule30" / "PHASE2" / "results" / "chaos15",
        project_root / "experiments" / "rule30" / "results" / "chaos14",
        project_root / "experiments" / "rule30" / "results" / "chaos15",
        project_root / "results" / "chaos14",
        project_root / "results" / "chaos15",
    ]
    
    for data_dir in possible_dirs:
        traj_15d_path = data_dir / "trajectory_15d.npy"
        traj_full_path = data_dir / "trajectory_full.npy"
        if traj_15d_path.exists() and traj_full_path.exists():
            return data_dir, traj_15d_path, traj_full_path
    
    return None, None, None


def extract_center_column(trajectory_full: np.ndarray) -> np.ndarray:
    """
    Extract center column values from full trajectory.
    
    Args:
        trajectory_full: (num_steps, 34) full state vectors
                        [freq_t (16), freq_tp1 (16), c_t (1), c_tp1 (1)]
    
    Returns:
        Center column values c_t of shape (num_steps,)
    """
    # c_t is at index 32 in the full state vector
    return trajectory_full[:, 32]


def fit_pca_and_extract(trajectory_15d: np.ndarray, 
                        n_components: int = 8,
                        verbose: bool = True) -> tuple:
    """
    Fit PCA and extract top components.
    
    Args:
        trajectory_15d: (num_steps, 15) free coordinates
        n_components: Number of PCA components to extract
        verbose: Print progress
    
    Returns:
        (pca_model, trajectory_pca, explained_variance)
    """
    if verbose:
        print(f"Fitting PCA with {n_components} components...")
    
    pca = PCA(n_components=n_components)
    trajectory_pca = pca.fit_transform(trajectory_15d)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    if verbose:
        print(f"Explained variance per component:")
        for i, (var, cumvar) in enumerate(zip(explained_variance, cumulative_variance), 1):
            print(f"  PC{i}: {var:.4f} ({var*100:.2f}%) | Cumulative: {cumvar:.4f} ({cumvar*100:.2f}%)")
        print(f"\nTotal explained variance: {cumulative_variance[-1]:.4f} ({cumulative_variance[-1]*100:.2f}%)")
    
    return pca, trajectory_pca, explained_variance


def compute_correlations(trajectory_pca: np.ndarray,
                        center_column: np.ndarray,
                        verbose: bool = True) -> np.ndarray:
    """
    Compute correlation between PCA components and center column.
    
    Args:
        trajectory_pca: (num_steps, n_components) PCA coordinates
        center_column: (num_steps,) center column values
        verbose: Print correlations
    
    Returns:
        Array of correlations of shape (n_components,)
    """
    correlations = np.zeros(trajectory_pca.shape[1])
    
    for i in range(trajectory_pca.shape[1]):
        corr = np.corrcoef(trajectory_pca[:, i], center_column)[0, 1]
        correlations[i] = corr
    
    if verbose:
        print(f"\nCorrelation with center column:")
        for i, corr in enumerate(correlations, 1):
            print(f"  PC{i}: {corr:.4f}")
    
    return correlations


def split_temporal_data(trajectory_pca: np.ndarray,
                        center_column: np.ndarray,
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1) -> dict:
    """
    Split data into train/validation/test sets (temporal, no shuffling).
    
    Args:
        trajectory_pca: (num_steps, n_components) PCA coordinates
        center_column: (num_steps,) center column values
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
    
    Returns:
        Dict with train/val/test splits
    """
    n = len(trajectory_pca)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        'train': {
            'pca': trajectory_pca[:n_train],
            'center': center_column[:n_train],
            'indices': np.arange(n_train)
        },
        'val': {
            'pca': trajectory_pca[n_train:n_train+n_val],
            'center': center_column[n_train:n_train+n_val],
            'indices': np.arange(n_train, n_train+n_val)
        },
        'test': {
            'pca': trajectory_pca[n_train+n_val:],
            'center': center_column[n_train+n_val:],
            'indices': np.arange(n_train+n_val, n)
        }
    }
    
    return splits


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract PCA trajectories from Phase 2 data"
    )
    
    parser.add_argument(
        '--n-components',
        type=int,
        default=8,
        help='Number of PCA components to extract (default: 8)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    
    args = parser.parse_args()
    
    # Find Phase 2 data
    data_dir, traj_15d_path, traj_full_path = find_phase2_data()
    
    if data_dir is None:
        print("Error: Phase 2 trajectory data not found.")
        print("Please run Phase 2 first (four_bit_chaos_tracker.py)")
        return
    
    if args.verbose:
        print(f"Loading data from: {data_dir}")
    
    # Load data
    trajectory_15d = np.load(traj_15d_path)
    trajectory_full = np.load(traj_full_path)
    
    if args.verbose:
        print(f"Loaded trajectory: {trajectory_15d.shape}")
        print(f"Loaded full state: {trajectory_full.shape}")
    
    # Extract center column
    center_column = extract_center_column(trajectory_full)
    
    # Fit PCA and extract components
    pca_model, trajectory_pca, explained_variance = fit_pca_and_extract(
        trajectory_15d,
        n_components=args.n_components,
        verbose=args.verbose
    )
    
    # Compute correlations
    correlations = compute_correlations(
        trajectory_pca,
        center_column,
        verbose=args.verbose
    )
    
    # Split data
    splits = split_temporal_data(trajectory_pca, center_column)
    
    if args.verbose:
        print(f"\nData splits:")
        print(f"  Train: {len(splits['train']['pca'])} samples")
        print(f"  Val:   {len(splits['val']['pca'])} samples")
        print(f"  Test:  {len(splits['test']['pca'])} samples")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    np.save(output_dir / 'trajectory_pca.npy', trajectory_pca)
    np.save(output_dir / 'center_column.npy', center_column)
    np.save(output_dir / 'correlations.npy', correlations)
    np.save(output_dir / 'explained_variance.npy', explained_variance)
    
    # Save PCA model
    with open(output_dir / 'pca_model.pkl', 'wb') as f:
        pickle.dump(pca_model, f)
    
    # Save splits
    with open(output_dir / 'data_splits.pkl', 'wb') as f:
        pickle.dump(splits, f)
    
    # Save metadata
    metadata = {
        'n_components': args.n_components,
        'trajectory_shape': trajectory_15d.shape,
        'pca_shape': trajectory_pca.shape,
        'explained_variance': explained_variance.tolist(),
        'correlations': correlations.tolist(),
        'data_source': str(data_dir)
    }
    
    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if args.verbose:
        print(f"\nResults saved to: {output_dir}")
        print(f"  - trajectory_pca.npy")
        print(f"  - center_column.npy")
        print(f"  - correlations.npy")
        print(f"  - explained_variance.npy")
        print(f"  - pca_model.pkl")
        print(f"  - data_splits.pkl")
        print(f"  - metadata.json")


if __name__ == "__main__":
    main()


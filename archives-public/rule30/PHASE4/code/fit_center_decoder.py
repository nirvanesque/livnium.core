#!/usr/bin/env python3
"""
Fit Center Column Decoder

Trains a non-linear decoder to map 8D PCA coordinates → center bit (0 or 1).

This solves the Phase 3 limitation: the geometry → bits mapping is non-linear.

Uses RandomForestClassifier as a non-linear readout (curved knife) to handle
the complex manifold structure of Rule 30's attractor.
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pickle
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def find_pca_data():
    """Find PCA trajectory and center column data."""
    possible_dirs = [
        project_root / "experiments" / "rule30" / "PHASE3" / "results",
        project_root / "experiments" / "rule30" / "results" / "chaos15",
        project_root / "experiments" / "rule30" / "results" / "chaos14",
        project_root / "experiments" / "rule30" / "PHASE2" / "results" / "chaos15",
        project_root / "experiments" / "rule30" / "PHASE2" / "results" / "chaos14",
    ]
    
    for data_dir in possible_dirs:
        traj_pca_path = data_dir / "trajectory_pca.npy"
        center_path = data_dir / "center_column.npy"
        
        # Also check for full trajectory (we can extract center from it)
        traj_full_path = data_dir / "trajectory_full.npy"
        
        if traj_pca_path.exists() and center_path.exists():
            return data_dir, traj_pca_path, center_path
        elif traj_pca_path.exists() and traj_full_path.exists():
            # Extract center column from full trajectory
            return data_dir, traj_pca_path, traj_full_path
    
    return None, None, None


def load_data(data_dir: Path, traj_pca_path: Path, center_path: Path, n_components: int = 8):
    """
    Load PCA trajectory and center column data.
    
    Args:
        data_dir: Directory containing data
        traj_pca_path: Path to PCA trajectory
        center_path: Path to center column (or full trajectory)
        n_components: Number of PCA components to use
    
    Returns:
        (X, y) where X is (n_samples, n_components) and y is (n_samples,) binary
    """
    # Load PCA trajectory
    trajectory_pca = np.load(traj_pca_path)
    
    # Use top n_components
    X = trajectory_pca[:, :n_components]
    
    # Load or extract center column
    if center_path.name == "trajectory_full.npy":
        # Extract center column from full trajectory (index 32)
        trajectory_full = np.load(center_path)
        center_column = trajectory_full[:, 32]
    else:
        center_column = np.load(center_path)
    
    # Convert center column to binary (0 or 1)
    # Center column is a probability/density, so we threshold at 0.5
    y = (center_column > 0.5).astype(int)
    
    return X, y


def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8):
    """
    Split data into train/test sets (temporal, no shuffling).
    
    Args:
        X: Features (n_samples, n_features)
        y: Targets (n_samples,)
        train_ratio: Fraction for training
    
    Returns:
        (X_train, X_test, y_train, y_test)
    """
    n = len(X)
    n_train = int(n * train_ratio)
    
    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]
    
    return X_train, X_test, y_train, y_test


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fit center column decoder (PCA → binary bit)"
    )
    
    parser.add_argument(
        '--n-components',
        type=int,
        default=8,
        help='Number of PCA components to use (default: 8)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Fraction of data for training (default: 0.8)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    
    args = parser.parse_args()
    
    # Find data
    data_dir, traj_pca_path, center_path = find_pca_data()
    
    if data_dir is None:
        print("Error: PCA trajectory data not found.")
        print("Please run Phase 3 first (extract_pca_trajectories.py)")
        return
    
    if args.verbose:
        print(f"Loading data from: {data_dir}")
    
    # Load data
    X, y = load_data(data_dir, traj_pca_path, center_path, n_components=args.n_components)
    
    if args.verbose:
        print(f"Loaded data:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=args.train_ratio)
    
    if args.verbose:
        print(f"\nData splits:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Test:  {len(X_test)} samples")
        print(f"  Train class distribution: {np.bincount(y_train)}")
        print(f"  Test class distribution: {np.bincount(y_test)}")
    
    # Train Random Forest decoder (non-linear readout)
    if args.verbose:
        print("\nTraining Random Forest decoder (non-linear readout)...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,  # Prevent overfitting
        random_state=42,
        class_weight='balanced'  # Force it to pay attention to "1"s
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    train_f1 = f1_score(y_train, train_pred)
    test_f1 = f1_score(y_test, test_pred)
    
    if args.verbose:
        print(f"\nModel Performance:")
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Test Accuracy:  {test_accuracy:.4f}")
        print(f"  Train F1-Score: {train_f1:.4f}")
        print(f"  Test F1-Score:  {test_f1:.4f}")
        
        print(f"\nClassification Report (Test Set):")
        print(classification_report(y_test, test_pred, target_names=['0', '1']))
        
        print(f"\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(y_test, test_pred)
        print(f"  True Negatives:  {cm[0, 0]}")
        print(f"  False Positives: {cm[0, 1]}")
        print(f"  False Negatives: {cm[1, 0]}")
        print(f"  True Positives:  {cm[1, 1]}")
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'center_decoder.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    metadata = {
        'n_components': args.n_components,
        'train_ratio': args.train_ratio,
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'train_f1': float(train_f1),
        'test_f1': float(test_f1),
        'data_source': str(data_dir),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'class_distribution_train': np.bincount(y_train).tolist(),
        'class_distribution_test': np.bincount(y_test).tolist()
    }
    
    with open(output_dir / 'decoder_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if args.verbose:
        print(f"\nModel saved to: {output_dir / 'center_decoder.pkl'}")
        print(f"Metadata saved to: {output_dir / 'decoder_metadata.json'}")


if __name__ == "__main__":
    main()


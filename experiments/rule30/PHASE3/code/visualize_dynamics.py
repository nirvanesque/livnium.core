#!/usr/bin/env python3
"""
Visualize Dynamics Models

This script generates all visualizations for Phase 3:
1. PC1 prediction plots
2. Trajectory comparisons
3. Center column comparisons
4. Error analysis
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def load_data(results_dir: Path):
    """Load all necessary data."""
    results_dir = Path(results_dir)
    
    # Load data splits
    with open(results_dir / 'data_splits.pkl', 'rb') as f:
        splits = pickle.load(f)
    
    # Load PCA trajectory
    trajectory_pca = np.load(results_dir / 'trajectory_pca.npy')
    center_column = np.load(results_dir / 'center_column.npy')
    
    # Load shadow trajectory if available
    shadow_trajectory = None
    shadow_center = None
    if (results_dir / 'shadow_trajectory_pca.npy').exists():
        shadow_trajectory = np.load(results_dir / 'shadow_trajectory_pca.npy')
        shadow_center = np.load(results_dir / 'shadow_center_column.npy')
    
    # Load PC1 predictions if available
    pc1_predictions = None
    for name in ['linear_test_predictions.npy', 'polynomial_degree_2_test_predictions.npy']:
        if (results_dir / name).exists():
            pc1_predictions = np.load(results_dir / name)
            break
    
    return splits, trajectory_pca, center_column, shadow_trajectory, shadow_center, pc1_predictions


def plot_pc1_prediction(splits, pc1_predictions, output_dir: Path, verbose=True):
    """Plot PC1 prediction results."""
    if pc1_predictions is None:
        if verbose:
            print("Skipping PC1 prediction plot (no predictions found)")
        return
    
    # Get test data
    test_pca = splits['test']['pca']
    test_pc1 = test_pca[:, 0]
    
    # Ensure predictions is 1D
    predictions_1d = np.array(pc1_predictions).flatten()
    
    # Align lengths - predictions are for PC1(t+1) given PC1(t), so we need to compare
    # test_pc1[1:] (next step) with predictions
    # Both should have length len(test_pc1) - 1 if predictions were made correctly
    max_len = min(len(test_pc1) - 1, len(predictions_1d))
    test_pc1_next = test_pc1[1:max_len+1]
    predictions = predictions_1d[:max_len]
    
    # Ensure they have the same length
    min_len = min(len(test_pc1_next), len(predictions))
    test_pc1_next = test_pc1_next[:min_len]
    predictions = predictions[:min_len]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Time series comparison
    ax = axes[0, 0]
    time_steps = np.arange(min_len)
    ax.plot(time_steps, test_pc1_next, 'b-', alpha=0.7, label='Real PC1', linewidth=1)
    ax.plot(time_steps, predictions, 'r--', alpha=0.7, label='Predicted PC1', linewidth=1)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('PC1 Value')
    ax.set_title('PC1 Prediction: Real vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    ax = axes[0, 1]
    ax.scatter(test_pc1_next, predictions, alpha=0.5, s=10)
    # Perfect prediction line
    min_val = min(test_pc1_next.min(), predictions.min())
    max_val = max(test_pc1_next.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Real PC1(t+1)')
    ax.set_ylabel('Predicted PC1(t+1)')
    ax.set_title('PC1 Prediction: Scatter Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    ax = axes[1, 0]
    residuals = test_pc1_next - predictions
    ax.plot(time_steps, residuals, 'g-', alpha=0.7, linewidth=1)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Residual (Real - Predicted)')
    ax.set_title('PC1 Prediction: Residuals')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Residual distribution
    ax = axes[1, 1]
    ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    ax.set_title('PC1 Prediction: Residual Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pc1_prediction.png', dpi=150)
    plt.close()
    
    if verbose:
        print(f"Saved PC1 prediction plot to: {output_dir / 'pc1_prediction.png'}")


def plot_trajectory_comparison(trajectory_pca, shadow_trajectory, output_dir: Path, verbose=True):
    """Plot trajectory comparison in PCA space."""
    if shadow_trajectory is None:
        if verbose:
            print("Skipping trajectory comparison plot (no shadow trajectory)")
        return
    
    # Align lengths
    min_len = min(len(trajectory_pca), len(shadow_trajectory))
    real_traj = trajectory_pca[:min_len]
    shadow_traj = shadow_trajectory[:min_len]
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: 2D projection (PC1 vs PC2)
    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(real_traj[:, 0], real_traj[:, 1], c=range(min_len), 
              cmap='viridis', s=1, alpha=0.6, label='Real')
    ax.scatter(shadow_traj[:, 0], shadow_traj[:, 1], c=range(min_len), 
              cmap='plasma', s=1, alpha=0.6, label='Shadow', marker='x')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Trajectory Comparison: PC1 vs PC2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: 3D projection (PC1, PC2, PC3)
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.scatter(real_traj[:, 0], real_traj[:, 1], real_traj[:, 2], 
              c=range(min_len), cmap='viridis', s=1, alpha=0.6, label='Real')
    ax.scatter(shadow_traj[:, 0], shadow_traj[:, 1], shadow_traj[:, 2], 
              c=range(min_len), cmap='plasma', s=1, alpha=0.6, label='Shadow', marker='x')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Trajectory Comparison: 3D View')
    ax.legend()
    
    # Plot 3: Time series of PC1
    ax = fig.add_subplot(2, 2, 3)
    time_steps = np.arange(min_len)
    ax.plot(time_steps, real_traj[:, 0], 'b-', alpha=0.7, label='Real PC1', linewidth=0.5)
    ax.plot(time_steps, shadow_traj[:, 0], 'r--', alpha=0.7, label='Shadow PC1', linewidth=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('PC1 Value')
    ax.set_title('PC1 Time Series Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Time series of PC2
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(time_steps, real_traj[:, 1], 'b-', alpha=0.7, label='Real PC2', linewidth=0.5)
    ax.plot(time_steps, shadow_traj[:, 1], 'r--', alpha=0.7, label='Shadow PC2', linewidth=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('PC2 Value')
    ax.set_title('PC2 Time Series Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_comparison.png', dpi=150)
    plt.close()
    
    if verbose:
        print(f"Saved trajectory comparison plot to: {output_dir / 'trajectory_comparison.png'}")


def plot_center_column_comparison(center_column, shadow_center, output_dir: Path, verbose=True):
    """Plot center column comparison."""
    if shadow_center is None:
        if verbose:
            print("Skipping center column comparison plot (no shadow center column)")
        return
    
    # Align lengths
    min_len = min(len(center_column), len(shadow_center))
    real_center = center_column[:min_len]
    shadow_center_aligned = shadow_center[:min_len]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Time series
    ax = axes[0, 0]
    time_steps = np.arange(min_len)
    ax.plot(time_steps, real_center, 'b-', alpha=0.7, label='Real', linewidth=0.5)
    ax.plot(time_steps, shadow_center_aligned, 'r--', alpha=0.7, label='Shadow', linewidth=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Center Column Value')
    ax.set_title('Center Column: Time Series Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Distribution (histogram)
    ax = axes[0, 1]
    ax.hist(real_center, bins=50, alpha=0.6, label='Real', color='blue', edgecolor='black')
    ax.hist(shadow_center_aligned, bins=50, alpha=0.6, label='Shadow', color='red', edgecolor='black')
    ax.set_xlabel('Center Column Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Center Column: Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Q-Q plot
    ax = axes[1, 0]
    # Sample for Q-Q plot (too many points otherwise)
    sample_size = min(1000, min_len)
    indices = np.linspace(0, min_len - 1, sample_size, dtype=int)
    real_sample = real_center[indices]
    shadow_sample = shadow_center_aligned[indices]
    
    # Sort for Q-Q plot
    real_sorted = np.sort(real_sample)
    shadow_sorted = np.sort(shadow_sample)
    
    ax.scatter(real_sorted, shadow_sorted, alpha=0.5, s=10)
    # Perfect match line
    min_val = min(real_sorted.min(), shadow_sorted.min())
    max_val = max(real_sorted.max(), shadow_sorted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_xlabel('Real Center Column (sorted)')
    ax.set_ylabel('Shadow Center Column (sorted)')
    ax.set_title('Center Column: Q-Q Plot')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Difference
    ax = axes[1, 1]
    difference = real_center - shadow_center_aligned
    ax.plot(time_steps, difference, 'g-', alpha=0.7, linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Difference (Real - Shadow)')
    ax.set_title('Center Column: Difference Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'center_column_comparison.png', dpi=150)
    plt.close()
    
    if verbose:
        print(f"Saved center column comparison plot to: {output_dir / 'center_column_comparison.png'}")


def plot_error_analysis(splits, output_dir: Path, verbose=True):
    """Plot error analysis for dynamics models."""
    # Try to load evaluation results
    results_dir = output_dir
    if not (results_dir / 'evaluation_results.json').exists():
        if verbose:
            print("Skipping error analysis plot (no evaluation results found)")
        return
    
    with open(results_dir / 'evaluation_results.json', 'r') as f:
        eval_results = json.load(f)
    
    if 'prediction_horizons' not in eval_results:
        if verbose:
            print("Skipping error analysis plot (no prediction horizon data)")
        return
    
    horizons_data = eval_results['prediction_horizons']
    
    # Extract data
    # Convert keys to integers for proper numerical sorting (1, 5, 10), not (1, 10, 5)
    # JSON keys are strings, so we access data using str(h)
    horizons = sorted([int(k) for k in horizons_data.keys()])
    
    mse_values = [horizons_data[str(h)]['mse'] for h in horizons]
    mae_values = [horizons_data[str(h)]['mae'] for h in horizons]
    r2_values = [horizons_data[str(h)]['r2'] for h in horizons]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: MSE vs horizon
    ax = axes[0]
    ax.plot(horizons, mse_values, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Prediction Horizon')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Prediction Error vs Horizon (MSE)')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(horizons)
    
    # Plot 2: MAE vs horizon
    ax = axes[1]
    ax.plot(horizons, mae_values, 'o-', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Prediction Horizon')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Prediction Error vs Horizon (MAE)')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(horizons)
    
    # Plot 3: R² vs horizon
    ax = axes[2]
    ax.plot(horizons, r2_values, 'o-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Prediction Horizon')
    ax.set_ylabel('R² Score')
    ax.set_title('Prediction Accuracy vs Horizon (R²)')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(horizons)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_analysis.png', dpi=150)
    plt.close()
    
    if verbose:
        print(f"Saved error analysis plot to: {output_dir / 'error_analysis.png'}")


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize dynamics models"
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='results',
        help='Directory containing models and data (default: results)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for plots (default: results)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    
    args = parser.parse_args()
    
    # Load data
    data_dir = Path(args.data_dir)
    splits, trajectory_pca, center_column, shadow_traj, shadow_center, pc1_predictions = load_data(data_dir)
    
    if args.verbose:
        print(f"Loaded data from: {data_dir}")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    if args.verbose:
        print("\nGenerating visualizations...")
    
    plot_pc1_prediction(splits, pc1_predictions, output_dir, verbose=args.verbose)
    plot_trajectory_comparison(trajectory_pca, shadow_traj, output_dir, verbose=args.verbose)
    plot_center_column_comparison(center_column, shadow_center, output_dir, verbose=args.verbose)
    plot_error_analysis(splits, output_dir, verbose=args.verbose)
    
    if args.verbose:
        print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()


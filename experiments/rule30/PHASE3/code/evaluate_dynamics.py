#!/usr/bin/env python3
"""
Evaluate Dynamics Models

This script:
1. Evaluates prediction accuracy (1-step, 5-step, 10-step ahead)
2. Compares real vs shadow trajectories
3. Computes distribution comparisons
4. Generates evaluation report
"""

import sys
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    
    return splits, trajectory_pca, center_column, shadow_trajectory, shadow_center


def evaluate_prediction_horizon(model_info, X_test, y_test, horizons=[1, 5, 10], verbose=True):
    """
    Evaluate model at different prediction horizons.
    
    Args:
        model_info: Model dict with 'model' key
        X_test: Test features
        y_test: Test targets
        horizons: List of prediction horizons to evaluate
    
    Returns:
        Dict with results for each horizon
    """
    model = model_info['model']
    use_poly = 'poly_features' in model_info
    
    results = {}
    
    for horizon in horizons:
        if horizon == 1:
            # 1-step ahead: direct prediction
            if use_poly:
                X_test_poly = model_info['poly_features'].transform(X_test)
                predictions = model.predict(X_test_poly)
            else:
                predictions = model.predict(X_test)
            
            targets = y_test
        else:
            # Multi-step ahead: iterative prediction
            predictions = []
            targets = []
            
            # Start from first point
            current_state = X_test[0].copy()
            
            for i in range(len(X_test) - horizon + 1):
                # Predict horizon steps ahead
                state = current_state.copy()
                for _ in range(horizon):
                    if use_poly:
                        state_poly = model_info['poly_features'].transform(state.reshape(1, -1))
                        state = model.predict(state_poly).flatten()
                    else:
                        state = model.predict(state.reshape(1, -1)).flatten()
                
                predictions.append(state)
                targets.append(y_test[i + horizon - 1])
                
                # Update current state (use actual next state for next iteration)
                if i + 1 < len(X_test):
                    current_state = X_test[i + 1]
            
            predictions = np.array(predictions)
            targets = np.array(targets)
        
        # Compute metrics
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        results[horizon] = {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
            'rmse': float(np.sqrt(mse))
        }
        
        if verbose:
            print(f"  Horizon {horizon}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")
    
    return results


def compare_distributions(real_data: np.ndarray, shadow_data: np.ndarray, 
                         name: str = "data", verbose=True) -> dict:
    """
    Compare distributions of real vs shadow data.
    
    Args:
        real_data: Real data array
        shadow_data: Shadow data array
        name: Name for reporting
        verbose: Print results
    
    Returns:
        Dict with comparison statistics
    """
    # Kolmogorov-Smirnov test
    ks_statistic, ks_pvalue = stats.ks_2samp(real_data, shadow_data)
    
    # Mean and std comparison
    real_mean = real_data.mean()
    shadow_mean = shadow_data.mean()
    real_std = real_data.std()
    shadow_std = shadow_data.std()
    
    # Relative differences
    mean_diff = abs(real_mean - shadow_mean) / (abs(real_mean) + 1e-10)
    std_diff = abs(real_std - shadow_std) / (abs(real_std) + 1e-10)
    
    results = {
        'ks_statistic': float(ks_statistic),
        'ks_pvalue': float(ks_pvalue),
        'real_mean': float(real_mean),
        'shadow_mean': float(shadow_mean),
        'real_std': float(real_std),
        'shadow_std': float(shadow_std),
        'mean_relative_diff': float(mean_diff),
        'std_relative_diff': float(std_diff),
        'distributions_match': ks_pvalue > 0.05
    }
    
    if verbose:
        print(f"\n{name} Distribution Comparison:")
        print(f"  KS statistic: {ks_statistic:.6f}, p-value: {ks_pvalue:.6f}")
        print(f"  Real: mean={real_mean:.6f}, std={real_std:.6f}")
        print(f"  Shadow: mean={shadow_mean:.6f}, std={shadow_std:.6f}")
        print(f"  Distributions match: {results['distributions_match']}")
    
    return results


def compare_trajectories(real_traj: np.ndarray, shadow_traj: np.ndarray, 
                        verbose=True) -> dict:
    """
    Compare real vs shadow trajectories in PCA space.
    
    Args:
        real_traj: Real PCA trajectory (num_steps, n_components)
        shadow_traj: Shadow PCA trajectory (num_steps, n_components)
        verbose: Print results
    
    Returns:
        Dict with comparison statistics
    """
    # Align trajectories (use minimum length)
    min_len = min(len(real_traj), len(shadow_traj))
    real_traj = real_traj[:min_len]
    shadow_traj = shadow_traj[:min_len]
    
    # Per-component comparison
    component_results = []
    for i in range(real_traj.shape[1]):
        real_comp = real_traj[:, i]
        shadow_comp = shadow_traj[:, i]
        
        mse = mean_squared_error(real_comp, shadow_comp)
        mae = mean_absolute_error(real_comp, shadow_comp)
        r2 = r2_score(real_comp, shadow_comp)
        corr = np.corrcoef(real_comp, shadow_comp)[0, 1]
        
        component_results.append({
            'component': i + 1,
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
            'correlation': float(corr)
        })
        
        if verbose:
            print(f"  PC{i+1}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}, Corr={corr:.4f}")
    
    # Overall trajectory distance
    trajectory_distance = np.mean(np.linalg.norm(real_traj - shadow_traj, axis=1))
    
    results = {
        'component_results': component_results,
        'mean_trajectory_distance': float(trajectory_distance)
    }
    
    return results


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate dynamics models"
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
        help='Output directory (default: results)'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['pc1', 'full'],
        default='full',
        help='Type of model to evaluate (default: full)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    
    args = parser.parse_args()
    
    # Load data
    data_dir = Path(args.data_dir)
    splits, trajectory_pca, center_column, shadow_traj, shadow_center = load_data(data_dir)
    
    if args.verbose:
        print(f"Loaded data from: {data_dir}")
        print(f"  Real trajectory: {trajectory_pca.shape}")
        if shadow_traj is not None:
            print(f"  Shadow trajectory: {shadow_traj.shape}")
    
    evaluation_results = {}
    
    # Evaluate prediction horizons
    if args.model_type == 'full':
        # Load full dynamics model
        model_path = None
        for name in ['linear_dynamics_model.pkl', 'polynomial_degree_2_dynamics_model.pkl']:
            if (data_dir / name).exists():
                model_path = data_dir / name
                break
        
        if model_path is None:
            print("Warning: Full dynamics model not found. Skipping horizon evaluation.")
        else:
            with open(model_path, 'rb') as f:
                model_info = pickle.load(f)
            
            if args.verbose:
                print("\n" + "="*60)
                print("EVALUATING PREDICTION HORIZONS")
                print("="*60)
            
            # Use test set for evaluation
            X_test = splits['test']['pca'][:-1, :8]  # Top 8 components
            y_test = splits['test']['pca'][1:, :8]
            
            horizon_results = evaluate_prediction_horizon(
                model_info, X_test, y_test, 
                horizons=[1, 5, 10, 20],
                verbose=args.verbose
            )
            evaluation_results['prediction_horizons'] = horizon_results
    
    # Compare shadow vs real
    if shadow_traj is not None:
        if args.verbose:
            print("\n" + "="*60)
            print("COMPARING SHADOW VS REAL TRAJECTORIES")
            print("="*60)
        
        # Compare trajectories
        traj_comparison = compare_trajectories(
            trajectory_pca[:len(shadow_traj)], shadow_traj,
            verbose=args.verbose
        )
        evaluation_results['trajectory_comparison'] = traj_comparison
        
        # Compare center column distributions
        if shadow_center is not None:
            center_comparison = compare_distributions(
                center_column[:len(shadow_center)], shadow_center,
                name="Center Column",
                verbose=args.verbose
            )
            evaluation_results['center_column_comparison'] = center_comparison
    
    # Save evaluation results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON
    def convert_to_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json(item) for item in obj]
        return obj
    
    evaluation_results_json = convert_to_json(evaluation_results)
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(evaluation_results_json, f, indent=2)
    
    if args.verbose:
        print(f"\nEvaluation results saved to: {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Fit Full Dynamics Model for Top 8 PCs

This script fits a dynamics model to predict all top k PCA components:
y_{t+1} = F(y_t) where y_t is (k,) vector

Uses vectorized models that predict all components simultaneously.
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def load_pca_data(results_dir: Path):
    """Load PCA trajectory data and splits."""
    results_dir = Path(results_dir)
    
    with open(results_dir / 'data_splits.pkl', 'rb') as f:
        splits = pickle.load(f)
    
    with open(results_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return splits, metadata


def prepare_full_data(splits: dict, n_components: int = 8):
    """
    Prepare data for full dynamics prediction.
    
    Args:
        splits: Data splits dict
        n_components: Number of PCA components to predict
    
    Returns:
        Dict with X_train, y_train, X_val, y_val, X_test, y_test
    """
    def prepare_split(split_data, n_components):
        pca = split_data['pca']
        X = pca[:, :n_components]  # All components as features
        y = pca[:, :n_components]  # All components as targets
        return X[:-1], y[1:]  # Predict next step
    
    data = {}
    for split_name in ['train', 'val', 'test']:
        X, y = prepare_split(splits[split_name], n_components)
        data[f'X_{split_name}'] = X
        data[f'y_{split_name}'] = y
    
    return data


def fit_linear_dynamics(X_train, y_train, X_val, y_val, verbose=True):
    """Fit linear dynamics: y_{t+1} = J @ y_t + b."""
    # Use MultiOutputRegressor to fit one model per output
    base_model = LinearRegression()
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    # Per-component metrics
    train_r2_per_component = [r2_score(y_train[:, i], train_pred[:, i]) 
                              for i in range(y_train.shape[1])]
    val_r2_per_component = [r2_score(y_val[:, i], val_pred[:, i]) 
                            for i in range(y_val.shape[1])]
    
    # Extract Jacobian matrix
    # For MultiOutputRegressor, each estimator has its own coefficients
    jacobian = np.array([est.coef_ for est in model.estimators_])
    bias = np.array([est.intercept_ for est in model.estimators_])
    
    if verbose:
        print(f"Linear Dynamics Model:")
        print(f"  Train MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
        print(f"  Val MSE:   {val_mse:.6f}, R²: {val_r2:.4f}")
        print(f"  Per-component R² (val): {[f'{r:.4f}' for r in val_r2_per_component]}")
        print(f"  Jacobian shape: {jacobian.shape}")
    
    return {
        'model': model,
        'type': 'linear',
        'jacobian': jacobian,
        'bias': bias,
        'train_mse': train_mse,
        'val_mse': val_mse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'train_r2_per_component': train_r2_per_component,
        'val_r2_per_component': val_r2_per_component
    }


def fit_polynomial_dynamics(X_train, y_train, X_val, y_val, degree=2, verbose=True):
    """Fit polynomial dynamics: y_{t+1} = P(y_t) where P is polynomial."""
    # Create polynomial features
    # For degree 3, use include_bias=False to avoid overfitting
    include_bias = degree < 3
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    
    # Fit MultiOutputRegressor with Ridge
    base_model = Ridge(alpha=1e-6)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train_poly, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train_poly)
    val_pred = model.predict(X_val_poly)
    
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    # Per-component metrics
    train_r2_per_component = [r2_score(y_train[:, i], train_pred[:, i]) 
                              for i in range(y_train.shape[1])]
    val_r2_per_component = [r2_score(y_val[:, i], val_pred[:, i]) 
                            for i in range(y_val.shape[1])]
    
    if verbose:
        print(f"Polynomial Dynamics Model (degree {degree}):")
        print(f"  Train MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
        print(f"  Val MSE:   {val_mse:.6f}, R²: {val_r2:.4f}")
        print(f"  Per-component R² (val): {[f'{r:.4f}' for r in val_r2_per_component]}")
        print(f"  Number of features: {X_train_poly.shape[1]}")
    
    return {
        'model': model,
        'poly_features': poly,
        'type': f'polynomial_degree_{degree}',
        'train_mse': train_mse,
        'val_mse': val_mse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'train_r2_per_component': train_r2_per_component,
        'val_r2_per_component': val_r2_per_component,
        'n_features': X_train_poly.shape[1]
    }


def evaluate_dynamics(model_info, X_test, y_test, verbose=True):
    """Evaluate dynamics model on test set."""
    model = model_info['model']
    model_type = model_info['type']
    
    if 'poly_features' in model_info:
        X_test_poly = model_info['poly_features'].transform(X_test)
        test_pred = model.predict(X_test_poly)
    else:
        test_pred = model.predict(X_test)
    
    test_mse = mean_squared_error(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    # Per-component metrics
    test_r2_per_component = [r2_score(y_test[:, i], test_pred[:, i]) 
                             for i in range(y_test.shape[1])]
    test_mae_per_component = [mean_absolute_error(y_test[:, i], test_pred[:, i]) 
                               for i in range(y_test.shape[1])]
    
    if verbose:
        print(f"\nTest Set Evaluation ({model_type}):")
        print(f"  MSE: {test_mse:.6f}")
        print(f"  MAE: {test_mae:.6f}")
        print(f"  R²:  {test_r2:.4f}")
        print(f"  Per-component R²: {[f'{r:.4f}' for r in test_r2_per_component]}")
        print(f"  Per-component MAE: {[f'{m:.6f}' for m in test_mae_per_component]}")
    
    return {
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_r2_per_component': test_r2_per_component,
        'test_mae_per_component': test_mae_per_component,
        'predictions': test_pred
    }


def analyze_stability(model_info, verbose=True):
    """
    Analyze stability of linear dynamics model.
    
    For linear model: y_{t+1} = J @ y_t + b
    Stability depends on eigenvalues of Jacobian J.
    """
    if model_info['type'] != 'linear':
        if verbose:
            print("Stability analysis only available for linear models.")
        return None
    
    jacobian = model_info['jacobian']
    eigenvalues = np.linalg.eigvals(jacobian)
    
    max_eigenvalue = np.max(np.abs(eigenvalues))
    is_stable = max_eigenvalue < 1.0
    
    if verbose:
        print(f"\nStability Analysis:")
        print(f"  Max |eigenvalue|: {max_eigenvalue:.6f}")
        print(f"  Stable: {is_stable}")
        print(f"  Eigenvalues: {eigenvalues}")
    
    return {
        'eigenvalues': eigenvalues.tolist(),
        'max_eigenvalue': float(max_eigenvalue),
        'is_stable': bool(is_stable)
    }


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fit full dynamics model for top k PCs"
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='results',
        help='Directory containing PCA data (default: results)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    
    parser.add_argument(
        '--n-components',
        type=int,
        default=8,
        help='Number of PCA components to model (default: 8)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['linear', 'polynomial'],
        default=['linear', 'polynomial'],
        help='Models to fit (default: linear polynomial)'
    )
    
    parser.add_argument(
        '--degree',
        type=int,
        default=None,
        help='Polynomial degree to fit (if None, fits both degree 2 and 3)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    
    args = parser.parse_args()
    
    # Load data
    data_dir = Path(args.data_dir)
    splits, metadata = load_pca_data(data_dir)
    
    if args.verbose:
        print(f"Loaded data from: {data_dir}")
        print(f"Number of components: {metadata['n_components']}")
    
    # Prepare full data
    data = prepare_full_data(splits, n_components=args.n_components)
    
    if args.verbose:
        print(f"\nData shapes:")
        print(f"  Train: {data['X_train'].shape} -> {data['y_train'].shape}")
        print(f"  Val:   {data['X_val'].shape} -> {data['y_val'].shape}")
        print(f"  Test:  {data['X_test'].shape} -> {data['y_test'].shape}")
    
    # Fit models
    results = {}
    
    if 'linear' in args.models:
        if args.verbose:
            print("\n" + "="*60)
            print("FITTING LINEAR DYNAMICS")
            print("="*60)
        results['linear'] = fit_linear_dynamics(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            verbose=args.verbose
        )
        results['linear']['test'] = evaluate_dynamics(
            results['linear'],
            data['X_test'], data['y_test'],
            verbose=args.verbose
        )
        results['linear']['stability'] = analyze_stability(
            results['linear'],
            verbose=args.verbose
        )
    
    if 'polynomial' in args.models:
        if args.verbose:
            print("\n" + "="*60)
            print("FITTING POLYNOMIAL DYNAMICS")
            print("="*60)
        # Use specified degree, or default to [2, 3]
        degrees = [args.degree] if args.degree is not None else [2, 3]
        for degree in degrees:
            key = f'polynomial_degree_{degree}'
            results[key] = fit_polynomial_dynamics(
                data['X_train'], data['y_train'],
                data['X_val'], data['y_val'],
                degree=degree,
                verbose=args.verbose
            )
            results[key]['test'] = evaluate_dynamics(
                results[key],
                data['X_test'], data['y_test'],
                verbose=args.verbose
            )
    
    # Find best model
    best_model = None
    best_r2 = -float('inf')
    
    for name, result in results.items():
        if 'test' in result and result['test']['test_r2'] > best_r2:
            best_r2 = result['test']['test_r2']
            best_model = name
    
    if args.verbose:
        print("\n" + "="*60)
        print(f"BEST MODEL: {best_model} (R² = {best_r2:.4f})")
        print("="*60)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    for name, result in results.items():
        model_data = result.copy()
        if 'test' in model_data:
            test_pred = model_data['test'].pop('predictions')
            np.save(output_dir / f'{name}_test_predictions.npy', test_pred)
        
        # Convert numpy arrays to lists for JSON
        if 'jacobian' in model_data:
            model_data['jacobian'] = model_data['jacobian'].tolist()
        if 'bias' in model_data:
            model_data['bias'] = model_data['bias'].tolist()
        
        with open(output_dir / f'{name}_dynamics_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
    
    # Save summary
    summary = {
        'best_model': best_model,
        'best_r2': best_r2,
        'n_components': args.n_components,
        'models': {name: {
            'type': result['type'],
            'val_r2': result.get('val_r2', None),
            'test_r2': result.get('test', {}).get('test_r2', None),
            'test_mse': result.get('test', {}).get('test_mse', None),
            'test_mae': result.get('test', {}).get('test_mae', None),
            'test_r2_per_component': result.get('test', {}).get('test_r2_per_component', None)
        } for name, result in results.items()}
    }
    
    with open(output_dir / 'full_dynamics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    if args.verbose:
        print(f"\nResults saved to: {output_dir}")
        print(f"  - {best_model}_dynamics_model.pkl (best model)")
        for name in results.keys():
            if name != best_model:
                print(f"  - {name}_dynamics_model.pkl")
        print(f"  - full_dynamics_summary.json")


if __name__ == "__main__":
    main()


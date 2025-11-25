#!/usr/bin/env python3
"""
Fit PC1 Dynamics Model

This script fits a dynamics model to predict PC1(t+1) from PC1(t), PC2(t), PC3(t).

Allowed strategies:
- Local linear approximation (Jacobian)
- Polynomial regression (degree ≤ 3)
- Kernel regression
- Sparse nonlinear autoregressive models

Forbidden:
- Neural networks
- Black-box deep learning
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge
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


def prepare_pc1_data(splits: dict, use_pc2_pc3: bool = True):
    """
    Prepare data for PC1 prediction.
    
    Args:
        splits: Data splits dict
        use_pc2_pc3: If True, use PC1, PC2, PC3 as features. If False, only PC1.
    
    Returns:
        Dict with X_train, y_train, X_val, y_val, X_test, y_test
    """
    def prepare_split(split_data, use_pc2_pc3):
        pca = split_data['pca']
        if use_pc2_pc3:
            X = pca[:, :3]  # PC1, PC2, PC3
        else:
            X = pca[:, 0:1]  # Only PC1
        y = pca[:, 0]  # PC1 at next time step (we'll shift)
        return X[:-1], y[1:]
    
    data = {}
    for split_name in ['train', 'val', 'test']:
        X, y = prepare_split(splits[split_name], use_pc2_pc3)
        data[f'X_{split_name}'] = X
        data[f'y_{split_name}'] = y
    
    return data


def fit_linear_model(X_train, y_train, X_val, y_val, verbose=True):
    """Fit linear model: y = X @ coef + intercept."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    if verbose:
        print(f"Linear Model:")
        print(f"  Train MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
        print(f"  Val MSE:   {val_mse:.6f}, R²: {val_r2:.4f}")
        print(f"  Coefficients: {model.coef_}")
        print(f"  Intercept: {model.intercept_:.6f}")
    
    return {
        'model': model,
        'type': 'linear',
        'train_mse': train_mse,
        'val_mse': val_mse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'coef': model.coef_.tolist(),
        'intercept': model.intercept_
    }


def fit_polynomial_model(X_train, y_train, X_val, y_val, degree=2, verbose=True):
    """Fit polynomial model: y = P(X) where P is polynomial of given degree."""
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    
    # Fit linear regression on polynomial features
    model = Ridge(alpha=1e-6)  # Small regularization for stability
    model.fit(X_train_poly, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train_poly)
    val_pred = model.predict(X_val_poly)
    
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    if verbose:
        print(f"Polynomial Model (degree {degree}):")
        print(f"  Train MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
        print(f"  Val MSE:   {val_mse:.6f}, R²: {val_r2:.4f}")
        print(f"  Number of features: {X_train_poly.shape[1]}")
    
    return {
        'model': model,
        'poly_features': poly,
        'type': f'polynomial_degree_{degree}',
        'train_mse': train_mse,
        'val_mse': val_mse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'n_features': X_train_poly.shape[1]
    }


def fit_kernel_ridge(X_train, y_train, X_val, y_val, kernel='rbf', verbose=True):
    """Fit kernel ridge regression model."""
    # Use cross-validation to select alpha
    alphas = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_alpha = None
    best_val_mse = float('inf')
    
    for alpha in alphas:
        model = KernelRidge(alpha=alpha, kernel=kernel, gamma='scale')
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_pred)
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_alpha = alpha
    
    # Fit final model
    model = KernelRidge(alpha=best_alpha, kernel=kernel, gamma='scale')
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    if verbose:
        print(f"Kernel Ridge Model (kernel={kernel}, alpha={best_alpha}):")
        print(f"  Train MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
        print(f"  Val MSE:   {val_mse:.6f}, R²: {val_r2:.4f}")
    
    return {
        'model': model,
        'type': f'kernel_ridge_{kernel}',
        'alpha': best_alpha,
        'train_mse': train_mse,
        'val_mse': val_mse,
        'train_r2': train_r2,
        'val_r2': val_r2
    }


def evaluate_model(model_info, X_test, y_test, verbose=True):
    """Evaluate model on test set."""
    model = model_info['model']
    model_type = model_info['type']
    
    if 'poly_features' in model_info:
        # Polynomial model
        X_test_poly = model_info['poly_features'].transform(X_test)
        test_pred = model.predict(X_test_poly)
    else:
        test_pred = model.predict(X_test)
    
    test_mse = mean_squared_error(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    if verbose:
        print(f"\nTest Set Evaluation ({model_type}):")
        print(f"  MSE: {test_mse:.6f}")
        print(f"  MAE: {test_mae:.6f}")
        print(f"  R²:  {test_r2:.4f}")
    
    return {
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'predictions': test_pred
    }


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fit PC1 dynamics model"
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
        '--use-pc2-pc3',
        action='store_true',
        help='Use PC2 and PC3 as features (default: only PC1)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['linear', 'polynomial', 'kernel'],
        default=['linear', 'polynomial'],
        help='Models to fit (default: linear polynomial)'
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
    
    # Prepare PC1 data
    data = prepare_pc1_data(splits, use_pc2_pc3=args.use_pc2_pc3)
    
    if args.verbose:
        print(f"\nData shapes:")
        print(f"  Train: {data['X_train'].shape}")
        print(f"  Val:   {data['X_val'].shape}")
        print(f"  Test:  {data['X_test'].shape}")
        if args.use_pc2_pc3:
            print("  Features: PC1, PC2, PC3")
        else:
            print("  Features: PC1 only")
    
    # Fit models
    results = {}
    
    if 'linear' in args.models:
        if args.verbose:
            print("\n" + "="*60)
            print("FITTING LINEAR MODEL")
            print("="*60)
        results['linear'] = fit_linear_model(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            verbose=args.verbose
        )
        results['linear']['test'] = evaluate_model(
            results['linear'],
            data['X_test'], data['y_test'],
            verbose=args.verbose
        )
    
    if 'polynomial' in args.models:
        if args.verbose:
            print("\n" + "="*60)
            print("FITTING POLYNOMIAL MODELS")
            print("="*60)
        for degree in [2, 3]:
            key = f'polynomial_degree_{degree}'
            results[key] = fit_polynomial_model(
                data['X_train'], data['y_train'],
                data['X_val'], data['y_val'],
                degree=degree,
                verbose=args.verbose
            )
            results[key]['test'] = evaluate_model(
                results[key],
                data['X_test'], data['y_test'],
                verbose=args.verbose
            )
    
    if 'kernel' in args.models:
        if args.verbose:
            print("\n" + "="*60)
            print("FITTING KERNEL RIDGE MODEL")
            print("="*60)
        results['kernel'] = fit_kernel_ridge(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            kernel='rbf',
            verbose=args.verbose
        )
        results['kernel']['test'] = evaluate_model(
            results['kernel'],
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
    
    # Save models (remove test predictions to save space)
    for name, result in results.items():
        model_data = result.copy()
        if 'test' in model_data:
            # Save test predictions separately
            test_pred = model_data['test'].pop('predictions')
            np.save(output_dir / f'{name}_test_predictions.npy', test_pred)
        
        # Save model
        with open(output_dir / f'{name}_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
    
    # Save summary
    summary = {
        'best_model': best_model,
        'best_r2': best_r2,
        'models': {name: {
            'type': result['type'],
            'val_r2': result.get('val_r2', None),
            'test_r2': result.get('test', {}).get('test_r2', None),
            'test_mse': result.get('test', {}).get('test_mse', None),
            'test_mae': result.get('test', {}).get('test_mae', None)
        } for name, result in results.items()},
        'features': 'PC1, PC2, PC3' if args.use_pc2_pc3 else 'PC1 only'
    }
    
    with open(output_dir / 'pc1_model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    if args.verbose:
        print(f"\nResults saved to: {output_dir}")
        print(f"  - {best_model}_model.pkl (best model)")
        for name in results.keys():
            if name != best_model:
                print(f"  - {name}_model.pkl")
        print(f"  - pc1_model_summary.json")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Validate Shadow Rule 30: Check if Motion + Decoder = Signal

Validates that the shadow dynamics produce a chaotic bit stream that
statistically resembles Rule 30.

Criteria:
- Center column density ≈ 0.5
- Bit autocorrelation ≈ 0 (white-noise like)
- No collapse to all 0s or all 1s
- No periodic cycles (rare but check)
"""

import sys
from pathlib import Path
import numpy as np
from scipy import stats
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def load_shadow_results(results_dir: Path):
    """Load shadow simulation results."""
    results_dir = Path(results_dir)
    
    if not (results_dir / 'shadow_center_column.npy').exists():
        raise FileNotFoundError(f"Shadow center column not found at {results_dir}")
    
    center_column = np.load(results_dir / 'shadow_center_column.npy')
    
    # Load statistics if available
    stats_file = results_dir / 'shadow_statistics.json'
    shadow_stats = None
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            shadow_stats = json.load(f)
    
    return center_column, shadow_stats


def compute_density(center_column: np.ndarray) -> float:
    """Compute fraction of ones in center column."""
    return float((center_column == 1).sum() / len(center_column))


def compute_autocorrelation(center_column: np.ndarray, max_lag: int = 100) -> dict:
    """
    Compute autocorrelation of center column bits.
    
    Returns autocorrelation at various lags.
    """
    n = len(center_column)
    autocorrs = {}
    
    # Convert to float for correlation computation
    center_float = center_column.astype(float)
    mean = center_float.mean()
    centered = center_float - mean
    
    # Compute autocorrelation at various lags
    lags = [1, 5, 10, 20, 50, 100]
    lags = [lag for lag in lags if lag < n // 2]
    
    for lag in lags:
        if lag < n:
            corr = np.corrcoef(centered[:-lag], centered[lag:])[0, 1]
            autocorrs[lag] = float(corr) if not np.isnan(corr) else 0.0
    
    return autocorrs


def check_collapse(center_column: np.ndarray) -> dict:
    """Check if center column collapsed to all 0s or all 1s."""
    unique_values = np.unique(center_column)
    n_unique = len(unique_values)
    
    return {
        'collapsed': n_unique == 1,
        'all_zeros': n_unique == 1 and unique_values[0] == 0,
        'all_ones': n_unique == 1 and unique_values[0] == 1,
        'n_unique_values': int(n_unique)
    }


def check_periodicity(center_column: np.ndarray, max_period: int = 100) -> dict:
    """
    Check for periodic cycles in center column.
    
    Returns period if found, None otherwise.
    """
    n = len(center_column)
    max_period = min(max_period, n // 2)
    
    for period in range(2, max_period + 1):
        # Check if sequence repeats with this period
        if n < period * 2:
            continue
        
        # Compare first period with second period
        first_period = center_column[:period]
        second_period = center_column[period:2*period]
        
        if np.array_equal(first_period, second_period):
            # Check if it continues
            is_periodic = True
            for i in range(2, n // period):
                current_period = center_column[i*period:(i+1)*period]
                if not np.array_equal(first_period, current_period):
                    is_periodic = False
                    break
            
            if is_periodic:
                return {
                    'is_periodic': True,
                    'period': int(period)
                }
    
    return {
        'is_periodic': False,
        'period': None
    }


def validate_shadow(center_column: np.ndarray, verbose: bool = True) -> dict:
    """
    Validate shadow Rule 30 against criteria.
    
    Returns dict with validation results.
    """
    results = {}
    
    # Criterion 1: Center column density ≈ 0.5
    density = compute_density(center_column)
    results['density'] = density
    results['density_valid'] = 0.3 <= density <= 0.7  # Allow some tolerance
    
    if verbose:
        print(f"\n1. Center Column Density: {density:.4f}")
        print(f"   Target: ~0.5, Valid: {results['density_valid']}")
    
    # Criterion 2: Bit autocorrelation ≈ 0 (white-noise like)
    autocorrs = compute_autocorrelation(center_column)
    max_autocorr = max(abs(v) for v in autocorrs.values()) if autocorrs else 0.0
    results['autocorrelation'] = autocorrs
    results['max_autocorrelation'] = max_autocorr
    results['autocorrelation_valid'] = max_autocorr < 0.1  # Low correlation
    
    if verbose:
        print(f"\n2. Autocorrelation (max abs): {max_autocorr:.4f}")
        print(f"   Target: <0.1 (white-noise like), Valid: {results['autocorrelation_valid']}")
        if verbose and autocorrs:
            print(f"   Autocorrs at lags: {autocorrs}")
    
    # Criterion 3: No collapse to all 0s or all 1s
    collapse_info = check_collapse(center_column)
    results['collapse'] = collapse_info
    results['collapse_valid'] = not collapse_info['collapsed']
    
    if verbose:
        print(f"\n3. Collapse Check:")
        print(f"   Collapsed: {collapse_info['collapsed']}")
        print(f"   All zeros: {collapse_info['all_zeros']}")
        print(f"   All ones: {collapse_info['all_ones']}")
        print(f"   Unique values: {collapse_info['n_unique_values']}")
        print(f"   Valid: {results['collapse_valid']}")
    
    # Criterion 4: No periodic cycles
    periodicity_info = check_periodicity(center_column)
    results['periodicity'] = periodicity_info
    results['periodicity_valid'] = not periodicity_info['is_periodic']
    
    if verbose:
        print(f"\n4. Periodicity Check:")
        print(f"   Is periodic: {periodicity_info['is_periodic']}")
        if periodicity_info['is_periodic']:
            print(f"   Period: {periodicity_info['period']}")
        print(f"   Valid: {results['periodicity_valid']}")
    
    # Overall validation
    all_valid = (
        results['density_valid'] and
        results['autocorrelation_valid'] and
        results['collapse_valid'] and
        results['periodicity_valid']
    )
    results['all_criteria_met'] = all_valid
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Overall Validation: {'✅ PASS' if all_valid else '❌ FAIL'}")
        print(f"{'='*60}")
    
    return results


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate Shadow Rule 30: Check Motion + Decoder = Signal"
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory containing shadow results (default: results)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for validation results (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    
    args = parser.parse_args()
    
    # Load shadow results
    results_dir = Path(args.results_dir)
    center_column, shadow_stats = load_shadow_results(results_dir)
    
    if args.verbose:
        print(f"Loaded shadow results from: {results_dir}")
        print(f"  Center column shape: {center_column.shape}")
        print(f"  Center column type: {center_column.dtype}")
    
    # Validate
    validation_results = validate_shadow(center_column, verbose=args.verbose)
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        if args.verbose:
            print(f"\nValidation results saved to: {output_path}")


if __name__ == "__main__":
    main()


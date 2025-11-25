#!/usr/bin/env python3
"""
Invariant Hunter V2 - Non-Linear Invariant Search

Searches for non-linear algebraic invariants in Rule 30 center column.
Tests quadratic, cubic, cross-frequency, and angular invariants.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Callable
from fractions import Fraction
from itertools import product, combinations
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.rule30_optimized import generate_center_column_direct
from experiments.rule30.pattern_analysis import compute_pattern_frequencies


def compute_pattern_features(sequence: List[int], pattern_length: int = 3) -> Dict[str, float]:
    """Compute pattern frequencies and derived features."""
    freqs = compute_pattern_frequencies(sequence, pattern_length)
    
    # Add derived features
    features = dict(freqs)
    
    # Density features
    features['density_ones'] = sum(sequence) / len(sequence) if sequence else 0.0
    features['density_zeros'] = 1.0 - features['density_ones']
    
    # Pattern differences
    if '111' in freqs and '000' in freqs:
        features['diff_111_000'] = freqs['111'] - freqs['000']
    
    if '101' in freqs and '010' in freqs:
        features['diff_101_010'] = freqs['101'] - freqs['010']
    
    # Pattern products
    if '011' in freqs and '110' in freqs:
        features['prod_011_110'] = freqs['011'] * freqs['110']
    
    if '001' in freqs and '100' in freqs:
        features['prod_001_100'] = freqs['001'] * freqs['100']
    
    return features


def enumerate_quadratic_invariants(
    patterns: List[str],
    max_coefficient: int = 5
) -> List[Dict]:
    """
    Enumerate quadratic invariants: Σ α_p · freq_p + Σ β_pq · freq_p · freq_q.
    
    Args:
        patterns: List of pattern strings
        max_coefficient: Maximum absolute coefficient value
        
    Returns:
        List of candidate invariants
    """
    candidates = []
    
    # Linear terms
    linear_coeffs = list(range(-max_coefficient, max_coefficient + 1))
    
    # Quadratic terms (products of patterns)
    # Limit to small number of quadratic terms to avoid explosion
    pattern_pairs = list(combinations(patterns, 2))[:10]  # Limit pairs
    
    # Enumerate combinations
    for linear_combo in product(linear_coeffs, repeat=min(3, len(patterns))):
        # Skip trivial (all zeros)
        if all(c == 0 for c in linear_combo):
            continue
        
        # Add quadratic terms
        for quad_combo in product([-1, 0, 1], repeat=min(2, len(pattern_pairs))):
            coefficients = {
                'linear': dict(zip(patterns[:len(linear_combo)], linear_combo)),
                'quadratic': dict(zip(pattern_pairs[:len(quad_combo)], quad_combo))
            }
            
            candidates.append({
                'type': 'quadratic',
                'coefficients': coefficients,
                'as_formula': _format_quadratic_formula(patterns, coefficients)
            })
    
    return candidates[:1000]  # Limit to avoid explosion


def enumerate_polynomial_invariants(
    patterns: List[str],
    max_degree: int = 2,
    max_coefficient: int = 3
) -> List[Dict]:
    """
    Enumerate polynomial invariants up to given degree.
    
    Args:
        patterns: List of pattern strings
        max_degree: Maximum polynomial degree
        max_coefficient: Maximum coefficient value
        
    Returns:
        List of candidate invariants
    """
    candidates = []
    
    # For now, focus on quadratic (degree 2)
    if max_degree >= 2:
        candidates.extend(enumerate_quadratic_invariants(patterns, max_coefficient))
    
    return candidates


def enumerate_cross_frequency_invariants(
    patterns: List[str],
    max_coefficient: int = 3
) -> List[Dict]:
    """
    Enumerate invariants based on cross-frequency relationships.
    
    Examples:
    - freq('111') - freq('000')
    - freq('011') * freq('110')
    - freq('101') + freq('010') - freq('001')
    """
    candidates = []
    
    # Pattern differences
    pattern_pairs = list(combinations(patterns, 2))
    
    for p1, p2 in pattern_pairs[:10]:  # Limit pairs
        for coeff in range(-max_coefficient, max_coefficient + 1):
            if coeff == 0:
                continue
            
            candidates.append({
                'type': 'difference',
                'pattern1': p1,
                'pattern2': p2,
                'coefficient': coeff,
                'as_formula': f"{coeff} * (freq('{p1}') - freq('{p2}'))"
            })
    
    # Pattern products
    for p1, p2 in pattern_pairs[:5]:  # Limit pairs
        for coeff in [-1, 1]:
            candidates.append({
                'type': 'product',
                'pattern1': p1,
                'pattern2': p2,
                'coefficient': coeff,
                'as_formula': f"{coeff} * freq('{p1}') * freq('{p2}')"
            })
    
    return candidates


def enumerate_angular_invariants(
    patterns: List[str],
    max_coefficient: int = 3
) -> List[Dict]:
    """
    Enumerate invariants based on angular/geometric relationships.
    
    These capture the angle-based nature of divergence.
    """
    candidates = []
    
    # Angular relationships often involve:
    # - Differences (angle separation)
    # - Ratios (angle ratios)
    # - Cross products (geometric products)
    
    # Pattern ratio invariants
    pattern_pairs = list(combinations(patterns, 2))[:5]
    
    for p1, p2 in pattern_pairs:
        candidates.append({
            'type': 'ratio',
            'pattern1': p1,
            'pattern2': p2,
            'as_formula': f"freq('{p1}') / (freq('{p2}') + 1e-10)"
        })
    
    # Weighted angle-like combinations
    # These mimic angular relationships
    for coeff1 in [-1, 1]:
        for coeff2 in [-1, 1]:
            if len(patterns) >= 2:
                candidates.append({
                    'type': 'weighted_angle',
                    'pattern1': patterns[0],
                    'pattern2': patterns[1],
                    'coeff1': coeff1,
                    'coeff2': coeff2,
                    'as_formula': f"{coeff1}*freq('{patterns[0]}') + {coeff2}*freq('{patterns[1]}')"
                })
    
    return candidates


def evaluate_invariant(
    candidate: Dict,
    sequences: List[List[int]],
    target_value: float = -0.572222233,
    tolerance: float = 1e-6
) -> Dict:
    """
    Evaluate a candidate invariant on sequences.
    
    Args:
        candidate: Candidate invariant dictionary
        sequences: List of sequences to test
        target_value: Expected invariant value
        tolerance: Tolerance for matching
        
    Returns:
        Dict with evaluation results
    """
    values = []
    
    for seq in sequences:
        features = compute_pattern_features(seq)
        
        # Evaluate based on candidate type
        if candidate['type'] == 'quadratic':
            value = 0.0
            coeffs = candidate['coefficients']
            
            # Linear terms
            for pattern, coeff in coeffs['linear'].items():
                value += coeff * features.get(pattern, 0.0)
            
            # Quadratic terms
            for (p1, p2), coeff in coeffs['quadratic'].items():
                value += coeff * features.get(p1, 0.0) * features.get(p2, 0.0)
            
            values.append(value)
        
        elif candidate['type'] == 'difference':
            p1 = candidate['pattern1']
            p2 = candidate['pattern2']
            coeff = candidate['coefficient']
            value = coeff * (features.get(p1, 0.0) - features.get(p2, 0.0))
            values.append(value)
        
        elif candidate['type'] == 'product':
            p1 = candidate['pattern1']
            p2 = candidate['pattern2']
            coeff = candidate['coefficient']
            value = coeff * features.get(p1, 0.0) * features.get(p2, 0.0)
            values.append(value)
        
        elif candidate['type'] == 'ratio':
            p1 = candidate['pattern1']
            p2 = candidate['pattern2']
            denom = features.get(p2, 0.0) + 1e-10
            value = features.get(p1, 0.0) / denom
            values.append(value)
        
        elif candidate['type'] == 'weighted_angle':
            p1 = candidate['pattern1']
            p2 = candidate['pattern2']
            c1 = candidate['coeff1']
            c2 = candidate['coeff2']
            value = c1 * features.get(p1, 0.0) + c2 * features.get(p2, 0.0)
            values.append(value)
        
        else:
            # Default: try to evaluate as formula string (simplified)
            values.append(0.0)
    
    if not values:
        return {'error': 'No values computed'}
    
    values = np.array(values)
    mean_val = np.mean(values)
    std_val = np.std(values)
    max_dev = np.max(np.abs(values - mean_val))
    
    is_conserved = std_val < tolerance
    matches_target = abs(mean_val - target_value) < tolerance
    
    return {
        'candidate': candidate,
        'mean_value': float(mean_val),
        'std_value': float(std_val),
        'max_deviation': float(max_dev),
        'is_conserved': is_conserved,
        'matches_target': matches_target,
        'values': values.tolist()
    }


def _format_quadratic_formula(patterns: List[str], coeffs: Dict) -> str:
    """Format quadratic invariant as formula string."""
    terms = []
    
    # Linear terms
    for pattern, coeff in coeffs['linear'].items():
        if coeff != 0:
            if coeff == 1:
                terms.append(f"freq('{pattern}')")
            elif coeff == -1:
                terms.append(f"-freq('{pattern}')")
            else:
                terms.append(f"{coeff}*freq('{pattern}')")
    
    # Quadratic terms
    for (p1, p2), coeff in coeffs['quadratic'].items():
        if coeff != 0:
            if coeff == 1:
                terms.append(f"freq('{p1}')*freq('{p2}')")
            elif coeff == -1:
                terms.append(f"-freq('{p1}')*freq('{p2}')")
            else:
                terms.append(f"{coeff}*freq('{p1}')*freq('{p2}')")
    
    if not terms:
        return "0"
    
    return " + ".join(terms)


def hunt_nonlinear_invariants(
    n_steps_list: List[int],
    target_value: float = -0.572222233,
    max_coefficient: int = 3,
    invariant_types: List[str] = None
) -> List[Dict]:
    """
    Hunt for non-linear invariants.
    
    Args:
        n_steps_list: List of sequence lengths
        target_value: Target invariant value
        max_coefficient: Maximum coefficient
        invariant_types: Types to search (default: all)
        
    Returns:
        List of passing candidates
    """
    if invariant_types is None:
        invariant_types = ['quadratic', 'difference', 'product', 'ratio', 'weighted_angle']
    
    print("="*70)
    print("NON-LINEAR INVARIANT HUNTER V2")
    print("="*70)
    print(f"Target value: {target_value:.9f}")
    print(f"Testing sequences: {n_steps_list}")
    print(f"Invariant types: {invariant_types}")
    print()
    
    # Generate sequences
    print("Generating Rule 30 sequences...")
    sequences = []
    for n in n_steps_list:
        seq = generate_center_column_direct(n, show_progress=False)
        sequences.append(seq)
        print(f"  Generated {n:,} steps: {len(seq):,} bits")
    
    print()
    
    # Get patterns
    features = compute_pattern_features(sequences[0])
    patterns = [p for p in features.keys() if p.startswith(('0', '1')) and len(p) == 3]
    
    print(f"Analyzing {len(patterns)} patterns: {patterns}")
    print()
    
    # Enumerate candidates
    all_candidates = []
    
    if 'quadratic' in invariant_types:
        print("Enumerating quadratic invariants...")
        quad_candidates = enumerate_quadratic_invariants(patterns, max_coefficient)
        all_candidates.extend(quad_candidates)
        print(f"  Generated {len(quad_candidates)} candidates")
    
    if 'difference' in invariant_types or 'product' in invariant_types:
        print("Enumerating cross-frequency invariants...")
        cross_candidates = enumerate_cross_frequency_invariants(patterns, max_coefficient)
        all_candidates.extend(cross_candidates)
        print(f"  Generated {len(cross_candidates)} candidates")
    
    if 'ratio' in invariant_types or 'weighted_angle' in invariant_types:
        print("Enumerating angular invariants...")
        angular_candidates = enumerate_angular_invariants(patterns, max_coefficient)
        all_candidates.extend(angular_candidates)
        print(f"  Generated {len(angular_candidates)} candidates")
    
    print(f"\nTotal candidates: {len(all_candidates)}")
    print()
    
    # Test candidates
    print("Testing candidates...")
    passing_candidates = []
    
    for i, candidate in enumerate(all_candidates):
        if (i + 1) % 100 == 0:
            print(f"  Tested {i+1:,}/{len(all_candidates):,} candidates...")
        
        result = evaluate_invariant(candidate, sequences, target_value)
        
        if result.get('is_conserved') and result.get('matches_target'):
            passing_candidates.append(result)
            print(f"\n  ✓ Found candidate: {result['candidate']['as_formula']}")
            print(f"    Mean: {result['mean_value']:.9f}, Std: {result['std_value']:.9e}")
    
    print()
    print(f"Found {len(passing_candidates)} passing candidates")
    
    return passing_candidates


def main():
    parser = argparse.ArgumentParser(
        description="Hunt for non-linear algebraic invariants in Rule 30",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=[1000, 10000],
        help='Sequence lengths to test (default: 1000 10000)'
    )
    
    parser.add_argument(
        '--target',
        type=float,
        default=-0.572222233,
        help='Target invariant value (default: -0.572222233)'
    )
    
    parser.add_argument(
        '--max-coeff',
        type=int,
        default=3,
        help='Maximum coefficient (default: 3)'
    )
    
    parser.add_argument(
        '--types',
        type=str,
        nargs='+',
        default=['quadratic', 'difference', 'product'],
        help='Invariant types to search (default: quadratic difference product)'
    )
    
    args = parser.parse_args()
    
    # Hunt for invariants
    passing = hunt_nonlinear_invariants(
        args.steps,
        target_value=args.target,
        max_coefficient=args.max_coeff,
        invariant_types=args.types
    )
    
    # Print summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    if passing:
        print(f"\nFound {len(passing)} candidate invariants:")
        for i, result in enumerate(passing[:10], 1):
            print(f"\n{i}. {result['candidate']['as_formula']}")
            print(f"   Type: {result['candidate'].get('type', 'unknown')}")
            print(f"   Mean: {result['mean_value']:.9f}")
            print(f"   Std:  {result['std_value']:.9e}")
        
        if len(passing) > 10:
            print(f"\n... and {len(passing) - 10} more")
        
        print("\n✓✓✓ NON-LINEAR INVARIANTS FOUND!")
        print("These candidates can now be verified with exact arithmetic")
        print("and then proven algebraically.")
    else:
        print("\nNo non-linear invariants found with given parameters.")
        print("Try:")
        print("  - Increasing --max-coeff")
        print("  - Testing different --types")
        print("  - Using longer sequences")
    
    print()


if __name__ == '__main__':
    main()


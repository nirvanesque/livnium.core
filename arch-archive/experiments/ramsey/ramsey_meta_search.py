"""
Ramsey Meta-Search: Outer Basin Reinforcement Loop

Implements the formula:
- P(i) = e^{-1/SW_i} / sum(e^{-1/SW_j})  (softmax over basin weights)
- SW_{t+1} = SW_t + α  (if correct)
- SW_{t+1} = (1 - β) * SW_t + N(0, σ)  (if wrong)

This is the OUTER loop over candidate universes/strategies.
The INNER loop is solve_ramsey_dynamic (single-universe descent).
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig

# Handle relative imports
try:
    from .ramsey_encoder import RamseyEncoder
    from .ramsey_tension import (
        compute_ramsey_tension,
        count_monochromatic_k3,
        count_monochromatic_k4,
        is_valid_ramsey_coloring
    )
    from .ramsey_dynamic_search import solve_ramsey_dynamic
except ImportError:
    from ramsey_encoder import RamseyEncoder
    from ramsey_tension import (
        compute_ramsey_tension,
        count_monochromatic_k3,
        count_monochromatic_k4,
        is_valid_ramsey_coloring
    )
    from ramsey_dynamic_search import solve_ramsey_dynamic


class CandidateUniverse:
    """
    Represents a candidate universe/strategy for Ramsey solving.
    
    Each candidate has:
    - SW_i: Basin weight (how much we trust this candidate)
    - Initial coloring/strategy
    - History of violations
    """
    def __init__(self, id: int, initial_coloring: Dict[Tuple[int, int], int] = None):
        self.id = id
        self.sw = 10.0  # Initial basin weight
        self.initial_coloring = initial_coloring
        self.best_violations = float('inf')
        self.total_attempts = 0
        self.successful_attempts = 0


def softmax_sample(candidates: List[CandidateUniverse], temperature: float = 1.0) -> CandidateUniverse:
    """
    Sample candidate using softmax over basin weights.
    
    Formula: P(i) = e^{-1/SW_i} / sum(e^{-1/SW_j})
    
    Args:
        candidates: List of candidate universes
        temperature: Temperature parameter (default 1.0)
        
    Returns:
        Sampled candidate
    """
    if not candidates:
        raise ValueError("No candidates to sample from")
    
    # Compute energies: E_i = 1 / SW_i
    energies = [1.0 / (c.sw + 1e-10) for c in candidates]  # +1e-10 to avoid division by zero
    
    # Compute probabilities: P(i) = e^{-E_i / T} / sum(e^{-E_j / T})
    exp_energies = [np.exp(-e / temperature) for e in energies]
    total = sum(exp_energies)
    probabilities = [e / total for e in exp_energies]
    
    # Sample
    return np.random.choice(candidates, p=probabilities)


def update_basin_weight(
    candidate: CandidateUniverse,
    violations: int,
    base_alpha: float = 0.10,
    base_beta: float = 0.15,
    sigma: float = 0.0
) -> None:
    """
    Update basin weight using the formula.
    
    Formula:
    - If correct (violations == 0): SW_{t+1} = SW_t + α
    - If wrong (violations > 0): SW_{t+1} = (1 - β) * SW_t + N(0, σ)
    
    Args:
        candidate: Candidate universe to update
        violations: Number of K₄ violations found
        base_alpha: Reinforcement strength
        base_beta: Decay strength
        sigma: Noise level (default 0.0 for Ramsey)
    """
    candidate.total_attempts += 1
    
    if violations == 0:
        # Correct: SW_{t+1} = SW_t + α
        candidate.sw += base_alpha
        candidate.successful_attempts += 1
    else:
        # Wrong: SW_{t+1} = (1 - β) * SW_t + N(0, σ)
        candidate.sw = (1.0 - base_beta) * candidate.sw
        if sigma > 0:
            candidate.sw += np.random.normal(0.0, sigma)
        # Ensure SW doesn't go negative
        if candidate.sw < 0:
            candidate.sw = 0.0
    
    # Track best
    if violations < candidate.best_violations:
        candidate.best_violations = violations


def solve_ramsey_meta(
    n_vertices: int,
    n_candidates: int = 10,
    inner_steps: int = 1000,
    outer_iterations: int = 20,
    verbose: bool = True,
    constraint_type: str = 'k4'
) -> Dict[str, Any]:
    """
    Solve Ramsey using meta-search over candidate universes.
    
    This implements the outer basin reinforcement loop:
    1. Sample candidate universe using softmax
    2. Run inner dynamic search
    3. Update candidate's SW based on violations
    4. Repeat
    
    Args:
        n_vertices: Number of vertices (n for Kₙ)
        n_candidates: Number of candidate universes
        inner_steps: Steps per inner dynamic search
        outer_iterations: Number of outer loop iterations
        verbose: Print progress
        
    Returns:
        Results dictionary
    """
    if verbose:
        print("="*70)
        print(f"Ramsey Meta-Search: K_{n_vertices}")
        print("="*70)
        print()
        print(f"  Outer loop: {outer_iterations} iterations")
        print(f"  Inner loop: {inner_steps} steps per candidate")
        print(f"  Candidates: {n_candidates} universes")
        print()
    
    # Estimate lattice size
    n_edges = n_vertices * (n_vertices - 1) // 2
    n_lattice = max(3, int(np.ceil((n_edges) ** (1/3))))
    if n_lattice % 2 == 0:
        n_lattice += 1
    
    # Create candidate universes
    candidates = []
    for i in range(n_candidates):
        candidates.append(CandidateUniverse(id=i))
    
    if verbose:
        print(f"  Created {len(candidates)} candidate universes")
        print()
    
    # Outer loop: meta-search over candidates
    best_overall_violations = float('inf')
    best_overall_coloring = None
    best_candidate_id = None
    
    for outer_iter in range(outer_iterations):
        # Sample candidate using softmax
        candidate = softmax_sample(candidates)
        
        if verbose and (outer_iter + 1) % 5 == 0:
            print(f"  Outer iteration {outer_iter + 1}/{outer_iterations}: "
                  f"Sampled candidate {candidate.id} (SW={candidate.sw:.3f})")
        
        # Create system for this candidate
        config = LivniumCoreConfig(
            lattice_size=n_lattice,
            enable_semantic_polarity=True
        )
        system = LivniumCoreSystem(config)
        
        # Create encoder
        encoder = RamseyEncoder(system, n_vertices)
        
        # Encode constraints based on type
        if constraint_type == 'k3':
            encoder.encode_k3_constraints()
        else:
            encoder.encode_k4_constraints()
        
        # Initialize with candidate's coloring if available
        if candidate.initial_coloring:
            encoder.encode_coloring(candidate.initial_coloring, initial_only=True)
        
        # Run inner dynamic search
        result = solve_ramsey_dynamic(
            system=system,
            encoder=encoder,
            max_steps=inner_steps,
            verbose=False,  # Suppress inner loop output
            constraint_type=constraint_type
        )
        
        # Extract results
        violations = result['violations']
        coloring = result['coloring']
        
        # Update candidate's basin weight using formula
        update_basin_weight(candidate, violations)
        
        # Track best overall
        if violations < best_overall_violations:
            best_overall_violations = violations
            best_overall_coloring = coloring
            best_candidate_id = candidate.id
        
        # Progress update
        if verbose and (outer_iter + 1) % 5 == 0:
            print(f"    Result: {violations} violations, "
                  f"Best overall: {best_overall_violations}")
            print(f"    Candidate {candidate.id}: SW={candidate.sw:.3f}, "
                  f"Best={candidate.best_violations}, "
                  f"Success rate={candidate.successful_attempts}/{candidate.total_attempts}")
            print()
        
        # Early exit if solved
        if violations == 0:
            if verbose:
                print(f"  ✓ Solution found at outer iteration {outer_iter + 1}")
            break
    
    # Final results
    is_valid = best_overall_violations == 0
    ramsey_tension = compute_ramsey_tension(
        best_overall_coloring or {},
        list(range(n_vertices)),
        constraint_type=constraint_type
    ) if best_overall_coloring else 1.0
    
    # Candidate statistics
    candidate_stats = {
        c.id: {
            'sw': c.sw,
            'best_violations': c.best_violations,
            'success_rate': c.successful_attempts / max(c.total_attempts, 1)
        }
        for c in candidates
    }
    
    results = {
        'solved': is_valid,
        'violations': best_overall_violations,
        'tension': ramsey_tension,
        'coloring': best_overall_coloring,
        'best_candidate_id': best_candidate_id,
        'candidate_stats': candidate_stats,
        'outer_iterations': outer_iterations
    }
    
    if verbose:
        print()
        print("="*70)
        print("RESULTS")
        print("="*70)
        print(f"  Valid coloring: {is_valid}")
        print(f"  Monochromatic K₄s: {best_overall_violations}")
        print(f"  Ramsey tension: {ramsey_tension:.4f}")
        print(f"  Best candidate: {best_candidate_id}")
        print()
        print("  Candidate Statistics:")
        for cid, stats in sorted(candidate_stats.items(), key=lambda x: x[1]['sw'], reverse=True):
            print(f"    Candidate {cid}: SW={stats['sw']:.3f}, "
                  f"Best={stats['best_violations']}, "
                  f"Success={stats['success_rate']:.2%}")
        print()
    
    return results


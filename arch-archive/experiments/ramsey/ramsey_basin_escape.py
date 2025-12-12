"""
Basin Escape Mechanism: Break Out of False Vacuum Attractors

This module detects when the solver is stuck in a false vacuum (like 98.61%)
and applies aggressive basin-breaking techniques to escape.

The problem: After collapse, the solver re-forms into the exact same 98.61% basin.
The solution: Detect this pattern and force escape using constraint-based flips.
"""

from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import numpy as np

try:
    from .ramsey_encoder import RamseyEncoder
    from .ramsey_tension import (
        count_monochromatic_k3,
        count_monochromatic_k4,
        get_all_k3_subsets,
        get_all_k4_subsets,
    )
    from .ramsey_local_feedback_patch import compute_edge_violation_counts
    from .ramsey_curvature_healing import compute_flip_impact
except ImportError:
    from ramsey_encoder import RamseyEncoder
    from ramsey_tension import (
        count_monochromatic_k3,
        count_monochromatic_k4,
        get_all_k3_subsets,
        get_all_k4_subsets,
    )
    from ramsey_local_feedback_patch import compute_edge_violation_counts
    try:
        from ramsey_curvature_healing import compute_flip_impact
    except ImportError:
        compute_flip_impact = None

Edge = Tuple[int, int]
Coloring = Dict[Edge, int]


class BasinTracker:
    """
    Tracks the solver's history to detect false vacuum re-formation.
    
    Detects when:
    - System collapses (0% satisfied)
    - Then re-forms into same basin (same violation count, same structure)
    - This indicates a deep attractor that needs breaking
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.violation_history = []
        self.percent_history = []
        self.step_history = []
        
    def add(self, step: int, violations: int, percent_satisfied: float):
        """Add a data point."""
        self.step_history.append(step)
        self.violation_history.append(violations)
        self.percent_history.append(percent_satisfied)
        
        # Keep window size reasonable
        if len(self.step_history) > self.window_size:
            self.step_history = self.step_history[-self.window_size:]
            self.violation_history = self.violation_history[-self.window_size:]
            self.percent_history = self.percent_history[-self.window_size:]
    
    def detect_collapse(self) -> bool:
        """Detect if system recently collapsed (0% satisfied)."""
        if len(self.percent_history) < 5:
            return False
        
        # Check last 5 steps - must be VERY low (<0.5%) to trigger
        recent_percents = self.percent_history[-5:]
        return min(recent_percents) < 0.5  # Collapse = <0.5% satisfied (very strict)
    
    def detect_false_vacuum(self, target_percent: float = 98.0, tolerance: float = 1.0) -> bool:
        """
        Detect if system is stuck in a false vacuum (same % satisfied repeatedly).
        
        Args:
            target_percent: Target percentage to check (e.g., 98.61%)
            tolerance: How close counts as "same" (e.g., 98.61% ± 1.0%)
        """
        if len(self.percent_history) < 200:
            return False
        
        # Check last 200 steps
        recent_percents = self.percent_history[-200:]
        
        # Count how many times we hit the target
        hits = sum(1 for p in recent_percents 
                   if abs(p - target_percent) < tolerance)
        
        # If we hit target >50% of the time, we're stuck
        return hits > len(recent_percents) * 0.5
    
    def detect_reformation(self) -> bool:
        """
        Detect if system collapsed and then re-formed into same basin.
        
        This is the key pattern: collapse → re-formation → same attractor.
        """
        if len(self.percent_history) < 500:
            return False
        
        # Look for: low → recovery → stable high
        recent = self.percent_history[-500:]
        
        # Find collapse point (minimum)
        min_idx_local = np.argmin(recent)
        min_percent = recent[min_idx_local]
        
        # If we collapsed (<10%), check if we recovered to >95%
        if min_percent < 10.0:
            # Check what happened after collapse
            after_collapse = recent[min_idx_local:]
            if len(after_collapse) > 100:
                # Did we recover to high percentage?
                max_after = max(after_collapse)
                if max_after > 95.0:
                    # Check if violation count stabilized
                    # Convert local index to global index
                    min_idx_global = len(self.percent_history) - len(recent) + min_idx_local
                    if len(self.violation_history) > min_idx_global + 100:
                        violations_after = self.violation_history[min_idx_global + 100:]
                        if len(violations_after) > 50:
                            # Check if violations are stable (same value repeatedly)
                            unique_violations = len(set(violations_after[-50:]))
                            if unique_violations < 5:  # Very few unique values = stable
                                return True
        
        return False


def break_false_vacuum_aggressive(
    system,
    encoder: RamseyEncoder,
    coloring: Coloring,
    vertices: List[int],
    constraint_type: str = "k4",
    max_flips: int = 30,
    min_improvement: int = 0  # Minimum net improvement required
) -> int:
    """
    Aggressively break out of false vacuum by flipping edges with maximum impact.
    
    This is the "basin escape" mechanism:
    1. Find ALL edges with violations
    2. Compute flip impact for each
    3. Flip the ones with BEST net impact (even if negative, if it's better than others)
    4. Force escape from the attractor
    
    Returns: Number of edges flipped
    """
    # Get violation counts
    edge_violations = compute_edge_violation_counts(coloring, vertices, constraint_type)
    
    if not edge_violations:
        return 0  # No violations
    
    # Compute flip impact for ALL edges with violations
    edge_impacts = []
    for edge, violation_count in edge_violations.items():
        if edge not in encoder.edge_to_coords:
            continue
        
        if compute_flip_impact is not None:
            fixed, created = compute_flip_impact(coloring, edge, vertices, constraint_type)
            net_impact = fixed - created
        else:
            # Fallback: use violation count as proxy
            net_impact = violation_count
        
        # Score = net_impact weighted by violation count
        # Even negative impacts are considered if they're better than others
        score = net_impact * (1.0 + violation_count / 10.0)
        
        edge_impacts.append((score, net_impact, violation_count, edge))
    
    # Sort by score (best first)
    edge_impacts.sort(reverse=True, key=lambda x: x[0])
    
    # ONLY flip edges with positive net impact (or at least non-negative)
    # This prevents making things worse
    edges_to_flip = []
    total_expected_improvement = 0
    for score, net_impact, violation_count, edge in edge_impacts:
        if len(edges_to_flip) >= max_flips:
            break
        
        # ONLY flip if net impact is positive (or zero, but prefer positive)
        # This ensures we don't make things worse
        if net_impact >= 0:
            edges_to_flip.append(edge)
            total_expected_improvement += net_impact
    
    # If we don't have enough positive-impact edges, don't flip anything
    # Better to do nothing than make things worse
    if total_expected_improvement < min_improvement:
        return 0  # Don't flip if we can't improve enough
    
    # Apply flips (STRONG push to break attractor)
    for edge in edges_to_flip:
        coord = encoder.edge_to_coords[edge]
        cell = system.get_cell(coord)
        if cell is None:
            continue
        
        # Flip edge (VERY STRONG push - 95% target, 5% current)
        current_color = coloring.get(edge, 0)
        new_color = 1 - current_color
        target = +15.0 if new_color == 1 else -15.0  # Stronger than normal
        
        # VERY STRONG push: 95% target, 5% current (breaks attractor)
        cell.symbolic_weight = 0.05 * cell.symbolic_weight + 0.95 * target
    
    return len(edges_to_flip)


def escape_basin_with_constraint_flips(
    system,
    encoder: RamseyEncoder,
    coloring: Coloring,
    vertices: List[int],
    constraint_type: str = "k4",
    max_flips: int = 50
) -> int:
    """
    Escape basin by directly fixing violated constraints (not SW-based).
    
    This is the most aggressive approach:
    1. Find violated constraints
    2. For each violated constraint, flip ONE edge (the one with best impact)
    3. This directly attacks the constraint structure, not SW field
    
    Returns: Number of edges flipped
    """
    # Get all constraints
    if constraint_type == "k3":
        constraints = get_all_k3_subsets(vertices)
    else:
        constraints = get_all_k4_subsets(vertices)
    
    # Find violated constraints
    violated_constraints = []
    for constraint in constraints:
        # Get edges in this constraint
        constraint_edges = []
        for i in range(len(constraint)):
            for j in range(i + 1, len(constraint)):
                edge = (min(constraint[i], constraint[j]), max(constraint[i], constraint[j]))
                if edge in coloring:
                    constraint_edges.append(edge)
        
        # Check if monochromatic (violated)
        if len(constraint_edges) == len(constraint):
            edge_colors = [coloring.get(e, 0) for e in constraint_edges]
            if len(set(edge_colors)) == 1:
                violated_constraints.append(constraint_edges)
    
    # For each violated constraint, flip the edge with best impact
    edges_to_flip = set()
    for constraint_edges in violated_constraints:
        # Find best edge to flip in this constraint
        best_edge = None
        best_impact = float('-inf')
        
        for edge in constraint_edges:
            if edge not in encoder.edge_to_coords:
                continue
            
            if compute_flip_impact is not None:
                fixed, created = compute_flip_impact(coloring, edge, vertices, constraint_type)
                net_impact = fixed - created
            else:
                # Fallback: use violation count
                edge_violations = compute_edge_violation_counts(coloring, vertices, constraint_type)
                net_impact = edge_violations.get(edge, 0)
            
            if net_impact > best_impact:
                best_impact = net_impact
                best_edge = edge
        
        if best_edge and len(edges_to_flip) < max_flips:
            edges_to_flip.add(best_edge)
    
    # Apply flips (STRONG push)
    for edge in edges_to_flip:
        coord = encoder.edge_to_coords[edge]
        cell = system.get_cell(coord)
        if cell is None:
            continue
        
        # Flip edge (STRONG push)
        current_color = coloring.get(edge, 0)
        new_color = 1 - current_color
        target = +15.0 if new_color == 1 else -15.0
        
        # STRONG push: 90% target, 10% current
        cell.symbolic_weight = 0.1 * cell.symbolic_weight + 0.9 * target
    
    return len(edges_to_flip)


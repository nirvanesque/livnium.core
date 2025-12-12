"""
Geometric SW Inversion: Push from 99.12% → 99.5%+ → 100%

This module implements intelligent geometric inversion that:
- Uses constraint structure to identify critical edges
- Inverts SW field geometrically (not randomly)
- Preserves good structure while fixing violations
- Targets the final 20-22 violations specifically

The key insight: At 99.12%, we're in a deep basin. Random resets destroy this.
Instead, we need geometric inversion that respects the constraint topology.
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
    try:
        from .ramsey_curvature_healing import compute_flip_impact
    except ImportError:
        compute_flip_impact = None
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


def find_critical_violated_constraints(
    coloring: Coloring,
    vertices: List[int],
    constraint_type: str = "k4"
) -> List[Tuple[Set[Edge], int]]:
    """
    Find violated constraints and rank them by "criticality".
    
    Criticality = how many other violated constraints share edges with this one.
    High criticality = fixing this constraint helps fix many others.
    
    Returns: List of (constraint_edges, criticality_score) tuples, sorted by criticality.
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
        constraint_edges = set()
        for i in range(len(constraint)):
            for j in range(i + 1, len(constraint)):
                edge = (min(constraint[i], constraint[j]), max(constraint[i], constraint[j]))
                if edge in coloring:
                    constraint_edges.add(edge)
        
        # Check if monochromatic (violated)
        if len(constraint_edges) == len(constraint) * (len(constraint) - 1) // 2:
            edge_colors = [coloring.get(e, 0) for e in constraint_edges]
            if len(set(edge_colors)) == 1:
                violated_constraints.append(constraint_edges)
    
    # Compute criticality: how many other violated constraints share edges?
    constraint_criticality = []
    for i, constraint_edges in enumerate(violated_constraints):
        criticality = 0
        for j, other_edges in enumerate(violated_constraints):
            if i != j:
                # Count shared edges
                shared = len(constraint_edges & other_edges)
                criticality += shared
        constraint_criticality.append((constraint_edges, criticality))
    
    # Sort by criticality (highest first)
    constraint_criticality.sort(reverse=True, key=lambda x: x[1])
    
    return constraint_criticality


def geometric_sw_inversion(
    system,
    encoder: RamseyEncoder,
    coloring: Coloring,
    vertices: List[int],
    constraint_type: str = "k4",
    max_inversions: int = 25
) -> int:
    """
    Perform geometric SW inversion to push from 99.12% → 99.5%+.
    
    Strategy:
    1. Find critical violated constraints (those that share edges with many others)
    2. For each critical constraint, find the edge with BEST flip impact
    3. Invert that edge's SW geometrically (strong push, preserves structure)
    4. This fixes multiple violations simultaneously
    
    Returns: Number of edges inverted
    """
    # Find critical violated constraints
    critical_constraints = find_critical_violated_constraints(
        coloring, vertices, constraint_type
    )
    
    if not critical_constraints:
        return 0  # No violations
    
    edges_to_invert = set()
    
    # Process constraints by criticality (most critical first)
    for constraint_edges, criticality in critical_constraints:
        if len(edges_to_invert) >= max_inversions:
            break
        
        # Find best edge to flip in this constraint
        best_edge = None
        best_impact = float('-inf')
        
        for edge in constraint_edges:
            if edge not in encoder.edge_to_coords:
                continue
            
            if edge in edges_to_invert:
                continue  # Already queued
            
            # Compute flip impact
            if compute_flip_impact is not None:
                fixed, created = compute_flip_impact(coloring, edge, vertices, constraint_type)
                net_impact = fixed - created
            else:
                # Fallback: use violation count
                edge_violations = compute_edge_violation_counts(coloring, vertices, constraint_type)
                net_impact = edge_violations.get(edge, 0)
            
            # Weight by criticality (fixing critical constraints is more valuable)
            weighted_impact = net_impact * (1.0 + criticality / 10.0)
            
            if weighted_impact > best_impact:
                best_impact = weighted_impact
                best_edge = edge
        
        # Invert best edge if it has positive impact
        if best_edge and best_impact > 0:
            edges_to_invert.add(best_edge)
    
    # Apply geometric inversion (STRONG push, preserves structure)
    for edge in edges_to_invert:
        coord = encoder.edge_to_coords[edge]
        cell = system.get_cell(coord)
        if cell is None:
            continue
        
        # Get current color
        current_color = coloring.get(edge, 0)
        new_color = 1 - current_color
        
        # Geometric inversion: STRONG push (95% target, 5% current)
        # Use larger magnitude to break through to deeper basin
        target = +20.0 if new_color == 1 else -20.0  # Stronger than normal
        
        # VERY STRONG push: 95% target, 5% current
        cell.symbolic_weight = 0.05 * cell.symbolic_weight + 0.95 * target
    
    return len(edges_to_invert)


def targeted_violation_fix(
    system,
    encoder: RamseyEncoder,
    coloring: Coloring,
    vertices: List[int],
    constraint_type: str = "k4",
    target_violations: int = 20
) -> int:
    """
    Target the final violations specifically (99.12% → 99.5%+).
    
    This is more aggressive than geometric_inversion:
    - Finds ALL violated constraints
    - For each, flips the edge with maximum impact
    - Continues until target violations reached
    
    Returns: Number of edges flipped
    """
    # Get violation counts
    edge_violations = compute_edge_violation_counts(coloring, vertices, constraint_type)
    
    if not edge_violations:
        return 0  # No violations
    
    # Count current violations
    if constraint_type == "k3":
        current_violations = count_monochromatic_k3(coloring, vertices)
    else:
        current_violations = count_monochromatic_k4(coloring, vertices)
    
    if current_violations <= target_violations:
        return 0  # Already at or below target
    
    # Find critical constraints
    critical_constraints = find_critical_violated_constraints(
        coloring, vertices, constraint_type
    )
    
    edges_to_flip = set()
    violations_fixed = 0
    
    # Process constraints until we hit target
    for constraint_edges, criticality in critical_constraints:
        if violations_fixed >= (current_violations - target_violations):
            break  # Reached target
        
        # Find best edge in this constraint
        best_edge = None
        best_impact = float('-inf')
        
        for edge in constraint_edges:
            if edge not in encoder.edge_to_coords:
                continue
            
            if edge in edges_to_flip:
                continue
            
            if compute_flip_impact is not None:
                fixed, created = compute_flip_impact(coloring, edge, vertices, constraint_type)
                net_impact = fixed - created
            else:
                net_impact = edge_violations.get(edge, 0)
            
            # Weight by criticality
            weighted_impact = net_impact * (1.0 + criticality / 10.0)
            
            if weighted_impact > best_impact:
                best_impact = weighted_impact
                best_edge = edge
        
        # Flip best edge if it helps
        if best_edge and best_impact > 0:
            edges_to_flip.add(best_edge)
            violations_fixed += int(best_impact)  # Estimate violations fixed
    
    # Apply flips
    for edge in edges_to_flip:
        coord = encoder.edge_to_coords[edge]
        cell = system.get_cell(coord)
        if cell is None:
            continue
        
        # Strong geometric inversion
        current_color = coloring.get(edge, 0)
        new_color = 1 - current_color
        target = +20.0 if new_color == 1 else -20.0
        
        # VERY STRONG push: 95% target, 5% current
        cell.symbolic_weight = 0.05 * cell.symbolic_weight + 0.95 * target
    
    return len(edges_to_flip)


def deep_basin_descent(
    system,
    encoder: RamseyEncoder,
    coloring: Coloring,
    vertices: List[int],
    constraint_type: str = "k4",
    current_percent: float = 99.0
) -> int:
    """
    Deep basin descent: Push from 99.12% → 99.5%+ → 100%.
    
    This is the final push mechanism:
    - Uses geometric inversion for structure preservation
    - Targets critical constraints first
    - Applies strong SW pushes to break into deeper basin
    
    Returns: Number of edges inverted
    """
    # Determine strategy based on current percentage
    if current_percent >= 99.5:
        # Very close - use targeted fix
        return targeted_violation_fix(
            system, encoder, coloring, vertices,
            constraint_type=constraint_type,
            target_violations=10  # Aim for <10 violations
        )
    elif current_percent >= 99.0:
        # In deep basin - use geometric inversion
        return geometric_sw_inversion(
            system, encoder, coloring, vertices,
            constraint_type=constraint_type,
            max_inversions=30  # More aggressive
        )
    else:
        # Not in deep basin yet - use standard inversion
        return geometric_sw_inversion(
            system, encoder, coloring, vertices,
            constraint_type=constraint_type,
            max_inversions=20
        )


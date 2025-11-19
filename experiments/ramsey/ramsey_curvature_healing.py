"""
Curvature-Guided Multi-Edge Healing for Ramsey Solver

This module implements the "compass" that gives the solver direction:
- Curvature-guided healing (geometric signals, not just violation counts)
- Multi-edge flips (heal clusters simultaneously)
- Global coherence (avoid breaking what's already good)
- Replace SW-based inference with constraint-based inference

The goal: Stop the chaotic oscillations and restore gradient descent.
"""

from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import numpy as np

try:
    from .ramsey_encoder import RamseyEncoder
    from .ramsey_tension import (
        compute_ramsey_tension,
        get_all_k3_subsets,
        get_all_k4_subsets,
    )
    from .ramsey_local_feedback_patch import compute_edge_violation_counts
except ImportError:
    from ramsey_encoder import RamseyEncoder
    from ramsey_tension import (
        compute_ramsey_tension,
        get_all_k3_subsets,
        get_all_k4_subsets,
    )
    from ramsey_local_feedback_patch import compute_edge_violation_counts

Edge = Tuple[int, int]
Coloring = Dict[Edge, int]


def compute_edge_curvature(
    encoder: RamseyEncoder,
    coloring: Coloring,
    vertices: List[int],
    constraint_type: str = "k4"
) -> Dict[Edge, float]:
    """
    Compute geometric curvature for each edge.
    
    Curvature = how "steep" the energy landscape is around this edge.
    High curvature = edge is in a conflicted region (many constraints pulling different ways).
    Low curvature = edge is in a stable region (constraints agree).
    
    This replaces SW-based inference with constraint-based geometric signals.
    """
    edge_curvature = {}
    
    # Get all constraints
    if constraint_type == "k3":
        constraints = get_all_k3_subsets(vertices)
    else:
        constraints = get_all_k4_subsets(vertices)
    
    # For each edge, compute how many constraints it participates in
    # and how many of those are violated
    edge_constraint_counts = defaultdict(int)
    edge_violation_counts = defaultdict(int)
    
    for constraint in constraints:
        # Get edges in this constraint
        constraint_edges = []
        for i in range(len(constraint)):
            for j in range(i + 1, len(constraint)):
                edge = (min(constraint[i], constraint[j]), max(constraint[i], constraint[j]))
                constraint_edges.append(edge)
        
        # Check if constraint is violated (monochromatic)
        edge_colors = [coloring.get(e, 0) for e in constraint_edges if e in coloring]
        if len(edge_colors) == len(constraint_edges):
            is_monochromatic = len(set(edge_colors)) == 1
            if is_monochromatic:
                # Constraint is violated
                for edge in constraint_edges:
                    edge_violation_counts[edge] += 1
                    edge_constraint_counts[edge] += 1
            else:
                # Constraint is satisfied
                for edge in constraint_edges:
                    edge_constraint_counts[edge] += 1
    
    # Curvature = violation_count / total_constraint_count
    # High curvature = many violations relative to total constraints
    for edge in encoder.edges:
        total_constraints = edge_constraint_counts.get(edge, 1)
        violations = edge_violation_counts.get(edge, 0)
        
        # Curvature: 0.0 = flat (no violations), 1.0 = maximum conflict
        curvature = violations / max(total_constraints, 1)
        edge_curvature[edge] = curvature
    
    return edge_curvature


def find_violation_clusters(
    coloring: Coloring,
    vertices: List[int],
    constraint_type: str = "k4",
    min_cluster_size: int = 3
) -> List[Set[Edge]]:
    """
    Find clusters of violated constraints.
    
    A cluster is a set of edges that participate in overlapping violated constraints.
    These should be healed together (multi-edge flip) to maintain coherence.
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
                violated_constraints.append(set(constraint_edges))
    
    # Build graph of overlapping violated constraints
    # Two constraints overlap if they share an edge
    clusters = []
    used_constraints = set()
    
    for i, constraint_edges in enumerate(violated_constraints):
        if i in used_constraints:
            continue
        
        # Start a new cluster
        cluster = set(constraint_edges)
        used_constraints.add(i)
        
        # Merge overlapping constraints
        changed = True
        while changed:
            changed = False
            for j, other_edges in enumerate(violated_constraints):
                if j in used_constraints:
                    continue
                
                # Check if they overlap
                if cluster & other_edges:
                    cluster |= other_edges
                    used_constraints.add(j)
                    changed = True
        
        # Only keep clusters above minimum size
        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
    
    return clusters


def compute_flip_impact(
    coloring: Coloring,
    edge: Edge,
    vertices: List[int],
    constraint_type: str = "k4"
) -> Tuple[int, int]:
    """
    Compute the impact of flipping an edge.
    
    Returns: (violations_fixed, violations_created)
    
    This tells us: if we flip this edge, how many constraints get fixed vs broken?
    """
    # Create test coloring with flipped edge
    test_coloring = coloring.copy()
    test_coloring[edge] = 1 - test_coloring.get(edge, 0)
    
    # Count violations before and after
    if constraint_type == "k3":
        constraints = get_all_k3_subsets(vertices)
    else:
        constraints = get_all_k4_subsets(vertices)
    
    violations_before = 0
    violations_after = 0
    
    for constraint in constraints:
        # Get edges in this constraint
        constraint_edges = []
        for i in range(len(constraint)):
            for j in range(i + 1, len(constraint)):
                e = (min(constraint[i], constraint[j]), max(constraint[i], constraint[j]))
                if e in coloring:
                    constraint_edges.append(e)
        
        if len(constraint_edges) == len(constraint):
            # Check before
            colors_before = [coloring.get(e, 0) for e in constraint_edges]
            if len(set(colors_before)) == 1:
                violations_before += 1
            
            # Check after
            colors_after = [test_coloring.get(e, 0) for e in constraint_edges]
            if len(set(colors_after)) == 1:
                violations_after += 1
    
    violations_fixed = violations_before - violations_after
    violations_created = violations_after - violations_before
    
    return violations_fixed, violations_created


def heal_with_curvature_guidance(
    system,
    encoder: RamseyEncoder,
    coloring: Coloring,
    vertices: List[int],
    constraint_type: str = "k4",
    max_edges_to_flip: int = 10,
    curvature_threshold: float = 0.3
) -> int:
    """
    Heal violations using curvature-guided multi-edge flips.
    
    This is the "compass" that gives the solver direction:
    1. Compute curvature for each edge (geometric signal)
    2. Find violation clusters (edges that should be healed together)
    3. For each cluster, compute flip impact (will this help or hurt?)
    4. Flip edges that have positive net impact (fix more than they break)
    
    Returns: Number of edges flipped
    """
    # Compute curvature (geometric signal, not SW)
    edge_curvature = compute_edge_curvature(encoder, coloring, vertices, constraint_type)
    
    # Find violation clusters (edges that participate in overlapping violations)
    clusters = find_violation_clusters(coloring, vertices, constraint_type, min_cluster_size=2)
    
    edges_flipped = 0
    edges_to_flip = set()
    
    # Process clusters by curvature (highest first)
    cluster_curvature = []
    for cluster in clusters:
        # Average curvature of edges in cluster
        cluster_avg_curvature = np.mean([edge_curvature.get(e, 0.0) for e in cluster])
        cluster_curvature.append((cluster_avg_curvature, cluster))
    
    # Sort by curvature (descending)
    cluster_curvature.sort(reverse=True, key=lambda x: x[0])
    
    # Heal clusters with high curvature
    for avg_curvature, cluster in cluster_curvature:
        if avg_curvature < curvature_threshold:
            continue  # Skip low-curvature clusters (already stable)
        
        if edges_flipped >= max_edges_to_flip:
            break
        
        # For each edge in cluster, compute flip impact
        edge_impacts = []
        for edge in cluster:
            if edge not in encoder.edge_to_coords:
                continue
            
            fixed, created = compute_flip_impact(coloring, edge, vertices, constraint_type)
            net_impact = fixed - created  # Positive = good, negative = bad
            
            edge_impacts.append((net_impact, fixed, created, edge))
        
        # Sort by net impact (best first)
        edge_impacts.sort(reverse=True, key=lambda x: x[0])
        
        # Flip edges with positive net impact
        for net_impact, fixed, created, edge in edge_impacts:
            if net_impact <= 0:
                break  # No more beneficial flips
            
            if edges_flipped >= max_edges_to_flip:
                break
            
            if edge in edges_to_flip:
                continue  # Already queued
            
            # Queue this edge for flipping
            edges_to_flip.add(edge)
            edges_flipped += 1
    
    # Apply flips
    for edge in edges_to_flip:
        if edge not in encoder.edge_to_coords:
            continue
        
        coord = encoder.edge_to_coords[edge]
        cell = system.get_cell(coord)
        if cell is None:
            continue
        
        # Flip edge (strong push)
        current_color = coloring.get(edge, 0)
        new_color = 1 - current_color
        target = +10.0 if new_color == 1 else -10.0
        
        # STRONG push: 90% target, 10% current
        cell.symbolic_weight = 0.1 * cell.symbolic_weight + 0.9 * target
    
    return edges_flipped


def heal_with_global_coherence(
    system,
    encoder: RamseyEncoder,
    coloring: Coloring,
    vertices: List[int],
    constraint_type: str = "k4",
    max_edges_to_flip: int = 15
) -> int:
    """
    Heal violations while maintaining global coherence.
    
    This version:
    1. Computes flip impact for ALL edges (not just in clusters)
    2. Selects edges with best net impact
    3. Ensures flips don't conflict with each other
    
    This is more aggressive but maintains coherence.
    """
    # Get violation counts (which edges are hot patches)
    edge_violations = compute_edge_violation_counts(coloring, vertices, constraint_type)
    
    if not edge_violations:
        return 0  # No violations
    
    # Compute flip impact for all edges with violations
    edge_candidates = []
    for edge, violation_count in edge_violations.items():
        if edge not in encoder.edge_to_coords:
            continue
        
        fixed, created = compute_flip_impact(coloring, edge, vertices, constraint_type)
        net_impact = fixed - created
        
        # Score = net_impact weighted by violation count
        score = net_impact * (1.0 + violation_count / 10.0)
        
        edge_candidates.append((score, net_impact, fixed, created, edge))
    
    # Sort by score (best first)
    edge_candidates.sort(reverse=True, key=lambda x: x[0])
    
    # Select edges with positive net impact
    edges_to_flip = []
    for score, net_impact, fixed, created, edge in edge_candidates:
        if net_impact <= 0:
            break  # No more beneficial flips
        
        if len(edges_to_flip) >= max_edges_to_flip:
            break
        
        edges_to_flip.append(edge)
    
    # Apply flips
    for edge in edges_to_flip:
        coord = encoder.edge_to_coords[edge]
        cell = system.get_cell(coord)
        if cell is None:
            continue
        
        # Flip edge (strong push)
        current_color = coloring.get(edge, 0)
        new_color = 1 - current_color
        target = +10.0 if new_color == 1 else -10.0
        
        # STRONG push: 90% target, 10% current
        cell.symbolic_weight = 0.1 * cell.symbolic_weight + 0.9 * target
    
    return len(edges_to_flip)


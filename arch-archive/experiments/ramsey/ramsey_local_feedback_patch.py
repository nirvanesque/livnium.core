"""
Ramsey Local Feedback Patch: Edge-Level K₄ Violation Counts

This patch adds edge-level violation counting to guide healing.
An edge that's part of 8 violated K₄s gets higher priority than one in 1.

Key Principle:
- target_SW(edge) = -λ * (#violated_K4s_that_include_this_edge)
- This creates a tension gradient pointing to the real contradictions
"""

from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Handle relative imports
try:
    from .ramsey_tension import (
        get_all_k4_subsets,
        get_k4_edges,
        count_monochromatic_k4,
    )
    from .ramsey_encoder import RamseyEncoder
except ImportError:
    from ramsey_tension import (
        get_all_k4_subsets,
        get_k4_edges,
        count_monochromatic_k4,
    )
    from ramsey_encoder import RamseyEncoder

Edge = Tuple[int, int]
Coloring = Dict[Edge, int]


def compute_edge_violation_counts(
    coloring: Coloring,
    vertices: List[int],
    constraint_type: str = "k4"
) -> Dict[Edge, int]:
    """
    Count how many violated K₄s (or K₃s) each edge is part of.
    
    Returns:
        Dictionary mapping edge → number of violated constraints containing it
    """
    edge_violations = defaultdict(int)
    
    if constraint_type == "k4":
        k4s = get_all_k4_subsets(vertices)
        for quad in k4s:
            edges = get_k4_edges(quad)
            if any(e not in coloring for e in edges):
                continue
            
            # Check if this K₄ is monochromatic (violation)
            colors = {coloring[e] for e in edges}
            if len(colors) == 1:  # All same color = violation
                # Count this violation for all edges in this K₄
                for e in edges:
                    edge_violations[e] += 1
    
    elif constraint_type == "k3":
        try:
            from .ramsey_tension import get_all_k3_subsets, get_k3_edges
        except ImportError:
            from ramsey_tension import get_all_k3_subsets, get_k3_edges
        triangles = get_all_k3_subsets(vertices)
        for tri in triangles:
            edges = get_k3_edges(tri)
            if any(e not in coloring for e in edges):
                continue
            
            # Check if this K₃ is monochromatic (violation)
            colors = {coloring[e] for e in edges}
            if len(colors) == 1:  # All same color = violation
                # Count this violation for all edges in this triangle
                for e in edges:
                    edge_violations[e] += 1
    
    return dict(edge_violations)


def apply_local_feedback(
    system,
    encoder: RamseyEncoder,
    coloring: Coloring,
    vertices: List[int],
    constraint_type: str = "k4",
    lambda_weight: float = 0.5
):
    """
    Apply local feedback: push edges away from violated constraints.
    
    For each edge:
      - If edge is in N violated K₄s → push SW away from current color
      - Push strength ∝ N (more violations = stronger push)
    
    Args:
        system: LivniumCoreSystem
        encoder: RamseyEncoder
        coloring: Current coloring
        vertices: List of vertices
        constraint_type: "k3" or "k4"
        lambda_weight: Strength of feedback (0.0 = no effect, 1.0 = full push)
    """
    # Compute violation counts per edge
    edge_violations = compute_edge_violation_counts(
        coloring, vertices, constraint_type
    )
    
    # Apply feedback to each edge
    for edge, violation_count in edge_violations.items():
        if violation_count == 0:
            continue  # No violations, no push needed
        
        if edge not in encoder.edge_to_coords:
            continue
        
        coord = encoder.edge_to_coords[edge]
        cell = system.get_cell(coord)
        if cell is None:
            continue
        
        # Current color (from SW)
        current_sw = cell.symbolic_weight
        current_color = 1 if current_sw >= 0.0 else 0
        
        # Push toward opposite color
        # Strength = lambda_weight * violation_count
        # Normalize by max possible violations (for K₄ in K₁₇, max is ~105)
        max_violations = 105 if constraint_type == "k4" else 10
        normalized_count = min(violation_count / max_violations, 1.0)
        push_strength = lambda_weight * normalized_count
        
        # Target: opposite color
        target_sw = -10.0 if current_color == 1 else +10.0
        
        # Apply push: blend current SW toward target
        # push_strength = 0.0 → no change
        # push_strength = 1.0 → full flip
        new_sw = (1.0 - push_strength) * current_sw + push_strength * target_sw
        cell.symbolic_weight = new_sw


def heal_with_violation_priority(
    system,
    encoder: RamseyEncoder,
    coloring: Coloring,
    vertices: List[int],
    constraint_type: str = "k4",
    max_heal: int = 10
) -> int:
    """
    Heal violations, prioritizing edges with MOST violations.
    
    This replaces the old heuristic (largest |SW|) with violation count.
    
    Returns:
        Number of violations healed
    """
    # Compute violation counts per edge
    edge_violations = compute_edge_violation_counts(
        coloring, vertices, constraint_type
    )
    
    if not edge_violations:
        return 0  # No violations
    
    # Sort edges by violation count (descending)
    sorted_edges = sorted(
        edge_violations.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    healed = 0
    
    for edge, violation_count in sorted_edges:
        if healed >= max_heal:
            break
        
        if edge not in encoder.edge_to_coords:
            continue
        
        coord = encoder.edge_to_coords[edge]
        cell = system.get_cell(coord)
        if cell is None:
            continue
        
        # Flip this edge (strong flip)
        current_color = coloring.get(edge, 0)
        new_color = 1 - current_color
        target = +10.0 if new_color == 1 else -10.0
        
        # STRONG push: 90% target, 10% current
        cell.symbolic_weight = 0.1 * cell.symbolic_weight + 0.9 * target
        
        healed += 1
    
    return healed


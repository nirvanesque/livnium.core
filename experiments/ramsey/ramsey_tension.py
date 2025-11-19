"""
Ramsey Tension: K₃ and K₄ Constraint Tension Fields

Computes tension based on monochromatic K₃ (triangles) and K₄ violations.
This bridges logical Ramsey constraints with geometric tension.

R(3,3) uses K₃ (triangles)
R(4,4) uses K₄ (4-cliques)
"""

from itertools import combinations
from typing import Dict, Tuple, List

Edge = Tuple[int, int]
Coloring = Dict[Edge, int]


def _norm_edge(u: int, v: int) -> Edge:
    """Always store edges sorted so (i,j) == (j,i)."""
    return (u, v) if u < v else (v, u)


# ---------- K3 (triangle) utilities ----------

def get_all_k3_subsets(vertices: List[int]) -> List[Tuple[int, int, int]]:
    """All triangles on vertex set."""
    return list(combinations(vertices, 3))


def get_k3_edges(triangle: Tuple[int, int, int]) -> List[Edge]:
    """Edges of a triangle (a,b,c)."""
    a, b, c = triangle
    return [
        _norm_edge(a, b),
        _norm_edge(a, c),
        _norm_edge(b, c),
    ]


def count_monochromatic_k3(coloring: Coloring, vertices: List[int]) -> int:
    """
    Count monochromatic triangles K3.

    A K3 = (a,b,c) is a violation if all 3 edges have the same color.
    """
    triangles = get_all_k3_subsets(vertices)
    violations = 0

    for tri in triangles:
        e1, e2, e3 = get_k3_edges(tri)
        # If any edge missing from coloring, skip this triangle
        if e1 not in coloring or e2 not in coloring or e3 not in coloring:
            continue
        c1 = coloring[e1]
        c2 = coloring[e2]
        c3 = coloring[e3]
        if c1 == c2 == c3:
            violations += 1

    return violations


# ---------- K4 (K4-based) utilities ----------

def get_all_k4_subsets(vertices: List[int]) -> List[Tuple[int, int, int, int]]:
    """All 4-vertex subsets (potential K₄s)."""
    return list(combinations(vertices, 4))


def get_k4_edges(k4: Tuple[int, int, int, int]) -> List[Edge]:
    """Edges of a K₄ (4-vertex complete subgraph)."""
    a, b, c, d = k4
    return [
        _norm_edge(a, b),
        _norm_edge(a, c),
        _norm_edge(a, d),
        _norm_edge(b, c),
        _norm_edge(b, d),
        _norm_edge(c, d),
    ]


def count_monochromatic_k4(coloring: Coloring, vertices: List[int]) -> int:
    """
    Count monochromatic K₄ subgraphs.
    
    A K₄ is monochromatic if all 6 edges have the same color.
    """
    k4s = get_all_k4_subsets(vertices)
    violations = 0

    for quad in k4s:
        edges = get_k4_edges(quad)
        # Require all edges present
        if any(e not in coloring for e in edges):
            continue
        colors = {coloring[e] for e in edges}
        if len(colors) == 1:  # all same color
            violations += 1

    return violations


# ---------- Unified Ramsey tension / validation ----------

def compute_ramsey_tension(
    coloring: Coloring,
    vertices: List[int],
    constraint_type: str = "k4",
) -> float:
    """
    Tension in [0,1]: fraction of violated constraints.

    constraint_type:
      - "k3": R(3,3)-style (triangles)
      - "k4": R(4,4)-style (K4s)
    """
    n = len(vertices)

    if constraint_type == "k3":
        total = 0
        triangles = get_all_k3_subsets(vertices)
        for tri in triangles:
            e1, e2, e3 = get_k3_edges(tri)
            if e1 not in coloring or e2 not in coloring or e3 not in coloring:
                continue
            total += 1
        if total == 0:
            return 0.0
        violations = count_monochromatic_k3(coloring, vertices)
        return violations / total

    elif constraint_type == "k4":
        total = 0
        k4s = get_all_k4_subsets(vertices)
        for quad in k4s:
            edges = get_k4_edges(quad)
            if any(e not in coloring for e in edges):
                continue
            total += 1
        if total == 0:
            return 0.0
        violations = count_monochromatic_k4(coloring, vertices)
        return violations / total

    else:
        raise ValueError(f"Unknown constraint_type: {constraint_type}")


def is_valid_ramsey_coloring(
    coloring: Coloring,
    vertices: List[int],
    constraint_type: str = "k4",
) -> bool:
    """Check if coloring is valid (no monochromatic subgraphs)."""
    if constraint_type == "k3":
        return count_monochromatic_k3(coloring, vertices) == 0
    elif constraint_type == "k4":
        return count_monochromatic_k4(coloring, vertices) == 0
    else:
        raise ValueError(f"Unknown constraint_type: {constraint_type}")


def ramsey_score(
    coloring: Coloring,
    vertices: List[int],
    constraint_type: str = "k4",
) -> float:
    """
    Simple scalar score: higher is better.

    score = 1 - tension  (so 1.0 = perfect; 0.0 = all constraints violated)
    """
    tension = compute_ramsey_tension(coloring, vertices, constraint_type)
    return 1.0 - tension

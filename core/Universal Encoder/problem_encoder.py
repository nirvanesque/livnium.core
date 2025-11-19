"""
Universal Problem Encoder: Convert Any Problem to Geometry

This is the main interface for encoding problems into geometric patterns:
- Constraints → Tension fields (energy landscape)
- Solutions → Basins (candidate attractors)

The encoder produces:
1. Tension fields (from constraints)
2. Candidate basins (from solution space)

These are then passed to multi-basin search.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from core.classical.livnium_core_system import LivniumCoreSystem
from .constraint_encoder import ConstraintEncoder, TensionField


@dataclass
class EncodedProblem:
    """
    Result of encoding a problem into geometry.
    
    Contains:
    - Tension fields (from constraints)
    - Candidate basins (from solution space)
    - Variable mappings (for decoding)
    """
    tension_fields: List[TensionField]
    candidate_basins: List[List[Tuple[int, int, int]]]
    variable_mappings: Dict[str, List[Tuple[int, int, int]]]
    problem_type: str
    metadata: Dict[str, Any]


class UniversalProblemEncoder:
    """
    Universal encoder that converts any problem into geometric patterns.
    
    Architecture:
    - Constraints → Tension fields (energy landscape)
    - Solutions → Basins (candidate attractors)
    """
    
    def __init__(self, system: LivniumCoreSystem):
        """
        Initialize universal problem encoder.
        
        Args:
            system: LivniumCoreSystem
        """
        self.system = system
        self.constraint_encoder = ConstraintEncoder(system)
    
    def encode(
        self,
        problem: Dict[str, Any]
    ) -> EncodedProblem:
        """
        Encode a problem into geometry.
        
        Args:
            problem: Problem specification with:
                - 'type': Problem type (e.g., 'graph_coloring', 'sat', 'ramsey')
                - 'variables': Variable definitions
                - 'constraints': Constraint definitions
                - 'candidates': Candidate solutions (optional)
                - Other problem-specific fields
        
        Returns:
            EncodedProblem with tension fields and candidate basins
        """
        problem_type = problem.get('type', 'unknown')
        
        # Route to specific encoder based on problem type
        if problem_type == 'graph_coloring':
            return self._encode_graph_coloring(problem)
        elif problem_type == 'sat':
            return self._encode_sat(problem)
        elif problem_type == 'ramsey':
            return self._encode_ramsey(problem)
        elif problem_type == 'constraint_satisfaction':
            return self._encode_constraint_satisfaction(problem)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
    
    def _encode_graph_coloring(
        self,
        problem: Dict[str, Any]
    ) -> EncodedProblem:
        """
        Encode graph coloring problem.
        
        Constraints: No monochromatic triangles (K₃)
        Solutions: Valid 2-colorings
        
        Args:
            problem: Graph coloring specification
        
        Returns:
            EncodedProblem
        """
        # Get graph structure
        vertices = problem.get('vertices', [])
        edges = problem.get('edges', [])
        
        # Map variables (edges) to coordinates
        variable_mappings = {}
        coords_list = list(self.system.lattice.keys())
        
        edge_to_coords = {}
        for i, edge in enumerate(edges):
            if i < len(coords_list):
                coords = [coords_list[i]]
                edge_to_coords[edge] = coords
                variable_mappings[f"edge_{edge}"] = coords
        
        # Encode constraints: No monochromatic triangles
        # For each triangle, create tension field
        tension_fields = []
        
        # Find all triangles
        from itertools import combinations
        triangles = list(combinations(vertices, 3))
        
        for triangle in triangles:
            # Get edges of triangle
            triangle_edges = [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[0], triangle[2])
            ]
            
            # Get coordinates for triangle edges
            triangle_coords = []
            for edge in triangle_edges:
                if edge in edge_to_coords:
                    triangle_coords.extend(edge_to_coords[edge])
            
            if not triangle_coords:
                continue
            
            # Create tension field: high tension if all edges same color
            def create_triangle_tension_fn(edge_coords_list):
                def compute_tension(system: LivniumCoreSystem) -> float:
                    # Get SW values (colors) for edges
                    sw_values = []
                    for coords in edge_coords_list:
                        cell = system.get_cell(coords)
                        if cell:
                            # Decode color: < 10 = color 0, >= 10 = color 1
                            color = 0 if cell.symbolic_weight < 10.0 else 1
                            sw_values.append(color)
                    
                    if len(sw_values) < 3:
                        return 0.0
                    
                    # Tension = 1.0 if all same color (monochromatic), 0.0 otherwise
                    if len(set(sw_values)) == 1:
                        return 1.0  # Monochromatic triangle
                    return 0.0  # Valid coloring
                
                return compute_tension
            
            field = self.constraint_encoder.encode_custom_constraint(
                constraint_id=f"triangle_{triangle}",
                involved_coords=triangle_coords,
                tension_fn=create_triangle_tension_fn(triangle_coords),
                description=f"No monochromatic triangle: {triangle}"
            )
            tension_fields.append(field)
        
        # Generate candidate basins (valid colorings)
        # For now, generate random colorings
        candidate_basins = []
        n_candidates = problem.get('n_candidates', 10)
        
        for _ in range(n_candidates):
            # Random coloring: assign random colors to edges
            basin_coords = []
            for edge, coords in edge_to_coords.items():
                basin_coords.extend(coords)
            candidate_basins.append(basin_coords)
        
        return EncodedProblem(
            tension_fields=tension_fields,
            candidate_basins=candidate_basins,
            variable_mappings=variable_mappings,
            problem_type='graph_coloring',
            metadata={'vertices': vertices, 'edges': edges}
        )
    
    def _encode_sat(
        self,
        problem: Dict[str, Any]
    ) -> EncodedProblem:
        """
        Encode SAT problem.
        
        Constraints: Clauses (tension if clause unsatisfied)
        Solutions: Variable assignments (basins)
        
        Args:
            problem: SAT specification
        
        Returns:
            EncodedProblem
        """
        # TODO: Implement SAT encoding
        raise NotImplementedError("SAT encoding not yet implemented")
    
    def _encode_ramsey(
        self,
        problem: Dict[str, Any]
    ) -> EncodedProblem:
        """
        Encode Ramsey problem.
        
        Constraints: No monochromatic K₄ (tension fields)
        Solutions: 2-colorings (basins)
        
        Args:
            problem: Ramsey specification
        
        Returns:
            EncodedProblem
        """
        # Similar to graph coloring but with K₄ constraints
        # TODO: Implement Ramsey-specific encoding
        return self._encode_graph_coloring(problem)  # For now, use graph coloring
    
    def _encode_constraint_satisfaction(
        self,
        problem: Dict[str, Any]
    ) -> EncodedProblem:
        """
        Encode general constraint satisfaction problem.
        
        Args:
            problem: CSP specification
        
        Returns:
            EncodedProblem
        """
        # TODO: Implement general CSP encoding
        raise NotImplementedError("CSP encoding not yet implemented")


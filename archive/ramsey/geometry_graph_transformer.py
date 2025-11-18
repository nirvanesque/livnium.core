"""
Geometry → Graph Transformer: The Missing Brain Stem

This module implements the critical missing layer that actually uses
Livnium Core geometry to drive graph mutations and search.

Without this, the solver is just "random + numba + thousands of individuals."
With this, geometry becomes the steering mechanism.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

from experiments.ramsey.ramsey_number_solver import RamseyGraph
from core.classical.livnium_core_system import LivniumCoreSystem, RotationAxis, CellClass
from core.classical.livnium_core_system import RotationGroup

import copy


@dataclass
class GeometryMutation:
    """Represents a geometry-driven mutation instruction."""
    edge_indices: List[Tuple[int, int]]  # Edges to mutate
    mutation_type: str  # 'flip', 'add', 'remove', 'swap'
    priority: float  # 0.0 to 1.0, based on geometry
    reason: str  # Why this mutation (for debugging)


class GeometryGraphTransformer:
    """
    Transforms graph structure based on Livnium Core geometry.
    
    This is the missing "brain stem" that wires geometry into computation.
    """
    
    def __init__(self, core_system: LivniumCoreSystem, n: int):
        """
        Initialize geometry-graph transformer.
        
        Args:
            core_system: Livnium Core System instance
            n: Number of vertices in graph
        """
        self.core_system = core_system
        self.n = n
        self.num_edges = n * (n - 1) // 2
        
        # Build edge index mapping for fast lookup
        self.edge_to_index: Dict[Tuple[int, int], int] = {}
        self.index_to_edge: List[Tuple[int, int]] = []
        idx = 0
        for u in range(n):
            for v in range(u + 1, n):
                self.edge_to_index[(u, v)] = idx
                self.index_to_edge.append((u, v))
                idx += 1
    
    def transform_graph_by_geometry(
        self,
        graph: RamseyGraph,
        old_cell_coords: Tuple[int, int, int],
        new_cell_coords: Tuple[int, int, int]
    ) -> RamseyGraph:
        """
        Transform graph structure based on cell coordinate change.
        
        This is the core missing function: geometry → graph structure.
        
        Args:
            graph: Graph to transform
            old_cell_coords: Previous cell coordinates
            new_cell_coords: New cell coordinates after rotation/motion
            
        Returns:
            Transformed graph
        """
        # Calculate motion vector in lattice space
        old_vec = np.array(old_cell_coords, dtype=float)
        new_vec = np.array(new_cell_coords, dtype=float)
        motion_vec = new_vec - old_vec
        
        # Get geometric properties
        old_cell = self.core_system.get_cell(old_cell_coords)
        new_cell = self.core_system.get_cell(new_cell_coords)
        
        old_face_exposure = old_cell.face_exposure if old_cell else 0
        new_face_exposure = new_cell.face_exposure if new_cell else 0
        old_class = old_cell.cell_class if old_cell else CellClass.CORE
        new_class = new_cell.cell_class if new_cell else CellClass.CORE
        
        # Generate geometry-driven mutations
        mutations = self._generate_geometry_mutations(
            graph, motion_vec, old_face_exposure, new_face_exposure,
            old_class, new_class
        )
        
        # Apply mutations
        transformed = graph.copy()
        for mutation in mutations:
            self._apply_mutation(transformed, mutation)
        
        return transformed
    
    def apply_rotation_to_graph_edges(
        self,
        graph: RamseyGraph,
        axis: RotationAxis,
        quarter_turns: int
    ) -> RamseyGraph:
        """
        Apply rotation to graph structure based on lattice rotation.
        
        When lattice rotates, graph edges should rotate their patterns.
        
        Args:
            graph: Graph to rotate
            axis: Rotation axis
            quarter_turns: Number of quarter-turns
            
        Returns:
            Rotated graph
        """
        # Create rotated graph
        rotated = graph.copy()
        
        # Map edges through rotation
        # Strategy: Rotate edge indices based on rotation group
        # For a graph with n vertices, we can think of edges as 2D coordinates
        # and rotate them in the edge space
        
        # Get rotation matrix for edge space
        # We'll use a simplified approach: rotate edge selection pattern
        edge_list = list(rotated.edge_coloring.keys())
        
        if len(edge_list) == 0:
            return rotated
        
        # Calculate rotation effect on edge selection
        # Higher face exposure cells → more edges affected
        # Rotation axis determines which edges rotate
        
        # For each edge, decide if it should be rotated based on geometry
        for (u, v) in edge_list:
            # 🔵 ROTATE SEMANTIC FIELD: Rotate semantic vector with geometry rotation
            sigma = getattr(rotated, 'edge_semantic_27', {}).get((u, v))
            if sigma is not None:
                # Rotate semantic field like a vector (circular shift based on polarity)
                # This gives the system semantic inertia - semantics rotate with geometry
                rotated_sigma = np.roll(sigma, int(quarter_turns * 7)) % 1.0
                current_color = rotated.get_edge_color(u, v)
                if current_color is not None:
                    rotated.set_edge_color(u, v, current_color, sigma27=rotated_sigma)
            
            # Map edge to geometric space
            edge_coords = self._edge_to_geometry_coords(u, v)
            
            # Rotate edge coordinates
            rotated_coords = RotationGroup.rotate_coordinates(
                edge_coords, axis, quarter_turns
            )
            
            # Determine if this edge should change
            # Higher face exposure → more likely to rotate
            cell = self.core_system.get_cell(rotated_coords)
            if cell:
                rotation_probability = cell.face_exposure / 3.0
                
                if np.random.random() < rotation_probability:
                    # Rotate this edge's color
                    current_color = rotated.get_edge_color(u, v)
                    if current_color is not None:
                        # Rotate color: 0→1 or 1→0 based on rotation direction
                        if quarter_turns % 2 == 1:  # 90° or 270° = flip
                            rotated.set_edge_color(u, v, 1 - current_color)
        
        return rotated
    
    def geometry_mutation(
        self,
        graph: RamseyGraph,
        cell_class: CellClass,
        face_exposure: int,
        polarity: float,
        motion_vec: Optional[Tuple[float, float, float]] = None
    ) -> RamseyGraph:
        """
        Apply geometry-driven mutation based on cell properties.
        
        This is the structured mutation that uses geometry, not randomness.
        
        Args:
            graph: Graph to mutate
            cell_class: Cell class (Core/Center/Edge/Corner)
            face_exposure: Face exposure (0-3)
            polarity: Semantic polarity (-1 to 1)
            motion_vec: Optional motion vector
            
        Returns:
            Mutated graph
        """
        mutated = graph.copy()
        
        # Generate mutations based on geometry
        mutations = self._generate_structured_mutations(
            graph, cell_class, face_exposure, polarity, motion_vec
        )
        
        # Apply mutations in priority order
        mutations.sort(key=lambda m: m.priority, reverse=True)
        for mutation in mutations[:max(1, int(len(mutations) * 0.3))]:  # Top 30%
            self._apply_mutation(mutated, mutation)
        
        return mutated
    
    def _generate_geometry_mutations(
        self,
        graph: RamseyGraph,
        motion_vec: np.ndarray,
        old_face_exposure: int,
        new_face_exposure: int,
        old_class: CellClass,
        new_class: CellClass
    ) -> List[GeometryMutation]:
        """Generate mutations based on geometry change."""
        mutations = []
        
        # Motion magnitude determines mutation intensity
        motion_magnitude = np.linalg.norm(motion_vec)
        
        # Face exposure change → structural change
        exposure_delta = new_face_exposure - old_face_exposure
        
        # Class change → different mutation strategy
        if old_class != new_class:
            # Major structural change
            num_mutations = int(motion_magnitude * 10) + abs(exposure_delta) * 2
        else:
            # Refinement
            num_mutations = max(1, int(motion_magnitude * 5))
        
        # Select edges to mutate based on motion direction
        edge_list = list(graph.edge_coloring.keys())
        if len(edge_list) == 0:
            return mutations
        
        # Map motion vector to edge space
        for i, (u, v) in enumerate(edge_list):
            # Calculate edge "position" in geometric space
            edge_pos = self._edge_to_geometry_coords(u, v)
            
            # Project motion onto edge
            edge_vec = np.array(edge_pos, dtype=float)
            projection = np.dot(motion_vec, edge_vec) / (np.linalg.norm(edge_vec) + 1e-10)
            
            # Higher projection → higher priority for mutation
            priority = abs(projection) * (1.0 + new_face_exposure / 3.0)
            
            if priority > 0.1:  # Threshold
                mutations.append(GeometryMutation(
                    edge_indices=[(u, v)],
                    mutation_type='flip',
                    priority=priority,
                    reason=f"motion_projection={projection:.3f}, exposure={new_face_exposure}"
                ))
        
        return mutations
    
    def _generate_structured_mutations(
        self,
        graph: RamseyGraph,
        cell_class: CellClass,
        face_exposure: int,
        polarity: float,
        motion_vec: Optional[Tuple[float, float, float]]
    ) -> List[GeometryMutation]:
        """Generate structured mutations based on cell properties."""
        mutations = []
        
        edge_list = list(graph.edge_coloring.keys())
        if len(edge_list) == 0:
            return mutations
        
        # Cell class determines mutation strategy
        if cell_class == CellClass.CORNER:
            # Corners: Aggressive exploration
            mutation_rate = 0.3
            mutation_type = 'flip'
        elif cell_class == CellClass.EDGE:
            # Edges: Moderate exploration
            mutation_rate = 0.2
            mutation_type = 'flip'
        elif cell_class == CellClass.CENTER:
            # Centers: Refinement
            mutation_rate = 0.1
            mutation_type = 'flip'
        else:  # CORE
            # Core: Minimal changes
            mutation_rate = 0.05
            mutation_type = 'flip'
        
        # Polarity modulates mutation
        if polarity > 0.5:
            # Moving toward solution: gentle refinement
            mutation_rate *= 0.5
        elif polarity < -0.5:
            # Moving away: aggressive escape
            mutation_rate *= 2.0
        
        # Face exposure modulates intensity
        mutation_rate *= (1.0 + face_exposure / 3.0)
        
        # Select edges based on geometry
        num_mutations = max(1, int(len(edge_list) * mutation_rate))
        
        for (u, v) in edge_list:
            # Calculate priority based on geometry
            edge_coords = self._edge_to_geometry_coords(u, v)
            cell = self.core_system.get_cell(edge_coords)
            
            priority = 0.5
            if cell:
                priority += cell.face_exposure / 6.0
            
            if motion_vec:
                edge_vec = np.array(edge_coords, dtype=float)
                motion = np.array(motion_vec, dtype=float)
                projection = np.dot(motion, edge_vec) / (np.linalg.norm(edge_vec) + 1e-10)
                priority += abs(projection) * 0.5
            
            mutations.append(GeometryMutation(
                edge_indices=[(u, v)],
                mutation_type=mutation_type,
                priority=priority,
                reason=f"class={cell_class.name}, exposure={face_exposure}, polarity={polarity:.2f}"
            ))
        
        return mutations
    
    def _edge_to_geometry_coords(self, u: int, v: int) -> Tuple[int, int, int]:
        """Map edge (u, v) to geometric coordinates."""
        # Map edge indices to 3D space
        # Use a deterministic mapping
        edge_idx = self.edge_to_index.get((u, v), u * self.n + v)
        
        # Map to lattice coordinates
        # Use modulo to map to valid lattice range
        lattice_size = self.core_system.lattice_size
        coord_range = list(range(-(lattice_size - 1) // 2, (lattice_size - 1) // 2 + 1))
        
        x = coord_range[edge_idx % len(coord_range)]
        y = coord_range[(edge_idx // len(coord_range)) % len(coord_range)]
        z = coord_range[(edge_idx // (len(coord_range) ** 2)) % len(coord_range)]
        
        return (x, y, z)
    
    def _apply_mutation(self, graph: RamseyGraph, mutation: GeometryMutation):
        """
        Apply a geometry mutation to a graph.
        
        ⚠️ NOTE: This method does NOT check validity - caller must validate after mutation.
        This is intentional to allow the caller to decide validation strategy.
        """
        for (u, v) in mutation.edge_indices:
            if mutation.mutation_type == 'flip':
                current = graph.get_edge_color(u, v)
                if current is not None:
                    # 🔵 ROTATE SEMANTIC FIELD: Rotate semantic vector with geometry
                    sigma = getattr(graph, 'edge_semantic_27', {}).get((u, v))
                    if sigma is not None:
                        # Rotate semantic field like a vector (circular shift)
                        # This gives the system semantic inertia - semantics rotate with geometry
                        rotated_sigma = np.roll(sigma, int(mutation.priority * 7)) % 1.0
                        graph.set_edge_color(u, v, 1 - current, sigma27=rotated_sigma)
                    else:
                        graph.set_edge_color(u, v, 1 - current)
            elif mutation.mutation_type == 'add':
                if graph.get_edge_color(u, v) is None:
                    graph.set_edge_color(u, v, np.random.randint(0, 2))
            elif mutation.mutation_type == 'remove':
                if (u, v) in graph.edge_coloring:
                    del graph.edge_coloring[(u, v)]
                    # Also remove semantic field
                    if hasattr(graph, 'edge_semantic_27') and (u, v) in graph.edge_semantic_27:
                        del graph.edge_semantic_27[(u, v)]
            elif mutation.mutation_type == 'swap':
                # Swap with another edge (not implemented yet)
                pass


def transform_graph_by_lattice_motion(
    graph: RamseyGraph,
    transformer: GeometryGraphTransformer,
    old_coords: Tuple[int, int, int],
    new_coords: Tuple[int, int, int]
) -> RamseyGraph:
    """
    Transform graph based on lattice cell motion.
    
    This is the main entry point for geometry-driven graph transformation.
    """
    return transformer.transform_graph_by_geometry(graph, old_coords, new_coords)


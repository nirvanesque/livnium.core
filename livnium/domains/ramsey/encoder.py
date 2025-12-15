"""
Ramsey Domain Encoder: Graph Coloring → Initial State

Converts graph edge colorings into state vectors.
Uses kernel.physics for constraint generation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RamseyEncoder(nn.Module):
    """
    Ramsey encoder that converts graph edge colorings to initial state.
    
    Takes a graph with n vertices and encodes edge colorings (red/blue)
    into state vectors. The encoding represents the constraint satisfaction
    problem: find a 2-coloring that avoids monochromatic k-cliques.
    
    Uses kernel physics for constraint generation.
    """
    
    def __init__(self, n: int, dim: int = 256):
        """
        Initialize Ramsey encoder.
        
        Args:
            n: Number of vertices in the graph
            dim: Dimension of state vectors
        """
        super().__init__()
        self.n = n
        self.dim = dim
        
        # Number of edges in complete graph K_n
        self.num_edges = n * (n - 1) // 2
        
        # Project edge coloring to state dimension
        # Each edge can be red (0) or blue (1), so we encode as binary vector
        self.proj = nn.Linear(self.num_edges, dim)
        
        # Optional MLP for richer representation
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
    
    def encode_coloring(self, edge_coloring: torch.Tensor) -> torch.Tensor:
        """
        Encode edge coloring to state vector.
        
        Args:
            edge_coloring: Edge coloring [B, num_edges] or [num_edges]
                          Values: 0 = red, 1 = blue
            
        Returns:
            State vector [B, dim] or [dim]
        """
        if edge_coloring.dim() == 1:
            edge_coloring = edge_coloring.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        # Project to state dimension
        h = self.proj(edge_coloring)
        
        # Apply MLP
        h = self.mlp(h)
        
        if squeeze:
            h = h.squeeze(0)
        
        return h
    
    def edge_coloring_to_vector(self, edge_coloring: dict) -> torch.Tensor:
        """
        Convert edge coloring dictionary to vector representation.
        
        Args:
            edge_coloring: Dictionary mapping (u, v) tuples to color (0 or 1)
            
        Returns:
            Vector representation [num_edges]
        """
        vec = torch.zeros(self.num_edges)
        edge_idx = 0
        
        for u in range(self.n):
            for v in range(u + 1, self.n):
                color = edge_coloring.get((u, v), edge_coloring.get((v, u), 0))
                vec[edge_idx] = float(color)
                edge_idx += 1
        
        return vec
    
    def generate_constraints(
        self,
        state: torch.Tensor,
        k: int,
        edge_coloring: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Generate constraints from state.
        
        For Ramsey domain, constraints are:
        - No monochromatic k-clique exists
        - All edges are colored (red or blue)
        
        Uses kernel.physics for constraint calculations.
        
        Args:
            state: Current state vector
            k: Clique size to avoid
            edge_coloring: Optional edge coloring vector [num_edges]
            
        Returns:
            Dictionary of constraints
        """
        # Import here to avoid circular dependencies
        from livnium.kernel.physics import alignment, divergence, tension
        from livnium.engine.ops_torch import TorchOps
        
        ops = TorchOps()
        
        # Create state wrapper for kernel physics
        class StateWrapper:
            def __init__(self, vec):
                self._vec = vec
            def vector(self):
                return self._vec
            def norm(self):
                return torch.norm(self._vec, p=2)
        
        state_wrapper = StateWrapper(state)
        
        # Compute state norm (energy)
        energy = state_wrapper.norm()
        
        # If edge coloring provided, compute constraint violation
        constraint_violation = torch.tensor(0.0, device=state.device)
        if edge_coloring is not None:
            # This would require checking for monochromatic cliques
            # For now, we return a placeholder
            constraint_violation = torch.tensor(0.0, device=state.device)
        
        return {
            "state": state,
            "energy": energy,
            "constraint_violation": constraint_violation,
            "k": k,
        }
    
    def build_initial_state(
        self,
        edge_coloring: torch.Tensor,
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build initial state from edge coloring.
        
        Args:
            edge_coloring: Edge coloring [B, num_edges] or [num_edges]
                          Values: 0 = red, 1 = blue
            add_noise: Whether to add symmetry-breaking noise
            
        Returns:
            Tuple of (initial_state, encoded_coloring, encoded_coloring)
            (v_a and v_b are the same for Ramsey domain)
        """
        v_coloring = self.encode_coloring(edge_coloring)
        
        # Initial state is the encoded coloring
        h0 = v_coloring
        
        if add_noise:
            from livnium.engine.config import defaults
            h0 = h0 + defaults.EPS_NOISE * torch.randn_like(h0)
        
        # For Ramsey, we use the same vector twice (no premise/hypothesis)
        return h0, v_coloring, v_coloring
    
    def check_monochromatic_clique(
        self,
        edge_coloring: torch.Tensor,
        k: int
    ) -> Tuple[bool, Optional[list]]:
        """
        Check if a monochromatic k-clique exists in the coloring.
        
        Args:
            edge_coloring: Edge coloring [num_edges] or [B, num_edges]
            k: Clique size to check
            
        Returns:
            Tuple of (has_clique, clique_vertices)
        """
        if edge_coloring.dim() == 2:
            # Batch: check first element
            edge_coloring = edge_coloring[0]
        
        # Convert vector to edge coloring dict
        coloring_dict = {}
        edge_idx = 0
        for u in range(self.n):
            for v in range(u + 1, self.n):
                color = int(edge_coloring[edge_idx].item())
                coloring_dict[(u, v)] = color
                coloring_dict[(v, u)] = color
                edge_idx += 1
        
        # Check all k-cliques
        from itertools import combinations
        
        for clique_vertices in combinations(range(self.n), k):
            # Check if all edges in clique have same color
            colors = set()
            for u in clique_vertices:
                for v in clique_vertices:
                    if u < v:
                        colors.add(coloring_dict[(u, v)])
            
            # If all edges same color, we have a monochromatic clique
            if len(colors) == 1:
                return True, list(clique_vertices)
        
        return False, None


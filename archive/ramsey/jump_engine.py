"""
Jump Engine: Direct Navigation Toward Solutions

This module implements gradient-like search that allows the solver to
make large leaps toward promising regions instead of wandering randomly.

Key concept: Energy function Φ that measures "distance from perfect solution"
(lower is better, like physical energy), then jump vectors that move directly
toward better solutions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

from experiments.ramsey.ramsey_number_solver import RamseyGraph


def energy(graph: RamseyGraph, n: int, k: int, num_edges: int) -> float:
    """
    Compute energy Φ(graph) - lower is better (like physical energy).
    
    This is the core "distance from perfect solution" metric.
    
    Args:
        graph: Graph to score
        n: Number of vertices
        k: Clique size to avoid
        num_edges: Total number of edges
        
    Returns:
        Energy value (lower = closer to perfect solution)
    """
    if num_edges == 0:
        return float('inf')
    
    # 1. Completeness reward (AGGRESSIVE - especially at high completeness)
    completeness = len(graph.edge_coloring) / float(num_edges)
    # Non-linear bonus: completeness^3 gives huge reward near 100%
    # This creates a "completeness gravity well" that pulls solutions toward 100%
    completeness_bonus = -5.0 * completeness  # Base reward
    # Extra aggressive bonus when > 60% (the plateau region)
    if completeness > 0.6:
        # Exponential bonus: the closer to 100%, the more reward
        plateau_bonus = -10.0 * (completeness - 0.6) ** 2  # Quadratic bonus in plateau region
        completeness_bonus += plateau_bonus
    if completeness > 0.9:
        # Massive bonus when > 90% (near completion)
        near_complete_bonus = -20.0 * (completeness - 0.9) ** 1.5
        completeness_bonus += near_complete_bonus
    
    # 2. Clique risk (almost-cliques increase energy)
    clique_risk = _compute_clique_risk(graph, n, k)
    clique_penalty = 3.0 * clique_risk
    
    # 3. Color balance reward (balanced colorings are better)
    balance_score = _compute_balance_score(graph)
    balance_reward = -1.0 * balance_score  # Negative because we want higher balance = lower energy
    
    # 4. Symmetry reward (symmetric patterns are often better)
    symmetry_score = _compute_symmetry_score(graph, n)
    symmetry_reward = -0.5 * symmetry_score  # Negative because we want higher symmetry = lower energy
    
    # Total energy (lower is better)
    total_energy = completeness_bonus + clique_penalty + balance_reward + symmetry_reward
    
    return total_energy


def _compute_clique_risk(graph: RamseyGraph, n: int, k: int) -> float:
    """Compute risk of forming cliques (higher = more dangerous)."""
    if len(graph.edge_coloring) == 0:
        return 0.0
    
    # Build adjacency lists for each color
    red_adj = {i: set() for i in range(n)}
    blue_adj = {i: set() for i in range(n)}
    
    for (u, v), color in graph.edge_coloring.items():
        if color == 0:  # Red
            red_adj[u].add(v)
            red_adj[v].add(u)
        elif color == 1:  # Blue
            blue_adj[u].add(v)
            blue_adj[v].add(u)
    
    # Count almost-cliques (k-1 vertices with many connections)
    risk = 0.0
    
    for vertex in range(n):
        # Check red almost-cliques
        red_neighbors = red_adj[vertex]
        if len(red_neighbors) >= k - 1:
            connections = 0
            for u in red_neighbors:
                for v in red_neighbors:
                    if u < v and v in red_adj[u]:
                        connections += 1
            
            max_connections = len(red_neighbors) * (len(red_neighbors) - 1) // 2
            if max_connections > 0:
                risk += (connections / max_connections) * (len(red_neighbors) / n)
        
        # Check blue almost-cliques
        blue_neighbors = blue_adj[vertex]
        if len(blue_neighbors) >= k - 1:
            connections = 0
            for u in blue_neighbors:
                for v in blue_neighbors:
                    if u < v and v in blue_adj[u]:
                        connections += 1
            
            max_connections = len(blue_neighbors) * (len(blue_neighbors) - 1) // 2
            if max_connections > 0:
                risk += (connections / max_connections) * (len(blue_neighbors) / n)
    
    return min(1.0, risk / n)  # Normalize


def _compute_balance_score(graph: RamseyGraph) -> float:
    """Compute color balance score (higher = more balanced)."""
    if len(graph.edge_coloring) == 0:
        return 0.0
    
    red_count = sum(1 for color in graph.edge_coloring.values() if color == 0)
    blue_count = sum(1 for color in graph.edge_coloring.values() if color == 1)
    total = red_count + blue_count
    
    if total == 0:
        return 0.0
    
    # Balance score: 1.0 if perfectly balanced, 0.0 if completely unbalanced
    ratio = min(red_count, blue_count) / max(red_count, blue_count) if max(red_count, blue_count) > 0 else 0.0
    return ratio


def _compute_symmetry_score(graph: RamseyGraph, n: int) -> float:
    """Compute symmetry score (higher = more symmetric)."""
    if len(graph.edge_coloring) == 0:
        return 0.0
    
    # Check if vertex degrees are balanced
    red_degrees = {i: 0 for i in range(n)}
    blue_degrees = {i: 0 for i in range(n)}
    
    for (u, v), color in graph.edge_coloring.items():
        if color == 0:  # Red
            red_degrees[u] += 1
            red_degrees[v] += 1
        elif color == 1:  # Blue
            blue_degrees[u] += 1
            blue_degrees[v] += 1
    
    # Compute variance in degrees (lower variance = more symmetric)
    red_deg_values = list(red_degrees.values())
    blue_deg_values = list(blue_degrees.values())
    
    if len(red_deg_values) == 0 or len(blue_deg_values) == 0:
        return 0.0
    
    red_variance = np.var(red_deg_values) if len(red_deg_values) > 1 else 0.0
    blue_variance = np.var(blue_deg_values) if len(blue_deg_values) > 1 else 0.0
    
    # Normalize: lower variance = higher score
    max_variance = n  # Maximum possible variance
    red_score = 1.0 - min(1.0, red_variance / max_variance) if max_variance > 0 else 0.0
    blue_score = 1.0 - min(1.0, blue_variance / max_variance) if max_variance > 0 else 0.0
    
    return (red_score + blue_score) / 2.0


@dataclass
class PotentialScore:
    """Potential function score for a graph coloring."""
    total: float  # Φ total
    completeness: float  # How many edges colored
    clique_risk: float  # Penalty for almost-cliques
    balance_reward: float  # Reward for balanced substructures
    distance_penalty: float  # Distance from best valid
    symmetry_score: float  # Symmetry matching


class PotentialFunction:
    """
    Computes potential Φ for a RamseyGraph.
    
    Φ = completeness - penalty_for_almost_cliques + reward_for_balanced_substructures
        - entropy_distance_from_best_valid + symmetry_match_score
    """
    
    def __init__(self, n: int, k: int, num_edges: int):
        """
        Initialize potential function.
        
        Args:
            n: Number of vertices
            k: Clique size to avoid
            num_edges: Total number of edges
        """
        self.n = n
        self.k = k
        self.num_edges = num_edges
    
    def compute_potential(
        self,
        graph: RamseyGraph,
        best_valid: Optional[RamseyGraph] = None
    ) -> PotentialScore:
        """
        Compute potential Φ for a graph.
        
        Args:
            graph: Graph to score
            best_valid: Optional best valid coloring for distance calculation
            
        Returns:
            PotentialScore with all components
        """
        # 1. Completeness (0.0 to 1.0)
        completeness = len(graph.edge_coloring) / float(self.num_edges) if self.num_edges > 0 else 0.0
        
        # 2. Clique risk penalty (negative, higher for dangerous patterns)
        clique_risk = self._compute_clique_risk(graph)
        
        # 3. Balance reward (positive, higher for balanced colorings)
        balance_reward = self._compute_balance_reward(graph)
        
        # 4. Distance penalty from best valid (if provided)
        distance_penalty = 0.0
        if best_valid:
            distance_penalty = self._compute_distance_penalty(graph, best_valid)
        
        # 5. Symmetry score (positive, higher for symmetric patterns)
        symmetry_score = self._compute_symmetry_score(graph)
        
        # Total potential
        total = (
            completeness * 1.0 +           # Weight: 1.0
            -clique_risk * 2.0 +           # Weight: -2.0 (penalty)
            balance_reward * 0.5 +         # Weight: 0.5
            -distance_penalty * 1.5 +       # Weight: -1.5 (penalty)
            symmetry_score * 0.3           # Weight: 0.3
        )
        
        return PotentialScore(
            total=total,
            completeness=completeness,
            clique_risk=clique_risk,
            balance_reward=balance_reward,
            distance_penalty=distance_penalty,
            symmetry_score=symmetry_score
        )
    
    def _compute_clique_risk(self, graph: RamseyGraph) -> float:
        """
        Compute risk of forming cliques.
        
        Returns higher values for graphs that are close to forming k-cliques.
        """
        if len(graph.edge_coloring) == 0:
            return 0.0
        
        # Build adjacency lists for each color
        red_adj = {i: set() for i in range(self.n)}
        blue_adj = {i: set() for i in range(self.n)}
        
        for (u, v), color in graph.edge_coloring.items():
            if color == 0:  # Red
                red_adj[u].add(v)
                red_adj[v].add(u)
            elif color == 1:  # Blue
                blue_adj[u].add(v)
                blue_adj[v].add(u)
        
        # Count almost-cliques (k-1 vertices with many connections)
        risk = 0.0
        
        for vertex in range(self.n):
            # Check red almost-cliques
            red_neighbors = red_adj[vertex]
            if len(red_neighbors) >= self.k - 1:
                # Count connections among neighbors
                connections = 0
                for u in red_neighbors:
                    for v in red_neighbors:
                        if u < v and v in red_adj[u]:
                            connections += 1
                
                # Risk increases with more connections
                max_connections = len(red_neighbors) * (len(red_neighbors) - 1) // 2
                if max_connections > 0:
                    risk += (connections / max_connections) * (len(red_neighbors) / self.n)
            
            # Check blue almost-cliques
            blue_neighbors = blue_adj[vertex]
            if len(blue_neighbors) >= self.k - 1:
                connections = 0
                for u in blue_neighbors:
                    for v in blue_neighbors:
                        if u < v and v in blue_adj[u]:
                            connections += 1
                
                max_connections = len(blue_neighbors) * (len(blue_neighbors) - 1) // 2
                if max_connections > 0:
                    risk += (connections / max_connections) * (len(blue_neighbors) / self.n)
        
        return min(1.0, risk / self.n)  # Normalize
    
    def _compute_balance_reward(self, graph: RamseyGraph) -> float:
        """
        Compute reward for balanced colorings.
        
        Returns higher values for graphs with roughly equal red/blue edges.
        """
        if len(graph.edge_coloring) == 0:
            return 0.0
        
        red_count = sum(1 for color in graph.edge_coloring.values() if color == 0)
        blue_count = sum(1 for color in graph.edge_coloring.values() if color == 1)
        total = red_count + blue_count
        
        if total == 0:
            return 0.0
        
        # Balance score: 1.0 if perfectly balanced, 0.0 if completely unbalanced
        ratio = min(red_count, blue_count) / max(red_count, blue_count) if max(red_count, blue_count) > 0 else 0.0
        return ratio
    
    def _compute_distance_penalty(self, graph: RamseyGraph, best_valid: RamseyGraph) -> float:
        """
        Compute distance penalty from best valid coloring.
        
        Returns higher values for graphs that are far from the best valid coloring.
        """
        # Compute Hamming distance (edge-wise difference)
        graph_edges = set(graph.edge_coloring.keys())
        best_edges = set(best_valid.edge_coloring.keys())
        
        # Edges in both: check if colors match
        common_edges = graph_edges & best_edges
        mismatches = 0
        for edge in common_edges:
            if graph.edge_coloring.get(edge) != best_valid.edge_coloring.get(edge):
                mismatches += 1
        
        # Edges in one but not the other
        unique_to_graph = graph_edges - best_edges
        unique_to_best = best_edges - graph_edges
        
        # Total distance
        total_distance = mismatches + len(unique_to_graph) + len(unique_to_best)
        max_distance = self.num_edges
        
        if max_distance == 0:
            return 0.0
        
        return total_distance / max_distance  # Normalize to [0, 1]
    
    def _compute_symmetry_score(self, graph: RamseyGraph) -> float:
        """
        Compute symmetry score.
        
        Returns higher values for graphs with symmetric patterns.
        """
        if len(graph.edge_coloring) == 0:
            return 0.0
        
        # Simple symmetry: check if vertex degrees are balanced
        red_degrees = {i: 0 for i in range(self.n)}
        blue_degrees = {i: 0 for i in range(self.n)}
        
        for (u, v), color in graph.edge_coloring.items():
            if color == 0:  # Red
                red_degrees[u] += 1
                red_degrees[v] += 1
            elif color == 1:  # Blue
                blue_degrees[u] += 1
                blue_degrees[v] += 1
        
        # Compute variance in degrees (lower variance = more symmetric)
        red_deg_values = list(red_degrees.values())
        blue_deg_values = list(blue_degrees.values())
        
        if len(red_deg_values) == 0 or len(blue_deg_values) == 0:
            return 0.0
        
        red_variance = np.var(red_deg_values) if len(red_deg_values) > 1 else 0.0
        blue_variance = np.var(blue_deg_values) if len(blue_deg_values) > 1 else 0.0
        
        # Normalize: lower variance = higher score
        max_variance = self.n  # Maximum possible variance
        red_score = 1.0 - min(1.0, red_variance / max_variance) if max_variance > 0 else 0.0
        blue_score = 1.0 - min(1.0, blue_variance / max_variance) if max_variance > 0 else 0.0
        
        return (red_score + blue_score) / 2.0


class JumpEngine:
    """
    Computes jump vectors that move directly toward better solutions.
    
    Uses energy function Φ to determine direction and magnitude of jumps.
    """
    
    def __init__(self, n: int, k: int, num_edges: int):
        """
        Initialize jump engine.
        
        Args:
            n: Number of vertices
            k: Clique size to avoid
            num_edges: Total number of edges
        """
        self.n = n
        self.k = k
        self.num_edges = num_edges
        
        # Track best energy seen
        self.best_energy: Optional[float] = None
        self.best_valid: Optional[RamseyGraph] = None
    
    def update_best(self, graph: RamseyGraph, is_valid: bool):
        """
        Update best valid coloring if this one has lower energy.
        
        Args:
            graph: Graph to check
            is_valid: Whether graph is valid
        """
        if is_valid:
            graph_energy = energy(graph, self.n, self.k, self.num_edges)
            if self.best_energy is None or graph_energy < self.best_energy:
                self.best_energy = graph_energy
                self.best_valid = graph.copy()
    
    def compute_jump_vector(
        self,
        graph: RamseyGraph,
        current_coords: Tuple[float, float, float],
        best_coords: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[float, float, float]:
        """
        Compute jump vector toward better solutions using energy Φ.
        
        Args:
            graph: Current graph
            current_coords: Current coordinates
            best_coords: Optional best coordinates (if not provided, uses best_valid)
            
        Returns:
            Jump vector (dx, dy, dz) to add to current coordinates
        """
        # Compute energy for current graph
        current_energy = energy(graph, self.n, self.k, self.num_edges)
        
        # Compute energy for best valid (if available)
        best_energy = None
        if self.best_valid:
            best_energy = energy(self.best_valid, self.n, self.k, self.num_edges)
        
        # Compute ΔΦ (energy difference)
        if best_energy is not None:
            delta_phi = current_energy - best_energy  # Positive = current is worse
        else:
            # No best yet: jump toward higher completeness
            completeness = len(graph.edge_coloring) / float(self.num_edges) if self.num_edges > 0 else 0.0
            delta_phi = 1.0 - completeness  # Higher completeness = lower energy
        
        # Get target coordinates
        if best_coords is None and self.best_valid:
            best_coords = self.best_valid.to_coordinates()
        
        if best_coords is None:
            # No target: use gradient toward completeness
            # Higher completeness = move toward center
            jump_magnitude = abs(delta_phi) * 0.1
            jump_vector = (
                -current_coords[0] * jump_magnitude,
                -current_coords[1] * jump_magnitude,
                -current_coords[2] * jump_magnitude
            )
        else:
            # Jump toward best coordinates
            current_vec = np.array(current_coords)
            best_vec = np.array(best_coords)
            direction = best_vec - current_vec
            
            # Normalize direction
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-10:
                direction = direction / direction_norm
            else:
                # Same coordinates: random small jump
                direction = np.random.randn(3)
                direction = direction / (np.linalg.norm(direction) + 1e-10)
            
            # Jump magnitude based on ΔΦ (larger energy gap = larger jump)
            jump_magnitude = abs(delta_phi) * 0.2  # Scale factor
            
            # Clamp jump magnitude
            jump_magnitude = min(jump_magnitude, 0.5)  # Max jump size
            
            # Jump direction: toward best if current is worse
            if delta_phi > 0:  # Current is worse, jump toward best
                jump_vector = tuple(float(x) for x in (direction * jump_magnitude))
            else:  # Current is better (shouldn't happen often), small random exploration
                jump_vector = tuple(float(x) for x in (direction * jump_magnitude * 0.1))
        
        # 🔵 SEMANTIC FIELD CONTRIBUTION: Add semantic curvature to jump vector
        # This gives the system memory, curvature bias, directional hinting, and stability fields
        semantic_contrib = np.zeros(3)
        edge_count = 0
        for (u, v), color in graph.edge_coloring.items():
            sigma = getattr(graph, 'edge_semantic_27', {}).get((u, v))
            if sigma is not None:
                # Semantic curvature contribution
                semantic_contrib += np.array([
                    sigma.mean(),
                    sigma.std(),
                    sigma.sum() % 1.0
                ])
                edge_count += 1
        
        if edge_count > 0:
            semantic_contrib = semantic_contrib / edge_count  # Average across edges
            jump_vector = tuple(
                float(x) for x in np.array(jump_vector) + 0.03 * semantic_contrib
            )
        
        return jump_vector
    
    def should_jump(
        self,
        graph: RamseyGraph,
        current_coords: Tuple[float, float, float],
        threshold: float = 0.1
    ) -> bool:
        """
        Decide if we should make a jump based on energy difference.
        
        Args:
            graph: Current graph
            current_coords: Current coordinates
            threshold: Minimum ΔΦ to trigger jump
            
        Returns:
            True if jump should be made
        """
        if self.best_valid is None:
            return False
        
        current_energy = energy(graph, self.n, self.k, self.num_edges)
        best_energy = energy(self.best_valid, self.n, self.k, self.num_edges)
        
        delta_phi = current_energy - best_energy  # Positive = current is worse
        
        return delta_phi > threshold
    
    def get_energy(self, graph: RamseyGraph) -> float:
        """Get energy for a graph."""
        return energy(graph, self.n, self.k, self.num_edges)
    
    def get_potential_score(self, graph: RamseyGraph) -> PotentialScore:
        """Get potential score for a graph."""
        return self.potential_fn.compute_potential(graph, self.best_valid)


def apply_jump(
    graph: RamseyGraph,
    coords: Tuple[float, float, float],
    jump_engine: JumpEngine,
    best_coords: Optional[Tuple[float, float, float]] = None
) -> Tuple[RamseyGraph, Tuple[float, float, float]]:
    """
    Apply jump to graph and coordinates.
    
    Args:
        graph: Graph to jump
        coords: Current coordinates
        jump_engine: Jump engine instance
        best_coords: Optional best coordinates
        
    Returns:
        (new_graph, new_coords) after jump
    """
    # Compute jump vector
    jump_vector = jump_engine.compute_jump_vector(graph, coords, best_coords)
    
    # Apply jump to coordinates
    new_coords = tuple(np.array(coords) + np.array(jump_vector))
    
    # Clamp coordinates to [0, 1]
    new_coords = tuple(np.clip(new_coords, 0.0, 1.0))
    
    # Graph mutation based on jump magnitude
    jump_magnitude = np.linalg.norm(jump_vector)
    
    if jump_magnitude > 0.1:  # Significant jump
        # Apply mutations to graph based on jump direction
        new_graph = graph.copy()
        
        # Determine mutation intensity from jump magnitude
        num_mutations = max(1, int(len(new_graph.edge_coloring) * jump_magnitude * 0.1))
        
        # Mutate edges
        edge_list = list(new_graph.edge_coloring.keys())
        if edge_list:
            for _ in range(min(num_mutations, len(edge_list))):
                u, v = edge_list[np.random.randint(len(edge_list))]
                current = new_graph.get_edge_color(u, v)
                if current is not None:
                    new_graph.set_edge_color(u, v, 1 - current)
    else:
        # Small jump: no graph mutation
        new_graph = graph.copy()
    
    return new_graph, new_coords


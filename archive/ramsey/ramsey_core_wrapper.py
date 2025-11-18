"""
Python wrapper for C-accelerated Ramsey core operations.

This module provides a clean Python interface to the C-accelerated bitset-based
clique checking and batch operations. It plugs seamlessly into the existing
RamseyGraph class without changing the Livnium meta-physics layer.

Usage:
    from ramsey_core_wrapper import RamseyCoreAccelerator
    
    accelerator = RamseyCoreAccelerator()
    if accelerator.available:
        # Use C-accelerated checking
        is_valid, clique = accelerator.check_coloring(graph, n, k)
    else:
        # Fallback to Python/Numba
        is_valid, clique = graph.has_monochromatic_clique(k)
"""

import numpy as np
from typing import Tuple, Optional, List

# Try to import C extension
try:
    import ramsey_core
    C_CORE_AVAILABLE = True
    C_ACCELERATOR_AVAILABLE = True  # Alias for compatibility
except ImportError:
    C_CORE_AVAILABLE = False
    C_ACCELERATOR_AVAILABLE = False  # Alias for compatibility
    ramsey_core = None


class RamseyCoreAccelerator:
    """
    C-accelerated Ramsey core operations.
    
    Provides fast bitset-based clique checking and batch validation
    that leverages M5 CPU optimizations (bitwise ops, POPCOUNT, etc.).
    
    Falls back gracefully to Python/Numba if C extension is not available.
    """
    
    def __init__(self):
        self.available = C_CORE_AVAILABLE
        if not self.available:
            print("⚠️  C-accelerated core not available, using Python/Numba fallback")
    
    def check_coloring(
        self, 
        graph,  # RamseyGraph instance
        n: int, 
        k: int
    ) -> Tuple[bool, Optional[List[int]]]:
        """
        Check if a RamseyGraph coloring is valid (no monochromatic k-clique).
        
        Uses C-accelerated bitset operations when available.
        
        Args:
            graph: RamseyGraph instance
            n: Number of vertices
            k: Clique size to avoid
            
        Returns:
            (is_valid, clique_vertices)
            - is_valid: True if no monochromatic k-clique exists
            - clique_vertices: List of vertices forming a clique if found, else None
        """
        if not self.available:
            # Fallback to existing Python method
            has_clique, clique = graph.has_monochromatic_clique(k)
            return not has_clique, clique
        
        # Convert graph to numpy array format for C extension
        num_edges = n * (n - 1) // 2
        edge_colors = np.zeros(num_edges, dtype=np.uint8)
        
        # Fill with 255 (uncolored) by default
        edge_colors.fill(255)
        
        # Map graph edges to array
        edge_idx = 0
        for u in range(n):
            for v in range(u + 1, n):
                color = graph.get_edge_color(u, v)
                if color is not None:
                    edge_colors[edge_idx] = color
                edge_idx += 1
        
        # Call C extension
        result = ramsey_core.check_ramsey_coloring(edge_colors, n, k)
        has_clique, clique_array = result
        
        if has_clique and clique_array is not None:
            clique = clique_array.tolist()
            return False, clique
        else:
            return True, None
    
    def batch_check_colorings(
        self,
        graphs: List,  # List of RamseyGraph instances
        n: int,
        k: int
    ) -> np.ndarray:
        """
        Check validity of a batch of RamseyGraph colorings.
        
        Uses C-accelerated batch operations for maximum throughput.
        This is where M5 really shines - processing thousands of omcubes in parallel.
        
        Args:
            graphs: List of RamseyGraph instances
            n: Number of vertices
            k: Clique size to avoid
            
        Returns:
            numpy array of bools (True = valid, False = invalid)
        """
        batch_size = len(graphs)
        num_edges = n * (n - 1) // 2
        
        if not self.available:
            # Fallback: check each graph individually
            results = np.zeros(batch_size, dtype=bool)
            for i, graph in enumerate(graphs):
                has_clique, _ = graph.has_monochromatic_clique(k)
                results[i] = not has_clique
            return results
        
        # Build batch array [batch_size, num_edges]
        edge_colorings = np.full((batch_size, num_edges), 255, dtype=np.uint8)
        
        for batch_idx, graph in enumerate(graphs):
            edge_idx = 0
            for u in range(n):
                for v in range(u + 1, n):
                    color = graph.get_edge_color(u, v)
                    if color is not None:
                        edge_colorings[batch_idx, edge_idx] = color
                    edge_idx += 1
        
        # Call C extension batch function
        results = ramsey_core.batch_check_ramsey_colorings(edge_colorings, n, k)
        return results


# Global accelerator instance
_global_accelerator = None

def get_accelerator() -> RamseyCoreAccelerator:
    """Get or create the global accelerator instance."""
    global _global_accelerator
    if _global_accelerator is None:
        _global_accelerator = RamseyCoreAccelerator()
    return _global_accelerator


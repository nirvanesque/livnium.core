"""
Ramsey Dataset Loader

Loads/generates graph coloring data for Ramsey number problems.
"""

import torch
from typing import Dict, Any, Optional
from livnium.datasets.base import LivniumDataset


class RamseyDataset(LivniumDataset):
    """
    Ramsey dataset loader.
    
    Generates graph coloring problems for Ramsey number constraint satisfaction.
    """
    
    def __init__(
        self,
        n: int = 10,
        k: int = 3,
        size: int = 1000,
        seed: Optional[int] = None,
    ):
        """
        Initialize Ramsey dataset.
        
        Args:
            n: Number of vertices in graph
            k: Clique size to avoid
            size: Number of samples
            seed: Random seed
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        self.n = n
        self.k = k
        self.size = size
        
        # Number of edges in complete graph K_n
        self.num_edges = n * (n - 1) // 2
        
        # Generate random edge colorings
        self.edge_colorings = torch.randint(0, 2, (size, self.num_edges))
        
        # Compute validity (simplified: just check if all edges are colored)
        # Real validity would require checking for monochromatic cliques
        self.labels = torch.ones(size, dtype=torch.long)  # Dummy: all valid
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "edge_coloring": self.edge_colorings[idx],
            "label": self.labels[idx],  # 0 = invalid, 1 = valid
            "n": self.n,
            "k": self.k,
        }


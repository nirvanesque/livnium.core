"""
Toy Dataset: Synthetic Data for Testing

Simple synthetic dataset for testing kernel+engine integration.
"""

import torch
from typing import Dict, Any, Optional
from livnium.datasets.base import LivniumDataset


class ToyDataset(LivniumDataset):
    """
    Simple synthetic dataset for toy domain.
    
    Generates random 2D input pairs with synthetic labels.
    """
    
    def __init__(self, size: int = 1000, dim: int = 2, seed: Optional[int] = None):
        """
        Initialize toy dataset.
        
        Args:
            size: Number of samples
            dim: Input dimension (default 2)
            seed: Random seed
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        self.size = size
        self.dim = dim
        
        # Generate synthetic data
        self.inputs_a = torch.randn(size, dim)
        self.inputs_b = torch.randn(size, dim)
        
        # Simple label: based on dot product sign
        self.labels = ((self.inputs_a * self.inputs_b).sum(dim=-1) > 0).long()
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "input_a": self.inputs_a[idx],
            "input_b": self.inputs_b[idx],
            "label": self.labels[idx],
        }


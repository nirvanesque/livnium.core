"""
Base Dataset Classes

Abstract base classes for LIVNIUM datasets.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
from torch.utils.data import Dataset


class LivniumDataset(Dataset, ABC):
    """
    Base dataset class for LIVNIUM domains.
    
    All domain datasets should inherit from this.
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single data sample.
        
        Returns:
            Dictionary with domain-specific keys (e.g., 'premise', 'hypothesis', 'label')
        """
        pass


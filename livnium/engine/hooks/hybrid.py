"""
Livnium Hybrid Hook: Bridging Physics Layers

Defines the interface for one-way research → production influence.
Allows experimental layers (Quantum, Recursive) to bias the stable CollapseEngine
without direct coupling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from typing import Optional

class CollapseBias(ABC):
    """
    Interface for providing external bias to the CollapseEngine.
    
    Research stack components (QuantumRegister, RecursiveGeometryEngine) 
    implement this to influence continuous vector dynamics.
    """
    
    @abstractmethod
    def bias_for_state(self, h: torch.Tensor, step_idx: int) -> torch.Tensor:
        """
        Compute a bias term to be added to the state update.
        
        Args:
            h: Current state vector [B, dim]
            step_idx: Current iteration step
            
        Returns:
            Bias tensor [B, dim]
        """
        pass

class NullBias(CollapseBias):
    """Default no-op bias."""
    def bias_for_state(self, h: torch.Tensor, step_idx: int) -> torch.Tensor:
        return torch.zeros_like(h)

@dataclass
class HybridConfig:
    """Config for enabling hybrid influences."""
    from livnium.engine.config.defaults import DEFAULT_HYBRID_BIAS
    enabled: bool = False
    bias_weight: float = DEFAULT_HYBRID_BIAS
    hook: Optional[CollapseBias] = None

"""
Engine Ops: Torch Implementation

Concrete implementation of Ops protocol for PyTorch tensors.
Engine provides this to kernel physics functions.
"""

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None

from livnium.kernel.ops import Ops


class TorchOps:
    """
    Torch implementation of Ops protocol.
    
    Provides tensor operations using PyTorch.
    """
    
    def __init__(self):
        """Initialize TorchOps."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Install torch to use TorchOps.")
    
    def dot(self, a, b) -> float:
        """
        Dot product of two vectors.
        
        Args:
            a: First vector (torch.Tensor)
            b: Second vector (torch.Tensor)
            
        Returns:
            Scalar dot product
        """
        a_flat = a.flatten().detach()
        b_flat = b.flatten().detach()
        return float(torch.dot(a_flat, b_flat))
    
    def norm(self, x, dim: int = -1):
        """
        Compute norm along specified dimension.
        
        Args:
            x: Input tensor
            dim: Dimension along which to compute norm
            
        Returns:
            Norm values
        """
        return torch.norm(x, p=2, dim=dim, keepdim=True)
    
    def clip(self, x, min_val: float, max_val: float):
        """
        Clip values to range [min_val, max_val].
        
        Args:
            x: Input tensor
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Clipped tensor
        """
        return torch.clamp(x, min=min_val, max=max_val)
    
    def where(self, condition, x, y):
        """
        Element-wise conditional selection.
        
        Args:
            condition: Boolean tensor
            x: Values where condition is True
            y: Values where condition is False
            
        Returns:
            Selected values
        """
        return torch.where(condition, x, y)
    
    def eps(self) -> float:
        """
        Return small epsilon value for numerical stability.
        
        Returns:
            Small epsilon (1e-8)
        """
        return 1e-8
    
    def normalize(self, x, dim: int = -1):
        """
        Normalize vector/tensor along specified dimension.
        
        Args:
            x: Input tensor
            dim: Dimension along which to normalize
            
        Returns:
            Normalized tensor (unit length along dim)
        """
        return F.normalize(x, p=2, dim=dim)


# Verify TorchOps implements Ops protocol
if TORCH_AVAILABLE:
    # This is a runtime check - if TorchOps doesn't match Ops, it will fail
    # when kernel tries to use it
    pass


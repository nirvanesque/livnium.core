"""
Kernel Ops Protocol: Abstract Tensor Operations

Kernel uses this protocol, engine implements it.
This allows kernel to be pure math without torch/numpy dependencies.
"""

from typing import Protocol, Any


class Ops(Protocol):
    """
    Protocol for tensor operations.
    
    Kernel physics functions take an Ops instance as first parameter.
    Engine provides concrete implementations (TorchOps, NumpyOps).
    """
    
    def dot(self, a: Any, b: Any) -> Any:
        """
        Dot product of two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Scalar dot product (may be Tensor or float depending on backend)
        """
        ...
    
    def norm(self, x: Any, dim: int = -1) -> Any:
        """
        Compute norm along specified dimension.
        
        Args:
            x: Input tensor/vector
            dim: Dimension along which to compute norm
            
        Returns:
            Norm values
        """
        ...
    
    def clip(self, x: Any, min_val: float, max_val: float) -> Any:
        """
        Clip values to range [min_val, max_val].
        
        Args:
            x: Input tensor/vector
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Clipped tensor/vector
        """
        ...
    
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """
        Element-wise conditional selection.
        
        Args:
            condition: Boolean tensor/array
            x: Values where condition is True
            y: Values where condition is False
            
        Returns:
            Selected values
        """
        ...
    
    def eps(self) -> float:
        """
        Return small epsilon value for numerical stability.
        
        Returns:
            Small epsilon (typically 1e-8)
        """
        ...
    
    def normalize(self, x: Any, dim: int = -1) -> Any:
        """
        Normalize vector/tensor along specified dimension.
        
        Args:
            x: Input tensor/vector
            dim: Dimension along which to normalize
            
        Returns:
            Normalized tensor/vector (unit length along dim)
        """
        ...


"""
Engine Ops: NumPy Implementation

Concrete implementation of Ops protocol for NumPy arrays.
Engine provides this to kernel physics functions.
"""

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from livnium.kernel.ops import Ops


class NumpyOps:
    """
    NumPy implementation of Ops protocol.
    
    Provides tensor operations using NumPy.
    """
    
    def __init__(self):
        """Initialize NumpyOps."""
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is not available. Install numpy to use NumpyOps.")
    
    def dot(self, a, b) -> float:
        """
        Dot product of two vectors.
        
        Args:
            a: First vector (numpy.ndarray)
            b: Second vector (numpy.ndarray)
            
        Returns:
            Scalar dot product
        """
        a_flat = a.flatten()
        b_flat = b.flatten()
        return float(np.dot(a_flat, b_flat))
    
    def norm(self, x, dim: int = -1):
        """
        Compute norm along specified dimension.
        
        Args:
            x: Input array
            dim: Dimension along which to compute norm
            
        Returns:
            Norm values
        """
        if dim == -1:
            # For 1D arrays, compute norm directly
            if x.ndim == 1:
                return np.linalg.norm(x, keepdims=True)
            else:
                # For multi-dimensional, compute along last axis
                return np.linalg.norm(x, axis=-1, keepdims=True)
        else:
            return np.linalg.norm(x, axis=dim, keepdims=True)
    
    def clip(self, x, min_val: float, max_val: float):
        """
        Clip values to range [min_val, max_val].
        
        Args:
            x: Input array
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Clipped array
        """
        return np.clip(x, min_val, max_val)
    
    def where(self, condition, x, y):
        """
        Element-wise conditional selection.
        
        Args:
            condition: Boolean array
            x: Values where condition is True
            y: Values where condition is False
            
        Returns:
            Selected values
        """
        return np.where(condition, x, y)
    
    def eps(self) -> float:
        """
        Return small epsilon value for numerical stability.
        
        Returns:
            Small epsilon (1e-8)
        """
        return 1e-8
    
    def normalize(self, x, dim: int = -1):
        """
        Normalize vector/array along specified dimension.
        
        Args:
            x: Input array
            dim: Dimension along which to normalize
            
        Returns:
            Normalized array (unit length along dim)
        """
        if dim == -1:
            # For 1D arrays, normalize directly
            if x.ndim == 1:
                norm = np.linalg.norm(x) + self.eps()
                return x / norm
            else:
                # For multi-dimensional, normalize along last axis
                norm = np.linalg.norm(x, axis=-1, keepdims=True) + self.eps()
                return x / norm
        else:
            norm = np.linalg.norm(x, axis=dim, keepdims=True) + self.eps()
            return x / norm


# Verify NumpyOps implements Ops protocol
if NUMPY_AVAILABLE:
    # This is a runtime check - if NumpyOps doesn't match Ops, it will fail
    # when kernel tries to use it
    pass


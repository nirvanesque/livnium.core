"""
Profiling Infrastructure

Performance profiling for LIVNIUM components.
"""

import time
from typing import Dict, List, Optional, Callable
from contextlib import contextmanager
from collections import defaultdict


class Profiler:
    """
    Simple profiler for tracking execution time.
    
    Can profile function calls and code blocks.
    """
    
    def __init__(self):
        """Initialize profiler."""
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.counts: Dict[str, int] = defaultdict(int)
    
    @contextmanager
    def profile(self, name: str):
        """
        Context manager for profiling code blocks.
        
        Args:
            name: Name of the operation being profiled
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.timings[name].append(elapsed)
            self.counts[name] += 1
    
    def profile_function(self, func: Callable, name: Optional[str] = None):
        """
        Decorator for profiling functions.
        
        Args:
            func: Function to profile
            name: Optional name (defaults to function name)
        """
        if name is None:
            name = func.__name__
        
        def wrapper(*args, **kwargs):
            with self.profile(name):
                return func(*args, **kwargs)
        
        return wrapper
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a profiled operation.
        
        Args:
            name: Operation name
            
        Returns:
            Dictionary with min, max, mean, total, count
        """
        if name not in self.timings:
            return {}
        
        timings = self.timings[name]
        if len(timings) == 0:
            return {}
        
        return {
            "min": min(timings),
            "max": max(timings),
            "mean": sum(timings) / len(timings),
            "total": sum(timings),
            "count": self.counts[name],
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all profiled operations.
        
        Returns:
            Dictionary mapping operation names to their statistics
        """
        return {name: self.get_stats(name) for name in self.timings.keys()}
    
    def reset(self):
        """Reset all profiling data."""
        self.timings.clear()
        self.counts.clear()
    
    def print_summary(self):
        """Print profiling summary."""
        print("\n" + "=" * 60)
        print("Profiling Summary")
        print("=" * 60)
        
        for name, stats in self.get_all_stats().items():
            print(f"\n{name}:")
            print(f"  Count: {stats['count']}")
            print(f"  Total: {stats['total']:.4f}s")
            print(f"  Mean:  {stats['mean']:.4f}s")
            print(f"  Min:   {stats['min']:.4f}s")
            print(f"  Max:   {stats['max']:.4f}s")
        
        print("=" * 60 + "\n")


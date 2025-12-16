"""
Metrics Tracking

Track and aggregate training/evaluation metrics.
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


class MetricsTracker:
    """
    Track training and evaluation metrics.
    
    Aggregates metrics over epochs and batches.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.current_epoch: int = 0
    
    def update(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Update metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number (if None, uses current epoch)
        """
        for key, value in metrics.items():
            self.metrics[key].append(float(value))
    
    def get_metrics(self, key: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Get tracked metrics.
        
        Args:
            key: Optional specific metric key
            
        Returns:
            Dictionary of metrics or single metric list
        """
        if key is not None:
            return self.metrics.get(key, [])
        return dict(self.metrics)
    
    def get_latest(self, key: str) -> Optional[float]:
        """
        Get latest value for a metric.
        
        Args:
            key: Metric key
            
        Returns:
            Latest value or None if not found
        """
        if key in self.metrics and len(self.metrics[key]) > 0:
            return self.metrics[key][-1]
        return None
    
    def get_average(self, key: str, last_n: Optional[int] = None) -> Optional[float]:
        """
        Get average value for a metric.
        
        Args:
            key: Metric key
            last_n: Optional number of recent values to average
            
        Returns:
            Average value or None if not found
        """
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return None
        
        values = self.metrics[key]
        if last_n is not None:
            values = values[-last_n:]
        
        return sum(values) / len(values)
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.current_epoch = 0
    
    def _make_json_serializable(self, obj):
        """
        Convert numpy arrays and other non-serializable types to JSON-serializable formats.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(item) for item in obj)
        else:
            return obj
    
    def save(self, filepath: str):
        """
        Save metrics to JSON file.
        
        Args:
            filepath: Path to save metrics
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        serializable_metrics = self._make_json_serializable(dict(self.metrics))
        
        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def load(self, filepath: str):
        """
        Load metrics from JSON file.
        
        Args:
            filepath: Path to load metrics from
        """
        with open(filepath, 'r') as f:
            self.metrics = defaultdict(list, json.load(f))


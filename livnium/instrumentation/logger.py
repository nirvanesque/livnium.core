"""
Logging Infrastructure

Structured logging for LIVNIUM training and evaluation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class LivniumLogger:
    """
    Structured logger for LIVNIUM.
    
    Provides logging with different levels and optional file output.
    """
    
    def __init__(
        self,
        name: str = "livnium",
        log_dir: Optional[str] = None,
        level: int = logging.INFO,
        log_to_file: bool = True,
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files (if None, uses current directory)
            level: Logging level
            log_to_file: Whether to log to file
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            if log_dir is None:
                log_dir = Path.cwd() / "logs"
            else:
                log_dir = Path(log_dir)
            
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"livnium_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = ""):
        """
        Log training metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step/epoch
            prefix: Optional prefix for metric names
        """
        metric_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                               for k, v in metrics.items()])
        prefix_str = f"{prefix} - " if prefix else ""
        self.info(f"{prefix_str}Step {step} - {metric_str}")


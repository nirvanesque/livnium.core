"""
Market Dataset Loader

Loads financial market time series data (OHLCV).
"""

import torch
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
from livnium.datasets.base import LivniumDataset


class MarketDataset(LivniumDataset):
    """
    Market dataset loader.
    
    Loads OHLCV (Open/High/Low/Close/Volume) time series data.
    """
    
    def __init__(
        self,
        data_path: str,
        window_size: int = 20,
        step_size: int = 1,
        create_dummy: bool = True,
    ):
        """
        Initialize market dataset.
        
        Args:
            data_path: Path to CSV file with OHLCV data
            window_size: Size of sliding window
            step_size: Step size for sliding window
            create_dummy: Whether to create dummy data if file doesn't exist
        """
        self.data_path = Path(data_path)
        self.window_size = window_size
        self.step_size = step_size
        
        # Load data
        if self.data_path.exists():
            self.df = pd.read_csv(self.data_path)
            # Ensure required columns exist
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in self.df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
        elif create_dummy:
            print(f"Warning: {data_path} not found, creating dummy data")
            self.df = self._create_dummy_data(1000)
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Create windows
        self.windows = self._create_windows()
    
    def _create_dummy_data(self, size: int) -> pd.DataFrame:
        """Create dummy OHLCV data."""
        import numpy as np
        
        dates = pd.date_range(start="2020-01-01", periods=size, freq="D")
        prices = 100 + np.cumsum(np.random.randn(size) * 2)
        
        return pd.DataFrame({
            "Date": dates,
            "Open": prices + np.random.randn(size) * 0.5,
            "High": prices + np.abs(np.random.randn(size) * 1.0),
            "Low": prices - np.abs(np.random.randn(size) * 1.0),
            "Close": prices + np.random.randn(size) * 0.5,
            "Volume": np.random.randint(1000000, 10000000, size),
        })
    
    def _create_windows(self) -> list:
        """Create sliding windows from time series."""
        windows = []
        for i in range(0, len(self.df) - self.window_size, self.step_size):
            windows.append(i)
        return windows
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        start_idx = self.windows[idx]
        end_idx = start_idx + self.window_size
        
        window_df = self.df.iloc[start_idx:end_idx]
        
        # Extract OHLCV
        open_prices = torch.tensor(window_df["Open"].values, dtype=torch.float32)
        high_prices = torch.tensor(window_df["High"].values, dtype=torch.float32)
        low_prices = torch.tensor(window_df["Low"].values, dtype=torch.float32)
        close_prices = torch.tensor(window_df["Close"].values, dtype=torch.float32)
        volumes = torch.tensor(window_df["Volume"].values, dtype=torch.float32)
        
        # Dummy label: based on price trend
        price_change = (close_prices[-1] - close_prices[0]) / close_prices[0]
        if price_change > 0.02:
            label = 0  # Bull
        elif price_change < -0.02:
            label = 1  # Bear
        else:
            label = 2  # Neutral
        
        return {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
            "label": label,
        }


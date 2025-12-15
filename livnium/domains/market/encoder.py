"""
Market Domain Encoder: Financial Time Series → Initial State

Converts financial market data (OHLCV) into state vectors.
Uses kernel.physics for constraint generation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


class MarketEncoder(nn.Module):
    """
    Market encoder that converts financial time series to initial state.
    
    Takes OHLCV (Open/High/Low/Close/Volume) data and computes features:
    - Returns (price changes)
    - Volatility (rolling std)
    - Volume z-scores
    - Price range (High-Low normalized)
    - Price ratio (Close/Open - 1)
    
    Uses kernel physics for constraint generation.
    """
    
    def __init__(
        self,
        dim: int = 256,
        window: int = 14,
        standardize: bool = True
    ):
        """
        Initialize market encoder.
        
        Args:
            dim: Dimension of state vectors
            window: Rolling window size for features
            standardize: Whether to standardize features
        """
        super().__init__()
        self.dim = dim
        self.window = window
        self.standardize = standardize
        
        # Project 5 features to state dimension
        self.proj = nn.Linear(5, dim)
        
        # Optional MLP for richer representation
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
    
    def compute_features(
        self,
        close: torch.Tensor,
        high: torch.Tensor,
        low: torch.Tensor,
        open_price: torch.Tensor,
        volume: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute market features from OHLCV data.
        
        Args:
            close: Close prices [B, T] or [T]
            high: High prices [B, T] or [T]
            low: Low prices [B, T] or [T]
            open_price: Open prices [B, T] or [T]
            volume: Volume [B, T] or [T]
            
        Returns:
            Features tensor [B, T, 5] or [T, 5]
        """
        # Ensure batch dimension
        squeeze = False
        if close.dim() == 1:
            close = close.unsqueeze(0)
            high = high.unsqueeze(0)
            low = low.unsqueeze(0)
            open_price = open_price.unsqueeze(0)
            volume = volume.unsqueeze(0)
            squeeze = True
        
        B, T = close.shape
        
        # Compute returns
        returns = (close[:, 1:] - close[:, :-1]) / (close[:, :-1] + 1e-8)
        # Pad first element
        returns = torch.cat([torch.zeros(B, 1, device=close.device), returns], dim=1)
        
        # Compute volatility (rolling std of returns)
        vol = torch.zeros_like(returns)
        for t in range(self.window, T):
            vol[:, t] = returns[:, t-self.window+1:t+1].std(dim=1)
        
        # Compute volume z-scores
        volume_mean = torch.zeros_like(volume)
        volume_std = torch.ones_like(volume)
        for t in range(self.window, T):
            volume_mean[:, t] = volume[:, t-self.window+1:t+1].mean(dim=1)
            volume_std[:, t] = volume[:, t-self.window+1:t+1].std(dim=1) + 1e-8
        volume_z = (volume - volume_mean) / volume_std
        
        # Price range (normalized by open)
        price_range = (high - low) / (open_price + 1e-8)
        
        # Price ratio
        price_ratio = (close / (open_price + 1e-8)) - 1
        
        # Stack features: [return, vol, volume_z, range, ratio]
        features = torch.stack([
            returns,
            vol,
            volume_z,
            price_range,
            price_ratio
        ], dim=-1)  # [B, T, 5]
        
        # Standardize if requested
        if self.standardize:
            # Column-wise z-score
            mean = features.mean(dim=1, keepdim=True)  # [B, 1, 5]
            std = features.std(dim=1, keepdim=True) + 1e-8  # [B, 1, 5]
            features = (features - mean) / std
        
        if squeeze:
            features = features.squeeze(0)
        
        return features
    
    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode features to state vector.
        
        Args:
            features: Feature tensor [B, T, 5] or [T, 5] or [B, 5] or [5]
            
        Returns:
            State vector [B, dim] or [dim]
        """
        # Handle different input shapes
        if features.dim() == 1:
            # Single timestep: [5]
            features = features.unsqueeze(0).unsqueeze(0)  # [1, 1, 5]
            squeeze_out = True
        elif features.dim() == 2:
            if features.shape[-1] == 5:
                # Batch of timesteps: [B, 5] -> use mean
                features = features.unsqueeze(1)  # [B, 1, 5]
                squeeze_out = False
            else:
                # Time series: [T, 5] -> use mean
                features = features.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, 5]
                squeeze_out = True
        else:
            # [B, T, 5] -> mean over time
            features = features.mean(dim=1)  # [B, 5]
            features = features.unsqueeze(1)  # [B, 1, 5]
            squeeze_out = False
        
        # Project to state dimension
        h = self.proj(features.squeeze(1))  # [B, dim] or [1, dim]
        
        # Apply MLP
        h = self.mlp(h)
        
        if squeeze_out:
            h = h.squeeze(0)
        
        return h
    
    def generate_constraints(
        self,
        state: torch.Tensor,
        basin_state: torch.Tensor
    ) -> dict:
        """
        Generate constraints from state and basin (historical average).
        
        Uses kernel.physics for alignment/divergence calculations.
        
        Args:
            state: Current state vector
            basin_state: Basin (EMA of historical states)
            
        Returns:
            Dictionary of constraints
        """
        # Import here to avoid circular dependencies
        from livnium.kernel.physics import alignment, divergence, tension
        from livnium.engine.ops_torch import TorchOps
        
        ops = TorchOps()
        
        # Normalize for pure directional analysis
        state_n = state / (torch.norm(state, p=2, dim=-1, keepdim=True) + 1e-8)
        basin_n = basin_state / (torch.norm(basin_state, p=2, dim=-1, keepdim=True) + 1e-8)
        
        # Create state wrappers for kernel physics
        class StateWrapper:
            def __init__(self, vec):
                self._vec = vec
            def vector(self):
                return self._vec
            def norm(self):
                return torch.norm(self._vec, p=2)
        
        # Handle batch dimension
        if state_n.dim() == 1:
            state_n = state_n.unsqueeze(0)
            basin_n = basin_n.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        # Compute physics using kernel
        align_values = []
        div_values = []
        tens_values = []
        
        for i in range(state_n.shape[0]):
            s_state = StateWrapper(state_n[i])
            b_state = StateWrapper(basin_n[i])
            align_val = alignment(ops, s_state, b_state)
            div_val = divergence(ops, s_state, b_state)
            tens_val = tension(ops, div_val)
            
            align_values.append(torch.tensor(align_val, device=state.device, dtype=state.dtype))
            div_values.append(torch.tensor(div_val, device=state.device, dtype=state.dtype))
            tens_values.append(torch.tensor(tens_val, device=state.device, dtype=state.dtype))
        
        align = torch.stack(align_values)
        div = torch.stack(div_values)
        tens = torch.stack(tens_values)
        
        if squeeze:
            align = align.squeeze(0)
            div = div.squeeze(0)
            tens = tens.squeeze(0)
        
        return {
            "alignment": align,
            "divergence": div,
            "tension": tens,
            "state": state,
            "basin": basin_state,
        }
    
    def build_initial_state(
        self,
        close: torch.Tensor,
        high: torch.Tensor,
        low: torch.Tensor,
        open_price: torch.Tensor,
        volume: torch.Tensor,
        basin_window: int = 7,
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build initial state from market data.
        
        Args:
            close: Close prices [B, T] or [T]
            high: High prices [B, T] or [T]
            low: Low prices [B, T] or [T]
            open_price: Open prices [B, T] or [T]
            volume: Volume [B, T] or [T]
            basin_window: Window for EMA basin calculation
            add_noise: Whether to add symmetry-breaking noise
            
        Returns:
            Tuple of (initial_state, current_state_vector, basin_state_vector)
        """
        # Compute features
        features = self.compute_features(close, high, low, open_price, volume)
        
        # Encode current state (use most recent timestep)
        if features.dim() == 3:
            # [B, T, 5] -> use last timestep
            current_features = features[:, -1, :]  # [B, 5]
        elif features.dim() == 2:
            # [T, 5] -> use last timestep
            current_features = features[-1:, :]  # [1, 5]
        else:
            # [5] -> single timestep
            current_features = features
        
        v_current = self.encode(current_features)
        
        # Compute basin (EMA of recent states)
        if features.dim() == 3:
            # [B, T, 5] -> encode each timestep, then EMA
            T = features.shape[1]
            states = []
            for t in range(max(0, T - basin_window), T):
                state_t = self.encode(features[:, t, :])
                states.append(state_t)
            if states:
                # EMA with alpha=0.1
                basin = states[0]
                alpha = 0.1
                for state in states[1:]:
                    basin = alpha * state + (1 - alpha) * basin
            else:
                basin = v_current
        else:
            # Single timestep or short series
            basin = v_current
        
        # Initial state is combination
        h0 = v_current + basin
        
        if add_noise:
            from livnium.engine.config import defaults
            h0 = h0 + defaults.EPS_NOISE * torch.randn_like(h0)
        
        return h0, v_current, basin


"""
Market Domain Head: Regime Classification Head

Takes collapsed state and outputs logits for market regime classification.
Uses kernel.physics for alignment calculations.

Regimes: Bull / Bear / Neutral / Panic / Euphoria (5 classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum


class Regime5(str, Enum):
    """Market regime classification."""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    PANIC = "panic"
    EUPHORIA = "euphoria"


class MarketHead(nn.Module):
    """
    Market regime classification head.
    
    Uses kernel.physics for alignment and tension calculations to classify
    market regimes based on geometric relationships between current state
    and historical basin.
    
    Outputs logits for 5 regimes: Bull, Bear, Neutral, Panic, Euphoria
    """
    
    def __init__(self, dim: int):
        """
        Initialize market head.
        
        Args:
            dim: Dimension of input state vector
        """
        super().__init__()
        self.dim = dim
        
        # Linear stack for classification
        # Features: [h_final, alignment, divergence, tension, dist, r_current, r_basin, r_final]
        # Total: dim + 7 features
        self.fc = nn.Sequential(
            nn.Linear(dim + 7, dim + 7),
            nn.ReLU(),
            nn.Linear(dim + 7, 5)  # 5 regimes
        )
    
    def forward(
        self,
        h_final: torch.Tensor,
        v_current: torch.Tensor,
        v_basin: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: state → regime logits.
        
        Uses kernel.physics for alignment/tension calculations.
        
        Args:
            h_final: Collapsed state vector [batch, dim] or [dim]
            v_current: Current state vector [batch, dim] or [dim]
            v_basin: Basin (EMA) state vector [batch, dim] or [dim]
            
        Returns:
            Logits tensor (batch, 5) for [Bull, Bear, Neutral, Panic, Euphoria]
        """
        # Ensure batch dimension
        squeeze = False
        if h_final.dim() == 1:
            h_final = h_final.unsqueeze(0)
            v_current = v_current.unsqueeze(0)
            v_basin = v_basin.unsqueeze(0)
            squeeze = True
        
        # Normalize for directional analysis
        v_current_n = F.normalize(v_current, dim=-1)
        v_basin_n = F.normalize(v_basin, dim=-1)
        
        # Use kernel.physics for alignment
        from livnium.kernel.physics import alignment, divergence, tension
        from livnium.engine.ops_torch import TorchOps
        
        ops = TorchOps()
        
        # Create state wrappers for kernel physics
        class StateWrapper:
            def __init__(self, vec):
                self._vec = vec
            def vector(self):
                return self._vec
            def norm(self):
                return torch.norm(self._vec, p=2)
        
        # Compute alignment using kernel.physics
        align_values = []
        div_values = []
        tens_values = []
        
        for i in range(v_current_n.shape[0]):
            current_state = StateWrapper(v_current_n[i])
            basin_state = StateWrapper(v_basin_n[i])
            align_val = alignment(ops, current_state, basin_state)
            div_val = divergence(ops, current_state, basin_state)
            tens_val = tension(ops, div_val)
            
            align_values.append(torch.tensor(align_val, device=v_current.device, dtype=v_current.dtype))
            div_values.append(torch.tensor(div_val, device=v_current.device, dtype=v_current.dtype))
            tens_values.append(torch.tensor(tens_val, device=v_current.device, dtype=v_current.dtype))
        
        align = torch.stack(align_values).unsqueeze(-1)  # [B, 1]
        div = torch.stack(div_values).unsqueeze(-1)  # [B, 1]
        tens = torch.stack(tens_values).unsqueeze(-1)  # [B, 1]
        
        # Radial geometry: distance and norms
        dist = (v_current - v_basin).norm(p=2, dim=-1, keepdim=True)
        r_current = v_current.norm(p=2, dim=-1, keepdim=True)
        r_basin = v_basin.norm(p=2, dim=-1, keepdim=True)
        r_final = h_final.norm(p=2, dim=-1, keepdim=True)
        
        # Feature set
        features = torch.cat([
            h_final,
            align,
            div,
            tens,
            dist,
            r_current,
            r_basin,
            r_final
        ], dim=-1)
        
        logits = self.fc(features)
        
        if squeeze:
            logits = logits.squeeze(0)
        
        return logits
    
    @staticmethod
    def classify_regime(
        alignment: float,
        tension: float,
        # Use defaults if not provided, but here we can't use defaults.XYZ as default arg directly 
        # unless we import it at top level.  Alternatively, use None and fill in body.
        # But to match the pattern:
        align_pos: float = None,
        align_strong_pos: float = 0.40,
        align_neg: float = None,
        align_strong_neg: float = -0.40,
        low_tension: float = 0.35,
        med_tension: float = 0.80,
        panic_tension: float = 1.00,
    ) -> Regime5:
        """
        Classify regime from alignment and tension (heuristic).
        
        This is a rule-based classifier. The neural network head learns
        a more sophisticated mapping, but this provides interpretability.
        
        Args:
            alignment: Alignment value from kernel.physics
            tension: Tension value from kernel.physics
            align_pos: Positive alignment threshold (defaults to MARKET_ALIGN_POS)
            align_strong_pos: Strong positive alignment threshold
            align_neg: Negative alignment threshold (defaults to MARKET_ALIGN_NEG)
            align_strong_neg: Strong negative alignment threshold
            low_tension: Low tension threshold
            med_tension: Medium tension threshold
            panic_tension: Panic tension threshold
            
        Returns:
            Regime5 classification
        """
        from livnium.engine.config import defaults
        
        if align_pos is None:
            align_pos = defaults.MARKET_ALIGN_POS
        if align_neg is None:
            align_neg = defaults.MARKET_ALIGN_NEG
        # PANIC: strong negative alignment + very high tension
        if alignment <= align_strong_neg and tension >= panic_tension:
            return Regime5.PANIC
        
        # EUPHORIA: strong positive alignment + at least medium tension
        if alignment >= align_strong_pos and tension >= med_tension:
            return Regime5.EUPHORIA
        
        # Bull: positive alignment, not too unstable
        if alignment >= align_pos and tension <= med_tension:
            return Regime5.BULL
        
        # Bear: negative alignment, some tension
        if alignment <= align_neg and tension >= low_tension:
            return Regime5.BEAR
        
        # Fallback
        return Regime5.NEUTRAL


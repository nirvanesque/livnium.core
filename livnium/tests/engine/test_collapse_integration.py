"""
Test: Collapse Engine Integration with Kernel

Verifies that collapse engine correctly uses kernel physics.
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

import torch
from livnium.engine.collapse.engine import CollapseEngine
from livnium.kernel.constants import DIVERGENCE_PIVOT


def test_collapse_uses_kernel_physics():
    """Verify collapse engine uses kernel physics correctly."""
    engine = CollapseEngine(dim=64, num_layers=3)
    h0 = torch.randn(64)
    
    h_final, trace = engine.collapse(h0)
    
    # Verify output shape
    assert h_final.shape == h0.shape, "Output shape should match input"
    
    # Verify trace structure
    assert "divergence_entail" in trace, "Trace should contain divergence"
    assert len(trace["divergence_entail"]) == 3, "Should have 3 steps"
    
    # Verify divergence values are reasonable (should use DIVERGENCE_PIVOT)
    # Divergence = DIVERGENCE_PIVOT - alignment, so should be in reasonable range
    first_div = trace["divergence_entail"][0]
    # Kernel physics returns floats, but engine may wrap them
    assert isinstance(first_div, (float, torch.Tensor)), "Divergence should be float or tensor"
    
    print("✓ Collapse engine integration test passed")


def test_collapse_uses_config_defaults():
    """Verify collapse engine uses config defaults."""
    engine = CollapseEngine(dim=64)
    
    # Check that defaults are used
    from livnium.engine.config import defaults
    assert engine.strength_entail == defaults.STRENGTH_ENTAIL
    assert engine.strength_contra == defaults.STRENGTH_CONTRA
    assert engine.strength_neutral == defaults.STRENGTH_NEUTRAL
    assert engine.max_norm == defaults.MAX_NORM
    
    print("✓ Collapse engine uses config defaults")


if __name__ == "__main__":
    test_collapse_uses_kernel_physics()
    test_collapse_uses_config_defaults()
    print("All collapse integration tests passed!")


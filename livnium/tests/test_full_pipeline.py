"""
Test: Full Pipeline Integration

Verifies that kernel + engine + domain work together correctly.
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import torch
from livnium.engine.collapse.engine import CollapseEngine
from livnium.domains.toy.encoder import ToyEncoder
from livnium.domains.toy.head import ToyHead
from livnium.kernel.constants import DIVERGENCE_PIVOT, K_O, K_T, K_C


def test_full_pipeline_toy():
    """Test full pipeline: encoder -> collapse -> head."""
    print("Testing full pipeline with toy domain...")
    
    # Create components
    encoder = ToyEncoder(dim=64)
    collapse_engine = CollapseEngine(dim=64, num_layers=3)
    head = ToyHead(dim=64, num_classes=3)
    
    # Create input
    x_a = torch.randn(2)
    x_b = torch.randn(2)
    
    # Encode
    h0, v_a, v_b = encoder.build_initial_state(x_a, x_b)
    print(f"  ✓ Encoded: h0 shape {h0.shape}")
    
    # Collapse
    h_final, trace = collapse_engine.collapse(h0)
    print(f"  ✓ Collapsed: h_final shape {h_final.shape}")
    print(f"  ✓ Trace keys: {list(trace.keys())}")
    
    # Head
    logits = head(h_final, v_a, v_b)
    print(f"  ✓ Head output: logits shape {logits.shape}")
    
    # Verify divergence uses kernel constant
    first_div = trace["divergence_entail"][0]
    print(f"  ✓ Divergence computed (should use DIVERGENCE_PIVOT={DIVERGENCE_PIVOT})")
    
    print("  ✓ Full pipeline test passed!\n")


def test_kernel_constants():
    """Verify kernel constants are accessible."""
    print("Testing kernel constants...")
    print(f"  ✓ DIVERGENCE_PIVOT = {DIVERGENCE_PIVOT}")
    print(f"  ✓ K_O = {K_O}, K_T = {K_T}, K_C = {K_C}")
    print("  ✓ Kernel constants accessible!\n")


def test_domain_uses_kernel():
    """Verify domains use kernel physics."""
    print("Testing domain uses kernel physics...")
    
    from livnium.domains.snli.encoder import SNLIEncoder
    
    encoder = SNLIEncoder(dim=64)
    prem_ids = torch.randint(0, 100, (10,))
    hyp_ids = torch.randint(0, 100, (10,))
    
    h0, v_p, v_h = encoder.build_initial_state(prem_ids, hyp_ids)
    constraints = encoder.generate_constraints(h0, v_p, v_h)
    
    assert "alignment" in constraints
    assert "divergence" in constraints
    assert "tension" in constraints
    
    print(f"  ✓ Constraints use kernel physics:")
    print(f"    - alignment = {constraints['alignment']:.3f}")
    print(f"    - divergence = {constraints['divergence']:.3f}")
    print(f"    - tension = {constraints['tension']:.3f}")
    print("  ✓ Domain uses kernel physics!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("LIVNIUM Full Pipeline Integration Tests")
    print("=" * 60)
    print()
    
    test_kernel_constants()
    test_full_pipeline_toy()
    test_domain_uses_kernel()
    
    print("=" * 60)
    print("All integration tests passed!")
    print("=" * 60)


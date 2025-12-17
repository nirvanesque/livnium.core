"""
Recursive Engine Smoke Test

Validates basic recursive geometry functionality without full fractal activation.
Tests the minimal viable recursive system: one parent, one child, energy conservation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from livnium.classical import LivniumCoreSystem, LivniumCoreConfig
from livnium.recursive import RecursiveGeometryEngine, MokshaEngine

def test_recursive_smoke():
    """Minimal recursive setup: depth=2, verify conservation and Moksha metric."""
    
    print("Recursive Engine Smoke Test")
    print("=" * 60)
    
    # 1. Create minimal base geometry (3x3x3)
    config = LivniumCoreConfig(lattice_size=3)
    base = LivniumCoreSystem(config)
    initial_sw = base.get_total_symbolic_weight()
    
    print(f"✓ Base geometry: {base.lattice_size}x{base.lattice_size}x{base.lattice_size}")
    print(f"  Initial SW: {initial_sw}")
    
    # 2. Create recursive engine (depth=2, not full fractal)
    engine = RecursiveGeometryEngine(base_geometry=base, max_depth=2)
    total_capacity = engine.get_total_capacity()
    
    print(f"✓ Recursive engine: depth={engine.max_depth}")
    print(f"  Total capacity: {total_capacity} cells")
    
    # 3. Verify energy conservation after rotation
    from livnium.classical.livnium_core_system import RotationAxis
    engine.apply_recursive_rotation(level_id=0, axis=RotationAxis.X, quarter_turns=1)
    
    final_sw = base.get_total_symbolic_weight()
    sw_preserved = abs(initial_sw - final_sw) < 1e-6
    
    print(f"✓ Energy conservation after rotation:")
    print(f"  SW preserved: {sw_preserved} (diff: {abs(initial_sw - final_sw):.10f})")
    
    # 4. Verify Moksha metric is computable
    moksha = MokshaEngine(engine)
    convergence_state = moksha.check_convergence()
    convergence_score = moksha.get_convergence_score()
    
    print(f"✓ Moksha metric computable:")
    print(f"  State: {convergence_state.name}")
    print(f"  Score: {convergence_score:.4f}")
    
    # 5. Verify no runaway recursion (child count is bounded)
    level_0_cells = len(engine.levels[0].geometry.lattice)
    assert level_0_cells == 27, f"Expected 27 cells at level 0, got {level_0_cells}"
    
    print(f"✓ Recursion bounded:")
    print(f"  Level 0 cells: {level_0_cells}")
    
    # Final assertion
    assert sw_preserved, "Symbolic weight not conserved!"
    assert convergence_score >= 0.0, "Moksha metric invalid!"
    
    print("=" * 60)
    print("✅ All smoke tests passed!")
    
    return True

if __name__ == "__main__":
    success = test_recursive_smoke()
    sys.exit(0 if success else 1)

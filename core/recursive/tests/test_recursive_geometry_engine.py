"""
Test assertions for RecursiveGeometryEngine.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.recursive.recursive_geometry_engine import RecursiveGeometryEngine, GeometryLevel
from core.classical.livnium_core_system import LivniumCoreSystem, RotationAxis
from core.config import LivniumCoreConfig


def test_basic_initialization():
    """Test basic recursive engine initialization."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    
    engine = RecursiveGeometryEngine(base_geometry, max_depth=2)
    
    assert engine.base_geometry == base_geometry, "Base geometry should be set"
    assert engine.max_depth == 2, "Max depth should be 2"
    assert 0 in engine.levels, "Level 0 should exist"
    assert engine.levels[0].level_id == 0, "Level 0 should have id 0"
    assert engine.levels[0].geometry == base_geometry, "Level 0 should have base geometry"


def test_hierarchy_building():
    """Test recursive hierarchy building."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    assert len(engine.levels) >= 1, "Should have at least level 0"
    assert 0 in engine.levels, "Level 0 should exist"
    
    level_0 = engine.levels[0]
    assert level_0.parent is None, "Level 0 should have no parent"
    assert isinstance(level_0.children, dict), "Level 0 should have children dict"


def test_geometry_level():
    """Test GeometryLevel class."""
    config = LivniumCoreConfig(lattice_size=3)
    geometry = LivniumCoreSystem(config)
    
    level = GeometryLevel(level_id=0, geometry=geometry)
    
    assert level.level_id == 0, "Level ID should be 0"
    assert level.geometry == geometry, "Geometry should match"
    assert level.parent is None, "Parent should be None"
    assert len(level.children) == 0, "Should have no children initially"
    assert level.scale_factor == 1, "Scale factor should be 1"
    
    total_cells = level.get_total_cells()
    assert total_cells == 27, "3×3×3 should have 27 cells"
    
    recursive_total = level.get_total_cells_recursive()
    assert recursive_total == 27, "Recursive total should equal cells when no children"


def test_subdivide_cell():
    """Test cell subdivision."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    
    engine = RecursiveGeometryEngine(base_geometry, max_depth=2)
    
    # Try to subdivide a cell
    result = engine.subdivide_cell(0, (0, 0, 0))
    
    # Should succeed if depth allows
    assert isinstance(result, bool), "Should return boolean"
    
    # Check if child was created
    level_0 = engine.levels[0]
    if result:
        assert (0, 0, 0) in level_0.children, "Child should be created if subdivision succeeded"


def test_recursive_rotation():
    """Test recursive rotation."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    # Get initial SW
    initial_sw = base_geometry.get_total_symbolic_weight()
    
    # Apply recursive rotation
    engine.apply_recursive_rotation(0, RotationAxis.X, quarter_turns=1)
    
    # SW should be preserved (rotation invariant)
    final_sw = base_geometry.get_total_symbolic_weight()
    assert abs(final_sw - initial_sw) < 1e-6, "SW should be preserved under rotation"


def test_recursive_observer():
    """Test recursive observer."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    # Get observer at level 0
    observer = engine.get_recursive_observer(0)
    assert observer == (0, 0, 0), "Level 0 observer should be at (0,0,0)"


def test_total_capacity():
    """Test total capacity calculation."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    
    engine = RecursiveGeometryEngine(base_geometry, max_depth=0)
    
    capacity = engine.get_total_capacity()
    assert capacity == 27, "Capacity should be 27 for 3×3×3 with depth 0"
    
    # With subdivision, capacity should increase
    engine2 = RecursiveGeometryEngine(base_geometry, max_depth=1)
    capacity2 = engine2.get_total_capacity()
    assert capacity2 >= 27, "Capacity should be >= 27 with subdivision"


def test_level_statistics():
    """Test level statistics."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    stats = engine.get_level_statistics()
    
    assert isinstance(stats, dict), "Should return dictionary"
    assert 0 in stats, "Should have stats for level 0"
    
    level_0_stats = stats[0]
    assert 'total_cells' in level_0_stats, "Should have total_cells"
    assert 'total_cells_recursive' in level_0_stats, "Should have total_cells_recursive"
    assert 'num_children' in level_0_stats, "Should have num_children"
    assert 'scale_factor' in level_0_stats, "Should have scale_factor"
    assert 'lattice_size' in level_0_stats, "Should have lattice_size"


def test_project_state_downward():
    """Test downward state projection."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    state = {'constraints': {'test': True}, 'values': {'sw': 100.0}}
    
    # Project downward (if level 1 exists)
    if 1 in engine.levels:
        projected = engine.project_state_downward(0, 1, state)
        assert isinstance(projected, dict), "Should return dictionary"


def test_project_state_upward():
    """Test upward state projection."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    state = {'constraints': {'test': True}, 'values': {'sw': 100.0}}
    
    # Project upward (if level 1 exists)
    if 1 in engine.levels:
        projected = engine.project_state_upward(1, 0, state)
        assert isinstance(projected, dict), "Should return dictionary"


def test_compress_entanglement():
    """Test entanglement compression."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    # Create entangled pairs
    entangled_pairs = [((0, 0, 0), (1, 1, 1)), ((0, 0, 1), (1, 1, 0))]
    
    compressed = engine.compress_entanglement(0, entangled_pairs)
    
    assert isinstance(compressed, dict), "Should return dictionary"


def test_moksha_check():
    """Test moksha convergence check."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    moksha = engine.check_moksha()
    assert isinstance(moksha, bool), "Should return boolean"


def test_final_truth():
    """Test final truth export."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    truth = engine.get_final_truth()
    assert isinstance(truth, dict), "Should return dictionary"


def test_subdivision_rule():
    """Test default subdivision rule."""
    config = LivniumCoreConfig(lattice_size=5)
    base_geometry = LivniumCoreSystem(config)
    
    engine = RecursiveGeometryEngine(base_geometry, max_depth=2)
    
    # Test subdivision rule directly
    child_geometry = engine._default_subdivision_rule(base_geometry, (0, 0, 0), depth=1)
    
    if child_geometry:
        assert isinstance(child_geometry, LivniumCoreSystem), "Should return LivniumCoreSystem"
        assert child_geometry.config.lattice_size < base_geometry.config.lattice_size, \
            "Child should be smaller than parent"


if __name__ == "__main__":
    print("Running RecursiveGeometryEngine tests...")
    
    test_basic_initialization()
    print("✓ Basic initialization")
    
    test_hierarchy_building()
    print("✓ Hierarchy building")
    
    test_geometry_level()
    print("✓ Geometry level")
    
    test_subdivide_cell()
    print("✓ Subdivide cell")
    
    test_recursive_rotation()
    print("✓ Recursive rotation")
    
    test_recursive_observer()
    print("✓ Recursive observer")
    
    test_total_capacity()
    print("✓ Total capacity")
    
    test_level_statistics()
    print("✓ Level statistics")
    
    test_project_state_downward()
    print("✓ Project state downward")
    
    test_project_state_upward()
    print("✓ Project state upward")
    
    test_compress_entanglement()
    print("✓ Compress entanglement")
    
    test_moksha_check()
    print("✓ Moksha check")
    
    test_final_truth()
    print("✓ Final truth")
    
    test_subdivision_rule()
    print("✓ Subdivision rule")
    
    print("\nAll tests passed! ✓")


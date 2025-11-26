"""
Test assertions for RecursiveConservation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.recursive.recursive_conservation import RecursiveConservation
from core.recursive.recursive_geometry_engine import RecursiveGeometryEngine
from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig


def test_basic_initialization():
    """Test basic conservation initialization."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    conservation = RecursiveConservation(engine)
    
    assert conservation.recursive_engine == engine, "Should reference recursive engine"


def test_verify_level_conservation():
    """Test level conservation verification."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    conservation = RecursiveConservation(engine)
    
    result = conservation.verify_level_conservation(0)
    
    assert isinstance(result, dict), "Should return dictionary"
    assert 'sw_conserved' in result, "Should have sw_conserved"
    assert 'class_counts_conserved' in result, "Should have class_counts_conserved"
    assert 'actual_sw' in result, "Should have actual_sw"
    assert 'expected_sw' in result, "Should have expected_sw"
    
    assert isinstance(result['sw_conserved'], bool), "sw_conserved should be boolean"
    assert isinstance(result['class_counts_conserved'], bool), "class_counts_conserved should be boolean"
    
    # For a valid geometry, conservation should hold
    assert result['sw_conserved'], "SW should be conserved for valid geometry"
    assert result['class_counts_conserved'], "Class counts should be conserved"


def test_verify_recursive_conservation():
    """Test recursive conservation verification."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    conservation = RecursiveConservation(engine)
    
    results = conservation.verify_recursive_conservation()
    
    assert isinstance(results, dict), "Should return dictionary"
    assert 0 in results, "Should have results for level 0"
    
    level_0_result = results[0]
    assert 'sw_conserved' in level_0_result, "Should have sw_conserved"
    assert 'class_counts_conserved' in level_0_result, "Should have class_counts_conserved"


def test_propagate_conservation_downward():
    """Test downward conservation propagation."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    conservation = RecursiveConservation(engine)
    
    result = conservation.propagate_conservation_downward(0)
    
    assert isinstance(result, bool), "Should return boolean"


def test_aggregate_conservation_upward():
    """Test upward conservation aggregation."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    conservation = RecursiveConservation(engine)
    
    result = conservation.aggregate_conservation_upward(0)
    
    assert isinstance(result, dict), "Should return dictionary"
    
    # If there are children, should have aggregated values
    if 1 in engine.levels:
        level_0 = engine.levels[0]
        if len(level_0.children) > 0:
            assert 'total_child_sw' in result, "Should have total_child_sw"
            assert 'total_child_counts' in result, "Should have total_child_counts"


def test_conservation_statistics():
    """Test conservation statistics."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    conservation = RecursiveConservation(engine)
    
    stats = conservation.get_conservation_statistics()
    
    assert isinstance(stats, dict), "Should return dictionary"
    assert 0 in stats, "Should have stats for level 0"
    
    level_0_stats = stats[0]
    assert 'conservation' in level_0_stats, "Should have conservation"
    assert 'aggregated_from_children' in level_0_stats, "Should have aggregated_from_children"


def test_invalid_level():
    """Test conservation with invalid level."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    conservation = RecursiveConservation(engine)
    
    result = conservation.verify_level_conservation(999)
    assert result == {}, "Should return empty dict for invalid level"
    
    result2 = conservation.propagate_conservation_downward(999)
    assert result2 == False, "Should return False for invalid level"
    
    result3 = conservation.aggregate_conservation_upward(999)
    assert result3 == {}, "Should return empty dict for invalid level"


if __name__ == "__main__":
    print("Running RecursiveConservation tests...")
    
    test_basic_initialization()
    print("✓ Basic initialization")
    
    test_verify_level_conservation()
    print("✓ Verify level conservation")
    
    test_verify_recursive_conservation()
    print("✓ Verify recursive conservation")
    
    test_propagate_conservation_downward()
    print("✓ Propagate conservation downward")
    
    test_aggregate_conservation_upward()
    print("✓ Aggregate conservation upward")
    
    test_conservation_statistics()
    print("✓ Conservation statistics")
    
    test_invalid_level()
    print("✓ Invalid level")
    
    print("\nAll tests passed! ✓")


"""
Test assertions for RecursiveProjection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.recursive.recursive_projection import RecursiveProjection
from core.recursive.recursive_geometry_engine import RecursiveGeometryEngine
from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig


def test_basic_initialization():
    """Test basic projection initialization."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    projection = RecursiveProjection(engine)
    
    assert projection.recursive_engine == engine, "Should reference recursive engine"


def test_project_downward():
    """Test downward projection."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    projection = RecursiveProjection(engine)
    
    state = {
        'constraints': {'test_constraint': True},
        'values': {'sw': 100.0, 'face_exposure': 2}
    }
    
    # Project downward if level 1 exists
    if 1 in engine.levels:
        projected = projection.project_downward(0, 1, state)
        assert isinstance(projected, dict), "Should return dictionary"
    
    # Test invalid projection (target <= source)
    try:
        projection.project_downward(1, 0, state)
        assert False, "Should raise ValueError for invalid projection"
    except ValueError:
        pass  # Expected


def test_project_upward():
    """Test upward projection."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    projection = RecursiveProjection(engine)
    
    state = {
        'constraints': {'test_constraint': True},
        'values': {'sw': 50.0, 'face_exposure': 1}
    }
    
    # Project upward if level 1 exists
    if 1 in engine.levels:
        projected = projection.project_upward(1, 0, state)
        assert isinstance(projected, dict), "Should return dictionary"
    
    # Test invalid projection (target >= source)
    try:
        projection.project_upward(0, 1, state)
        assert False, "Should raise ValueError for invalid projection"
    except ValueError:
        pass  # Expected


def test_project_with_missing_levels():
    """Test projection with missing levels."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=0)
    
    projection = RecursiveProjection(engine)
    
    state = {'constraints': {}, 'values': {}}
    
    # Try to project to non-existent level
    result = projection.project_downward(0, 1, state)
    assert result == {}, "Should return empty dict for missing level"


def test_project_constraints():
    """Test constraint projection."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    projection = RecursiveProjection(engine)
    
    # Test with constraints only
    state = {'constraints': {'test': True}}
    
    if 1 in engine.levels:
        projected = projection.project_downward(0, 1, state)
        assert isinstance(projected, dict), "Should return dictionary"


def test_project_values():
    """Test value projection."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    projection = RecursiveProjection(engine)
    
    # Test with values only
    state = {'values': {'sw': 100.0}}
    
    if 1 in engine.levels:
        projected = projection.project_downward(0, 1, state)
        assert isinstance(projected, dict), "Should return dictionary"


def test_aggregate_values():
    """Test value aggregation."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    projection = RecursiveProjection(engine)
    
    state = {'values': {'sw': 50.0}}
    
    if 1 in engine.levels:
        aggregated = projection.project_upward(1, 0, state)
        assert isinstance(aggregated, dict), "Should return dictionary"


if __name__ == "__main__":
    print("Running RecursiveProjection tests...")
    
    test_basic_initialization()
    print("✓ Basic initialization")
    
    test_project_downward()
    print("✓ Project downward")
    
    test_project_upward()
    print("✓ Project upward")
    
    test_project_with_missing_levels()
    print("✓ Project with missing levels")
    
    test_project_constraints()
    print("✓ Project constraints")
    
    test_project_values()
    print("✓ Project values")
    
    test_aggregate_values()
    print("✓ Aggregate values")
    
    print("\nAll tests passed! ✓")


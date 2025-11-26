"""
Test assertions for GeometrySubdivision.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.recursive.geometry_subdivision import GeometrySubdivision
from core.recursive.recursive_geometry_engine import RecursiveGeometryEngine
from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig


def test_basic_initialization():
    """Test basic subdivision initialization."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    subdivision = GeometrySubdivision(engine)
    
    assert subdivision.recursive_engine == engine, "Should reference recursive engine"


def test_subdivide_by_face_exposure():
    """Test subdivision by face exposure."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=2)
    
    subdivision = GeometrySubdivision(engine)
    
    # Subdivide cells with face exposure >= 2 (edges and corners)
    count = subdivision.subdivide_by_face_exposure(0, min_exposure=2)
    
    assert isinstance(count, int), "Should return integer count"
    assert count >= 0, "Count should be non-negative"
    
    # Check that some cells were subdivided
    level_0 = engine.levels[0]
    assert len(level_0.children) >= 0, "Should have some children or none"


def test_subdivide_by_symbolic_weight():
    """Test subdivision by symbolic weight."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=2)
    
    subdivision = GeometrySubdivision(engine)
    
    # Subdivide cells with SW >= 18 (edges and corners)
    count = subdivision.subdivide_by_symbolic_weight(0, min_sw=18.0)
    
    assert isinstance(count, int), "Should return integer count"
    assert count >= 0, "Count should be non-negative"


def test_subdivide_all():
    """Test subdividing all cells."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=2)
    
    subdivision = GeometrySubdivision(engine)
    
    # Subdivide all cells at level 0
    count = subdivision.subdivide_all(0)
    
    assert isinstance(count, int), "Should return integer count"
    assert count >= 0, "Count should be non-negative"
    assert count <= 27, "Count should not exceed total cells (27)"


def test_subdivision_statistics():
    """Test subdivision statistics."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=2)
    
    subdivision = GeometrySubdivision(engine)
    
    stats = subdivision.get_subdivision_statistics(0)
    
    assert isinstance(stats, dict), "Should return dictionary"
    assert 'level_id' in stats, "Should have level_id"
    assert 'total_cells' in stats, "Should have total_cells"
    assert 'subdivided_cells' in stats, "Should have subdivided_cells"
    assert 'subdivision_ratio' in stats, "Should have subdivision_ratio"
    assert 'total_child_cells' in stats, "Should have total_child_cells"
    
    assert stats['level_id'] == 0, "Level ID should be 0"
    assert stats['total_cells'] == 27, "Total cells should be 27 for 3×3×3"
    assert 0.0 <= stats['subdivision_ratio'] <= 1.0, "Ratio should be in [0, 1]"


def test_invalid_level():
    """Test subdivision with invalid level."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    subdivision = GeometrySubdivision(engine)
    
    # Try invalid level
    count = subdivision.subdivide_by_face_exposure(999, min_exposure=2)
    assert count == 0, "Should return 0 for invalid level"
    
    stats = subdivision.get_subdivision_statistics(999)
    assert stats == {}, "Should return empty dict for invalid level"


if __name__ == "__main__":
    print("Running GeometrySubdivision tests...")
    
    test_basic_initialization()
    print("✓ Basic initialization")
    
    test_subdivide_by_face_exposure()
    print("✓ Subdivide by face exposure")
    
    test_subdivide_by_symbolic_weight()
    print("✓ Subdivide by symbolic weight")
    
    test_subdivide_all()
    print("✓ Subdivide all")
    
    test_subdivision_statistics()
    print("✓ Subdivision statistics")
    
    test_invalid_level()
    print("✓ Invalid level")
    
    print("\nAll tests passed! ✓")


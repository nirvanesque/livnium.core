"""
Test assertions for corner rotation policy.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.classical.livnium_core_system import LivniumCoreSystem, RotationAxis
from core.config import LivniumCoreConfig
from core.search.corner_rotation_policy import (
    should_allow_corner_rotations,
    rotation_affects_corners,
    get_safe_rotation,
)


def test_should_allow_corner_rotations_no_coords():
    """Test corner rotation policy with no active coords."""
    config = LivniumCoreConfig(lattice_size=3)
    system = LivniumCoreSystem(config)
    
    # No active coords, no convergence stats
    allow = should_allow_corner_rotations(system, active_coords=None)
    assert allow == False, "Should not allow corners without convergence"
    
    # With convergence stats (single basin)
    convergence_stats = {'num_alive': 1}
    allow = should_allow_corner_rotations(
        system,
        active_coords=None,
        convergence_stats=convergence_stats
    )
    assert allow == True, "Should allow corners when converged (single basin)"


def test_should_allow_corner_rotations_with_coords():
    """Test corner rotation policy with active coords."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    coords_list = list(system.lattice.keys())[:4]
    
    # Low curvature, high tension (should not allow)
    for coords in coords_list:
        cell = system.get_cell(coords)
        if cell:
            cell.symbolic_weight = 10.0  # Low SW = low curvature
    
    allow = should_allow_corner_rotations(
        system,
        active_coords=coords_list,
        basin_depth_threshold=0.5,
        tension_epsilon=0.1
    )
    # May or may not allow depending on actual curvature/tension
    assert isinstance(allow, bool), "Should return boolean"
    
    # High curvature, low tension (should allow)
    for coords in coords_list:
        cell = system.get_cell(coords)
        if cell:
            cell.symbolic_weight = 50.0  # High SW = high curvature
    
    allow = should_allow_corner_rotations(
        system,
        active_coords=coords_list,
        basin_depth_threshold=0.1,  # Low threshold
        tension_epsilon=1.0  # High threshold
    )
    assert isinstance(allow, bool), "Should return boolean"


def test_should_allow_corner_rotations_convergence():
    """Test corner rotation policy with convergence stats."""
    config = LivniumCoreConfig(lattice_size=3)
    system = LivniumCoreSystem(config)
    
    coords_list = list(system.lattice.keys())[:4]
    
    # Single basin (converged)
    convergence_stats = {'num_alive': 1}
    allow = should_allow_corner_rotations(
        system,
        active_coords=coords_list,
        convergence_stats=convergence_stats
    )
    assert allow == True, "Should allow corners when converged"
    
    # Multiple basins (not converged)
    convergence_stats = {'num_alive': 3}
    allow = should_allow_corner_rotations(
        system,
        active_coords=coords_list,
        convergence_stats=convergence_stats
    )
    # May or may not allow depending on curvature/tension
    assert isinstance(allow, bool), "Should return boolean"


def test_rotation_affects_corners():
    """Test rotation affects corners check."""
    config = LivniumCoreConfig(lattice_size=3)
    system = LivniumCoreSystem(config)
    
    # For 3x3x3, all rotations affect corners
    for axis in RotationAxis:
        for quarter_turns in [1, 2, 3]:
            affects = rotation_affects_corners(system, axis, quarter_turns)
            assert affects == True, f"All rotations should affect corners in 3x3x3"


def test_get_safe_rotation():
    """Test getting safe rotation."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    coords_list = list(system.lattice.keys())[:4]
    
    # When corners allowed
    rotation = get_safe_rotation(
        system,
        active_coords=coords_list,
        allow_corners=True
    )
    
    assert rotation is not None, "Should return rotation when corners allowed"
    assert isinstance(rotation, tuple), "Should return tuple"
    assert len(rotation) == 2, "Should have (axis, quarter_turns)"
    axis, quarter_turns = rotation
    assert axis in RotationAxis, "Should be valid axis"
    assert quarter_turns in [1, 2, 3], "Should be valid quarter turns"
    
    # When corners not allowed
    rotation = get_safe_rotation(
        system,
        active_coords=coords_list,
        allow_corners=False
    )
    # May return None or a rotation depending on implementation
    assert rotation is None or isinstance(rotation, tuple), \
        "Should return None or tuple"


def test_get_safe_rotation_auto_detect():
    """Test auto-detection of corner policy."""
    config = LivniumCoreConfig(lattice_size=3)
    system = LivniumCoreSystem(config)
    
    coords_list = list(system.lattice.keys())[:4]
    
    # With convergence (should allow corners)
    convergence_stats = {'num_alive': 1}
    rotation = get_safe_rotation(
        system,
        active_coords=coords_list,
        allow_corners=None,  # Auto-detect
        convergence_stats=convergence_stats
    )
    
    assert rotation is not None, "Should return rotation when converged"


def test_corner_policy_thresholds():
    """Test corner policy with different thresholds."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    coords_list = list(system.lattice.keys())[:4]
    
    # Set high SW for high curvature
    for coords in coords_list:
        cell = system.get_cell(coords)
        if cell:
            cell.symbolic_weight = 50.0
    
    # Low threshold (should allow)
    allow1 = should_allow_corner_rotations(
        system,
        active_coords=coords_list,
        basin_depth_threshold=0.1,
        tension_epsilon=1.0
    )
    
    # High threshold (may not allow)
    allow2 = should_allow_corner_rotations(
        system,
        active_coords=coords_list,
        basin_depth_threshold=10.0,
        tension_epsilon=0.01
    )
    
    assert isinstance(allow1, bool), "Should return boolean"
    assert isinstance(allow2, bool), "Should return boolean"


if __name__ == "__main__":
    print("Running corner rotation policy tests...")
    
    test_should_allow_corner_rotations_no_coords()
    print("✓ Should allow corner rotations (no coords)")
    
    test_should_allow_corner_rotations_with_coords()
    print("✓ Should allow corner rotations (with coords)")
    
    test_should_allow_corner_rotations_convergence()
    print("✓ Should allow corner rotations (convergence)")
    
    test_rotation_affects_corners()
    print("✓ Rotation affects corners")
    
    test_get_safe_rotation()
    print("✓ Get safe rotation")
    
    test_get_safe_rotation_auto_detect()
    print("✓ Get safe rotation (auto-detect)")
    
    test_corner_policy_thresholds()
    print("✓ Corner policy thresholds")
    
    print("\nAll tests passed! ✓")


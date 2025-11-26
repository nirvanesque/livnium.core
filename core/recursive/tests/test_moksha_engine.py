"""
Test assertions for MokshaEngine.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.recursive.moksha_engine import MokshaEngine, ConvergenceState, FixedPointState
from core.recursive.recursive_geometry_engine import RecursiveGeometryEngine
from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig


def test_basic_initialization():
    """Test basic moksha engine initialization."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    moksha = MokshaEngine(engine)
    
    assert moksha.recursive_engine == engine, "Should reference recursive engine"
    assert moksha.convergence_threshold == 0.999, "Default threshold should be 0.999"
    assert moksha.stability_window == 10, "Default stability window should be 10"
    assert not moksha.moksha_reached, "Moksha should not be reached initially"
    assert moksha.moksha_state is None, "Moksha state should be None initially"


def test_convergence_state_enum():
    """Test ConvergenceState enum."""
    assert ConvergenceState.SEARCHING.value == "searching"
    assert ConvergenceState.CONVERGING.value == "converging"
    assert ConvergenceState.MOKSHA.value == "moksha"
    assert ConvergenceState.DIVERGING.value == "diverging"


def test_fixed_point_state():
    """Test FixedPointState dataclass."""
    fp = FixedPointState(
        level_id=0,
        coordinates=(0, 0, 0),
        state_hash="test_hash",
        convergence_score=0.5
    )
    
    assert fp.level_id == 0, "Level ID should be 0"
    assert fp.coordinates == (0, 0, 0), "Coordinates should match"
    assert fp.state_hash == "test_hash", "State hash should match"
    assert fp.convergence_score == 0.5, "Convergence score should be 0.5"
    
    # Test hashability
    assert isinstance(hash(fp), int), "Should be hashable"


def test_check_convergence():
    """Test convergence checking."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    moksha = MokshaEngine(engine)
    
    # Initially should be searching (not enough history)
    state = moksha.check_convergence()
    assert state == ConvergenceState.SEARCHING, "Should be searching initially"
    
    # After multiple checks, should still be searching (system not stable)
    for _ in range(5):
        moksha.check_convergence()
    
    state2 = moksha.check_convergence()
    assert state2 in [ConvergenceState.SEARCHING, ConvergenceState.CONVERGING, 
                      ConvergenceState.DIVERGING], "Should be in valid state"


def test_get_convergence_score():
    """Test convergence score calculation."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    moksha = MokshaEngine(engine)
    
    score = moksha.get_convergence_score()
    
    assert isinstance(score, float), "Should return float"
    assert 0.0 <= score <= 1.0, "Score should be in [0, 1]"
    
    # Initially should be 0 (not converged)
    assert score == 0.0, "Initial score should be 0"


def test_export_final_truth():
    """Test final truth export."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    moksha = MokshaEngine(engine)
    
    truth = moksha.export_final_truth()
    
    assert isinstance(truth, dict), "Should return dictionary"
    assert 'moksha' in truth, "Should have moksha flag"
    assert 'message' in truth, "Should have message"
    
    assert truth['moksha'] == False, "Should be False initially"
    assert 'not yet reached' in truth['message'].lower(), "Message should indicate not reached"


def test_should_terminate():
    """Test termination check."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    moksha = MokshaEngine(engine)
    
    should_terminate = moksha.should_terminate()
    
    assert isinstance(should_terminate, bool), "Should return boolean"
    assert not should_terminate, "Should not terminate initially"


def test_reset():
    """Test moksha engine reset."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    moksha = MokshaEngine(engine)
    
    # Add some history
    for _ in range(5):
        moksha.check_convergence()
    
    assert len(moksha.state_history) > 0, "Should have history"
    
    # Reset
    moksha.reset()
    
    assert len(moksha.state_history) == 0, "History should be cleared"
    assert not moksha.moksha_reached, "Moksha should be False"
    assert moksha.moksha_state is None, "Moksha state should be None"
    assert len(moksha.fixed_points) == 0, "Fixed points should be cleared"


def test_moksha_statistics():
    """Test moksha statistics."""
    config = LivniumCoreConfig(lattice_size=3)
    base_geometry = LivniumCoreSystem(config)
    engine = RecursiveGeometryEngine(base_geometry, max_depth=1)
    
    moksha = MokshaEngine(engine)
    
    stats = moksha.get_moksha_statistics()
    
    assert isinstance(stats, dict), "Should return dictionary"
    assert 'moksha_reached' in stats, "Should have moksha_reached"
    assert 'convergence_score' in stats, "Should have convergence_score"
    assert 'state_history_size' in stats, "Should have state_history_size"
    assert 'fixed_points_count' in stats, "Should have fixed_points_count"
    assert 'convergence_threshold' in stats, "Should have convergence_threshold"
    
    assert stats['moksha_reached'] == False, "Should be False initially"
    assert stats['convergence_score'] == 0.0, "Score should be 0 initially"


if __name__ == "__main__":
    print("Running MokshaEngine tests...")
    
    test_basic_initialization()
    print("✓ Basic initialization")
    
    test_convergence_state_enum()
    print("✓ Convergence state enum")
    
    test_fixed_point_state()
    print("✓ Fixed point state")
    
    test_check_convergence()
    print("✓ Check convergence")
    
    test_get_convergence_score()
    print("✓ Get convergence score")
    
    test_export_final_truth()
    print("✓ Export final truth")
    
    test_should_terminate()
    print("✓ Should terminate")
    
    test_reset()
    print("✓ Reset")
    
    test_moksha_statistics()
    print("✓ Moksha statistics")
    
    print("\nAll tests passed! ✓")


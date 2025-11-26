"""
Test assertions for MeasurementEngine.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from core.quantum.measurement_engine import MeasurementEngine, MeasurementResult
from core.quantum.quantum_cell import QuantumCell


def test_basic_initialization():
    """Test basic measurement engine initialization."""
    engine = MeasurementEngine()
    
    assert len(engine.measurement_history) == 0, "Should start with empty history"


def test_measure_cell():
    """Test measuring a single cell."""
    engine = MeasurementEngine()
    
    # Create superposition
    cell = QuantumCell(
        coordinates=(0, 0, 0),
        amplitudes=np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex),
        num_levels=2
    )
    
    result = engine.measure_cell(cell, collapse=True)
    
    assert isinstance(result, MeasurementResult), "Should return MeasurementResult"
    assert result.cell == (0, 0, 0), "Cell coordinates should match"
    assert result.measured_state in [0, 1], "Measurement should be 0 or 1"
    assert 0.0 <= result.probability <= 1.0, "Probability should be valid"
    assert result.collapsed, "Should be collapsed"
    
    # Check history
    assert len(engine.measurement_history) == 1, "Should record measurement"


def test_measure_without_collapse():
    """Test measurement without collapse."""
    engine = MeasurementEngine()
    
    cell = QuantumCell(
        coordinates=(0, 0, 0),
        amplitudes=np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex),
        num_levels=2
    )
    
    original_state = cell.get_state_vector().copy()
    result = engine.measure_cell(cell, collapse=False)
    
    assert result.collapsed == False, "Should not collapse"
    # State should remain unchanged
    assert np.allclose(cell.get_state_vector(), original_state), "State should be unchanged"


def test_measure_all_cells():
    """Test measuring all cells."""
    engine = MeasurementEngine()
    
    cells = {
        (0, 0, 0): QuantumCell((0, 0, 0), np.array([1.0, 0.0], dtype=complex)),
        (1, 0, 0): QuantumCell((1, 0, 0), np.array([0.0, 1.0], dtype=complex)),
    }
    
    results = engine.measure_all_cells(cells, collapse=True)
    
    assert len(results) == 2, "Should measure all cells"
    assert (0, 0, 0) in results, "Should include first cell"
    assert (1, 0, 0) in results, "Should include second cell"


def test_expectation_value():
    """Test expectation value calculation."""
    engine = MeasurementEngine()
    
    # |+⟩ = (|0⟩ + |1⟩)/√2
    cell = QuantumCell(
        coordinates=(0, 0, 0),
        amplitudes=np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex),
        num_levels=2
    )
    
    # Pauli Z operator
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # ⟨+|Z|+⟩ = 0
    expectation = engine.get_expectation_value(cell, Z)
    assert abs(expectation) < 1e-6, "Expectation should be 0 for |+⟩"


def test_variance():
    """Test variance calculation."""
    engine = MeasurementEngine()
    
    cell = QuantumCell(
        coordinates=(0, 0, 0),
        amplitudes=np.array([1.0, 0.0], dtype=complex),
        num_levels=2
    )
    
    # Pauli Z operator
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    variance = engine.get_variance(cell, Z)
    assert variance >= 0.0, "Variance should be non-negative"


def test_measurement_statistics():
    """Test measurement statistics."""
    engine = MeasurementEngine()
    
    cell = QuantumCell((0, 0, 0), np.array([1.0, 0.0], dtype=complex))
    
    # Make multiple measurements
    for _ in range(5):
        engine.measure_cell(cell, collapse=True)
    
    stats = engine.get_measurement_statistics()
    
    assert 'total_measurements' in stats, "Should have total count"
    assert stats['total_measurements'] == 5, "Should have 5 measurements"
    assert 'unique_outcomes' in stats, "Should have unique outcomes"


def test_measurement_result():
    """Test MeasurementResult dataclass."""
    result = MeasurementResult(
        cell=(0, 0, 0),
        measured_state=1,
        probability=0.5,
        collapsed=True
    )
    
    assert result.cell == (0, 0, 0), "Cell should match"
    assert result.measured_state == 1, "State should match"
    assert result.probability == 0.5, "Probability should match"
    assert result.collapsed == True, "Collapsed should match"
    
    # Test repr
    repr_str = repr(result)
    assert "MeasurementResult" in repr_str, "Repr should include class name"


if __name__ == "__main__":
    print("Running MeasurementEngine tests...")
    
    test_basic_initialization()
    print("✓ Basic initialization")
    
    test_measure_cell()
    print("✓ Measure cell")
    
    test_measure_without_collapse()
    print("✓ Measure without collapse")
    
    test_measure_all_cells()
    print("✓ Measure all cells")
    
    test_expectation_value()
    print("✓ Expectation value")
    
    test_variance()
    print("✓ Variance")
    
    test_measurement_statistics()
    print("✓ Measurement statistics")
    
    test_measurement_result()
    print("✓ Measurement result")
    
    print("\nAll tests passed! ✓")


"""
Test assertions for QuantumCell.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from core.quantum.quantum_cell import QuantumCell


def test_basic_initialization():
    """Test basic quantum cell initialization."""
    cell = QuantumCell(coordinates=(0, 0, 0), amplitudes=None, num_levels=2)
    
    assert cell.coordinates == (0, 0, 0), "Coordinates should match"
    assert cell.num_levels == 2, "Should have 2 levels for qubit"
    assert len(cell.amplitudes) == 2, "Should have 2 amplitudes"
    
    # Default state should be |0⟩
    probs = cell.get_probabilities()
    assert abs(probs[0] - 1.0) < 1e-6, "Default state should be |0⟩"
    assert abs(probs[1] - 0.0) < 1e-6, "Default state should be |0⟩"


def test_normalization():
    """Test state normalization."""
    # Create unnormalized state
    cell = QuantumCell(
        coordinates=(0, 0, 0),
        amplitudes=np.array([2.0, 1.0], dtype=complex),
        num_levels=2
    )
    
    # Should be normalized
    probs = cell.get_probabilities()
    total_prob = np.sum(probs)
    assert abs(total_prob - 1.0) < 1e-6, "Probabilities should sum to 1"


def test_get_probabilities():
    """Test probability calculation."""
    cell = QuantumCell(
        coordinates=(0, 0, 0),
        amplitudes=np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex),
        num_levels=2
    )
    
    probs = cell.get_probabilities()
    assert len(probs) == 2, "Should have 2 probabilities"
    assert abs(probs[0] - 0.5) < 1e-6, "Should be 50/50"
    assert abs(probs[1] - 0.5) < 1e-6, "Should be 50/50"
    assert abs(np.sum(probs) - 1.0) < 1e-6, "Should sum to 1"


def test_measure():
    """Test measurement and collapse."""
    cell = QuantumCell(
        coordinates=(0, 0, 0),
        amplitudes=np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex),
        num_levels=2
    )
    
    # Measure multiple times (should collapse)
    measured = cell.measure()
    assert measured in [0, 1], "Measurement should be 0 or 1"
    
    # After measurement, state should be collapsed
    probs = cell.get_probabilities()
    assert abs(probs[measured] - 1.0) < 1e-6, "Collapsed state should have probability 1"
    assert abs(probs[1 - measured] - 0.0) < 1e-6, "Other state should have probability 0"


def test_apply_unitary():
    """Test unitary gate application."""
    cell = QuantumCell(coordinates=(0, 0, 0), amplitudes=None, num_levels=2)
    
    # Apply Hadamard gate
    H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    cell.apply_unitary(H)
    
    # Should create superposition
    probs = cell.get_probabilities()
    assert abs(probs[0] - 0.5) < 1e-6, "Hadamard should create 50/50 superposition"
    assert abs(probs[1] - 0.5) < 1e-6, "Hadamard should create 50/50 superposition"
    
    # State should still be normalized
    total_prob = np.sum(probs)
    assert abs(total_prob - 1.0) < 1e-6, "Should remain normalized"


def test_set_get_state_vector():
    """Test state vector getter/setter."""
    cell = QuantumCell(coordinates=(0, 0, 0), amplitudes=None, num_levels=2)
    
    # Set custom state
    new_state = np.array([1/np.sqrt(2), 1j/np.sqrt(2)], dtype=complex)
    cell.set_state_vector(new_state)
    
    # Get state back
    retrieved = cell.get_state_vector()
    assert np.allclose(retrieved, new_state), "State should match"
    
    # Should be normalized
    probs = cell.get_probabilities()
    assert abs(np.sum(probs) - 1.0) < 1e-6, "Should be normalized"


def test_meta_interference():
    """Test meta-interference (non-unitary bias)."""
    cell = QuantumCell(
        coordinates=(0, 0, 0),
        amplitudes=np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex),
        num_levels=2
    )
    
    # Apply meta-interference to bias toward |0⟩
    cell.apply_meta_interference(target_state=0, bias_strength=0.5)
    
    # Should bias toward target
    probs = cell.get_probabilities()
    assert probs[0] > 0.5, "Should bias toward |0⟩"
    
    # Should still be normalized
    assert abs(np.sum(probs) - 1.0) < 1e-6, "Should remain normalized"


def test_get_fidelity():
    """Test fidelity calculation."""
    cell = QuantumCell(
        coordinates=(0, 0, 0),
        amplitudes=np.array([1.0, 0.0], dtype=complex),
        num_levels=2
    )
    
    # Fidelity with |0⟩ should be 1
    target = np.array([1.0, 0.0], dtype=complex)
    fidelity = cell.get_fidelity(target)
    assert abs(fidelity - 1.0) < 1e-6, "Fidelity with |0⟩ should be 1"
    
    # Fidelity with |1⟩ should be 0
    target2 = np.array([0.0, 1.0], dtype=complex)
    fidelity2 = cell.get_fidelity(target2)
    assert abs(fidelity2 - 0.0) < 1e-6, "Fidelity with |1⟩ should be 0"


def test_qudit():
    """Test multi-level system (qudit)."""
    cell = QuantumCell(
        coordinates=(0, 0, 0),
        amplitudes=np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], dtype=complex),
        num_levels=3
    )
    
    assert cell.num_levels == 3, "Should have 3 levels"
    assert len(cell.amplitudes) == 3, "Should have 3 amplitudes"
    
    probs = cell.get_probabilities()
    assert len(probs) == 3, "Should have 3 probabilities"
    assert abs(np.sum(probs) - 1.0) < 1e-6, "Should sum to 1"


if __name__ == "__main__":
    print("Running QuantumCell tests...")
    
    test_basic_initialization()
    print("✓ Basic initialization")
    
    test_normalization()
    print("✓ Normalization")
    
    test_get_probabilities()
    print("✓ Get probabilities")
    
    test_measure()
    print("✓ Measure")
    
    test_apply_unitary()
    print("✓ Apply unitary")
    
    test_set_get_state_vector()
    print("✓ Set/get state vector")
    
    test_meta_interference()
    print("✓ Meta interference")
    
    test_get_fidelity()
    print("✓ Get fidelity")
    
    test_qudit()
    print("✓ Qudit")
    
    print("\nAll tests passed! ✓")


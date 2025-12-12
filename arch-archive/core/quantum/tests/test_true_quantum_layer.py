"""
Test assertions for TrueQuantumRegister.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from core.quantum.true_quantum_layer import TrueQuantumRegister
from core.quantum.quantum_gates import QuantumGates


def test_basic_initialization():
    """Test basic quantum register initialization."""
    register = TrueQuantumRegister(qubit_indices=[0, 1, 2])
    
    assert register.num_qubits == 3, "Should have 3 qubits"
    assert register.dim == 8, "Dimension should be 2^3 = 8"
    assert len(register.state) == 8, "State vector should have 8 elements"
    
    # Initial state should be |000⟩
    assert abs(register.state[0] - 1.0) < 1e-6, "Should start in |000⟩"
    assert np.allclose(register.state[1:], 0), "Other states should be 0"


def test_normalize():
    """Test state normalization."""
    register = TrueQuantumRegister(qubit_indices=[0, 1])
    register.state = np.array([2.0, 1.0, 0.0, 0.0], dtype=complex)
    
    register.normalize()
    
    norm = np.sqrt(np.sum(np.abs(register.state)**2))
    assert abs(norm - 1.0) < 1e-6, "State should be normalized"


def test_apply_gate():
    """Test applying single-qubit gate."""
    register = TrueQuantumRegister(qubit_indices=[0, 1])
    
    # Apply Hadamard to qubit 0
    H = QuantumGates.hadamard()
    register.apply_gate(H, target_id=0)
    
    # Should create superposition on first qubit
    # |00⟩ → (|00⟩ + |10⟩)/√2
    assert abs(register.state[0] - 1/np.sqrt(2)) < 1e-6, "Should have superposition"
    assert abs(register.state[2] - 1/np.sqrt(2)) < 1e-6, "Should have superposition"


def test_apply_cnot():
    """Test applying CNOT gate."""
    register = TrueQuantumRegister(qubit_indices=[0, 1])
    
    # Prepare |10⟩ state
    X = QuantumGates.pauli_x()
    register.apply_gate(X, target_id=0)  # |00⟩ → |10⟩
    
    # Apply CNOT: control=0, target=1
    register.apply_cnot(control_id=0, target_id=1)
    
    # Should be |11⟩
    assert abs(register.state[3] - 1.0) < 1e-6, "CNOT|10⟩ should be |11⟩"


def test_measure_qubit():
    """Test measuring a qubit."""
    register = TrueQuantumRegister(qubit_indices=[0, 1])
    
    # Create superposition
    H = QuantumGates.hadamard()
    register.apply_gate(H, target_id=0)
    
    # Measure qubit 0
    outcome = register.measure_qubit(target_id=0)
    
    assert outcome in [0, 1], "Measurement should be 0 or 1"
    
    # State should be collapsed
    norm = np.sqrt(np.sum(np.abs(register.state)**2))
    assert abs(norm - 1.0) < 1e-6, "State should be normalized after measurement"


def test_get_qubit_state():
    """Test getting reduced state of a qubit."""
    register = TrueQuantumRegister(qubit_indices=[0, 1])
    
    # Prepare |00⟩ + |11⟩ (Bell state)
    H = QuantumGates.hadamard()
    register.apply_gate(H, target_id=0)
    register.apply_cnot(control_id=0, target_id=1)
    
    # Get state of qubit 0
    state = register.get_qubit_state(qubit_id=0)
    
    assert len(state) == 2, "Should return 2-element state vector"
    norm = np.sqrt(np.sum(np.abs(state)**2))
    assert abs(norm - 1.0) < 1e-6, "Reduced state should be normalized"


def test_get_fidelity():
    """Test fidelity calculation."""
    register = TrueQuantumRegister(qubit_indices=[0])
    
    # Set to |0⟩
    state_0 = np.array([1.0, 0.0], dtype=complex)
    fidelity = register.get_fidelity(state_0, qubit_id=0)
    
    assert abs(fidelity - 1.0) < 1e-6, "Fidelity with |0⟩ should be 1"
    
    # Fidelity with |1⟩ should be 0
    state_1 = np.array([0.0, 1.0], dtype=complex)
    fidelity2 = register.get_fidelity(state_1, qubit_id=0)
    assert abs(fidelity2 - 0.0) < 1e-6, "Fidelity with |1⟩ should be 0"


def test_meta_interference():
    """Test meta-interference (non-unitary bias)."""
    register = TrueQuantumRegister(qubit_indices=[0, 1])
    
    # Create superposition
    H = QuantumGates.hadamard()
    register.apply_gate(H, target_id=0)
    
    # Apply meta-interference to bias toward |00⟩
    register.apply_meta_interference(target_pattern=0, bias_strength=0.5)
    
    # Should bias toward target
    prob_00 = abs(register.state[0])**2
    assert prob_00 > 0.5, "Should bias toward |00⟩"
    
    # Should still be normalized
    norm = np.sqrt(np.sum(np.abs(register.state)**2))
    assert abs(norm - 1.0) < 1e-6, "Should remain normalized"


def test_get_full_state():
    """Test getting full state vector."""
    register = TrueQuantumRegister(qubit_indices=[0, 1])
    
    state = register.get_full_state()
    
    assert len(state) == 4, "Should return 4-element state vector"
    assert np.allclose(state, register.state), "Should match internal state"


def test_invalid_qubit_id():
    """Test error handling for invalid qubit IDs."""
    register = TrueQuantumRegister(qubit_indices=[0, 1])
    
    try:
        register.apply_gate(QuantumGates.hadamard(), target_id=999)
        assert False, "Should raise ValueError for invalid qubit ID"
    except ValueError:
        pass  # Expected
    
    try:
        register.measure_qubit(target_id=999)
        assert False, "Should raise ValueError for invalid qubit ID"
    except ValueError:
        pass  # Expected


if __name__ == "__main__":
    print("Running TrueQuantumRegister tests...")
    
    test_basic_initialization()
    print("✓ Basic initialization")
    
    test_normalize()
    print("✓ Normalize")
    
    test_apply_gate()
    print("✓ Apply gate")
    
    test_apply_cnot()
    print("✓ Apply CNOT")
    
    test_measure_qubit()
    print("✓ Measure qubit")
    
    test_get_qubit_state()
    print("✓ Get qubit state")
    
    test_get_fidelity()
    print("✓ Get fidelity")
    
    test_meta_interference()
    print("✓ Meta interference")
    
    test_get_full_state()
    print("✓ Get full state")
    
    test_invalid_qubit_id()
    print("✓ Invalid qubit ID")
    
    print("\nAll tests passed! ✓")


"""
Test assertions for QuantumGates.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from core.quantum.quantum_gates import QuantumGates, GateType


def test_hadamard():
    """Test Hadamard gate."""
    H = QuantumGates.hadamard()
    
    assert H.shape == (2, 2), "Hadamard should be 2×2"
    assert QuantumGates.is_unitary(H), "Hadamard should be unitary"
    
    # H|0⟩ = (|0⟩ + |1⟩)/√2
    state_0 = np.array([1.0, 0.0], dtype=complex)
    result = H @ state_0
    expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    assert np.allclose(result, expected), "H|0⟩ should create superposition"


def test_pauli_gates():
    """Test Pauli gates (X, Y, Z)."""
    X = QuantumGates.pauli_x()
    Y = QuantumGates.pauli_y()
    Z = QuantumGates.pauli_z()
    
    assert QuantumGates.is_unitary(X), "Pauli X should be unitary"
    assert QuantumGates.is_unitary(Y), "Pauli Y should be unitary"
    assert QuantumGates.is_unitary(Z), "Pauli Z should be unitary"
    
    # X|0⟩ = |1⟩
    state_0 = np.array([1.0, 0.0], dtype=complex)
    result = X @ state_0
    expected = np.array([0.0, 1.0], dtype=complex)
    assert np.allclose(result, expected), "X|0⟩ should be |1⟩"
    
    # Z|0⟩ = |0⟩
    result = Z @ state_0
    assert np.allclose(result, state_0), "Z|0⟩ should be |0⟩"


def test_phase_gate():
    """Test phase gate."""
    phi = np.pi / 4
    P = QuantumGates.phase(phi)
    
    assert P.shape == (2, 2), "Phase gate should be 2×2"
    assert QuantumGates.is_unitary(P), "Phase gate should be unitary"
    
    # P(φ)|1⟩ = e^(iφ)|1⟩
    state_1 = np.array([0.0, 1.0], dtype=complex)
    result = P @ state_1
    expected = np.array([0.0, np.exp(1j * phi)], dtype=complex)
    assert np.allclose(result, expected), "Phase gate should add phase"


def test_rotation_gates():
    """Test rotation gates (Rx, Ry, Rz)."""
    theta = np.pi / 2
    
    Rx = QuantumGates.rotation_x(theta)
    Ry = QuantumGates.rotation_y(theta)
    Rz = QuantumGates.rotation_z(theta)
    
    assert QuantumGates.is_unitary(Rx), "Rx should be unitary"
    assert QuantumGates.is_unitary(Ry), "Ry should be unitary"
    assert QuantumGates.is_unitary(Rz), "Rz should be unitary"


def test_cnot():
    """Test CNOT gate."""
    CNOT = QuantumGates.cnot(control=0, target=1)
    
    assert CNOT.shape == (4, 4), "CNOT should be 4×4"
    assert QuantumGates.is_unitary(CNOT), "CNOT should be unitary"
    
    # CNOT|10⟩ = |11⟩
    state_10 = np.array([0.0, 0.0, 1.0, 0.0], dtype=complex)
    result = CNOT @ state_10
    expected = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex)
    assert np.allclose(result, expected), "CNOT|10⟩ should be |11⟩"


def test_cz():
    """Test controlled-Z gate."""
    CZ = QuantumGates.cz()
    
    assert CZ.shape == (4, 4), "CZ should be 4×4"
    assert QuantumGates.is_unitary(CZ), "CZ should be unitary"


def test_swap():
    """Test SWAP gate."""
    SWAP = QuantumGates.swap()
    
    assert SWAP.shape == (4, 4), "SWAP should be 4×4"
    assert QuantumGates.is_unitary(SWAP), "SWAP should be unitary"
    
    # SWAP|01⟩ = |10⟩
    state_01 = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    result = SWAP @ state_01
    expected = np.array([0.0, 0.0, 1.0, 0.0], dtype=complex)
    assert np.allclose(result, expected), "SWAP|01⟩ should be |10⟩"


def test_identity():
    """Test identity gate."""
    I = QuantumGates.identity(num_levels=2)
    
    assert I.shape == (2, 2), "Identity should be 2×2"
    assert np.allclose(I, np.eye(2)), "Identity should be I"
    
    # I|ψ⟩ = |ψ⟩
    state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    result = I @ state
    assert np.allclose(result, state), "Identity should preserve state"


def test_is_unitary():
    """Test unitary check."""
    # Valid unitary
    H = QuantumGates.hadamard()
    assert QuantumGates.is_unitary(H), "Hadamard should be unitary"
    
    # Non-unitary matrix
    non_unitary = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
    assert not QuantumGates.is_unitary(non_unitary), "Non-unitary should fail check"


def test_get_gate():
    """Test get_gate method."""
    H = QuantumGates.get_gate(GateType.HADAMARD)
    assert QuantumGates.is_unitary(H), "Should return valid gate"
    
    X = QuantumGates.get_gate(GateType.PAULI_X)
    assert QuantumGates.is_unitary(X), "Should return valid gate"
    
    P = QuantumGates.get_gate(GateType.PHASE, phi=np.pi/4)
    assert QuantumGates.is_unitary(P), "Should return valid gate"
    
    Rx = QuantumGates.get_gate(GateType.ROTATION_X, theta=np.pi/2)
    assert QuantumGates.is_unitary(Rx), "Should return valid gate"


def test_gate_type_enum():
    """Test GateType enum."""
    assert GateType.HADAMARD.value == "H"
    assert GateType.PAULI_X.value == "X"
    assert GateType.CNOT.value == "CNOT"


if __name__ == "__main__":
    print("Running QuantumGates tests...")
    
    test_hadamard()
    print("✓ Hadamard")
    
    test_pauli_gates()
    print("✓ Pauli gates")
    
    test_phase_gate()
    print("✓ Phase gate")
    
    test_rotation_gates()
    print("✓ Rotation gates")
    
    test_cnot()
    print("✓ CNOT")
    
    test_cz()
    print("✓ CZ")
    
    test_swap()
    print("✓ SWAP")
    
    test_identity()
    print("✓ Identity")
    
    test_is_unitary()
    print("✓ Is unitary")
    
    test_get_gate()
    print("✓ Get gate")
    
    test_gate_type_enum()
    print("✓ Gate type enum")
    
    print("\nAll tests passed! ✓")


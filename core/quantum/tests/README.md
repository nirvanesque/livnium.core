# Quantum Module Tests

Assertion-based tests for the quantum layer.

## Test Files

- **`test_quantum_cell.py`**: Tests for `QuantumCell`
  - Basic initialization
  - State normalization
  - Probability calculation
  - Measurement and collapse
  - Unitary gate application
  - Meta-interference
  - Fidelity calculation
  - Multi-level systems (qudits)

- **`test_quantum_gates.py`**: Tests for `QuantumGates`
  - Hadamard gate
  - Pauli gates (X, Y, Z)
  - Phase gate
  - Rotation gates (Rx, Ry, Rz)
  - CNOT gate
  - CZ gate
  - SWAP gate
  - Identity gate
  - Unitarity checks
  - Gate factory method

- **`test_quantum_lattice.py`**: Tests for `QuantumLattice`
  - Basic initialization
  - Quantum feature requirements
  - Gate application (single and all cells)
  - Entanglement creation
  - Measurement operations
  - State summary

- **`test_entanglement_manager.py`**: Tests for `EntanglementManager`
  - Basic initialization
  - Bell pair creation
  - Entanglement by distance
  - Entanglement by face exposure
  - Entanglement queries
  - Breaking entanglement
  - Statistics

- **`test_measurement_engine.py`**: Tests for `MeasurementEngine`
  - Basic initialization
  - Single cell measurement
  - Measurement without collapse
  - Measuring all cells
  - Expectation values
  - Variance calculation
  - Measurement statistics

- **`test_true_quantum_layer.py`**: Tests for `TrueQuantumRegister`
  - Basic initialization
  - State normalization
  - Single-qubit gates
  - CNOT gate (true entanglement)
  - Qubit measurement
  - Reduced state calculation
  - Fidelity calculation
  - Meta-interference
  - Error handling

- **`test_geometry_quantum_coupling.py`**: Tests for `GeometryQuantumCoupling`
  - Basic initialization
  - Quantum state initialization from geometry
  - Polarity to phase mapping
  - Face exposure to entanglement strength
  - Symbolic weight to amplitude modulation
  - Cell class to initial state
  - Φ calculations (straight, rotated, dual)
  - Geometry-quantum synchronization

## Running Tests

Run individual test files:
```bash
python3 core/quantum/tests/test_quantum_cell.py
python3 core/quantum/tests/test_quantum_gates.py
python3 core/quantum/tests/test_quantum_lattice.py
python3 core/quantum/tests/test_entanglement_manager.py
python3 core/quantum/tests/test_measurement_engine.py
python3 core/quantum/tests/test_true_quantum_layer.py
python3 core/quantum/tests/test_geometry_quantum_coupling.py
```

## Test Style

All tests use Python `assert` statements. Each test function:
- Has a descriptive name starting with `test_`
- Uses clear assertions with helpful error messages
- Tests one specific aspect of functionality
- Can be run independently

## Test Coverage

The tests cover:
- ✅ Quantum state management (superposition, normalization)
- ✅ Quantum gates (unitary operations)
- ✅ Entanglement (Bell pairs, geometric entanglement)
- ✅ Measurement (Born rule, collapse)
- ✅ True tensor product mechanics
- ✅ Geometry-quantum coupling (Φ, polarity, face exposure)
- ✅ Meta-interference (non-unitary optimization)
- ✅ Error handling and edge cases


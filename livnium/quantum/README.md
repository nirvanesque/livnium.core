# Livnium Quantum Layer (Experimental)

> [!WARNING]
> **Experimental Research Layer**
> 
> This module implements a classical simulation of tensor-product quantum mechanics for research and educational purposes. It is optional, not required by Livnium, and not intended for production use. APIs may change and performance is not guaranteed.

## Overview

The `livnium/quantum` package provides a real, mathematically rigorous implementation of quantum mechanics primitives, distinct from the "quantum-inspired" physics embedding models found elsewhere in Livnium.

This layer is designed to allow Livnium to explore true quantum algorithms (like Grover's search or quantum teleportation) within the same geometric framework used for semantic space.

## Features

### Core Quantum Mechanics (`livnium.quantum.core`)
*   **True Tensor Product States**: Implements full $2^n$ state vectors for n-qubit registers.
*   **Universal Gate Set**: Includes Hadamard, Pauli-X/Y/Z, Phase, CNOT, and arbitrary rotations.
*   **Born Rule Measurement**: Probabilistic measurement $P(i) = |\alpha_i|^2$ with state collapse.
*   **Entanglement**: Can create and verify Bell states (e.g., $|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$).

### Lattice Integration (`livnium.quantum.lattice`)
*   *Note: These components interact with archived parts of the Livnium system and may require adaptation.*
*   **Quantum Lattice**: Maps spatial cells to qubits.
*   **Geometry Coupling**: Theoretical framework for coupling geometric exposure to entanglement entropy.

## Usage

### Basic Bell State Example

```python
from livnium.quantum import QuantumRegister, QuantumGates

# Initialize a 2-qubit register
qr = QuantumRegister([0, 1])

# Apply Hadamard to qubit 0: |0⟩ -> (|0⟩+|1⟩)/√2
qr.apply_gate(QuantumGates.hadamard(), 0)

# Apply CNOT (control=0, target=1): (|00⟩+|11⟩)/√2
qr.apply_cnot(0, 1)

# Measure qubit 0
result = qr.measure_qubit(0)
print(f"Measured: {result}")
```

### Running the Demo

A complete demonstration is available in the examples directory:

```bash
python3 livnium/examples/quantum_bell_state.py
```

## Performance Note

This implementation uses `numpy` for matrix operations. While it includes optional `numba` acceleration hooks, simulating quantum systems is exponentially expensive ($O(2^n)$). This module is intended for small-scale simulations (typically n < 20 qubits) to validate logic and interactions.

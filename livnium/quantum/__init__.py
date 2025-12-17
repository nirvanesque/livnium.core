"""
Livnium Quantum Layer (Experimental)

Real tensor-product quantum mechanics integrated with Livnium geometry.

This layer provides:
- Real quantum gates (H, X, Y, Z, CNOT, etc.)
- True multi-qubit entanglement using tensor products
- Born rule measurement with state collapse

Example:
    >>> from livnium.quantum import QuantumRegister, QuantumGates
    >>> qr = QuantumRegister(num_qubits=2)
    >>> qr.apply_gate(0, QuantumGates.hadamard())
    >>> qr.apply_cnot(0, 1)  # Create Bell state
    >>> result = qr.measure_qubit(0)

Note:
    This is an experimental layer. It extends Livnium with quantum computing
    capabilities but is not required for standard Livnium operation.
    
    For advanced features like QuantumLattice, import directly:
    >>> from livnium.quantum.lattice.quantum_lattice import QuantumLattice
"""

from livnium.quantum.core import QuantumRegister, QuantumGates, QuantumCell

__all__ = [
    "QuantumRegister",
    "QuantumGates",
    "QuantumCell",
]

__version__ = "experimental"

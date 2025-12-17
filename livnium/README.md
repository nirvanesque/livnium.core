# LIVNIUM: Law-Governed Reasoning Platform

LIVNIUM is a reasoning system that replaces "searching for answers" with **geometric constraint satisfaction**. It operates by removing impossible futures until only one path can fall.

## The Core Thesis
Reasoning is not just token prediction; it is an evolution of state through a landscape of constraints. Livnium formalizes this as **Physics-Governed Collapse**:
1. **Represent**: Data is encoded as state vectors in high-dimensional space.
2. **Constrain**: Geometric laws (alignment, divergence, tension) define the "stress" between claims.
3. **Collapse**: An engine evolves the system to minimize tension, reaching a stable "Truth" basin.

---

## Dual-Stack Architecture

Livnium maintains a strict separation between stable production dynamics and experimental research.

### 1. The Production Stack (Stable)
Focused on speed, performance, and continuous vector-space reasoning.
- **[Kernel (LUGK)](livnium/kernel/)**: The immutable constitution. Pure math formulas (alignment, divergence, tension). Imports nothing but `typing`.
- **[Engine (LUGE)](livnium/engine/)**: Runtime dynamics. Implements the `CollapseEngine` and `BasinField` using Torch/Numpy.
- **[Domains](livnium/domains/)**: Domain-specific semantics (NLI, Document, Market) implemented as plugins.

### 2. The Research Stack (Experimental)
Explores discrete geometry, fractal scaling, and true quantum priors.
- **[Classical Core](livnium/classical/)**: The 3D geometric lattice foundation.
- **[Recursive Engine](livnium/recursive/)**: "Fractal Machine" for nested structural context and Fixed-Point (Moksha) detection.
- **[Quantum Layer](livnium/quantum/)**: Real multi-qubit entanglement and Born-rule measurement for hardware-grade ambiguity.

---

## Flagship Demonstrations

Run these to see the platform in action. Each proves a different layer of the system:

### 1. Semantic Reconciliation: Contradiction Collapse
**Domain**: `document`
**Command**: `python3 livnium/examples/document_contradiction_demo.py`
**Shows**: Using mutual attraction/repulsion forces to resolve conflicting claims in a contract dispute.

### 2. Geometric Truth: Recursive Moksha
**Layer**: `recursive`
**Command**: `python3 livnium/examples/recursive_moksha_demo.py`
**Shows**: A fractal geometry reaching a fixed-point invariant state across nested levels.

### 3. Absolute Ambiguity: Quantum Bell State
**Layer**: `quantum`
**Command**: `python3 livnium/examples/quantum_bell_state.py`
**Shows**: True linear-algebraic entanglement between qubits, demonstrating real tensor-product physics.

---

## Governance & Compliance

Livnium stays stable through strict **Constitutional Enforcement**. We use automated gates to ensure that experimental layers never contaminate the immutable kernel.

```bash
# Verify kernel purity (no torch/numpy imports)
python3 livnium/tests/kernel/test_kernel_import_clean.py

# Scan for "magic constants" (forces all thresholds into engine config)
python3 livnium/tests/kernel/test_no_magic_constants.py

# Validate Layer 0 stability (Recursive Engine Smoke Test)
python3 livnium/tests/test_recursive_smoke.py
```

## The One Rule

> **Never let engine convenience leak upward into the kernel.**

If it feels convenient to put something in the kernel, it doesn't belong there. Laws are inconvenient by nature.

---

## Getting Started

See [livnium/domains/DOMAIN_TEMPLATE.md](livnium/domains/DOMAIN_TEMPLATE.md) to build your first physics-governed domain, or explore the [Document Pipeline](livnium/integration/README.md) for real-world integration.


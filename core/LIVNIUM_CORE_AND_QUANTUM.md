# Livnium Core System & Quantum Layer: Complete Guide

## Table of Contents

1. [Livnium Core System Overview](#livnium-core-system-overview)
2. [Core System Components](#core-system-components)
3. [Quantum Layer](#quantum-layer)
4. [Configurable Modes & Features](#configurable-modes--features)
5. [Feature Dependencies](#feature-dependencies)
6. [Usage Examples](#usage-examples)
7. [Configuration Reference](#configuration-reference)

---

## Livnium Core System Overview

The **Livnium Core System** is a geometric computing system based on an N×N×N spatial lattice with symbolic weight, rotations, and an observer system. It implements all 7 axioms of the Livnium specification.

### Key Characteristics

- **Fully Generalized**: Works for any odd N ≥ 3 (3, 5, 7, 9, 11, ...)
- **Symbolic Weight**: SW = 9·f (where f is face exposure)
- **90° Rotations**: 24-element rotation group
- **Observer System**: Global Observer at (0,0,0) + Local Observers
- **Invariants**: Total SW and class counts are conserved under rotations

### The 7 Axioms

1. **A1: Canonical Spatial Alphabet**
   - N×N×N lattice with Σ(N) alphabet (N³ symbols)
   - Coordinates: (x, y, z) where x, y, z ∈ {-(N-1)/2, ..., (N-1)/2}

2. **A2: Observer Anchor**
   - Global Observer fixed at (0, 0, 0)
   - Provides reference frame for all operations

3. **A3: Symbolic Weight Law**
   - SW = 9·f (independent of N)
   - Face exposure f ∈ {0, 1, 2, 3}

4. **A4: Dynamic Law**
   - Only 90° quarter-turns allowed
   - 24-element rotation group

5. **A5: Semantic Polarity**
   - cos(θ) between motion and observer
   - N-invariant

6. **A6: Activation Rule**
   - Local Observer designation
   - N-invariant

7. **A7: Cross-Lattice Coupling**
   - Wreath-product transformations
   - Infrastructure ready

---

## Core System Components

### 1. LatticeCell

A single cell in the N×N×N lattice.

**Properties**:
- `coordinates`: (x, y, z) tuple
- `symbol`: Symbol from alphabet (optional)
- `face_exposure`: f ∈ {0, 1, 2, 3} (number of faces on boundary)
- `symbolic_weight`: SW = 9·f
- `cell_class`: Core, Center, Edge, or Corner

**Cell Classes**:
- **CORE** (f=0): SW = 0, interior cells
- **CENTER** (f=1): SW = 9, face centers
- **EDGE** (f=2): SW = 18, edges
- **CORNER** (f=3): SW = 27, corners

### 2. Observer System

**Global Observer**:
- Fixed at (0, 0, 0)
- Provides reference frame
- Used for semantic polarity calculations

**Local Observer**:
- Can be designated at any cell
- Used for activation rules
- Derived from global observer

### 3. RotationGroup

24-element rotation group for cube rotations.

**Rotation Axes**:
- X-axis rotations
- Y-axis rotations
- Z-axis rotations

**Rotation Types**:
- 90° quarter-turns (1, 2, 3, or 4 quarter-turns)
- All rotations preserve invariants

### 4. Invariants

**Total Symbolic Weight**:
```
ΣSW(N) = 54(N-2)² + 216(N-2) + 216
```

**Class Counts**:
- Core: (N-2)³
- Centers: 6(N-2)²
- Edges: 12(N-2)
- Corners: 8

**Verified Values**:
- N=3: ΣSW = 486
- N=5: ΣSW = 1350
- N=7: ΣSW = 2646

All rotations preserve these invariants.

---

## Quantum Layer

The **Quantum Layer** adds quantum computing capabilities to the geometric lattice, with full integration between geometry and quantum mechanics.

### Architecture

```
LivniumCoreSystem (Geometric Layer)
    ↓
QuantumLattice (Quantum Layer)
    ├── QuantumCell (per cell)
    ├── QuantumGates (unitary operations)
    ├── EntanglementManager (correlations)
    ├── MeasurementEngine (Born rule)
    └── GeometryQuantumCoupling (Livnium-specific)
```

### Components

#### 1. QuantumCell

Represents quantum state per cell: |ψ⟩ = α|0⟩ + β|1⟩

**Features**:
- Complex amplitudes: α, β ∈ ℂ
- Normalization: |α|² + |β|² = 1
- State operations: apply_unitary(), measure(), get_probabilities()
- Fidelity calculation: |⟨ψ|φ⟩|²

**State Representation**:
```python
amplitudes = np.array([alpha, beta], dtype=complex)
```

#### 2. QuantumGates

Complete unitary gate library.

**Single-Qubit Gates**:
- **Hadamard (H)**: Creates superposition
- **Pauli-X**: Bit flip
- **Pauli-Y**: Y rotation
- **Pauli-Z**: Phase flip
- **Phase (S)**: π/2 phase gate
- **T Gate**: π/4 phase gate

**Rotation Gates**:
- **Rx(θ)**: Rotation about X-axis
- **Ry(θ)**: Rotation about Y-axis
- **Rz(θ)**: Rotation about Z-axis

**Two-Qubit Gates**:
- **CNOT**: Controlled-NOT
- **CZ**: Controlled-Z
- **SWAP**: Swap two qubits

**Multi-Qubit Gates**:
- **Toffoli**: 3-qubit controlled gate
- **Fredkin**: Controlled swap

**Unitary Verification**: All gates verified (U†U = I)

#### 3. EntanglementManager

Manages multi-cell quantum entanglement.

**Features**:
- **Bell States**: |Φ⁺⟩, |Φ⁻⟩, |Ψ⁺⟩, |Ψ⁻⟩
- **Geometric Entanglement**: Distance-based connections
- **Face-Exposure Entanglement**: Higher f → more connections
- **Entanglement Graph**: Track all entangled pairs

**Entanglement Types**:
- Pairwise entanglement (Bell pairs)
- Geometric entanglement (based on distance)
- Face-exposure based (higher exposure → more connections)

#### 4. MeasurementEngine

Implements quantum measurement with Born rule.

**Features**:
- **Born Rule**: P(i) = |αᵢ|²
- **State Collapse**: |ψ⟩ → |i⟩ after measurement
- **Expectation Values**: ⟨ψ|O|ψ⟩
- **Variance**: ⟨O²⟩ - ⟨O⟩²

**Measurement Process**:
1. Calculate probabilities from amplitudes
2. Sample according to Born rule
3. Collapse state to measured value
4. Return measurement result

#### 5. GeometryQuantumCoupling

**Livnium-specific integration** between geometry and quantum.

**Coupling Rules**:

1. **Face Exposure → Entanglement**
   - Higher f → more entanglement connections
   - Corner cells (f=3) have maximum connections

2. **Symbolic Weight → Amplitude**
   - SW modulates amplitude strength
   - Higher SW → stronger quantum states

3. **Polarity → Phase**
   - Semantic polarity affects quantum phase
   - Observer-dependent phase shifts

4. **Observer → Measurement Basis**
   - Observer position rotates measurement basis
   - Local observer affects measurement outcomes

5. **Class → Initial State**
   - Core cells: |0⟩
   - Center cells: (|0⟩ + |1⟩)/√2
   - Edge cells: More complex superpositions
   - Corner cells: Maximum superposition

6. **Geometric Rotation → Quantum Gate**
   - 90° geometric rotation → quantum rotation gate
   - Rotations preserve quantum-geometric coupling

### What Makes This Unique

**Not just a quantum simulator** — it's a **quantum-inspired geometric computer** where:

1. **Geometry drives quantum**: Face exposure determines entanglement topology
2. **Symbolic Weight modulates amplitudes**: Higher SW → stronger quantum states
3. **Polarity affects phase**: Semantic meaning influences quantum phase
4. **Observer influences measurement**: Observer position rotates measurement basis
5. **Class determines initial state**: Cell class maps to quantum superposition

This is the **Livnium way** — quantum mechanics integrated with symbolic geometry.

---

## Configurable Modes & Features

All features in the Livnium Core System can be enabled/disabled through `LivniumCoreConfig`. This allows fine-grained control over system capabilities.

### Configuration Categories

#### 1. Core Structure

```python
enable_3x3x3_lattice: bool = True          # A1: Canonical Spatial Alphabet
enable_symbol_alphabet: bool = True        # 27-symbol alphabet (Σ = {0, a...z})
```

**Purpose**: Enable/disable basic lattice structure and symbol alphabet.

#### 2. Symbolic Weight System

```python
enable_symbolic_weight: bool = True         # A3: Symbolic Weight Law (SW = 9·f)
enable_face_exposure: bool = True          # Face exposure calculation (f ∈ {0,1,2,3})
enable_class_structure: bool = True        # Core/Center/Edge/Corner classes
```

**Purpose**: Control symbolic weight calculations and cell classification.

**Dependencies**:
- `enable_symbolic_weight` requires `enable_face_exposure`
- `enable_class_structure` requires `enable_face_exposure`

#### 3. Dynamic Law (Rotations)

```python
enable_90_degree_rotations: bool = True     # A4: Only 90° quarter-turns
enable_rotation_group: bool = True         # 24-element rotation group
```

**Purpose**: Enable/disable rotation operations.

#### 4. Observer System

```python
enable_global_observer: bool = True        # A2: Global Observer at (0,0,0)
enable_local_observer: bool = True         # A6: Local Observer designation
enable_observer_coordinates: bool = True   # Observer-based coordinate system
```

**Purpose**: Control observer system functionality.

**Dependencies**:
- `enable_local_observer` requires `enable_global_observer`
- `enable_semantic_polarity` requires `enable_global_observer`

#### 5. Semantic Polarity

```python
enable_semantic_polarity: bool = True      # A5: cos(θ) between motion and observer
```

**Purpose**: Enable semantic polarity calculations.

#### 6. Cross-Lattice Coupling

```python
enable_cross_lattice_coupling: bool = True # A7: Wreath-product transformations
```

**Purpose**: Enable cross-lattice coupling (infrastructure ready).

#### 7. Quantum Features

```python
enable_quantum: bool = False               # Master switch for quantum layer
enable_superposition: bool = False         # Complex amplitudes per cell
enable_quantum_gates: bool = False         # Unitary gate operations
enable_entanglement: bool = False          # Multi-cell entanglement
enable_measurement: bool = False           # Born rule + collapse
enable_geometry_quantum_coupling: bool = False  # Geometry ↔ Quantum mapping
```

**Purpose**: Control quantum layer features.

**Dependencies**:
- `enable_quantum_gates` requires `enable_superposition`
- `enable_entanglement` requires `enable_superposition`
- `enable_measurement` requires `enable_superposition`
- `enable_geometry_quantum_coupling` requires `enable_quantum`

#### 8. Memory Layer

```python
enable_memory: bool = False                # Master switch for memory layer
enable_working_memory: bool = False        # Working memory per cell
enable_long_term_memory: bool = False      # Long-term memory consolidation
enable_memory_coupling: bool = False       # Memory-geometry coupling
```

**Purpose**: Control memory layer features.

**Dependencies**:
- `enable_long_term_memory` requires `enable_memory`
- `enable_memory_coupling` requires `enable_memory`

#### 9. Reasoning Layer

```python
enable_reasoning: bool = False             # Master switch for reasoning layer
enable_search: bool = False                # Search engine
enable_rules: bool = False                 # Rule engine
enable_problem_solving: bool = False       # Problem solver
```

**Purpose**: Control reasoning layer features.

**Dependencies**:
- `enable_rules` requires `enable_reasoning`
- `enable_problem_solving` requires `enable_reasoning`

#### 10. Semantic Layer

```python
enable_semantic: bool = False              # Master switch for semantic layer
enable_feature_extraction: bool = False    # Feature extraction
enable_meaning_graph: bool = False        # Meaning graph
enable_inference: bool = False             # Inference engine
```

**Purpose**: Control semantic layer features.

**Dependencies**:
- `enable_meaning_graph` requires `enable_semantic`
- `enable_inference` requires `enable_semantic`

#### 11. Meta Layer

```python
enable_meta: bool = False                  # Master switch for meta layer
enable_introspection: bool = False         # Introspection
enable_anomaly_detection: bool = False    # Anomaly detection
enable_calibration: bool = False           # Auto-calibration
```

**Purpose**: Control meta layer features.

**Dependencies**:
- `enable_introspection` requires `enable_meta`
- `enable_anomaly_detection` requires `enable_meta`
- `enable_calibration` requires `enable_meta`

#### 12. Runtime Layer

```python
enable_runtime: bool = False               # Master switch for runtime orchestrator
enable_episodes: bool = False              # Episode management
```

**Purpose**: Control runtime orchestration.

**Dependencies**:
- `enable_episodes` requires `enable_runtime`

#### 13. Recursive Geometry (Layer 0)

```python
enable_recursive_geometry: bool = False    # Enable recursive geometry engine
recursive_max_depth: int = 3               # Maximum recursion depth
recursive_subdivision_rule: str = "default" # Subdivision rule
enable_moksha: bool = False                # Enable fixed-point convergence (moksha)
moksha_convergence_threshold: float = 0.999 # Convergence threshold
moksha_stability_window: int = 10          # Stability window for convergence
```

**Purpose**: Control recursive geometry engine (Layer 0).

**Features**:
- Fractal compression (exponential capacity with linear memory)
- Fixed-point convergence (moksha)
- Recursive problem solving

#### 14. Invariants

```python
enable_sw_conservation: bool = True        # ΣSW conservation
enable_class_count_conservation: bool = True  # Class counts conservation
```

**Purpose**: Control invariant checking.

**Dependencies**:
- `enable_sw_conservation` requires `enable_symbolic_weight`
- `enable_class_count_conservation` requires `enable_class_structure`

#### 15. Hierarchical Extension

```python
enable_hierarchical_extension: bool = False  # Level-0 (macro) + Level-1 (micro)
hierarchical_macro_size: int = 3            # Macro lattice size (3×3×3)
hierarchical_micro_size: int = 3            # Micro lattice size (3×3×3)
```

**Purpose**: Enable hierarchical (macro/micro) structure.

#### 16. Lattice Size

```python
lattice_size: int = 3  # N×N×N lattice (must be odd, ≥ 3)
```

**Purpose**: Set lattice size (supports N=3, 5, 7, 9, 11, ...).

**Validation**: Must be odd and ≥ 3.

#### 17. Equilibrium Constant

```python
equilibrium_constant: float = 10.125  # K = 10.125 (for 3×3×3)
```

**Purpose**: Set equilibrium constant (for 3×3×3).

---

## Feature Dependencies

The configuration system validates dependencies automatically. Here's the dependency graph:

### Core Dependencies

```
enable_symbolic_weight → enable_face_exposure
enable_class_structure → enable_face_exposure
enable_semantic_polarity → enable_global_observer
enable_local_observer → enable_global_observer
enable_sw_conservation → enable_symbolic_weight
enable_class_count_conservation → enable_class_structure
```

### Quantum Dependencies

```
enable_quantum_gates → enable_superposition
enable_entanglement → enable_superposition
enable_measurement → enable_superposition
enable_geometry_quantum_coupling → enable_quantum
```

### Memory Dependencies

```
enable_long_term_memory → enable_memory
enable_memory_coupling → enable_memory
```

### Reasoning Dependencies

```
enable_rules → enable_reasoning
enable_problem_solving → enable_reasoning
```

### Semantic Dependencies

```
enable_meaning_graph → enable_semantic
enable_inference → enable_semantic
```

### Meta Dependencies

```
enable_introspection → enable_meta
enable_anomaly_detection → enable_meta
enable_calibration → enable_meta
```

### Runtime Dependencies

```
enable_episodes → enable_runtime
```

---

## Usage Examples

### Example 1: Basic Classical System

```python
from core import LivniumCoreSystem, LivniumCoreConfig

# Default configuration (all classical features enabled)
config = LivniumCoreConfig()
system = LivniumCoreSystem(config)

# Get cell information
cell = system.get_cell((0, 0, 0))
print(f"Face exposure: {cell.face_exposure}")
print(f"Symbolic Weight: {cell.symbolic_weight}")
print(f"Class: {cell.cell_class}")

# Rotate the system
system.rotate(RotationAxis.X, quarter_turns=1)

# Verify invariants
total_sw = system.get_total_symbolic_weight()
print(f"Total SW: {total_sw}")  # Should be 486 for N=3
```

### Example 2: Quantum Layer Only

```python
from core import (
    LivniumCoreSystem, LivniumCoreConfig,
    QuantumLattice, GateType
)

# Enable quantum features
config = LivniumCoreConfig(
    lattice_size=3,
    enable_quantum=True,
    enable_superposition=True,
    enable_quantum_gates=True,
    enable_entanglement=True,
    enable_measurement=True,
    enable_geometry_quantum_coupling=True
)

# Create systems
core = LivniumCoreSystem(config)
qlattice = QuantumLattice(core)

# Apply Hadamard gate to create superposition
qlattice.apply_gate((0, 0, 0), GateType.HADAMARD)

# Entangle two cells
qlattice.entangle_cells((0, 0, 0), (1, 0, 0))

# Measure
result = qlattice.measure_cell((0, 0, 0))
print(f"Measured: {result.measured_state}")
print(f"Probability: {result.probability}")
```

### Example 3: Minimal Configuration

```python
from core import LivniumCoreSystem, LivniumCoreConfig

# Minimal configuration - only basic lattice
config = LivniumCoreConfig(
    enable_symbolic_weight=False,
    enable_face_exposure=False,
    enable_class_structure=False,
    enable_90_degree_rotations=False,
    enable_rotation_group=False,
    enable_global_observer=False,
    enable_local_observer=False,
    enable_semantic_polarity=False,
    enable_cross_lattice_coupling=False,
    enable_sw_conservation=False,
    enable_class_count_conservation=False
)

system = LivniumCoreSystem(config)
# Only basic lattice structure available
```

### Example 4: Quantum with Specific Gates

```python
from core import (
    LivniumCoreSystem, LivniumCoreConfig,
    QuantumLattice, GateType
)

config = LivniumCoreConfig(
    enable_quantum=True,
    enable_superposition=True,
    enable_quantum_gates=True,
    enable_entanglement=False,  # Disable entanglement
    enable_measurement=True,
    enable_geometry_quantum_coupling=False  # Disable coupling
)

core = LivniumCoreSystem(config)
qlattice = QuantumLattice(core)

# Apply different gates
qlattice.apply_gate((0, 0, 0), GateType.HADAMARD)
qlattice.apply_gate((1, 0, 0), GateType.PAULI_X)
qlattice.apply_gate((0, 1, 0), GateType.PAULI_Y)
qlattice.apply_gate((0, 0, 1), GateType.PAULI_Z)

# Apply rotation gate
qlattice.apply_gate((0, 0, 0), GateType.ROTATION_X, angle=np.pi/4)
```

### Example 5: Custom Lattice Size

```python
from core import LivniumCoreSystem, LivniumCoreConfig

# 5×5×5 lattice
config = LivniumCoreConfig(lattice_size=5)
system = LivniumCoreSystem(config)

# Total SW for N=5
total_sw = system.get_total_symbolic_weight()
print(f"Total SW (5×5×5): {total_sw}")  # Should be 1350

# Get class counts
class_counts = system.get_class_counts()
print(f"Core cells: {class_counts[CellClass.CORE]}")      # (5-2)³ = 27
print(f"Center cells: {class_counts[CellClass.CENTER]}")   # 6(5-2)² = 54
print(f"Edge cells: {class_counts[CellClass.EDGE]}")       # 12(5-2) = 36
print(f"Corner cells: {class_counts[CellClass.CORNER]}")    # 8
```

### Example 6: Full System with All Layers

```python
from core import (
    LivniumCoreSystem, LivniumCoreConfig,
    Orchestrator, EpisodeManager
)

# Enable all layers
config = LivniumCoreConfig(
    # Classical (always enabled)
    enable_recursive_geometry=True,  # Layer 0
    enable_moksha=True,
    
    # Quantum
    enable_quantum=True,
    enable_superposition=True,
    enable_quantum_gates=True,
    enable_entanglement=True,
    enable_measurement=True,
    enable_geometry_quantum_coupling=True,
    
    # Memory
    enable_memory=True,
    enable_working_memory=True,
    enable_long_term_memory=True,
    enable_memory_coupling=True,
    
    # Reasoning
    enable_reasoning=True,
    enable_search=True,
    enable_rules=True,
    enable_problem_solving=True,
    
    # Semantic
    enable_semantic=True,
    enable_feature_extraction=True,
    enable_meaning_graph=True,
    enable_inference=True,
    
    # Meta
    enable_meta=True,
    enable_introspection=True,
    enable_anomaly_detection=True,
    enable_calibration=True,
    
    # Runtime
    enable_runtime=True,
    enable_episodes=True
)

core = LivniumCoreSystem(config)
orchestrator = Orchestrator(core)
episode_manager = EpisodeManager(orchestrator)

episode = episode_manager.start_episode()
episode = episode_manager.run_episode(max_timesteps=100)
```

---

## Configuration Reference

### Complete Configuration Class

```python
@dataclass
class LivniumCoreConfig:
    # Core Structure
    enable_3x3x3_lattice: bool = True
    enable_symbol_alphabet: bool = True
    
    # Symbolic Weight
    enable_symbolic_weight: bool = True
    enable_face_exposure: bool = True
    enable_class_structure: bool = True
    
    # Dynamic Law
    enable_90_degree_rotations: bool = True
    enable_rotation_group: bool = True
    
    # Observer System
    enable_global_observer: bool = True
    enable_local_observer: bool = True
    enable_observer_coordinates: bool = True
    
    # Semantic Polarity
    enable_semantic_polarity: bool = True
    
    # Cross-Lattice Coupling
    enable_cross_lattice_coupling: bool = True
    
    # Quantum Features
    enable_quantum: bool = False
    enable_superposition: bool = False
    enable_quantum_gates: bool = False
    enable_entanglement: bool = False
    enable_measurement: bool = False
    enable_geometry_quantum_coupling: bool = False
    
    # Memory Layer
    enable_memory: bool = False
    enable_working_memory: bool = False
    enable_long_term_memory: bool = False
    enable_memory_coupling: bool = False
    
    # Reasoning Layer
    enable_reasoning: bool = False
    enable_search: bool = False
    enable_rules: bool = False
    enable_problem_solving: bool = False
    
    # Semantic Layer
    enable_semantic: bool = False
    enable_feature_extraction: bool = False
    enable_meaning_graph: bool = False
    enable_inference: bool = False
    
    # Meta Layer
    enable_meta: bool = False
    enable_introspection: bool = False
    enable_anomaly_detection: bool = False
    enable_calibration: bool = False
    
    # Runtime
    enable_runtime: bool = False
    enable_episodes: bool = False
    
    # Recursive Geometry (Layer 0)
    enable_recursive_geometry: bool = False
    recursive_max_depth: int = 3
    recursive_subdivision_rule: str = "default"
    enable_moksha: bool = False
    moksha_convergence_threshold: float = 0.999
    moksha_stability_window: int = 10
    
    # Invariants
    enable_sw_conservation: bool = True
    enable_class_count_conservation: bool = True
    
    # Hierarchical Extension
    enable_hierarchical_extension: bool = False
    hierarchical_macro_size: int = 3
    hierarchical_micro_size: int = 3
    
    # Lattice Size
    lattice_size: int = 3  # Must be odd, ≥ 3
    
    # Equilibrium Constant
    equilibrium_constant: float = 10.125
```

### Quick Reference: Default Values

| Feature | Default | Category |
|---------|---------|----------|
| `enable_3x3x3_lattice` | `True` | Core Structure |
| `enable_symbol_alphabet` | `True` | Core Structure |
| `enable_symbolic_weight` | `True` | Symbolic Weight |
| `enable_face_exposure` | `True` | Symbolic Weight |
| `enable_class_structure` | `True` | Symbolic Weight |
| `enable_90_degree_rotations` | `True` | Dynamic Law |
| `enable_rotation_group` | `True` | Dynamic Law |
| `enable_global_observer` | `True` | Observer System |
| `enable_local_observer` | `True` | Observer System |
| `enable_semantic_polarity` | `True` | Semantic Polarity |
| `enable_cross_lattice_coupling` | `True` | Cross-Lattice |
| `enable_quantum` | `False` | Quantum |
| `enable_superposition` | `False` | Quantum |
| `enable_quantum_gates` | `False` | Quantum |
| `enable_entanglement` | `False` | Quantum |
| `enable_measurement` | `False` | Quantum |
| `enable_geometry_quantum_coupling` | `False` | Quantum |
| `enable_memory` | `False` | Memory |
| `enable_reasoning` | `False` | Reasoning |
| `enable_semantic` | `False` | Semantic |
| `enable_meta` | `False` | Meta |
| `enable_runtime` | `False` | Runtime |
| `enable_recursive_geometry` | `False` | Recursive |
| `enable_moksha` | `False` | Recursive |
| `lattice_size` | `3` | Lattice Size |

---

## Summary

The Livnium Core System provides:

1. **Complete Geometric Foundation**: N×N×N lattice with symbolic weight, rotations, and observer system
2. **Full Quantum Layer**: Superposition, gates, entanglement, measurement, and geometry-quantum coupling
3. **Fine-Grained Control**: All features can be enabled/disabled independently
4. **Dependency Validation**: Automatic validation of feature dependencies
5. **Generalized Design**: Works for any odd N ≥ 3

**Key Principle**: The system is modular and configurable, allowing you to use only what you need while maintaining full compatibility between enabled features.


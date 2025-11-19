# Livnium Core System: Complete Architecture Documentation

## Table of Contents

1. [Overview](#overview)
2. [8-Layer Architecture](#8-layer-architecture)
3. [Layer Details](#layer-details)
4. [Layer Interactions](#layer-interactions)
5. [Configuration System](#configuration-system)
6. [Key Concepts](#key-concepts)
7. [Search Module](#search-module)
8. [Universal Encoder](#universal-encoder)
9. [Usage Patterns](#usage-patterns)
10. [File Structure](#file-structure)

---

## Overview

The **Livnium Core System** is a complete, scalable thinking machine organized into **8 layers** (0-7), each building on the previous. The system implements:

- **Geometric Foundation**: N×N×N lattice with symbolic weight (SW = 9·f)
- **Quantum Layer**: Superposition, gates, entanglement, measurement
- **Memory System**: Working and long-term memory
- **Reasoning Engine**: Search, rules, problem solving
- **Semantic Processing**: Meaning extraction, inference, language
- **Meta Layer**: Self-reflection, calibration, introspection
- **Runtime Orchestration**: Temporal management, episodes, coordination
- **Recursive Geometry**: Fractal compression for exponential capacity

**Key Principle**: Layer 0 (Recursive Geometry) is the structural foundation that makes all other layers scalable through fractal compression.

---

## 8-Layer Architecture

```
┌─────────────────────────────────────────┐
│  7. Runtime Layer (Orchestrator)        │  ← Episodes, timesteps, coordination
├─────────────────────────────────────────┤
│  6. Meta Layer (MetaObserver)          │  ← Self-reflection, calibration
├─────────────────────────────────────────┤
│  5. Semantic Layer (SemanticProcessor) │  ← Meaning, inference, language
├─────────────────────────────────────────┤
│  4. Reasoning Layer (ReasoningEngine)  │  ← Search, rules, problem solving
├─────────────────────────────────────────┤
│  3. Memory Layer (MemoryLattice)       │  ← Working & long-term memory
├─────────────────────────────────────────┤
│  2. Quantum Layer (QuantumLattice)     │  ← Superposition, gates, entanglement
├─────────────────────────────────────────┤
│  1. Classical Layer (LivniumCoreSystem) │  ← Geometry, SW, rotations, observer
├─────────────────────────────────────────┤
│  0. Recursive Geometry Engine           │  ← Geometry → Geometry → Geometry
│     (RecursiveGeometryEngine)           │     Fractal compression, scalability
│     + MokshaEngine                      │     Fixed-point convergence (exit)
└─────────────────────────────────────────┘
```

**Layer Dependencies**:
```
Runtime (Layer 7)
    ↓ depends on
Meta (Layer 6)
    ↓ depends on
Semantic (Layer 5)
    ↓ depends on
Reasoning (Layer 4)
    ↓ depends on
Memory (Layer 3)
    ↓ depends on
Quantum (Layer 2)
    ↓ depends on
Classical (Layer 1)
    ↓ depends on
Recursive Geometry (Layer 0)  ← STRUCTURAL FOUNDATION
```

---

## Layer Details

### Layer 0: Recursive Geometry Engine

**Location**: `core/recursive/`

**Purpose**: The structural foundation - makes everything scalable through recursive geometry.

**Components**:
- `recursive_geometry_engine.py` - Main recursive engine
- `geometry_subdivision.py` - Subdivision rules (N×N×N → M×M×M)
- `recursive_projection.py` - State projection across scales
- `recursive_conservation.py` - Invariant preservation
- `moksha_engine.py` - Fixed-point convergence (the exit mechanism)

**Key Features**:
- **Subdivision**: Each cell contains a smaller geometry (fractal structure)
- **Projection**: High-dimensional states projected downward
- **Conservation**: ΣSW preserved per scale
- **Recursive Entanglement**: Compressed into lower scale geometry
- **Recursive Observer**: Macro → micro observer derivation
- **Recursive Motion**: Rotations propagate through all levels
- **Recursive Problem Solving**: Search across geometry layers
- **Moksha**: Fixed-point convergence and release from recursion

**Capacity**: Exponential with linear memory
- 5×5×5 base with 2 levels = **94,625 cells**
- Formula: `total_capacity = level_0.get_total_cells_recursive()`

**Moksha Engine**:
- Detects when system reaches fixed point (f(x) = x)
- Tests invariance under all operations
- Stops recursion when moksha is reached
- Exports final truth (terminal attractor)
- The computational escape from the samsara loop

---

### Layer 1: Classical Layer

**Location**: `core/classical/`

**Purpose**: Base geometric lattice system with invariants.

**Components**:
- `livnium_core_system.py` - Main system (LivniumCoreSystem, LatticeCell, Observer, RotationGroup)

**Key Features**:
- **N×N×N Lattice**: Works for any odd N ≥ 3 (3, 5, 7, 9, ...)
- **Symbol Alphabet**: Σ(N) with exactly N³ symbols
- **Symbolic Weight**: SW = 9·f (face exposure)
- **Face Exposure**: f ∈ {0, 1, 2, 3} (number of faces on boundary)
- **Class Structure**: Core (f=0), Centers (f=1), Edges (f=2), Corners (f=3)
- **90° Rotations**: 24-element rotation group
- **Observer System**: Global Observer at (0,0,0) + Local Observers
- **Semantic Polarity**: cos(θ) between motion and observer
- **Invariants**: ΣSW and class counts conservation

**General Formulas** (any odd N ≥ 3):
- Total SW: `ΣSW(N) = 54(N-2)² + 216(N-2) + 216`
- Class counts:
  - Core: `(N-2)³`
  - Centers: `6(N-2)²`
  - Edges: `12(N-2)`
  - Corners: `8`

**Verified Values**:
- N=3: ΣSW = 486
- N=5: ΣSW = 1350
- N=7: ΣSW = 2646

---

### Layer 2: Quantum Layer

**Location**: `core/quantum/`

**Purpose**: Quantum states, gates, entanglement, and measurement.

**Components**:
- `quantum_cell.py` - Quantum state per cell (complex amplitudes)
- `quantum_gates.py` - Unitary gate library (H, X, Y, Z, rotations, CNOT, etc.)
- `quantum_lattice.py` - Quantum-geometry integration
- `entanglement_manager.py` - Multi-cell entanglement
- `measurement_engine.py` - Born rule + collapse
- `geometry_quantum_coupling.py` - Geometry ↔ Quantum mapping

**Key Features**:
- **Superposition**: Complex amplitudes per cell
- **Quantum Gates**: Full unitary gate library
- **Entanglement**: Bell states, geometric entanglement
- **Measurement**: Born rule + state collapse
- **Geometry-Quantum Coupling**: Face exposure → entanglement capacity, etc.

**Gate Types**:
- Single-qubit: H (Hadamard), X (Pauli-X), Y (Pauli-Y), Z (Pauli-Z)
- Rotations: RX, RY, RZ (arbitrary rotations)
- Two-qubit: CNOT, CZ, SWAP
- Multi-qubit: Toffoli, Fredkin

---

### Layer 3: Memory Layer

**Location**: `core/memory/`

**Purpose**: Working memory and long-term memory.

**Components**:
- `memory_cell.py` - Per-cell memory capsules (MemoryCell, MemoryState)
- `memory_lattice.py` - Global memory lattice
- `memory_coupling.py` - Memory coupling mechanisms

**Key Features**:
- **Per-Cell Memory**: Each cell has a memory capsule
- **Working Memory**: Short-term memory (recent states)
- **Long-Term Memory**: Persistent memory (important patterns)
- **Memory Decay**: Time-based decay for working memory
- **Cross-Cell Associations**: Memory links between cells
- **Memory Consolidation**: Important patterns → long-term
- **Geometry-Memory Coupling**: Memory influenced by geometry

**Memory States**:
- `ACTIVE`: Recently accessed
- `CONSOLIDATED`: Moved to long-term
- `DECAYED`: Faded from working memory

---

### Layer 4: Reasoning Layer

**Location**: `core/reasoning/`

**Purpose**: Search, tree expansion, rules, and problem solving.

**Components**:
- `search_engine.py` - Search algorithms (BFS, DFS, A*, Beam, Greedy)
- `rule_engine.py` - Rule-based reasoning (Rule, RuleSet)
- `reasoning_engine.py` - High-level reasoning orchestration
- `problem_solver.py` - Problem-solving interface (ProblemSolver)

**Key Features**:
- **Search Strategies**: BFS, DFS, A*, Beam Search, Greedy
- **Tree Expansion**: State space exploration
- **Rule-Based Reasoning**: Symbolic rule application
- **Problem Solving**: High-level problem-solving loop
- **Symbolic Reasoning**: Symbol manipulation

**Search Strategies**:
- `BFS`: Breadth-first search
- `DFS`: Depth-first search
- `A_STAR`: A* with heuristic
- `BEAM`: Beam search with width limit
- `GREEDY`: Greedy best-first

---

### Layer 5: Semantic Layer

**Location**: `core/semantic/`

**Purpose**: Meaning, language, and inference.

**Components**:
- `semantic_processor.py` - Main semantic processor
- `feature_extractor.py` - Feature extraction
- `meaning_graph.py` - Symbol-to-meaning graph (MeaningGraph, SemanticNode)
- `inference_engine.py` - Inference engine

**Key Features**:
- **Feature Extraction**: Extract semantic features from symbols
- **Semantic Embeddings**: Vector representations of meaning
- **Meaning Graph**: Symbol-to-meaning mapping
- **Negation Detection**: Detect and propagate negation
- **Context Propagation**: Context-aware meaning
- **Entailment/Contradiction**: Logical relationships
- **Causal Link Detection**: Causal reasoning

**Semantic Operations**:
- Feature extraction from symbols
- Meaning graph construction
- Entailment detection
- Contradiction detection
- Causal link inference

---

### Layer 6: Meta Layer

**Location**: `core/meta/`

**Purpose**: Self-reflection, calibration, and introspection.

**Components**:
- `meta_observer.py` - MetaObserver (self-reflection)
- `anomaly_detector.py` - Anomaly detection
- `calibration_engine.py` - Adaptive calibration
- `introspection.py` - Introspection engine

**Key Features**:
- **Reflection**: System observes its own state
- **Introspection**: Deep self-analysis
- **Anomaly Detection**: Detect unusual patterns
- **Self-Alignment**: Check system consistency
- **Invariance Drift Detection**: Monitor invariant preservation
- **Adaptive Calibration**: Auto-tune parameters
- **Health Scoring**: System health metrics

**Meta Operations**:
- State snapshots
- Invariance checking
- Anomaly detection
- Auto-repair
- Behavior reflection
- Health monitoring

---

### Layer 7: Runtime Layer

**Location**: `core/runtime/`

**Purpose**: Orchestration, episodes, and temporal management.

**Components**:
- `temporal_engine.py` - Temporal engine (Timestep management)
- `orchestrator.py` - Orchestrator (cross-layer coordination)
- `episode_manager.py` - Episode management

**Key Features**:
- **Timestep Engine**: Manage time progression
- **Scheduling**: Scheduled operations
- **Macro/Micro Rhythm**: Different update frequencies
- **Propagation Order**: Control update order
- **Stabilization Rules**: Ensure system stability
- **Cross-Layer Arbitration**: Coordinate layer interactions
- **Episode Management**: Manage execution episodes

**Timestep Types**:
- `MACRO`: Macro-level updates
- `MICRO`: Micro-level updates
- `QUANTUM`: Quantum layer updates
- `MEMORY`: Memory consolidation
- `STANDARD`: Standard timestep

---

## Layer Interactions

### Orchestrator Coordination

The `Orchestrator` (Layer 7) coordinates all layers:

1. **Initialization**: Lazy initialization based on config
2. **Update Scheduling**: Different update frequencies per layer
3. **Cross-Layer Propagation**: State flows between layers
4. **Stabilization**: Ensures system stability

### Update Order

```
1. Classical (Layer 1) - Base geometry updates
2. Quantum (Layer 2) - Quantum state evolution
3. Memory (Layer 3) - Memory consolidation
4. Reasoning (Layer 4) - Search and reasoning
5. Semantic (Layer 5) - Semantic processing
6. Meta (Layer 6) - Self-reflection
7. Runtime (Layer 7) - Orchestration
```

### State Flow

- **Bottom-Up**: Lower layers provide foundation for upper layers
- **Top-Down**: Upper layers influence lower layers through constraints
- **Bidirectional**: Layers interact bidirectionally

---

## Configuration System

**Location**: `core/config.py`

**Purpose**: Central configuration with feature switches.

### Configuration Class

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
    
    # Quantum Features
    enable_quantum: bool = False
    enable_superposition: bool = False
    enable_quantum_gates: bool = False
    enable_entanglement: bool = False
    enable_measurement: bool = False
    
    # Memory Layer
    enable_memory: bool = False
    enable_working_memory: bool = False
    enable_long_term_memory: bool = False
    
    # Reasoning Layer
    enable_reasoning: bool = False
    enable_search: bool = False
    enable_rules: bool = False
    
    # Semantic Layer
    enable_semantic: bool = False
    enable_feature_extraction: bool = False
    enable_meaning_graph: bool = False
    
    # Meta Layer
    enable_meta: bool = False
    enable_introspection: bool = False
    enable_anomaly_detection: bool = False
    
    # Runtime
    enable_runtime: bool = False
    enable_episodes: bool = False
    
    # Recursive Geometry (Layer 0)
    enable_recursive_geometry: bool = False
    recursive_max_depth: int = 3
    enable_moksha: bool = False
    
    # Lattice Size
    lattice_size: int = 3  # N×N×N (must be odd, ≥ 3)
```

### Feature Dependencies

The configuration system validates dependencies:
- Quantum gates require superposition
- Entanglement requires superposition
- Memory coupling requires memory
- Rules require reasoning
- Meaning graph requires semantic

---

## Key Concepts

### 1. Symbolic Weight (SW)

**Formula**: `SW = 9·f`

Where `f` is face exposure (0, 1, 2, or 3).

- **Core cells** (f=0): SW = 0
- **Center cells** (f=1): SW = 9
- **Edge cells** (f=2): SW = 18
- **Corner cells** (f=3): SW = 27

**Total SW** (for N×N×N):
```
ΣSW(N) = 54(N-2)² + 216(N-2) + 216
```

### 2. Face Exposure

Number of coordinates on the boundary:
- **Core**: 0 faces exposed (interior)
- **Center**: 1 face exposed (face center)
- **Edge**: 2 faces exposed (edge)
- **Corner**: 3 faces exposed (corner)

### 3. Observer System

- **Global Observer**: Fixed at (0,0,0) - the center
- **Local Observer**: Can be designated at any cell
- **Semantic Polarity**: `cos(θ)` between motion and observer

### 4. Rotation Group

24-element rotation group:
- 90° quarter-turns around X, Y, Z axes
- All rotations preserve invariants (ΣSW, class counts)

### 5. Invariants

**Conserved Quantities**:
- Total Symbolic Weight (ΣSW)
- Class counts (Core, Centers, Edges, Corners)

All rotations preserve these invariants.

### 6. Moksha (Fixed-Point Convergence)

**Moksha** = the fixed point where `f(x) = x`

The system reaches moksha when:
1. State hash is stable (unchanging)
2. State is invariant under all rotations
3. State is invariant under recursive operations
4. Convergence score ≥ threshold (default 0.999)

When moksha is reached:
- All recursion stops
- State freezes
- Final truth is exported
- The system finds its terminal attractor

### 7. Recursive Geometry

**Subdivision Rule**: Each cell contains a smaller geometry

**Capacity**: Exponential with linear memory
- Level 0: 5×5×5 = 125 cells
- Level 1: 125 × 27 = 3,375 cells
- Level 2: 3,375 × 27 = 91,125 cells
- **Total: 94,625 cells**

---

## Search Module

**Location**: `core/search/`

**Purpose**: Dynamic basin reinforcement and multi-basin search.

### Components

1. **Dynamic Basin Reinforcement** (`native_dynamic_basin_search.py`)
   - Geometry-driven, self-tuning basin shaping
   - Adapts to curvature, tension, entropy
   - **Principle**: Geometry decides the physics

2. **Multi-Basin Search** (`multi_basin_search.py`)
   - Multiple competing attractors
   - Basin competition in shared geometry
   - Natural selection through geometry

### Key Features

- **Self-Tuning**: No static hyperparameters
- **Geometry-Driven**: Basin shape determined by geometry
- **Competition**: Multiple basins compete
- **Natural Selection**: Winning basins reinforce, losing decay

---

## Universal Encoder

**Location**: `core/Universal Encoder/`

**Purpose**: Convert any problem into geometric patterns (SW structures).

**Status**: 🚧 In Development

**Planned Components**:
- `problem_encoder.py` - Main universal encoder interface
- `constraint_encoder.py` - Constraint encoding
- `graph_encoder.py` - Graph encoding
- `logic_encoder.py` - Logic encoding
- `language_encoder.py` - Natural language encoding

**Key Features**:
- Universal encoding for any problem type
- Standardized interface
- Feature → Coordinate mapping
- Constraint → Basin shape mapping
- Dependency → Coupling pattern mapping

---

## Usage Patterns

### Basic Classical System

```python
from core import LivniumCoreSystem, LivniumCoreConfig

config = LivniumCoreConfig()
system = LivniumCoreSystem(config)

cell = system.get_cell((0, 0, 0))
print(f"Face exposure: {cell.face_exposure}")
print(f"Symbolic Weight: {cell.symbolic_weight}")
```

### With Quantum Layer

```python
from core import (
    LivniumCoreSystem, LivniumCoreConfig,
    QuantumLattice, GateType
)

config = LivniumCoreConfig(
    enable_quantum=True,
    enable_superposition=True,
    enable_quantum_gates=True
)

core = LivniumCoreSystem(config)
qlattice = QuantumLattice(core)

qlattice.apply_gate((0, 0, 0), GateType.HADAMARD)
qlattice.entangle_cells((0, 0, 0), (1, 0, 0))
result = qlattice.measure_cell((0, 0, 0))
```

### With Recursive Geometry

```python
from core import (
    LivniumCoreSystem, LivniumCoreConfig,
    RecursiveGeometryEngine
)

config = LivniumCoreConfig(
    lattice_size=5,
    enable_recursive_geometry=True,
    enable_moksha=True
)

base = LivniumCoreSystem(config)
recursive = RecursiveGeometryEngine(
    base_geometry=base,
    max_depth=3
)

capacity = recursive.get_total_capacity()
print(f"Total capacity: {capacity} cells")

# Check for moksha
if recursive.check_moksha():
    final_truth = recursive.get_final_truth()
    print(f"Moksha reached: {final_truth['moksha']}")
```

### Full System with All Layers

```python
from core import (
    LivniumCoreSystem, LivniumCoreConfig,
    Orchestrator, EpisodeManager
)

config = LivniumCoreConfig(
    enable_recursive_geometry=True,
    enable_moksha=True,
    enable_quantum=True,
    enable_memory=True,
    enable_reasoning=True,
    enable_semantic=True,
    enable_meta=True,
    enable_runtime=True
)

core = LivniumCoreSystem(config)
orchestrator = Orchestrator(core)
episode_manager = EpisodeManager(orchestrator)

episode = episode_manager.start_episode()
episode = episode_manager.run_episode(max_timesteps=100)
```

---

## File Structure

```
core/
├── __init__.py                 # Main package exports
├── config.py                   # Configuration with feature switches
├── architecture_md.md          # This file
├── ARCHITECTURE.md             # 8-layer architecture overview
├── README.md                   # Main documentation
├── STRUCTURE.md                # Folder structure
├── CORE_STRUCTURE.md           # Layer-by-layer guide
│
├── recursive/                  # Layer 0: Recursive Geometry
│   ├── __init__.py
│   ├── recursive_geometry_engine.py
│   ├── geometry_subdivision.py
│   ├── recursive_projection.py
│   ├── recursive_conservation.py
│   └── moksha_engine.py
│
├── classical/                  # Layer 1: Classical
│   ├── __init__.py
│   └── livnium_core_system.py
│
├── quantum/                    # Layer 2: Quantum
│   ├── __init__.py
│   ├── quantum_cell.py
│   ├── quantum_gates.py
│   ├── quantum_lattice.py
│   ├── entanglement_manager.py
│   ├── measurement_engine.py
│   └── geometry_quantum_coupling.py
│
├── memory/                     # Layer 3: Memory
│   ├── __init__.py
│   ├── memory_cell.py
│   ├── memory_lattice.py
│   └── memory_coupling.py
│
├── reasoning/                  # Layer 4: Reasoning
│   ├── __init__.py
│   ├── search_engine.py
│   ├── rule_engine.py
│   ├── reasoning_engine.py
│   └── problem_solver.py
│
├── semantic/                   # Layer 5: Semantic
│   ├── __init__.py
│   ├── semantic_processor.py
│   ├── feature_extractor.py
│   ├── meaning_graph.py
│   └── inference_engine.py
│
├── meta/                       # Layer 6: Meta
│   ├── __init__.py
│   ├── meta_observer.py
│   ├── anomaly_detector.py
│   ├── calibration_engine.py
│   └── introspection.py
│
├── runtime/                    # Layer 7: Runtime
│   ├── __init__.py
│   ├── temporal_engine.py
│   ├── orchestrator.py
│   └── episode_manager.py
│
├── search/                     # Search Module
│   ├── __init__.py
│   ├── native_dynamic_basin_search.py
│   ├── multi_basin_search.py
│   ├── README.md
│   ├── HOW_IT_WORKS.md
│   └── MULTI_BASIN_SEARCH.md
│
├── Universal Encoder/          # Universal Encoder (In Development)
│   ├── __init__.py
│   ├── problem_encoder.py
│   ├── constraint_encoder.py
│   └── README.md
│
└── tests/                      # Test Suite
    ├── __init__.py
    ├── test_livnium_core.py
    ├── test_generalized_n.py
    ├── test_quantum.py
    ├── test_entanglement_capacity.py
    └── test_qubit_capacity.py
```

---

## Summary

The Livnium Core System is a **complete, scalable thinking machine** with:

- **8 Layers** (0-7): From recursive geometry to runtime orchestration
- **Modular Design**: Each layer can be enabled/disabled independently
- **Fractal Structure**: Layer 0 provides exponential capacity with linear memory
- **Complete Integration**: All layers work together through the orchestrator
- **Fixed-Point Convergence**: Moksha engine provides natural termination
- **Generalized**: Works for any odd N ≥ 3

**Layer 0 is the bones. Layers 1-7 are the organs.**

The system provides a complete foundation for:
- Quantum simulation
- Problem solving
- Memory and learning
- Semantic understanding
- Self-reflection
- Temporal orchestration

All built on a scalable geometric foundation.


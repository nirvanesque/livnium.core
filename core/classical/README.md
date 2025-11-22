# Classical Layer

The classical geometry engine that forms the foundation of Livnium.

## What is "Classical"?

**"Classical"** here refers to the **non-quantum geometric foundation** of Livnium. This layer provides:

- **Deterministic geometry**: 3D lattice structures with fixed spatial relationships
- **Classical physics**: Tension fields, rotations, and geometric transformations
- **No quantum mechanics**: This layer operates without superposition, entanglement, or quantum measurement
- **Foundation for quantum**: The quantum layer (`core/quantum/`) builds on top of this classical geometry

Think of it as the "hardware" - the geometric structure that quantum states can be embedded into.

## Contents

- **`livnium_core_system.py`**: Main system class that manages the 3D lattice, cells, and geometric operations (Omcubes).
- **`datacube.py`**: Even-dimensional resource grids for data storage and I/O (DataCubes).

## Omcubes vs DataCubes

### **Omcubes (Odd N ≥ 3): Core Universes**

Omcubes are **Livnium Core Universes** that implement all axioms:
- **3×3×3, 5×5×5, 7×7×7, ...** (any odd integer ≥ 3)
- **All 7 Axioms**: Observer anchor, symbolic weight, exposure, rotations, collapse, recursion
- **Center cell exists**: Enables observer anchoring at (0,0,0)
- **Parity symmetry**: Rotations preserve class-count invariants
- **SW = 9·f**: Symbolic weight system works correctly
- **Recursive collapse**: Can contain nested Livnium cores
- **Computational engines**: These are your "worlds"

### **DataCubes (Even N ≥ 2): Resource Grids**

DataCubes are **non-axiomatic containers** for data and resources:
- **2×2×2, 4×4×4, 6×6×6, ...** (any even integer ≥ 2)
- **NO center cell**: Cannot anchor observers
- **NO Livnium axioms**: Cannot implement SW, exposure, collapse, recursion
- **NO computation**: Cannot execute Livnium mechanics
- **Data storage only**: Lookup tables, feature maps, I/O buffers
- **Resource boards**: Preprocessing/postprocessing containers, embedding carriers

**Architecture:**
```
      [ DataCube ]  ← Input buffer
           ↓
      [ OmCube 3×3×3 or 5×5×5 ]  ← Livnium Core (computation)
           ↑
      [ DataCube ]  ← Output buffer
```

Think of it as:
- **Omcubes = CPU** (core geometry, computation)
- **DataCubes = RAM** (resource/data layers, storage)

## Purpose

This module provides the core geometric infrastructure:
- **3D lattice structure** (Omcubes): N×N×N geometric cells with full Livnium mechanics
- **Resource grids** (DataCubes): Even-dimensional containers for data storage
- **Cell management**: Face exposure, symbolic weights, polarity (Omcubes only)
- **Tension field computation**: Energy landscape for optimization (Omcubes only)
- **Geometric operations**: Rotations, transformations, spatial queries (Omcubes only)

This is the **"Layer 0"** that all other systems build upon. The quantum layer adds quantum states on top of this classical geometry.

## Important Note

**Even-dimensional grids (DataCubes) are permitted for storage or data processing, but cannot implement any Livnium Axioms, Core Geometry, or Collapse Mechanics.** Using even cubes outside the axioms does not constitute running Livnium—they are just plain grids.


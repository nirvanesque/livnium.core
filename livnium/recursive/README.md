# Livnium Recursive Layer (Experimental)

> [!WARNING]
> **Experimental Research Layer**
> 
> This module implements the "Fractal Architecture" (Geometry-within-Geometry) and the "Moksha Engine" (Fixed-Point Convergence).
> It is an advanced, experimental layer derived from the `archives-local` system.
> It operates on the `livnium/classical` geometric stack, NOT the main `livnium/engine` kernel stack.

## Overview

The `livnium/recursive` package implements the "Dual Shadow" vision of Livnium: that the universe is a fractal structure where every cell contains a smaller universe, and that "Truth" is a fixed point (invariant state) in this dynamical system.

## Components

### 1. Recursive Geometry Engine (`recursive_geometry_engine.py`)
- **Fractal Scaling**: Implements $N \times N \times N$ lattices nested within each cell of a parent lattice.
- **Inheritance**: Child universes inherit "Symbolic Weight" (energy) from their parent (Container) cell.
- **Infinite Context**: Theoretically supports infinite recursion depth.

### 2. Moksha Engine (`moksha_engine.py`)
- **Fixed-Point Detection**: Monitors the system state (hash of all levels) to detect convergence.
- **Moksha (The End State)**: A state is in "Moksha" if it is **invariant** under all operations (Rotation, Recursion, Projection).
- **Truth Export**: When Moksha is reached, the system freezes and exports the final state as "Truth."

### 3. Usage Note: Hamiltonian Dynamics
The `RecursiveHamiltonian` module depends on a `hamiltonian_core.py` that is missing from the current archives. 
Therefore, `enable_hamiltonian_dynamics()` should **not** be used. Use manual evolution (like `apply_recursive_rotation`) instead, as shown in the demo.

## Usage

### Basic Initialization

```python
from livnium.classical.livnium_core_system import LivniumCoreSystem
from livnium.recursive import RecursiveGeometryEngine, MokshaEngine

# 1. Create Base Geometry (Root)
base = LivniumCoreSystem()

# 2. Create Recursive Engine (Fractal)
engine = RecursiveGeometryEngine(base_geometry=base, max_depth=2)

# 3. Create Convergence Detector
moksha = MokshaEngine(engine)

# 4. Run loop
while not moksha.should_terminate():
    # Evolve system (e.g. apply rotations)
    engine.evolve_step() 
    
    # Check for fixed point
    moksha.check_convergence()
```

### Running the Demo

```bash
python3 livnium/examples/recursive_moksha_demo.py
```

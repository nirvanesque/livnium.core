# Constraint-Graph Physics: The Correct Ramsey Engine

## The Problem

The initial Ramsey test was using **lattice-based physics** (cube topology) for a **constraint-graph problem** (graph topology). This was fundamentally wrong.

### What Was Wrong

1. **Lattice φ-gradients** - Computed from cube neighbors, not graph neighbors
2. **Cube rotations** - Don't correspond to constraint changes in graph space
3. **Manhattan propagation** - Tension spread in cube topology, not constraint topology
4. **Cell-based freeze** - Detected low SW in cells, not monochromatic triangles

### The Root Issue

**Fast Task Test** = Om-cube structural projection (lattice physics)  
**Ramsey Task** = Constraint graph dynamics (graph physics)

**They use different geometry.**

---

## The Solution: Constraint-Graph Physics

### New Module: `constraint_graph.py`

Provides constraint-graph-based tension propagation and healing, **NOT** lattice-based.

### Key Functions

#### 1. `build_triangle_adjacency(vertices, edges)`

Builds triangle adjacency graph:
- Two triangles are adjacent if they share an edge
- Defines constraint topology for Ramsey problems

#### 2. `build_edge_adjacency(edges)`

Builds edge adjacency graph:
- Two edges are adjacent if they share a vertex
- Defines propagation topology for edge-based tension

#### 3. `get_constraint_phi(coloring, edge, edge_adjacency)`

Computes φ (polarity) for an edge based on **constraint topology**:
- φ represents "tension" or "contradiction potential"
- Based on neighbor edges' colors, not cube position
- Returns φ in [-1, 1] representing constraint tension

#### 4. `propagate_constraint_tension(...)`

Spreads tension along **constraint-graph topology**, NOT cube topology:
- Tension flows through edge adjacency (graph space)
- NOT through Manhattan neighbors (cube space)
- Propagates along steepest constraint gradients

#### 5. `heal_triangle_conflict(...)`

Heals triangle conflicts by adjusting edge colors:
- Acts on triangle constraints, not cube cells
- Changes edge colors to break monochromatic triangles
- Constraint-based healing, not cell-based

#### 6. `find_monochromatic_triangles(coloring, vertices)`

Finds all monochromatic triangles in the coloring.

#### 7. `compute_constraint_tension(...)`

Computes global constraint tension:
- High tension = many potential contradictions
- Low tension = stable coloring

---

## What Changed in `test_ramsey_r33.py`

### 1. Removed Lattice-Based Imports

**Before:**
```python
from fast_task_test import propagate_tension, heal_conflict
```

**After:**
```python
from constraint_graph import (
    propagate_constraint_tension,
    heal_triangle_conflict,
    build_triangle_adjacency,
    build_edge_adjacency
)
```

### 2. Built Constraint Graph Topology

**Added:**
```python
self.triangle_adjacency = build_triangle_adjacency(self.vertices, self.edges)
self.edge_adjacency = build_edge_adjacency(self.edges)
```

### 3. Fixed Freeze Detection

**Before:**
```python
def is_frozen(self) -> bool:
    # Check if any edge cell has very low SW
    for coords in self.edge_to_cell.values():
        cell = self.system.get_cell(coords)
        if cell and cell.symbolic_weight < 1.0:
            return True
    return False
```

**After:**
```python
def is_frozen(self) -> bool:
    """
    Check if system is frozen (constraint violation).
    
    In Ramsey context, freeze means:
    - Monochromatic triangles exist (constraint violation)
    - NOT low symbolic_weight in cube cells
    """
    coloring = self.decode_coloring()
    mono_triangles = find_monochromatic_triangles(coloring, self.vertices)
    return len(mono_triangles) > 0
```

### 4. Replaced Cube Rotations with Edge Flips

**Before:**
```python
# Try random rotation
axis = random.choice(list(RotationAxis))
system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
```

**After:**
```python
# In constraint graph space, we modify edge colors directly
# NOT through cube rotations (rotations don't correspond to constraint changes)

# Strategy: Try flipping edges that are part of monochromatic triangles
coloring = task.decode_coloring()
mono_triangles = find_monochromatic_triangles(coloring, task.vertices)

if mono_triangles:
    # Heal triangle conflicts (constraint-based healing)
    for triangle in mono_triangles[:1]:
        heal_triangle_conflict(coloring, triangle, task.edge_to_cell, system)
```

### 5. Replaced Lattice Propagation with Constraint Propagation

**Before:**
```python
propagate_tension(system, active)  # Lattice-based
```

**After:**
```python
propagate_constraint_tension(
    coloring,
    task.edge_adjacency,  # Constraint graph topology
    task.edge_to_cell,
    system,
    source_edges  # Edges from monochromatic triangles
)
```

---

## Why This Is Correct

### Physics Alignment

- **Ramsey problems** live in graph space, not cube space
- **Tension flows** along edge/triangle relationships
- **Healing acts** on triangle conflicts, not cell conflicts
- **φ-gradients** computed from constraint topology

### Real Physics Analogs

This matches how:
- **CDCL SAT solvers** propagate conflicts along clause graph
- **Graph-coloring solvers** eliminate bad motifs through edge adjacency
- **Constraint satisfaction** spreads errors through constraint graph
- **Statistical physics** spreads energy through interaction graph

---

## Summary

The system now uses:

✅ **Constraint-graph topology** - Edges/triangles, not cube neighbors  
✅ **Constraint-based φ** - Computed from edge adjacency, not cube position  
✅ **Triangle-based healing** - Acts on monochromatic triangles, not cells  
✅ **Graph-based propagation** - Tension flows along constraint relationships  
✅ **Edge-based operations** - Direct edge flips, not cube rotations  

**The Ramsey engine now uses TRUE Livnium constraint-graph physics.**


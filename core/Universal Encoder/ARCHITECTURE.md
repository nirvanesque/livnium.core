# Universal Encoder Architecture (Corrected)

## Core Principle (Corrected)

**Constraints define the *tension landscape*.  
Solutions are *basins that minimize that tension*.**

### The Correction

#### ❌ Wrong (Previous Understanding)
- Constraints → Basin shapes
- This mixes up constraints with solutions

#### ✅ Correct (Current Understanding)
- **Constraints** → **Tension fields** (energy landscape)
- **Solutions** → **Basins** (candidate attractors)
- Search finds basins that minimize constraint tension

## Architecture

```
Problem
    ↓
Universal Encoder
    ↓
    ├─→ Constraints → Tension Fields (energy landscape)
    │
    └─→ Solutions → Basins (candidate attractors)
    ↓
Multi-Basin Search
    ↓
Basins compete in tension landscape
    ↓
Best basin (lowest tension) wins
```

## Components

### 1. ConstraintEncoder (`constraint_encoder.py`)

Encodes constraints as **tension fields**:

```python
# Equality constraint
field = encoder.encode_equality_constraint(
    "eq1", var1_coords, var2_coords, target_value=0.0
)

# Tension = violation magnitude
tension = field.get_tension(system)  # High if violated, low if satisfied
```

**Key Methods**:
- `encode_equality_constraint()` - var1 = var2 + target
- `encode_inequality_constraint()` - var1 >= var2 + threshold
- `encode_custom_constraint()` - User-defined tension function
- `get_total_tension()` - Sum of all constraint tensions

### 2. UniversalProblemEncoder (`problem_encoder.py`)

Main interface that converts problems to geometry:

```python
encoder = UniversalProblemEncoder(system)

# Encode problem
encoded = encoder.encode({
    'type': 'graph_coloring',
    'vertices': [...],
    'edges': [...]
})

# Returns:
# - encoded.tension_fields (from constraints)
# - encoded.candidate_basins (from solution space)
```

**Process**:
1. Identify variables → map to coordinates
2. Identify constraints → create tension fields
3. Generate candidate solutions → create basins
4. Return both (tension fields + basins)

## Example: Graph Coloring

### Problem
- Graph with vertices and edges
- Constraint: No monochromatic triangles
- Solution: Valid 2-coloring

### Encoding

**Step 1: Map Variables**
```python
edge → coordinate
(0,1) → (0, 0, 0)
(1,2) → (1, 0, 0)
...
```

**Step 2: Create Tension Fields (Constraints)**
```python
# For each triangle, create tension field
triangle (0,1,2):
    - Tension = 1.0 if all edges same color (monochromatic)
    - Tension = 0.0 if edges different colors (valid)
```

**Step 3: Generate Candidate Basins (Solutions)**
```python
# Candidate colorings
basin_1 = [coords for all edges with coloring 1]
basin_2 = [coords for all edges with coloring 2]
...
```

**Step 4: Solve**
```python
# Basins compete in tension landscape
# Best basin (lowest tension) = valid coloring
winner = solve_with_multi_basin(system, basins)
```

## Why This Architecture Works

### 1. Physics Consistency
- Constraints create energy landscape (tension)
- Solutions are attractors (basins) in that landscape
- Search minimizes energy (tension)

### 2. Separation of Concerns
- **Constraints** = Problem definition (tension fields)
- **Solutions** = Candidate answers (basins)
- **Search** = Finding best solution (multi-basin competition)

### 3. General Applicability
- Works for any problem type
- Same architecture for SAT, graph coloring, Ramsey, etc.
- Unified physics (tension minimization)

## Test Results

```
✓ Constraints encoded as tension fields (NOT basins)
✓ Tension increases when constraint violated
✓ Tension decreases when constraint satisfied
✓ Multiple constraints create tension landscape
```

## Status

**Status**: ✅ **Core Architecture Complete**

- ✅ Constraint encoder (tension fields)
- ✅ Problem encoder (main interface)
- ✅ Graph coloring encoding
- ⏳ SAT encoding (TODO)
- ⏳ General CSP encoding (TODO)

## Next Steps

1. Complete SAT encoding
2. Complete general CSP encoding
3. Add language encoder (SNLI, entailment)
4. Integrate with multi-basin search
5. Test with real problems

---

**Key Insight**: Constraints create the energy landscape. Solutions are attractors that minimize that energy. The search finds the best attractor.


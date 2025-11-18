# Layer 0 Recursive Problem Solving Integration

## What Changed

The experiment now uses **Layer 0's recursive problem solving** - the "real trick that lets you solve big spaces cheaply."

## Key Insight from Layer 0

> **"Recursive Problem Solving: Search → happens across layers of geometry"**

Instead of searching in a single N×N×N lattice, we now:
- Create recursive geometry levels
- Search across layers (macro → micro)
- Use fractal compression
- Solve big spaces cheaply

## New Update Methods

### 1. `"recursive"` - Full Recursive Problem Solving
- Uses `RecursiveGeometryEngine` with multiple levels
- Searches across geometry layers
- Best for large N (7, 9, 11+)

### 2. `"hybrid_recursive"` - Recommended for Large N ⭐
- Uses recursive problem solving for N >= 5
- Falls back to simple updates for N = 3
- **Best for large systems**

## Architecture Layers Now Used

✅ **Layer 0 (Recursive)**: 
   - MokshaEngine for fast convergence
   - RecursiveGeometryEngine for recursive problem solving
   - Geometry subdivision for fractal compression

✅ **Layer 1 (Classical)**: LivniumCoreSystem for base geometry  
✅ **Layer 4 (Reasoning)**: ProblemSolver for intelligent search  

## How Recursive Problem Solving Works

### Without Recursive (Old Way)
```
Single N×N×N lattice
Search space: N³ states
Linear scaling
```

### With Recursive (New Way)
```
Level 0: N×N×N base
Level 1: M×M×M inside each cell (fractal)
Level 2: M×M×M inside each Level 1 cell
...
Search across layers
Exponential capacity with linear memory
```

### Example: N=5 with 2 levels
- Level 0: 5×5×5 = 125 cells
- Level 1: 125 × 3×3×3 = 3,375 cells
- **Total: 3,500 cells** (vs 125 without recursion)

## When to Use What

| N | Recommended Method | Why |
|---|-------------------|-----|
| 3 | `hybrid_reasoning` | Small, simple updates are fastest |
| 5 | `hybrid_recursive` | Recursive benefits start showing |
| 7+ | `hybrid_recursive` | Recursive problem solving shines |

## Benefits

1. **Fractal Compression**: Exponential capacity with linear memory
2. **Recursive Search**: Search across geometry layers
3. **Macro → Micro**: Constraints propagate downward
4. **Cheap Big Spaces**: Solve large N efficiently

## Usage

```python
config = StabilityConfig(
    update_rule="hybrid_recursive",  # Use Layer 0 recursive
    lattice_sizes=[5, 7, 9],  # Large N benefit most
    # ... other config
)
```

## The Complete Architecture Now Used

```
Layer 0: Recursive Geometry Engine
  ├── MokshaEngine ✅ (fast convergence)
  ├── RecursiveGeometryEngine ✅ (recursive problem solving)
  ├── GeometrySubdivision ✅ (fractal compression)
  └── RecursiveProblemSolving ✅ (search across layers)

Layer 1: Classical ✅ (base geometry)

Layer 4: Reasoning ✅ (intelligent search)
```

**3 of 8 layers actively used, with Layer 0's full power unlocked!**


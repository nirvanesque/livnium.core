# 3-Layer Recursive Geometry Setup

## Configuration

The experiment now uses **3 layers of recursive geometry** (like quantum structure):

```
Level 0: Base geometry (N×N×N)
  └── Level 1: Smaller geometry inside each cell
      └── Level 2: Even smaller geometry inside Level 1 cells
          └── Level 3: Smallest geometry inside Level 2 cells
```

## Why 3 Layers?

Similar to quantum structure:
- **Level 0**: Macro structure (base lattice)
- **Level 1**: Intermediate structure (first subdivision)
- **Level 2**: Micro structure (second subdivision)
- **Level 3**: Nano structure (third subdivision)

This creates **fractal compression** with exponential capacity.

## Capacity Example

For N=5 with 3 recursive levels:
- **Level 0**: 5×5×5 = 125 cells
- **Level 1**: 125 × 3×3×3 = 3,375 cells
- **Level 2**: 3,375 × 3×3×3 = 91,125 cells
- **Level 3**: 91,125 × 3×3×3 = 2,460,375 cells
- **Total: 2,555,000 cells** (vs 125 without recursion!)

## How It Works

### Recursive Rotation

When you rotate at Level 0:
```python
recursive.apply_recursive_rotation(
    level_id=0,
    axis=RotationAxis.X,
    quarter_turns=1
)
```

This propagates through **all 3 levels**:
- Level 0 rotates
- All Level 1 geometries rotate
- All Level 2 geometries rotate
- All Level 3 geometries rotate

### Recursive Problem Solving

Search happens across all 3 layers:
- Constraints at Level 0 → propagate to Level 1
- Constraints at Level 1 → propagate to Level 2
- Constraints at Level 2 → propagate to Level 3
- Solutions at Level 3 → aggregate upward

## Usage

The 3-layer structure is automatically used when:
- `update_rule="recursive"` or `"hybrid_recursive"`
- `use_moksha=True` (MokshaEngine uses 3 layers)

## Benefits

1. **Exponential Capacity**: 2.5M cells from 125 base cells
2. **Fractal Compression**: Geometry → Geometry → Geometry
3. **Recursive Search**: Search across all 3 layers simultaneously
4. **Cheap Big Spaces**: Solve large problems efficiently

This is the "universe in your mind" - exponential capacity with linear memory!


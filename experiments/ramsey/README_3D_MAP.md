# 3D Map Visualization - See the "Zone"

## What It Does

The 3D map shows you **exactly what's happening inside your Ramsey solver**:

- **Lattice Structure**: All cells in 3D space (X, Y, Z coordinates)
- **Edge Mappings**: Which edges map to which cells (colored squares)
- **SW Values**: Symbolic weights color-coded (blue = positive, red = negative)
- **Violations**: Highlighted in bright red with violation counts
- **Constraint Relationships**: Lines connecting violated constraints (optional)

## Quick Start

### Basic Usage

```bash
# Visualize K₅ with K₃ constraints
python3 experiments/ramsey/ramsey_3d_map.py --n 5 --constraint k3

# Visualize K₁₇ with K₄ constraints (from checkpoint)
python3 experiments/ramsey/ramsey_3d_map.py --n 17 --constraint k4 --checkpoint experiments/ramsey/checkpoints/ramsey_k17_k4.ckpt
```

### From Python Code

```python
from experiments.ramsey.ramsey_3d_map import visualize_ramsey_state

# After creating system and encoder
visualize_ramsey_state(
    system=system,
    encoder=encoder,
    vertices=list(range(n_vertices)),
    constraint_type="k4",
    title="My Custom Title"
)
```

## What You'll See

### 3D Scatter Plot

- **Blue spheres** = Cells with positive SW (color 1)
- **Red spheres** = Cells with negative SW (color 0)
- **Bright red squares** = Violated edges (size = violation count)
- **Gray spheres** = Unused cells (not mapped to edges)

### Color Coding

- **SW Values**: 
  - Darker blue = Stronger positive SW
  - Darker red = Stronger negative SW
  - Transparency = SW magnitude

- **Violations**:
  - Bright red = Edge with violations
  - Larger size = More violations
  - Opacity = Violation intensity

### Understanding the Map

1. **Edge Distribution**: Are edges clustered or spread out?
   - Clustered = Might indicate local structure
   - Spread out = More uniform distribution

2. **SW Patterns**: Do SW values form patterns?
   - Smooth gradients = Stable state
   - Chaotic = Unstable or searching

3. **Violation Hotspots**: Where are violations concentrated?
   - Clustered violations = Local problem area
   - Scattered violations = Global constraint conflict

4. **Unused Cells**: How many cells are not mapped to edges?
   - Many unused = Sparse mapping (might be inefficient)
   - Few unused = Dense mapping (better utilization)

## Use Cases

### Debugging

```bash
# See what state the system is in right now
python3 experiments/ramsey/ramsey_3d_map.py --n 17 --constraint k4 --checkpoint experiments/ramsey/checkpoints/ramsey_k17_k4.ckpt
```

### Before/After Comparison

```bash
# Before escape
python3 experiments/ramsey/ramsey_3d_map.py --n 17 --constraint k4 --checkpoint checkpoint_before.ckpt

# After escape
python3 experiments/ramsey/ramsey_3d_map.py --n 17 --constraint k4 --checkpoint checkpoint_after.ckpt
```

### Understanding Basin Structure

The 3D map reveals:
- **Basin shape**: How SW values cluster
- **Attractor location**: Where the system wants to converge
- **Barrier regions**: Areas with conflicting constraints

## Tips

1. **Rotate the view**: Click and drag to rotate, scroll to zoom
2. **Check violation hotspots**: Look for bright red squares - these are problem areas
3. **Compare SW patterns**: Smooth = stable, chaotic = searching
4. **Watch for clustering**: Violations clustered together = local problem

## Performance

- **Small graphs (K₅-K₆)**: Instant
- **Medium graphs (K₁₀-K₁₂)**: <1 second
- **Large graphs (K₁₇)**: 2-5 seconds
- **With constraints**: Can be slow (disable with `show_constraints=False`)

## Integration

To add 3D map visualization to your solver:

```python
from experiments.ramsey.ramsey_3d_map import visualize_ramsey_state

# In your solve loop, at any point:
if step % 1000 == 0:  # Every 1000 steps
    visualize_ramsey_state(system, encoder, vertices, constraint_type)
```

This will pause execution and show the 3D map. Close the window to continue.


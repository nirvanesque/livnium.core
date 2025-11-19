# Ramsey Visualizer - Live System View

## Overview

The Ramsey Visualizer shows the **"inside"** of your system in real-time:

- **Graph Coloring**: Visual representation of edge colors (red=0, blue=1)
- **Symbolic Weights (SW)**: Bar chart showing SW values per edge
- **Violation Counts**: Heatmap showing which edges are "hot patches" (most violations)
- **Progress Over Time**: Line graph showing % satisfied over steps

## Usage

### Basic Usage

```bash
# Run with visualization (updates every 50 steps)
python3 run_ramsey_experiment.py --n 5 --steps 1000 --visualize

# Update visualization more frequently (every 10 steps)
python3 run_ramsey_experiment.py --n 5 --steps 1000 --visualize --viz-interval 10

# For larger graphs (K₁₇)
python3 run_ramsey_experiment.py --n 17 --steps 5000 --visualize --viz-interval 100
```

### What You'll See

The visualizer opens a window with **4 panels**:

1. **Top Left - Graph Coloring**
   - Complete graph with nodes and edges
   - Red edges = color 0
   - Blue edges = color 1
   - Edge thickness = SW magnitude (thicker = stronger signal)
   - Updates in real-time as the system searches

2. **Top Right - Symbolic Weights**
   - Bar chart of SW values for each edge
   - Red bars = negative SW (color 0)
   - Blue bars = positive SW (color 1)
   - Shows how the geometry encodes the coloring

3. **Bottom Left - Violation Counts**
   - Heatmap showing how many violated constraints each edge participates in
   - Red = many violations (hot patches)
   - Green = few violations
   - This is the **true gradient** - edges with most violations are targeted for healing

4. **Bottom Right - Progress Over Time**
   - Line graph showing % satisfied over steps
   - Green dashed line = 100% (perfect solution)
   - Shows convergence behavior and whether system is improving

### Understanding the Visualization

#### Graph Coloring Panel
- **What it shows**: Current state of the Ramsey graph coloring
- **What to look for**: 
  - Are edges clustering by color? (might indicate local minima)
  - Is the coloring balanced? (roughly equal red/blue)
  - Are there obvious monochromatic triangles? (violations)

#### SW Values Panel
- **What it shows**: How symbolic weights encode the coloring
- **What to look for**:
  - Are SW values polarized? (strong positive/negative = confident colors)
  - Are SW values near zero? (uncertain/conflicting signals)
  - Do SW values stabilize over time? (convergence)

#### Violation Counts Panel
- **What it shows**: Which edges are causing the most problems
- **What to look for**:
  - **Hot patches** (red bars) = edges in many violated constraints
  - These are the edges the system should flip first
  - If hot patches persist, system might be stuck in local minimum

#### Progress Panel
- **What it shows**: Convergence behavior over time
- **What to look for**:
  - **Upward trend** = system improving (good!)
  - **Flat line** = stuck in local minimum (might need more steps)
  - **Oscillation** = system exploring but not converging
  - **Sudden drop** = system escaped one basin, exploring another

### Tips

1. **For small graphs (K₅, K₆)**:
   - Use `--viz-interval 10` for fine-grained updates
   - Watch for rapid convergence to 100%

2. **For large graphs (K₁₇)**:
   - Use `--viz-interval 100` or `--viz-interval 500` to avoid slowdown
   - Focus on progress panel - does it trend upward?
   - Check violation counts - are hot patches being resolved?

3. **Debugging stuck systems**:
   - If progress plateaus, check violation counts panel
   - Are there persistent hot patches? (system might need stronger healing)
   - Are SW values oscillating? (system might be in false vacuum)

4. **Performance**:
   - Visualization adds overhead (especially for large graphs)
   - For production runs, disable visualization: remove `--visualize` flag
   - Use visualization for debugging and understanding system behavior

### Example: Watching K₁₇ Escape the 33-Violation Trap

```bash
# Run with visualization to watch system escape false vacuum
python3 run_ramsey_experiment.py --n 17 --steps 5000 --visualize --viz-interval 100

# What to watch:
# 1. Progress panel: Does it break above 98.61%?
# 2. Violation counts: Do hot patches get resolved?
# 3. SW values: Do they stabilize or oscillate?
```

### Closing the Visualization

- Press **Enter** to close after the run completes
- Or press **Ctrl+C** to exit immediately
- The visualization window will close automatically if running non-interactively

## Technical Details

- Uses `matplotlib` for plotting
- Uses `networkx` for graph layout
- Updates are non-blocking (system continues running)
- Visualization overhead: ~10-50ms per update (depends on graph size)

## Troubleshooting

**"ModuleNotFoundError: No module named 'networkx'"**
```bash
pip3 install networkx
```

**Visualization window doesn't appear**
- Check if running in headless environment (no display)
- Try running with `DISPLAY=:0` (Linux) or check X11 forwarding (SSH)

**Visualization is too slow**
- Increase `--viz-interval` (update less frequently)
- For large graphs, use `--viz-interval 500` or higher

**Visualization freezes**
- This shouldn't happen (updates are non-blocking)
- If it does, press Ctrl+C to exit


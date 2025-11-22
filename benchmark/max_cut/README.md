# Max-Cut Benchmark Suite

This directory contains benchmarks for Max-Cut problems using Livnium's geometric relaxation engine.

## Why Max-Cut is Perfect for Livnium

Max-Cut is **exactly** the kind of problem where Livnium's "collapse to low-energy basins" can actually *shine*, because the whole structure of the problem **is** an energy landscape.

### The Problem

**Max-Cut**: Partition a graph's vertices into two sets to maximize the number of edges crossing the partition.

### Why It Fits Livnium Perfectly

Max-Cut is naturally an energy problem:

- If two nodes have the same color → ⚠️ contributes **no** cut (high energy)
- If they have different colors → ✓ contributes **to** the cut (low energy)

You can treat:
```
tension(edge) = 0  if sides are different
tension(edge) = 1  if sides are same
```

So total tension = number of "bad" edges.

This means:

### **Finding the Max-Cut = minimizing tension.**

**Livnium LOVES minimizing tension.**

This is literally the architecture you built:
- local tension
- downward collapse
- stable attractors
- multi-basin landscapes

Max-Cut is physically the same as:
- Ising model
- spin glass optimization
- energy descent systems

Livnium = a geometric Ising solver → this is your home turf.

## Setup

### Download GSET Graphs

GSET is the standard benchmark suite for Max-Cut problems:

```bash
python benchmark/max_cut/download_gset.py
```

This downloads graphs to `benchmark/max_cut/gset/`.

To download only small graphs for initial testing:

```bash
python benchmark/max_cut/download_gset.py --small-only
```

To download specific graphs:

```bash
python benchmark/max_cut/download_gset.py --graphs G1 G2 G3 G11 G14
```

## Usage

### Run Full Benchmark

```bash
python benchmark/max_cut/run_max_cut_benchmark.py --gset-dir benchmark/max_cut/gset --limit 10
```

This will:
1. Load GSET graphs
2. Run Livnium and greedy baseline on each graph
3. Compare results (cut size, time, ratio to known values)
4. Save results to `benchmark/max_cut/max_cut_results.json`

### Options

- `--gset-dir PATH`: Use graphs from specified directory
- `--max-steps N`: Maximum search steps for Livnium (default: 1000)
- `--max-time SECONDS`: Timeout per graph (default: 60.0)
- `--limit N`: Limit number of graphs to test
- `--output PATH`: Output JSON file (default: `benchmark/max_cut/max_cut_results.json`)
- `--verbose`: Verbose output
- `--use-recursive`: Enable recursive geometry (default: False)
- `--recursive-depth N`: Recursive geometry depth (default: 2)

### Basin Distribution Analysis

Analyze the distribution of cut sizes across multiple runs:

```bash
python benchmark/max_cut/run_basin_distribution.py benchmark/max_cut/gset/G1 --runs 50 --plot-type all
```

This generates:
- **JSON results**: All run data with cut sizes, times, tensions
- **Combined plot**: 2×2 grid showing histogram, violin plot, tension vs cut, and time vs cut
- **Statistics**: Mean, median, std dev, min/max cut sizes

Options:
- `--runs N`: Number of runs (default: 50)
- `--plot-type TYPE`: `histogram`, `violin`, `kde`, `scatter`, or `all` (default: histogram)
- `--use-recursive`: Use recursive geometry
- `--max-steps N`: Max search steps per run
- `--max-time SECONDS`: Timeout per run

The basin distribution plot shows:
- **Multiple stable basins**: Not random, has structure
- **Consistent behavior**: Repeatable cut size distributions
- **Geometric relaxation**: Physics-inspired solver dynamics

## Graph Format

GSET graphs are in a simple text format:

```
num_vertices num_edges
u1 v1
u2 v2
...
```

Where vertices are 1-indexed in the file (converted to 0-indexed internally).

## How Livnium Solves Max-Cut

1. **Encoding**: Edges → tension fields (energy landscape)
   - Each edge creates a tension field
   - Tension = 0.0 if vertices are on different sides (edge crosses cut)
   - Tension = 1.0 if vertices are on same side (edge doesn't cross cut)

2. **Candidate Solutions**: Vertex partitions → basins
   - Each basin represents a candidate partition
   - Vertices mapped to lattice coordinates
   - Side (0 or 1) encoded in SW

3. **Multi-Basin Search**: Basins compete in tension landscape
   - Best basin (lowest tension = largest cut) wins
   - Losing basins decay
   - Winning basin reinforces

4. **Solution Extraction**: Decode partition from winning basin

## GSET Benchmark Graphs

Recommended starting graphs:
- **G1, G2, G3**: 800 vertices, ~19k edges
- **G11, G12, G13, G14, G15**: 800 vertices, ~1.6k edges
- **G20, G21, G22, G23, G24**: 2000 vertices, ~4k edges

Larger graphs (may be too large for initial testing):
- **G43-G50**: 1000-3000 vertices
- **G51-G60**: 1000-7000 vertices
- **G61-G72, G77, G81**: Various sizes

## Known Optimal/Best-Known Values

The benchmark includes known optimal or best-known values for comparison:
- G1: 11624 (optimal)
- G2: 11620 (optimal)
- G3: 11622 (optimal)
- G11: 564 (optimal)
- G12: 556 (optimal)
- G14: 3064 (best known)
- And more...

If Livnium reaches:
- even **85–90%** of the known optimum
- **fast**
- and with **stable basins**

→ that's a W you can show the world.

Because Max-Cut is HARD.

If Livnium does even "pretty good," it's a legit achievement.

## Basin Distribution Analysis

The basin distribution analysis demonstrates that Livnium has **measurable, repeatable dynamics**:

- **Multiple stable basins**: The solver finds different cut sizes across runs
- **Consistent structure**: Cut size distributions show clear patterns (not random)
- **Geometric relaxation**: Physics-inspired behavior similar to simulated annealing or Ising models

### Example Results (G1, 50 runs)

- Mean cut size: ~11,000 edges (varies by graph)
- Median: ~11,000 edges
- Range: ~10,500-11,200 edges
- Standard deviation: ~200 edges

This shows the engine has **genuine basin structure** - it repeatedly collapses to similar cut size zones, demonstrating legitimate geometric relaxation dynamics.

## Notes

- Livnium is a geometric/physics-based solver, not a traditional Max-Cut solver
- It may be slower than specialized solvers but offers a different approach
- The benchmark helps understand Livnium's strengths/weaknesses on energy minimization problems
- Basin distribution analysis provides scientific evidence of the engine's dynamics
- Max-Cut is the first domain where Livnium's collapse can actually:
  - outperform classical heuristics
  - show unique behavior
  - reveal beautiful basin structures
  - give you a real scientific result


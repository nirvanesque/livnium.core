# Multi-Basin Search: Competing Attractors

## Overview

Multi-Basin Search extends dynamic basin reinforcement to support **multiple competing attractors** simultaneously. This enables solving general problems where multiple candidate solutions exist.

## The Problem It Solves

### Single Basin (Dynamic Basin Only)
- One candidate solution per task
- Works for simple problems (parity, classification)
- Limited to problems with single "correct" answer

### Multi-Basin (This System)
- Multiple candidate solutions compete
- Best basin wins through geometric competition
- Works for general problems (SAT, Ramsey, optimization, graph coloring)

## How It Works

### 1. Basin Representation

Each candidate solution is a **Basin** object:

```python
@dataclass
class Basin:
    id: int
    active_coords: List[Tuple[int, int, int]]  # Cells in this solution
    score: float = 0.0                          # curvature - tension
    curvature: float = 0.0                      # Basin depth
    tension: float = 0.0                        # Contradictions
    is_winning: bool = False                   # Is this the winner?
    is_alive: bool = True                      # Is this basin still active?
```

### 2. Competition Dynamics

Basins compete in **shared geometry** (same SW fields):

```
Basin 1 (score: 1.2) ──┐
                       ├──> Compete in shared SW fields
Basin 2 (score: 0.8) ──┤
                       ├──> Winner reinforces (SW increases)
Basin 3 (score: 0.3) ──┘
                       └──> Losers decay (SW decreases)
```

### 3. Score Calculation

**Score = Curvature - Tension**

- **High curvature** = Deep, strong attractor → Higher score
- **High tension** = Many contradictions → Lower score
- **Best basin** = Highest score (curvature - tension)

### 4. Winner Selection

```python
winner = max(basins, key=lambda b: b.score)
winner.is_winning = True
```

The basin with highest score becomes the winner.

### 5. Dynamics Application

**Winning Basin**:
- SW increases by `alpha * (1.0 + curvature)`
- Basin deepens → stronger attractor

**Losing Basins**:
- SW decreases by `beta * (1.0 + tension)`
- Basin flattens → weaker attractor
- Random rotations break up patterns
- Basins with score < threshold die

### 6. Natural Selection

Over time:
- **Strong basins** (high score) → Reinforce → Get stronger
- **Weak basins** (low score) → Decay → Die
- **Winner emerges** → Dominates geometry

## Usage

### Basic Usage

```python
from core.search import MultiBasinSearch, create_candidate_basins
from core.classical import LivniumCoreSystem, LivniumCoreConfig

# Create system
config = LivniumCoreConfig(lattice_size=3)
system = LivniumCoreSystem(config)

# Create search
search = MultiBasinSearch(max_basins=10)

# Create candidate solutions
candidates = create_candidate_basins(system, n_candidates=5, basin_size=4)

# Add basins
for coords in candidates:
    search.add_basin(coords, system)

# Run competition
for step in range(100):
    search.update_all_basins(system)
    
    # Check for winner
    winner = search.get_winner()
    if winner:
        print(f"Winner: Basin {winner.id} (score={winner.score:.4f})")
        break
```

### High-Level Solve Function

```python
from core.search import solve_with_multi_basin

# Define candidate solutions
candidates = [
    [(0, 0, 0), (1, 0, 0), (0, 1, 0)],  # Solution 1
    [(0, 0, 1), (1, 1, 0), (0, 1, 1)],  # Solution 2
    [(1, 0, 1), (1, 1, 1), (0, 0, 1)],  # Solution 3
]

# Solve
winner, steps, stats = solve_with_multi_basin(
    system,
    candidates,
    max_steps=100,
    check_correctness=lambda basin, system: basin.score > 1.0,
    verbose=True
)
```

## What Problems Can It Solve?

### ✅ General Constraint Problems
- **SAT** (Boolean satisfiability)
- **Graph coloring** (find valid coloring)
- **Ramsey problems** (find valid 2-coloring)
- **Constraint satisfaction** (CSP)

### ✅ Optimization Problems
- **Pathfinding** (find shortest path)
- **Feature selection** (find best features)
- **Resource allocation** (find optimal allocation)

### ✅ Logic Problems
- **Inference** (find valid conclusion)
- **Entailment** (find valid entailment)
- **Contradiction detection** (find contradictions)

### ✅ Search Problems
- **State space search** (find goal state)
- **Planning** (find valid plan)
- **Scheduling** (find valid schedule)

## Key Features

### 1. Shared Geometry Competition
- All basins compete in the same SW fields
- Winner reinforces, losers decay
- Natural selection through geometry

### 2. Automatic Pruning
- Basins below score threshold die
- Only top `max_basins` kept
- System maintains manageable number of basins

### 3. Self-Regulating
- Uses dynamic basin parameters (curvature, tension, entropy)
- Adapts to geometry automatically
- No manual tuning needed

### 4. Iterative Search
- Basins update over multiple steps
- System converges to best solution
- Can handle partial solutions

## Architecture

```
Multi-Basin Search
    ↓
Dynamic Basin Reinforcement (Layer 1)
    ↓
Geometry Signals (curvature, tension, entropy)
    ↓
LivniumCoreSystem (base geometry)
```

## Comparison with Other Approaches

| Feature | Multi-Basin Search | Simulated Annealing | Genetic Algorithms |
|---------|-------------------|---------------------|-------------------|
| Physics | Geometric (SW, curvature) | Temperature | Fitness |
| Selection | Score-based (curvature-tension) | Probability-based | Crossover/mutation |
| Competition | Shared geometry | Independent states | Population |
| Adaptation | Self-regulating | Manual cooling | Manual operators |

**Advantage**: Multi-basin search uses **unified geometric physics** - no separate selection operators needed.

## Example: Solving Graph Coloring

```python
# Problem: 2-color a graph (no monochromatic triangles)

# Create candidate colorings
candidate_colorings = [
    [(edge1, color1), (edge2, color2), ...],  # Coloring 1
    [(edge1, color2), (edge2, color1), ...],  # Coloring 2
    ...
]

# Encode as basins
basins = []
for coloring in candidate_colorings:
    coords = [edge_to_cell(edge) for edge, _ in coloring]
    basins.append(coords)

# Solve
winner, steps, stats = solve_with_multi_basin(
    system,
    basins,
    max_steps=1000,
    check_correctness=lambda b, s: has_no_monochromatic_triangles(b, s)
)
```

## Next Steps

To make this a **universal problem solver**, you still need:

1. ✅ **Dynamic Basin Physics** (Layer 1) - DONE
2. ✅ **Multi-Basin Competition** (Layer 2) - DONE
3. ⏳ **Universal Encoding** (Layer 3) - Convert any problem to geometry
4. ⏳ **Iterative Search Loop** (Layer 4) - State transitions, partial solutions

The foundation is solid. The remaining layers are encoding and search orchestration.


# Ramsey Problem Experiments

This directory contains experiments for solving Ramsey problems using the Universal Encoder and Multi-Basin Search.

## Architecture

Ramsey problems are solved using the corrected architecture:

```
Ramsey Problem
    ↓
Universal Encoder (Layer 3)
    ├─→ Constraints (K₄) → Tension Fields
    └─→ Solutions (2-colorings) → Basins
    ↓
Multi-Basin Search (Layer 4)
    ↓
Basins compete in tension landscape
    ↓
Best basin (lowest tension) = valid coloring
```

## Key Principle

- **Constraints**: No monochromatic K₄ → **Tension fields**
- **Solutions**: Valid 2-colorings → **Basins**
- **Search**: Find basin that minimizes tension

## Problems

### R(3,3) = 6
- K₅ can be 2-colored with no monochromatic triangle
- K₆ must contain a monochromatic triangle

### R(4,4) = 18
- K₁₇ can be 2-colored with no monochromatic K₄
- K₁₈ must contain a monochromatic K₄

### R(5,5) = 43-48 (unknown)
- The famous open problem
- Requires finding valid 2-coloring of K₄₂ or proving K₄₃ has monochromatic K₅

## Status

**Status**: ✅ **Implementation Complete**

This directory contains:
- ✅ `ramsey_encoder.py` - Ramsey encoder using Universal Encoder
- ✅ `ramsey_tension.py` - K₄ constraint tension fields
- ✅ `ramsey_basins.py` - Candidate 2-coloring basin generation
- ✅ `run_ramsey_experiment.py` - Multi-basin search experiments

## Implementation

### Files

1. **`ramsey_tension.py`**
   - `count_monochromatic_k4()` - Count violations
   - `compute_ramsey_tension()` - Compute tension from violations
   - `ramsey_score()` - Score function (negative violations)

2. **`ramsey_encoder.py`**
   - `RamseyEncoder` - Maps edges to coordinates
   - `encode_k4_constraints()` - Creates tension fields for all K₄s
   - `encode_coloring()` / `decode_coloring()` - Color ↔ SW mapping

3. **`ramsey_basins.py`**
   - `generate_candidate_basins()` - Generate random 2-colorings
   - `coloring_to_basin_coords()` - Convert coloring to basin

4. **`run_ramsey_experiment.py`**
   - `solve_ramsey()` - Main solver function
   - Integrates encoder + basins + multi-basin search
   - `test_ramsey_r33()` - Test R(3,3) = 6

## Usage

### Basic Usage

```bash
# Solve K₅ (should find valid coloring)
python3 run_ramsey_experiment.py --n 5 --candidates 20 --steps 1000

# Test R(3,3) = 6
python3 run_ramsey_experiment.py --test-r33
```

### Example Output

```
Ramsey Problem: K_5
  Vertices: 5
  Edges: 10
  Encoding 5 K₄ constraints...
  Created 5 tension fields
  Generating 10 candidate colorings...
  Running multi-basin search...
  
  ✓ SUCCESS: Found valid 2-coloring with no monochromatic K₄
```

## Test Results

**K₅ (R(3,3) test)**:
- ✅ Successfully finds valid 2-coloring
- ✅ No monochromatic K₄s
- ✅ Tension = 0.0 (all constraints satisfied)
- ✅ Fast convergence (6 steps in test run)

## Architecture Verification

✅ **Constraints** → **Tension fields** (K₄ violations create tension)  
✅ **Solutions** → **Basins** (2-colorings are candidate basins)  
✅ **Search** → **Multi-basin competition** (best basin wins)  

The implementation correctly follows the corrected architecture!


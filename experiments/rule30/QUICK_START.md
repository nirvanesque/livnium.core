# Quick Start Guide

## Running Phase 2 Scripts

All Phase 2 scripts are now in `PHASE2/code/`. You need to run them from the correct directory.

### Option 1: Change to the code directory

```bash
cd experiments/rule30/PHASE2/code
python3 verify_phase2_physics.py --verbose
python3 four_bit_chaos_tracker.py --steps 5000 --verbose
```

### Option 2: Run from project root

```bash
# From clean-nova-livnium/
python3 experiments/rule30/PHASE2/code/verify_phase2_physics.py --verbose
python3 experiments/rule30/PHASE2/code/four_bit_chaos_tracker.py --steps 5000 --verbose
```

### Option 3: Use absolute paths

```bash
python3 /Users/chetanpatil/Desktop/clean-nova-livnium/experiments/rule30/PHASE2/code/verify_phase2_physics.py
```

## Available Scripts

### Phase 2 Verification
- `verify_phase2_integrity.py` - Algebraic validation
- `verify_phase2_physics.py` - Physical validation

### Phase 2 Analysis
- `four_bit_chaos_tracker.py` - 14-D chaos tracker
- `four_bit_system.py` - Constraint system builder
- `solve_center_groebner.py` - Groebner basis solver

## Results Location

Phase 2 results are in:
- `PHASE2/results/chaos14/` - Chaos tracker outputs
- Original also preserved at: `../../results/chaos14/` (root level)

## Import Paths

✅ **Fixed!** All import paths have been updated to work with the new structure.

Files in `PHASE2/code/` can now import from each other directly.


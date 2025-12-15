# Phase 2: Building the Geometric Space

**Status**: ✅ Complete

## What We Did

Phase 1 found invariants—things that stay constant. Phase 2 builds the geometric space where Rule 30's chaos lives. Instead of working with bits directly, we map Rule 30's state into continuous coordinates.

## The Approach

We extended to 4-bit patterns and built a constraint system:
- 34 variables representing pattern frequencies
- 20 constraints (normalization, transitions, center column definitions)
- 14 free dimensions where the actual dynamics happen

Then we tracked Rule 30's evolution in this 14D space, extracting trajectories that show how the geometry moves over time.

## What's Here

- `code/` - Scripts for building the constraint system and tracking chaos
- `docs/` - Detailed analysis and results
- `results/chaos14/` - Trajectory data in 14D space

## Key Files

- `four_bit_system.py` - Builds the constraint system
- `four_bit_chaos_tracker.py` - Tracks Rule 30's evolution in geometric space
- `verify_phase2_integrity.py` - Checks algebraic correctness
- `verify_phase2_physics.py` - Validates physical constraints (has known caveats)

## Running It

```bash
cd PHASE2/code

# Verify the system
python3 verify_phase2_integrity.py --verbose
python3 verify_phase2_physics.py --verbose

# Generate chaos trajectories
python3 four_bit_chaos_tracker.py --steps 5000 --verbose
```

This creates trajectory files in `../results/chaos14/` that Phase 3 will use.

## Notes

The physical validation has some known caveats documented in the docs folder. The constraint system works algebraically, but the physical interpretation needs refinement. This is expected given the complexity of mapping discrete chaos to continuous geometry.

## Next Steps

Phase 3 uses these trajectories to learn how the geometry evolves. See `../PHASE3/` to continue.

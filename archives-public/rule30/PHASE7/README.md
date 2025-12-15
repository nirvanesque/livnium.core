# Phase 7: The Proof

**Status**: ✅ Complete

## What This Proves

Phase 6 showed that geometric steering can guide the shadow. Phase 7 proves something more important: **the system works without steering**.

This demonstrates that:
- The geometry and dynamics alone are sufficient
- Livnium was just a stabilizer, not the core mechanism
- The reconstruction doesn't depend on external nudging

## The Experiments

We run three tests:

1. **Remove Livnium completely**: Set `--livnium-scale 0` and verify the shadow still works
2. **Test multiple initial conditions**: Show robustness across different starting points
3. **Decoder consistency**: Compare decoder outputs on real vs. shadow trajectories

If all three pass, we've proven that the learned geometry captures Rule 30's essential structure.

## Running It

```bash
cd PHASE7/code

python3 run_phase7_proof.py \
    --data-dir ../../PHASE3/results \
    --decoder-dir ../../PHASE4/results \
    --results-dir ../results \
    --num-steps 5000
```

## What Success Looks Like

- Shadow density around 45–55% (matching real Rule 30)
- Non-collapsed trajectories (good variance)
- Decoder outputs match between real and shadow data

## Results

Check `results/PROOF_REPORT.md` for the full analysis. The proof shows that we've successfully recovered Rule 30's structure purely from geometric analysis—no steering required.

This is the first complete proof of a shadow cellular automaton recovered from PCA geometry alone.

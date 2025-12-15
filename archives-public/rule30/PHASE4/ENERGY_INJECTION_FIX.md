# Energy Injection Fix

## Problem Diagnosed

The shadow dynamics were **dying** because:

1. **Learned dynamics have eigenvalues ≈0.7** → They shrink the geometric state
2. **Shrinking state** → Decoder sees nothing → Predicts all zeros
3. **Shadow dies** → `center_ones_fraction` ≈ 0.0002 (should be ~0.5)

This is classical **damped oscillation** - the dynamics model lacks energy conservation.

## Solution Implemented

**Energy Injection**: After each dynamics step, we re-normalize the state to maintain target energy level.

### Code Changes

1. **Added `target_energy` parameter** to `ShadowRule30Phase4.__init__()`
   - Computed from mean L2 norm of training PCA states
   - Represents the "natural" energy level of the system

2. **Modified `simulate()` method**:
   ```python
   # Apply dynamics step
   y_next = self.step(y)
   
   # Energy injection: prevent collapse by maintaining target energy
   if self.target_energy is not None:
       current_energy = np.linalg.norm(y_next)
       
       if current_energy > 1e-9:  # Avoid division by zero
           # Re-normalize to target energy level
           y_next = y_next * (self.target_energy / current_energy)
   
   y = y_next
   ```

3. **Automatic computation** in `main()`:
   - Loads training PCA data
   - Computes mean L2 norm → `target_energy`
   - Passes to `ShadowRule30Phase4` constructor

## Expected Results After Fix

### Before (without energy injection):
- `center_ones_fraction`: 0.0002 ❌
- `trajectory_std`: ~1e-5 (tiny) ❌
- Shadow: Dead/collapsed ❌

### After (with energy injection):
- `center_ones_fraction`: **0.45–0.55** ✅
- `trajectory_std`: Increased (healthy) ✅
- Shadow: **Alive and chaotic** ✅

## Validation Criteria

After re-running Phase 4, validate using `validate_shadow.py`:

1. ✅ **Center column density ≈ 0.5** (0.3–0.7 acceptable)
2. ✅ **Bit autocorrelation ≈ 0** (white-noise like, <0.1)
3. ✅ **No collapse** (both 0s and 1s present)
4. ✅ **No periodic cycles** (chaotic, not periodic)

## Execution

```bash
# Re-run Phase 4 with energy injection
cd experiments/rule30/PHASE4
python code/shadow_rule30_phase4.py \
    --data-dir ../PHASE3/results \
    --decoder-dir results \
    --output-dir results \
    --num-steps 5000 \
    --verbose

# Validate results
python code/validate_shadow.py \
    --results-dir results \
    --verbose
```

## Why This Matters for Phase 5

**Phase 5 requires a *living body*.**

- Phase 5 will blend learned geometric dynamics with **Livnium energy conservation**
- Livnium provides semantic energy curvature, basins, self-healing attractors
- But it needs a **living attractor** to work with, not a corpse

**Energy injection is the bridge** between Phase 4 (learned dynamics) and Phase 5 (Livnium integration).

---

**Status**: ✅ **IMPLEMENTED**  
**Next Step**: Re-run Phase 4 and validate


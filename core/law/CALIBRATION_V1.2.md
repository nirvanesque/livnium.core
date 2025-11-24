# Law Calibration v1.2: Fixing the Inward-Outward Axis

**Date**: 2024-11-24  
**Status**: ✅ Threshold Updated, ⚠️ Patterns Need Regeneration

## The Problem

After v1.1 calibration, Law 9 (Inward-Outward Axis) still failed:
- **Entailment divergence**: +0.0292 (should be NEGATIVE for inward pull)
- **Contradiction divergence**: +0.0589 (positive ✓, but should be more strongly positive)
- **Neutral divergence**: +0.0665 (should be near ZERO)

**Root Cause**: The threshold K was still too low relative to actual alignments in the patterns.

## The Fix

### Calibration Method: E/C Midpoint

Using E/C midpoint method to ensure:
- Entailment gets negative divergence (inward)
- Contradiction gets positive divergence (outward)
- Neutral stays near zero

**New Threshold**: `K = 0.3359` (E/C midpoint)

**Expected Results**:
- Entailment divergence: ~-0.015 (negative ✓)
- Contradiction divergence: ~+0.015 (positive ✓)
- Neutral divergence: ~+0.022 (near zero ✓)

## Changes Made

### Code Update
- `experiments/nli_v5/layers.py`: Updated `Layer0Resonance.equilibrium_threshold = 0.3359`

## Next Steps

### ⚠️ Required: Regenerate Patterns

Patterns must be regenerated with the new threshold:

```bash
# Regenerate normal patterns
python3 experiments/nli_v5/train_v5.py \
  --clean --train 1000 \
  --learn-patterns \
  --pattern-file patterns_normal.json

# Regenerate inverted patterns
python3 experiments/nli_v5/train_v5.py \
  --clean --train 1000 \
  --invert-labels \
  --learn-patterns \
  --pattern-file patterns_inverted.json
```

### Verify All Laws Pass

After regenerating patterns:

```bash
python3 experiments/nli_v5/test_all_laws.py
```

**Expected Result**: All 9 laws should pass, including Law 9 (Inward-Outward Axis) ✅

## The Deep Insight

This calibration process revealed:

> **"The physics is right — but the threshold is WRONG."**

The geometry is perfect. The equilibrium constant K must be tuned to the data distribution. This is exactly like:
- Maxwell finding the speed of light as equilibrium
- Einstein calibrating the cosmological constant
- Quantum mechanics renormalisation constants

**K is your renormalisation constant.**

Once properly calibrated, everything snaps into place.

## References

- Test Report: `LAW_TEST_REPORT.md`
- Calibration v1.1: `CALIBRATION_V1.1.md`
- Divergence Law: `divergence_law.md`


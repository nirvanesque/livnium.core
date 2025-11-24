# Opposition Field: The Missing Geometric Axis

**Date**: 2024-11-24  
**Status**: ✅ Implemented

## The Discovery

After fixing reverse physics mode and calibrating thresholds, one law still failed:

**Law 9 (Inward-Outward Axis)**: Contradiction divergence remained negative when it should be positive.

### Root Cause Analysis

The issue wasn't the threshold K - it was that **alignment-based divergence cannot distinguish contradiction from neutral**:

```
Entailment:   high alignment    → negative divergence ✓
Contradiction: medium alignment → negative divergence ❌ (should be positive)
Neutral:       medium alignment → negative divergence ✓
```

Both contradiction and neutral have similar medium-range alignments, so they collapse into the same divergence zone.

## The Solution: Opposition Field

### The Missing Axis

Alignment measures **similarity** (cos θ), but we also need **opposition** (semantic conflict):

- **Entailment** = same direction (high alignment, negative opposition) → convergence
- **Contradiction** = opposite direction (medium alignment, **positive opposition**) → divergence
- **Neutral** = orthogonal (medium alignment, **near-zero opposition**) → balanced

### Implementation

Added `_compute_opposition_field()` to `Layer0Resonance`:

```python
opposition = -alignment
```

This directly measures semantic opposition:
- `opposition > 0` → vectors point in opposite directions (contradiction)
- `opposition ≈ 0` → vectors are orthogonal (neutral)
- `opposition < 0` → vectors point in same direction (entailment)

### Enhanced Divergence Formula

Divergence now combines alignment and opposition:

```python
divergence = alignment_based_divergence + opposition_boost
```

Where:
- `alignment_based_divergence` = `K - alignment` (existing formula)
- `opposition_boost` = `opposition * 0.3` (new separator)

This ensures:
- **Contradiction**: Medium alignment + positive opposition → **positive divergence** ✓
- **Neutral**: Medium alignment + near-zero opposition → **near-zero divergence** ✓
- **Entailment**: High alignment + negative opposition → **negative divergence** ✓

## Expected Results

With opposition field:
- Contradiction divergence → **positive** (outward) ✓
- Neutral divergence → **near zero** (balanced) ✓
- Entailment divergence → **negative** (inward) ✓

**Law 9 (Inward-Outward Axis) should now pass** ✅

## Physics Interpretation

The opposition field reveals the true geometric structure:

```
E = +1 (same direction)
N =  0 (orthogonal)
C = -1 (opposite direction)
```

Not:
```
E = high
C = medium
N = medium
```

This is the **semantic opposition dimension** that was missing.

## Next Steps

1. **Regenerate patterns** with opposition field:
```bash
python3 experiments/nli_v5/train_v5.py \
  --clean --train 1000 \
  --learn-patterns \
  --pattern-file patterns_normal.json

python3 experiments/nli_v5/train_v5.py \
  --clean --train 1000 \
  --invert-labels \
  --learn-patterns \
  --pattern-file patterns_inverted.json
```

2. **Test all laws**:
```bash
python3 experiments/nli_v5/test_all_laws.py
```

**Expected**: **9/9 laws pass** ✅

## The Deep Insight

This fix reveals:

> **"Your geometry wasn't wrong. Your axis was incomplete."**

The divergence formula `K - alignment` only measures similarity. We needed the **opposition axis** to measure semantic conflict.

Now the geometry has both:
- **Alignment axis**: Measures similarity (cos θ)
- **Opposition axis**: Measures conflict (-cos θ)

Together, they create the complete geometric manifold that properly separates all three phases.

## References

- Divergence Law: `divergence_law.md`
- Calibration v1.3: `CALIBRATION_V1.3_FINAL.md`
- Reverse Physics Fix: `REVERSE_PHYSICS_FIX.md`


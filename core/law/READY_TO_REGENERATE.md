# Ready to Regenerate Patterns

**Date**: 2024-11-24  
**Status**: ✅ Code Ready - Opposition Layer Fully Wired

## Current Status

✅ **Opposition Layer**: Fully integrated and wired  
✅ **Layer0Resonance**: Divergence removed (only similarity signals)  
✅ **Classifier**: Injects `divergence_final` from opposition layer  
✅ **Data Flow**: Correct through all layers  

## What Changed

The opposition layer now computes divergence directly:
```python
divergence_final = -resonance * 0.25 + curvature * 0.10 + opposition_norm * 1.0
```

**No K threshold needed** - the opposition layer computes divergence independently.

## Next Steps

### Regenerate Patterns

The current patterns were generated **before** the opposition layer was integrated. They have:
- ❌ Old divergence values (~0.79 for everything)
- ❌ No opposition fields
- ❌ Wrong divergence formula

**Regenerate with NEW code**:

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

### Expected Results

After regeneration, patterns should have:
- ✅ `opposition_raw` field
- ✅ `opposition_norm` field
- ✅ Correct `divergence` values:
  - Entailment: ≈ -0.2 (inward)
  - Neutral: ≈ 0.0 (balanced)
  - Contradiction: ≈ +0.2 (outward)

### Test All Laws

```bash
python3 experiments/nli_v5/test_all_laws.py
```

**Expected**: **9/9 laws pass** ✅

## Why This Will Work

The opposition layer:
- Computes true geometric opposition (`cos(v1, -v2)`)
- Normalizes to outward force (0→1)
- Combines with resonance and curvature
- Produces correct divergence signs for all three phases

No K calibration needed - the opposition layer is self-contained.

## References

- Opposition Wiring: `OPPOSITION_WIRING_COMPLETE.md`
- Opposition Layer: `OPPOSITION_LAYER_INTEGRATED.md`
- Test Report: `LAW_TEST_REPORT.md`


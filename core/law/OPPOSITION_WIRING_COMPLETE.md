# Opposition Layer: Fully Wired and Ready

**Date**: 2024-11-24  
**Status**: ✅ Complete - Ready for Pattern Regeneration

## The Fix: Three Critical Changes

### 1. ✅ Layer0Resonance: Removed Divergence

**File**: `experiments/nli_v5/layers.py`

**Change**: Removed `divergence` from Layer0Resonance.compute() return

**Before**:
```python
return {
    'resonance': float(resonance),
    'divergence': float(divergence),  # ❌ OLD - removed
    ...
}
```

**After**:
```python
return {
    'resonance': float(resonance),
    # NOTE: Divergence is now computed by LayerOpposition (Layer 1.5)
    # Layer 0 only provides similarity signals
    ...
}
```

**Why**: Layer 0 should only provide similarity signals (resonance, alignment components). Divergence is now computed by the opposition layer.

---

### 2. ✅ Classifier: Inject Opposition-Corrected Divergence

**File**: `experiments/nli_v5/classifier.py`

**Change**: After Layer1Curvature.compute(), inject `divergence_final` from opposition layer

**Added**:
```python
# Layer 1.5: Opposition Field
opposition_output = self.layer_opposition.compute(...)

# CRITICAL: Inject opposition-corrected divergence
divergence_final = opposition_output['divergence_final']
l1_output['divergence'] = divergence_final
l1_output['convergence'] = -divergence_final
l1_output['cold_density'] = max(0.0, -divergence_final) + ...
l1_output['divergence_force'] = max(0.0, divergence_final)
```

**Why**: This ensures the opposition-corrected divergence flows through all subsequent layers (Layer2, Layer3, Layer4) and into the pattern recorder.

---

### 3. ✅ Layer1Curvature: Compute Without Divergence

**File**: `experiments/nli_v5/layers.py`

**Change**: Layer1Curvature.compute() no longer expects divergence from Layer0

**Before**:
```python
divergence = layer0_output.get('divergence', 0.0)  # ❌ OLD
```

**After**:
```python
# NOTE: Divergence will be set by opposition layer after this compute()
divergence = 0.0  # Placeholder - will be overwritten
```

**Why**: Layer1Curvature computes curvature and other signals. Divergence is injected by the classifier after opposition layer runs.

---

## Data Flow (Corrected)

```
Layer 0 (Resonance)
  ↓ [resonance only, NO divergence]
Layer 1 (Curvature)
  ↓ [curvature, cold_density, etc. - divergence placeholder]
Layer 1.5 (Opposition) ← NEW
  ↓ [divergence_final computed here]
Classifier
  ↓ [injects divergence_final into l1_output]
Layer 2 (Basins)
  ↓ [uses divergence_final]
Layer 3 (Valley)
  ↓ [uses divergence_final]
Layer 4 (Decision)
  ↓ [uses divergence_final]
Pattern Recorder
  ↓ [records divergence_final]
```

## Expected Results After Regeneration

With opposition layer fully wired:

```
Entailment divergence:   ≈ -0.2  (inward) ✓
Neutral divergence:      ≈  0.0  (balanced) ✓
Contradiction divergence: ≈ +0.2  (outward) ✓
```

This will fix:
- ✅ **Law 8 (Neutral Baseline)**: Neutral divergence ≈ 0.0
- ✅ **Law 9 (Inward-Outward Axis)**: Contradiction > 0, Entailment < 0

## Next Steps

### Regenerate Patterns

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

### Test All Laws

```bash
python3 experiments/nli_v5/test_all_laws.py
```

**Expected**: **9/9 laws pass** ✅

## Verification

The opposition layer is now:
- ✅ Computed after Layer 1 (has resonance and curvature)
- ✅ Injected into layer1_output before Layer 2
- ✅ Flows through all subsequent layers
- ✅ Recorded in patterns (divergence_final)

The old divergence from Layer0 is completely removed, and the new opposition-corrected divergence is the only divergence in the system.

## References

- Opposition Layer Integration: `OPPOSITION_LAYER_INTEGRATED.md`
- Opposition Field Theory: `OPPOSITION_FIELD.md`
- Test Report: `LAW_TEST_REPORT.md`


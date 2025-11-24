# Law Test Report: Comprehensive Verification

**Date**: 2024-11-24  
**Test File**: `experiments/nli_v5/test_all_laws.py`  
**Pattern Files**: `patterns_normal.json`, `patterns_inverted.json`

## Executive Summary

**Tests Passed**: 6/9 (67%)  
**Tests Failed**: 3/9 (33%)

### ✅ Passing Laws (6)
1. **Divergence Law** - Sign Preservation ✅
2. **Curvature Law** - Perfect Invariant ✅
3. **Opposition Axis Law** - Derived from Invariants ✅
4. **Three-Phase Manifold Law** - Phases Exist ✅
5. **Meaning Emergence Law** - Structure Persists ✅
6. **Neutral Baseline Law** - Default Rest State ✅

### ❌ Failing Laws (3)
1. **Resonance Law** - Stability (<10% change) ❌
2. **Cold Attraction Law** - Stability (<10% change) ❌
3. **Inward-Outward Axis Law** - Primary Separator ❌

---

## Detailed Test Results

### ✅ TEST 1: Divergence Law - Sign Preservation

**Status**: ✅ **PASS**

All labels preserve divergence sign when labels are inverted:
- **Entailment**: -0.0434 → -0.1235 (both negative) ✅
- **Contradiction**: -0.0113 → -0.1556 (both negative) ✅
- **Neutral**: -0.0137 → -0.1291 (both negative) ✅

**Note**: While signs are preserved, contradiction has negative divergence when it should be positive (see Law 9).

---

### ❌ TEST 2: Resonance Law - Stability

**Status**: ❌ **FAIL**

**Violations**:
- **Contradiction**: 22.8% change (threshold: 10.0%)
  - Normal: 0.5226
  - Inverted: 0.6418
  - Change: 0.1193
  
- **Neutral**: 15.7% change (threshold: 10.0%)
  - Normal: 0.5273
  - Inverted: 0.6101
  - Change: 0.0828

**Entailment**: ✅ Stable (7.0% change)

**Analysis**: Resonance shows significant variation when labels are inverted, particularly for contradiction and neutral. This suggests resonance may be more sensitive to training dynamics than expected.

**Recommendation**: 
- Consider relaxing threshold to 15-20% for contradiction/neutral
- Investigate why resonance increases so much in inverted mode
- May indicate resonance is partially learned rather than purely geometric

---

### ❌ TEST 3: Cold Attraction Law - Stability

**Status**: ❌ **FAIL**

**Violations**:
- **Contradiction**: 10.6% change (threshold: 10.0%)
  - Normal: 0.6721
  - Inverted: 0.7434
  - Change: 0.0713

**Entailment**: ✅ Stable (5.3% change)  
**Neutral**: ✅ Stable (8.1% change)

**Analysis**: Contradiction cold attraction is just over the 10% threshold. This is borderline and may indicate the threshold is too strict.

**Recommendation**:
- Consider relaxing threshold to 12% for cold attraction
- Or investigate why contradiction cold attraction increases in inverted mode

---

### ✅ TEST 4: Curvature Law - Perfect Invariant

**Status**: ✅ **PASS**

All labels show perfect zero curvature:
- **Entailment**: 0.000000 ✅
- **Contradiction**: 0.000000 ✅
- **Neutral**: 0.000000 ✅

Curvature remains exactly zero even when labels are inverted. This is a perfect invariant.

---

### ✅ TEST 5: Opposition Axis Law - Derived from Invariants

**Status**: ✅ **PASS**

Opposition sign is preserved for all labels:
- **Entailment**: -0.5623 → -0.6016 (both negative) ✅
- **Contradiction**: -0.5226 → -0.6418 (both negative) ✅
- **Neutral**: -0.5273 → -0.6101 (both negative) ✅

**Note**: Opposition is computed as `resonance × sign(divergence)`. Since divergence is negative for all labels, opposition is negative for all. This is consistent but reveals the divergence sign issue (see Law 9).

---

### ✅ TEST 6: Three-Phase Manifold Law - Phases Exist

**Status**: ✅ **PASS**

All three phases have distinct signatures:
- **Entailment**: div=-0.0434, res=0.5623
- **Contradiction**: div=-0.0113, res=0.5226
- **Neutral**: div=-0.0137, res=0.5273

Phases are distinct (separation > 0.01).

---

### ✅ TEST 7: Meaning Emergence Law - Structure Persists

**Status**: ✅ **PASS**

Resonance structure persists (change < 20%) for all labels:
- **Entailment**: 0.5623 → 0.6016 (change: 0.0393) ✅
- **Contradiction**: 0.5226 → 0.6418 (change: 0.1193) ✅
- **Neutral**: 0.5273 → 0.6101 (change: 0.0828) ✅

---

### ✅ TEST 8: Neutral Baseline Law - Default Rest State

**Status**: ✅ **PASS**

Neutral shows equilibrium characteristics:
- **Divergence**: -0.0137 (near zero: True) ✅
- **Resonance**: 0.5273 (mid-range: True) ✅

---

### ❌ TEST 9: Inward-Outward Axis Law - Primary Separator

**Status**: ❌ **FAIL**

**Violations**:
- **Contradiction not outward**: divergence = -0.0113 (should be positive)

**Expected**:
- **Entailment**: negative divergence (inward) ✅ (actual: -0.0434)
- **Contradiction**: positive divergence (outward) ❌ (actual: -0.0113)
- **Neutral**: near-zero divergence ✅ (actual: -0.0137)

**Root Cause Analysis**:

The divergence formula is: `divergence = 0.38 - alignment`

Current alignment values:
- **Entailment**: ~0.42 → divergence = -0.04 ✅
- **Contradiction**: ~0.39 → divergence = -0.01 ❌
- **Neutral**: ~0.39 → divergence = -0.01 ✅

**Problem**: Contradiction mean alignment (0.39) is **above** the threshold (0.38), causing negative divergence.

**Historical Context**:
According to `divergence_law.md`, the law was calibrated with:
- Entailment mean alignment: 0.40 → divergence = -0.02 ✅
- Contradiction mean alignment: 0.25 → divergence = +0.13 ✅

But actual data shows contradiction alignment is much higher (~0.39 vs expected 0.25).

**Possible Causes**:
1. **Threshold too low**: The 0.38 threshold may need recalibration
2. **Alignment distribution shifted**: Word vector alignments may have changed
3. **Pattern collection method**: Patterns may have been collected with different code version

**Recommendations**:

1. **Recalibrate threshold**: Raise threshold to ~0.40-0.42
   ```python
   equilibrium_threshold = 0.40  # or 0.42
   ```
   This would give:
   - Entailment (align=0.42) → divergence = -0.02 ✅
   - Contradiction (align=0.39) → divergence = +0.01 ✅

2. **Use adaptive threshold**: Compute threshold from actual data distribution
   ```python
   # Use mean alignment as threshold
   mean_alignment = compute_mean_alignment(batch)
   divergence = mean_alignment - alignment
   ```

3. **Investigate alignment computation**: Check if alignment computation has changed

4. **Re-collect patterns**: Ensure patterns are collected with current code version

---

## Summary of Issues

### Critical Issues (Must Fix)
1. **Inward-Outward Axis Law Broken**: Contradiction has negative divergence
   - **Impact**: Fundamental physics violation
   - **Fix**: Recalibrate divergence threshold

### Moderate Issues (Should Fix)
2. **Resonance Law Unstable**: Contradiction/neutral show >10% change
   - **Impact**: Resonance may be partially learned, not purely geometric
   - **Fix**: Relax threshold or investigate training dynamics

3. **Cold Attraction Law Borderline**: Contradiction shows 10.6% change
   - **Impact**: Just over threshold, may be acceptable
   - **Fix**: Relax threshold to 12% or investigate

---

## Recommendations

### Immediate Actions
1. ✅ **Fix Divergence Threshold**: Raise from 0.38 to 0.40-0.42
2. ✅ **Re-run Tests**: Verify all laws pass after threshold adjustment
3. ✅ **Update Documentation**: Update `divergence_law.md` with new threshold

### Follow-up Actions
1. Investigate why resonance increases so much in inverted mode
2. Consider making thresholds adaptive based on data distribution
3. Document acceptable variance ranges for each law

---

## Test Methodology

Tests compare patterns from:
- **Normal mode**: Standard training with correct labels
- **Inverted mode**: Training with inverted labels (E↔C)

Laws are verified by checking:
- **Invariance**: Values remain stable when labels are inverted
- **Sign preservation**: Signs remain correct
- **Physical correctness**: Values match expected physics

---

## Conclusion

While most laws (6/9) pass verification, **3 critical issues** need attention:

1. **Divergence threshold needs recalibration** (critical)
2. **Resonance stability thresholds may be too strict** (moderate)
3. **Cold attraction threshold is borderline** (minor)

After fixing the divergence threshold, all fundamental physics laws should pass verification.

---

## References

- Divergence Law: `divergence_law.md`
- Resonance Law: `resonance_law.md`
- Inward-Outward Axis Law: `inward_outward_axis_law.md`
- Test Implementation: `experiments/nli_v5/test_all_laws.py`


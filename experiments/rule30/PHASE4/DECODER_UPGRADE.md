# Decoder Upgrade: Linear → Non-Linear

## Problem Diagnosed (Gemini)

The **Linear Decoder (Logistic Regression) failed**:

- **Density Collapse**: Model predicts "1" only **0.7%** of the time (Real Rule 30 is ~50%)
- **Poor Accuracy**: Test accuracy ~59%
- **Asymmetry**: Precision for "0" is low (0.55), but Precision for "1" is perfect (1.00)
  - The linear model effectively drew a line that classifies almost *everything* as "0" to be safe

**Scientific Conclusion**: Rule 30's geometry is **not linearly separable**. You cannot slice the 8D attractor with a flat knife to separate "0"s from "1s". They are likely mixed together in a complex manifold (like a spiral or a Swiss roll).

## Solution Implemented

**Replaced `LogisticRegression` with `RandomForestClassifier`**:

- **Non-linear readout**: Handles complex manifold boundaries
- **Class-balanced**: `class_weight='balanced'` forces attention to "1"s
- **Robust configuration**: `n_estimators=100`, `max_depth=10` (prevents overfitting)
- **No black-box physics**: Only used as a *readout* tool, not for dynamics

## Changes Made

**File**: `code/fit_center_decoder.py`

1. **Import changed**:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   ```

2. **Model instantiation**:
   ```python
   model = RandomForestClassifier(
       n_estimators=100,
       max_depth=10,  # Prevent overfitting
       random_state=42,
       class_weight='balanced'  # Force it to pay attention to "1"s
   )
   ```

## Expected Improvements

- **Accuracy**: 70–90% (up from ~59%)
- **Decoder sensitivity**: Now sensitive to "1"s
- **Center ones fraction**: 0.4–0.6 (healthy Rule 30, up from 0.7%)

## Next Steps

1. **Re-train decoder**:
   ```bash
   python code/fit_center_decoder.py --n-components 8 --output-dir results --verbose
   ```

2. **Re-run shadow simulation**:
   ```bash
   python code/shadow_rule30_phase4.py \
       --data-dir ../PHASE3/results \
       --decoder-dir results \
       --output-dir results \
       --num-steps 5000 \
       --verbose
   ```

3. **Validate results**:
   - Check `center_ones_fraction` ≈ 0.45–0.55
   - Check decoder accuracy > 70%

## Why This Works

- **Random Forest = Curved Knife**: Can handle non-linear boundaries in the 8D PCA space
- **Class Balancing**: Prevents the model from collapsing to "all zeros"
- **Ensemble Method**: Multiple decision trees capture complex patterns
- **Interpretable**: Still transparent (feature importance available), not a black box

---

**Status**: ✅ **IMPLEMENTED**  
**Next**: Re-train and re-run Phase 4


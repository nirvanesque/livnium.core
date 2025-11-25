# Phase 4: Non-Linear Decoding

**Status**: 🚀 **IMPLEMENTED**

## Overview

Phase 4 adds a **non-linear decoder** to map the 8D PCA geometry back to binary bits (0/1) for the center column.

**Key Insight from Phase 3**: The geometry → bits mapping is **non-linear**. Chaos generates bits through thresholding, not linear mixing.

## What We Learned from Phase 3

1. ✅ **Geometry is predictable (short-term)**: PC1–PC3 have real signal (R² ~ 0.57)
2. ✅ **Chaos behaves like chaos**: Prediction horizon collapses around step 5 (Lyapunov divergence)
3. ✅ **Shadow attractor shape is accurate**: PCA scatter plots overlap
4. ❌ **Reconstruction fails**: Linear mapping (`center ≈ PC1 * correlation`) doesn't work

**The Real Takeaway**: We successfully modeled the hidden continuous geometry of Rule 30, but need a *non-linear decoder* to turn that geometry back into bits.

## Implementation

### Files

- `code/fit_center_decoder.py` - Trains Logistic Regression to predict center bit (0/1) from PCA coordinates
- `code/shadow_rule30_phase4.py` - Shadow Rule 30 with decoder integration

### Quick Start

1. **Train the decoder:**
   ```bash
   cd experiments/rule30/PHASE4
   python code/fit_center_decoder.py --n-components 8 --output-dir results --verbose
   ```

2. **Run Shadow Rule 30 with decoder (with energy injection):**
   ```bash
   python code/shadow_rule30_phase4.py \
       --data-dir ../PHASE3/results \
       --decoder-dir results \
       --output-dir results \
       --num-steps 5000 \
       --verbose
   ```

3. **Validate shadow dynamics:**
   ```bash
   python code/validate_shadow.py \
       --results-dir results \
       --verbose
   ```

## Energy Injection Fix

**Problem**: Learned dynamics have eigenvalues ≈0.7, causing geometric state to shrink → decoder sees nothing → predicts all zeros → shadow dies.

**Solution**: Energy injection maintains target energy level (mean L2 norm from training data) after each dynamics step. This prevents collapse and keeps the shadow alive.

**What to watch for:**
- `center_ones_fraction` should rise from ~0.0002 to **0.45–0.55**
- `trajectory_std` should increase
- `shadow_center_column.npy` should look noisy and alive

## Expected Results

- **Decoder Accuracy**: Should achieve >50% accuracy (better than random)
- **Binary Output**: Center column is now actual bits (0 or 1), not probabilities
- **Non-linear Mapping**: Proves geometry → bits requires thresholding

## Why Keep It Small?

Phase 3 infrastructure is already perfect. We only need a **decoder** — not a rebuild, not a rewrite.

- Phase 2 = finding space
- Phase 3 = finding motion  
- Phase 4 = adding the lens to *see back into bit world*

Small. Fast. Surgical.


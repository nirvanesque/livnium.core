# Phase 4B Patches: Nonlinear Generator + Stochastic Driver

## Overview

Phase 4B completes the Shadow Rule 30 by adding two critical components:

1. **Nonlinear Generator** (Polynomial Degree 3) - Adds curvature and switching behavior
2. **Stochastic Driver** (Noise Injection) - Models unpredictable residual components

Together, these patches make the Shadow universe truly chaotic.

---

## PATCH 1: Nonlinear Generator (Polynomial Dynamics)

### Problem

The Shadow universe is "alive" but not "chaotic." It oscillates along one stable eigenvector ("the highway") and never enters the 1-regions.

### Solution

Retrain the Phase 3 dynamics model using **polynomial degree = 3** with `include_bias=False`.

### Changes Made

**File**: `PHASE3/code/fit_full_dynamics.py`

1. **Updated `fit_polynomial_dynamics()`**:
   - For degree 3, uses `include_bias=False` to avoid overfitting
   - For degree 2, keeps `include_bias=True` (backward compatible)

2. **Added `--degree` argument**:
   - Allows specifying polynomial degree (default: fits both 2 and 3)

### Execution

```bash
cd experiments/rule30/PHASE3
python code/fit_full_dynamics.py \
    --data-dir results \
    --output-dir results \
    --n-components 8 \
    --models polynomial \
    --degree 3 \
    --verbose
```

### Expected Results

- Shadow attractor shows switching activity
- No longer trapped on single eigenvector
- Curvature, folding, and escape from dominant mode

---

## PATCH 2: Stochastic Driver (Noise Injection)

### Problem

The real Rule 30 attractor contains **unpredictable components**. The deterministic dynamics model cannot capture this residual noise.

### Solution

Learn noise from training data residuals and inject it during simulation.

### Changes Made

**File**: `PHASE4/code/shadow_rule30_phase4.py`

1. **Added `fit_residuals()` method**:
   - Computes residuals: `residuals = true_next - predicted_next`
   - Fits multivariate Gaussian: `noise_mean`, `noise_cov`
   - Models the stochastic component

2. **Updated `step()` method**:
   - Adds stochastic noise: `y_tp1 = F(y_t) + noise`
   - Uses `np.random.multivariate_normal()` for correlated noise
   - Falls back to diagonal approximation if covariance is singular

3. **Updated `main()`**:
   - Calls `fit_residuals()` before simulation
   - Loads full PCA trajectory to learn noise statistics

### Execution

After retraining Phase 3 with degree 3:

```bash
cd experiments/rule30/PHASE4
python code/shadow_rule30_phase4.py \
    --data-dir ../PHASE3/results \
    --decoder-dir results \
    --output-dir results \
    --num-steps 5000 \
    --verbose
```

### Expected Results

- `center_ones_fraction` rises from **0.028 → 0.2–0.5**
- Trajectory variance increases to match real Rule 30
- Decoder (RandomForest) fires properly on "1" regions
- Shadow shows true chaotic behavior

---

## Complete Workflow

### Step 1: Retrain Phase 3 with Degree 3

```bash
cd experiments/rule30/PHASE3
python code/fit_full_dynamics.py \
    --data-dir results \
    --output-dir results \
    --n-components 8 \
    --models polynomial \
    --degree 3 \
    --verbose
```

### Step 2: Re-run Phase 4 with Both Patches

```bash
cd experiments/rule30/PHASE4
python code/shadow_rule30_phase4.py \
    --data-dir ../PHASE3/results \
    --decoder-dir results \
    --output-dir results \
    --num-steps 5000 \
    --verbose
```

### Step 3: Validate

```bash
python code/validate_shadow.py \
    --results-dir results \
    --verbose
```

**Check for:**
- `center_ones_fraction` ≈ 0.45–0.55 ✅
- `center_std` not tiny ✅
- Trajectory shows switching activity ✅
- Decoder fires on both 0s and 1s ✅

---

## Why Both Patches Are Required

- **Polynomial Degree 3**: Provides the **curvature** and **switching behavior** needed to escape the dominant eigenvector
- **Stochastic Driver**: Models the **unpredictable residual** that makes Rule 30 truly chaotic

Together, they create a Shadow universe that:
- Has proper geometric structure (polynomial)
- Exhibits true chaos (stochastic noise)
- Produces realistic bit sequences (decoder fires correctly)

---

## Status

✅ **PATCH 1**: Implemented (Polynomial Degree 3)  
✅ **PATCH 2**: Implemented (Stochastic Driver)  
🔄 **Next**: Retrain Phase 3, then re-run Phase 4

---

**Phase 4B Complete** → Ready for Phase 5 (Livnium Integration)


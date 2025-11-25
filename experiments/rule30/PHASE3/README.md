# Phase 3: Dynamics Modeling — The Motion Law

**Status**: 🚀 **IMPLEMENTED**

## Overview

Phase 3 moves from *mapping* the chaos (Phase 2) to *predicting* the chaos. We learn the "motion law" in the PCA space that governs how Rule 30's geometric coordinates evolve over time.

## Objectives

1. **Work in PCA space, NOT raw 15D space**
   - 95% of variance lives in 8 dimensions
   - PC1–PC3 capture 70% of dynamics
   - Model only the top 8 PCs

2. **Learn the motion law**
   - Function: `y_{t+1} ≈ F(y_t)` where `y_t` = top k PCA components
   - Dynamics from real Rule 30 grid simulation
   - Model must be simple, transparent, interpretable

3. **Attack PC1 first**
   - PC1 explains ≈35% variance
   - PC1 correlates ≈±0.7 with center column
   - Predict PC1(t+1) from {PC1(t), PC2(t), PC3(t)}

4. **Build the "Shadow Rule 30"**
   - Model operates entirely in PCA space
   - Never touches bitwise grid
   - Produces synthetic time series matching real center column distribution

## Structure

- `code/` - Implementation files
- `docs/` - Documentation
- `results/` - Phase 3 output

## Implementation Files

### Core Scripts
- `extract_pca_trajectories.py` - Extract and prepare PCA data from Phase 2
- `fit_pc1_dynamics.py` - Fit PC1 prediction model
- `fit_full_dynamics.py` - Fit full 8D dynamics model
- `shadow_rule30.py` - Shadow Rule 30 implementation
- `evaluate_dynamics.py` - Evaluation and metrics
- `visualize_dynamics.py` - Generate all visualizations
- `generate_report.py` - Generate comprehensive report

### Documentation
- `PHASE3_PLAN.md` - Detailed phase plan
- `PHASE3_RESULTS.md` - Generated results report (after running)

## Quick Start

1. **Extract PCA trajectories from Phase 2 data:**
   ```bash
   python code/extract_pca_trajectories.py --n-components 8 --output-dir results --verbose
   ```

2. **Fit PC1 dynamics model:**
   ```bash
   python code/fit_pc1_dynamics.py --data-dir results --output-dir results --use-pc2-pc3 --verbose
   ```

3. **Fit full dynamics model:**
   ```bash
   python code/fit_full_dynamics.py --data-dir results --output-dir results --n-components 8 --verbose
   ```

4. **Run Shadow Rule 30:**
   ```bash
   python code/shadow_rule30.py --data-dir results --output-dir results --num-steps 5000 --verbose
   ```

5. **Evaluate models:**
   ```bash
   python code/evaluate_dynamics.py --data-dir results --output-dir results --verbose
   ```

6. **Generate visualizations:**
   ```bash
   python code/visualize_dynamics.py --data-dir results --output-dir results --verbose
   ```

7. **Generate report:**
   ```bash
   python code/generate_report.py --data-dir results --output PHASE3_RESULTS.md --verbose
   ```

## Dependencies

- Phase 2 results (trajectory data)
- scikit-learn (PCA, regression)
- numpy, scipy
- matplotlib (visualization)

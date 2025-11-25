# Phase 3: Dynamics Modeling — The Motion Law

**Status**: 🚀 **IN PROGRESS**

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

## Data Preparation

### Input Data
- **Source**: Phase 2 trajectory data
  - `trajectory_15d.npy`: (num_steps, 15) free coordinates
  - `trajectory_full.npy`: (num_steps, 34) full state vectors
- **Location**: `PHASE2/results/chaos14/` or `PHASE2/results/chaos15/`

### Processing Steps
1. Load 15D trajectory from Phase 2
2. Fit PCA on full trajectory (compute once, reuse)
3. Extract top 8 PC components
4. Split into train/validation/test sets (80/10/10)
5. Extract center column values for correlation analysis

## Modeling Approaches

### Strategy 1: Local Linear Approximation (Jacobian)
- Fit: `y_{t+1} = J @ y_t + b`
- Where `J` is (k×k) Jacobian matrix, `b` is bias
- Simple, interpretable, captures local dynamics

### Strategy 2: Polynomial Regression (degree ≤ 3)
- Fit: `y_{t+1} = P(y_t)` where P is polynomial
- Captures nonlinearities
- Still interpretable (coefficients show interactions)

### Strategy 3: Symbolic Regression (PySR)
- Automatically discover symbolic expressions
- Most interpretable if successful
- May be slow for high-dimensional data

### Strategy 4: Kernel Regression
- Non-parametric, flexible
- Less interpretable but can capture complex dynamics

### Strategy 5: Sparse Nonlinear Autoregressive Models
- Combines interpretability with flexibility
- Uses feature selection to find important terms

### Forbidden Approaches
- ❌ Neural networks
- ❌ Black-box deep learning
- ❌ Models that "fit" but do not explain

## Validation Steps

1. **Train/Test Split**
   - 80% training, 10% validation, 10% test
   - Ensure temporal ordering (no shuffling)

2. **Error Metrics**
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - R² score
   - Prediction horizon analysis (1-step, 5-step, 10-step ahead)

3. **Stability Analysis**
   - Check if learned dynamics are stable
   - Analyze eigenvalues of Jacobian (for linear models)
   - Long-term trajectory divergence

4. **Reconstruction Error**
   - Compare predicted PCA → reconstructed 15D → full state
   - Measure center column prediction accuracy

## Visualizations

1. **PC1 Prediction**
   - Time series: real vs predicted PC1
   - Scatter: PC1(t+1) vs predicted PC1(t+1)
   - Residual analysis

2. **Trajectory Comparison**
   - Real vs Shadow Rule 30 trajectories in PCA space
   - 2D/3D projections colored by time

3. **Center Column Comparison**
   - Real vs predicted center column time series
   - Distribution comparison (histograms)

4. **Error Analysis**
   - Prediction error over time
   - Error distribution
   - Error vs PC magnitude

## Outputs

### Code Files
1. `extract_pca_trajectories.py` - Extract and prepare PCA data
2. `fit_pc1_dynamics.py` - Fit PC1 prediction model
3. `fit_full_dynamics.py` - Fit full 8D dynamics model
4. `shadow_rule30.py` - Shadow Rule 30 implementation
5. `evaluate_dynamics.py` - Evaluation and metrics
6. `visualize_dynamics.py` - Generate all visualizations

### Results
1. Trained models (saved as pickle files)
2. Prediction results (numpy arrays)
3. Visualizations (PNG files)
4. `PHASE3_RESULTS.md` - Comprehensive report

## Success Criteria

1. **PC1 Predictability**
   - R² > 0.5 for PC1 prediction
   - MAE < 0.1 × std(PC1)

2. **Shadow Rule 30**
   - Synthetic center column distribution matches real (KS test p > 0.05)
   - PCA trajectory visually resembles real attractor

3. **Interpretability**
   - Model coefficients/terms are interpretable
   - Clear understanding of what drives dynamics

## Timeline

1. **Week 1**: Data preparation + PC1 modeling
2. **Week 2**: Full dynamics + Shadow Rule 30
3. **Week 3**: Evaluation + Report

## Dependencies

- Phase 2 results (trajectory data)
- scikit-learn (PCA, regression)
- numpy, scipy
- matplotlib (visualization)
- Optional: PySR (symbolic regression)


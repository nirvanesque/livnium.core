# Phase 3 Quick Start Guide

This guide walks you through running Phase 3 from start to finish.

## Prerequisites

1. **Phase 2 must be complete**
   - Ensure `PHASE2/results/chaos14/` or `PHASE2/results/chaos15/` contains:
     - `trajectory_15d.npy`
     - `trajectory_full.npy`

2. **Required Python packages:**
   ```bash
   pip install numpy scipy scikit-learn matplotlib
   ```

## Step-by-Step Execution

### Step 1: Extract PCA Trajectories

Extract PCA trajectories from Phase 2 data and prepare for modeling.

```bash
cd experiments/rule30/PHASE3
python code/extract_pca_trajectories.py \
    --n-components 8 \
    --output-dir results \
    --verbose
```

**What this does:**
- Loads 15D trajectory from Phase 2
- Fits PCA on full trajectory
- Extracts top 8 PC components
- Computes correlation with center column
- Splits data into train/val/test sets

**Output:**
- `results/trajectory_pca.npy` - PCA coordinates
- `results/pca_model.pkl` - Fitted PCA model
- `results/data_splits.pkl` - Train/val/test splits
- `results/metadata.json` - Metadata

### Step 2: Fit PC1 Dynamics Model

Fit a model to predict PC1(t+1) from PC1(t), PC2(t), PC3(t).

```bash
python code/fit_pc1_dynamics.py \
    --data-dir results \
    --output-dir results \
    --use-pc2-pc3 \
    --models linear polynomial \
    --verbose
```

**What this does:**
- Fits linear and polynomial models for PC1 prediction
- Evaluates on validation and test sets
- Selects best model based on R² score

**Output:**
- `results/linear_model.pkl` - Linear model
- `results/polynomial_degree_2_model.pkl` - Polynomial model
- `results/pc1_model_summary.json` - Model comparison

### Step 3: Fit Full Dynamics Model

Fit a model to predict all top 8 PCs simultaneously.

```bash
python code/fit_full_dynamics.py \
    --data-dir results \
    --output-dir results \
    --n-components 8 \
    --models linear polynomial \
    --verbose
```

**What this does:**
- Fits linear and polynomial dynamics for all components
- Evaluates per-component performance
- Analyzes stability (for linear models)

**Output:**
- `results/linear_dynamics_model.pkl` - Linear dynamics
- `results/polynomial_degree_2_dynamics_model.pkl` - Polynomial dynamics
- `results/full_dynamics_summary.json` - Model comparison

### Step 4: Run Shadow Rule 30

Simulate dynamics entirely in PCA space.

```bash
python code/shadow_rule30.py \
    --data-dir results \
    --output-dir results \
    --num-steps 5000 \
    --initial-condition from_data \
    --verbose
```

**What this does:**
- Simulates trajectory using learned dynamics
- Reconstructs center column from PCA coordinates
- Computes statistics

**Output:**
- `results/shadow_trajectory_pca.npy` - Shadow trajectory
- `results/shadow_center_column.npy` - Shadow center column
- `results/shadow_statistics.json` - Statistics

### Step 5: Evaluate Models

Evaluate prediction accuracy and compare real vs shadow.

```bash
python code/evaluate_dynamics.py \
    --data-dir results \
    --output-dir results \
    --model-type full \
    --verbose
```

**What this does:**
- Evaluates prediction horizons (1-step, 5-step, 10-step ahead)
- Compares real vs shadow trajectories
- Compares center column distributions

**Output:**
- `results/evaluation_results.json` - Evaluation metrics

### Step 6: Generate Visualizations

Create all visualization plots.

```bash
python code/visualize_dynamics.py \
    --data-dir results \
    --output-dir results \
    --verbose
```

**Output:**
- `results/pc1_prediction.png` - PC1 prediction plots
- `results/trajectory_comparison.png` - Real vs shadow trajectories
- `results/center_column_comparison.png` - Center column comparison
- `results/error_analysis.png` - Error vs prediction horizon

### Step 7: Generate Report

Generate comprehensive markdown report.

```bash
python code/generate_report.py \
    --data-dir results \
    --output docs/PHASE3_RESULTS.md \
    --verbose
```

**Output:**
- `docs/PHASE3_RESULTS.md` - Comprehensive results report

## Expected Results

### Success Criteria

1. **PC1 Predictability**
   - R² > 0.5 for PC1 prediction
   - MAE < 0.1 × std(PC1)

2. **Shadow Rule 30**
   - Synthetic center column distribution matches real (KS test p > 0.05)
   - PCA trajectory visually resembles real attractor

3. **Interpretability**
   - Model coefficients/terms are interpretable
   - Clear understanding of what drives dynamics

### Typical Output

- **PC1 R²**: ~0.6-0.8 (depending on model)
- **Full dynamics R²**: ~0.5-0.7
- **Prediction horizon**: R² > 0.5 for 1-5 steps ahead
- **Shadow trajectory**: Visually similar to real, distribution matches

## Troubleshooting

### Issue: Phase 2 data not found

**Solution:** Ensure Phase 2 has been run and trajectory files exist in:
- `PHASE2/results/chaos14/` or
- `PHASE2/results/chaos15/`

### Issue: Import errors

**Solution:** Ensure you're running from the correct directory and Phase 2 code is accessible:
```bash
cd experiments/rule30/PHASE3
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../PHASE2/code"
```

### Issue: Low R² scores

**Possible causes:**
- Insufficient training data (try longer Phase 2 simulation)
- Model too simple (try polynomial degree 3)
- Need more PCA components (increase `--n-components`)

## Next Steps

After completing Phase 3:

1. Review `docs/PHASE3_RESULTS.md` for detailed analysis
2. Examine visualizations in `results/`
3. Consider improvements:
   - Symbolic regression for more interpretable models
   - Better center column reconstruction
   - Stability analysis of learned dynamics


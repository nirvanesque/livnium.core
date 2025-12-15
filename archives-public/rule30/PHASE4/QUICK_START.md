# Phase 4 Quick Start

## Prerequisites

- Phase 3 must be complete (PCA trajectories extracted)
- Required packages: `numpy`, `scikit-learn`

## Execution Steps

### Step 1: Train the Decoder

```bash
cd experiments/rule30/PHASE4
python code/fit_center_decoder.py \
    --n-components 8 \
    --output-dir results \
    --verbose
```

**What this does:**
- Loads PCA trajectory and center column from Phase 3
- Converts center column to binary (threshold at 0.5)
- Trains Logistic Regression to predict bit (0/1) from PCA coordinates
- Evaluates accuracy and F1-score
- Saves model to `results/center_decoder.pkl`

**Expected output:**
- Classification report showing accuracy
- Model saved to `results/center_decoder.pkl`

### Step 2: Run Shadow Rule 30 with Decoder

```bash
python code/shadow_rule30_phase4.py \
    --data-dir ../PHASE3/results \
    --decoder-dir results \
    --output-dir results \
    --num-steps 5000 \
    --verbose
```

**What this does:**
- Loads PCA model, dynamics model, and decoder from Phase 3/4
- Simulates trajectory in PCA space
- Uses decoder to predict binary bits (0/1) instead of linear approximation
- Saves shadow trajectory and center column bits

**Expected output:**
- Shadow trajectory in PCA space
- Center column as binary bits (0 or 1)
- Statistics showing fraction of ones

## What Changed from Phase 3?

**Phase 3**: Used linear approximation
```python
center_column = pc1 * correlation  # ❌ Doesn't work
```

**Phase 4**: Uses non-linear decoder
```python
center_bits = decoder.predict(trajectory_pca)  # ✅ Binary prediction
```

## Success Criteria

- **Decoder Accuracy**: >50% (better than random)
- **Binary Output**: Center column is actual bits (0 or 1)
- **Non-linear Mapping**: Proves geometry → bits requires thresholding

## Troubleshooting

### Issue: Phase 3 data not found

**Solution**: Ensure Phase 3 has been run:
```bash
cd ../PHASE3
python code/extract_pca_trajectories.py --output-dir results --verbose
```

### Issue: Decoder accuracy is low

**Possible causes:**
- Insufficient training data
- Need more PCA components (try `--n-components 10`)
- Class imbalance (check class distribution in output)

### Issue: Import errors

**Solution**: Ensure you're in the correct directory:
```bash
cd experiments/rule30/PHASE4
```


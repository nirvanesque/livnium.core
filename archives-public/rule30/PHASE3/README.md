# Phase 3: Learning How It Moves

**Status**: ✅ Complete

## The Goal

Phase 2 gave us trajectories in 14D space. Phase 3 answers: how does this geometry evolve over time? Can we predict where it goes next?

## The Strategy

Instead of working in full 14D space, we use PCA to find the most important directions:
- 95% of the variance lives in just 8 dimensions
- The first 3 components (PC1–PC3) capture 70% of the dynamics
- PC1 alone explains 35% and correlates strongly with the center column

We learn a polynomial model that predicts the next state from the current one: `y_{t+1} ≈ F(y_t)`.

## What We Built

- **PCA extraction**: Reduces 14D trajectories to 8D
- **Dynamics models**: Polynomial models that predict how coordinates evolve
- **Shadow Rule 30**: A system that runs entirely in geometric space, never touching bits

The shadow produces synthetic trajectories that match Rule 30's statistics—proof that the geometry captures the essential dynamics.

## Running It

```bash
cd PHASE3

# Extract PCA coordinates
python3 code/extract_pca_trajectories.py --n-components 8 --output-dir results --verbose

# Fit dynamics models
python3 code/fit_pc1_dynamics.py --data-dir results --output-dir results --use-pc2-pc3 --verbose
python3 code/fit_full_dynamics.py --data-dir results --output-dir results --n-components 8 --verbose

# Generate shadow trajectory
python3 code/shadow_rule30.py --data-dir results --output-dir results --num-steps 5000 --verbose

# Evaluate and visualize
python3 code/evaluate_dynamics.py --data-dir results --output-dir results --verbose
python3 code/visualize_dynamics.py --data-dir results --output-dir results --verbose
```

See `QUICK_START.md` in this directory for more details.

## Results

- Dynamics model with R² ~ 0.57 (good for chaotic systems)
- Shadow trajectories that match real Rule 30's geometry
- Visualizations showing the learned attractor shape

## Next Steps

Phase 4 adds a decoder to map geometry back to bits. See `../PHASE4/` to continue.

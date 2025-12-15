# Quick Start: Running the Shadow Rule 30 Pipeline

This guide walks you through running the complete pipeline from raw chaos data to proof experiments.

## Prerequisites

- Python 3.10 or higher
- Basic scientific Python stack: `numpy`, `scipy`, `scikit-learn`, `matplotlib`
- Install with: `pip install numpy scipy scikit-learn matplotlib`

All commands assume you're starting from the repo root.

## Phase 2: Generate the Geometry

First, we need to map Rule 30's behavior into continuous space.

```bash
cd archives-public/rule30/PHASE2/code

# Verify the system integrity
python3 verify_phase2_integrity.py --verbose

# Check physical constraints (has known caveats, see docs)
python3 verify_phase2_physics.py --verbose

# Generate chaos trajectories
python3 four_bit_chaos_tracker.py --steps 5000 --verbose
```

This creates trajectory data in `../results/chaos14/`. The trajectories represent Rule 30's state as points in geometric space.

## Phase 3: Learn How It Moves

Now we learn the "motion law"—how the geometry evolves over time.

```bash
cd ../../PHASE3

# Extract PCA coordinates (reduces dimensionality)
python3 code/extract_pca_trajectories.py --n-components 8 --output-dir results --verbose

# Fit dynamics model for PC1 (most important component)
python3 code/fit_pc1_dynamics.py --data-dir results --output-dir results --use-pc2-pc3 --verbose

# Fit full 8D dynamics model
python3 code/fit_full_dynamics.py --data-dir results --output-dir results --n-components 8 --verbose

# Generate shadow trajectory using learned dynamics
python3 code/shadow_rule30.py --data-dir results --output-dir results --num-steps 5000 --verbose

# Evaluate how well the model works
python3 code/evaluate_dynamics.py --data-dir results --output-dir results --verbose

# Create visualizations
python3 code/visualize_dynamics.py --data-dir results --output-dir results --verbose

# Generate phase report
python3 code/generate_report.py --data-dir results --output docs/PHASE3_RESULTS.md --verbose
```

This gives you a polynomial model that predicts how the geometry moves, plus a "shadow" trajectory that follows those dynamics.

## Phase 4: Decode Back to Bits

The shadow lives in geometry space. We need to map it back to actual bits.

```bash
cd ../PHASE4

# Train decoder and run shadow with energy injection
bash run_all.sh

# Validate the results
python3 code/validate_shadow.py --results-dir results --verbose
```

This produces:
- `results/center_decoder.pkl` - The decoder model
- `results/shadow_statistics.json` - Statistics comparing shadow to real Rule 30
- `results/shadow_center_column.npy` - The decoded bit sequence

The decoder achieves ~94% accuracy and the shadow matches Rule 30's density (~49%).

## Phase 6: Optional Livnium Steering

Add geometric steering to guide the dynamics (optional).

```bash
cd ../PHASE6/code

python3 shadow_rule30_phase6.py \
  --data-dir ../../PHASE3/results \
  --decoder-dir ../../PHASE4/results \
  --output-dir ../results \
  --num-steps 5000 \
  --livnium-scale 0.01 \
  --livnium-type vector \
  --verbose
```

This shows how geometric forces can influence the shadow's behavior.

## Phase 7: Proof Without Steering

Finally, prove the system works without Livnium—the shadow should still match Rule 30 even without steering.

```bash
cd ../PHASE7/code

python3 run_phase7_proof.py \
  --data-dir ../../PHASE3/results \
  --decoder-dir ../../PHASE4/results \
  --results-dir ../results \
  --num-steps 5000
```

This generates:
- Decoder consistency checks
- Density comparisons (shadow vs. real Rule 30)
- A proof report in `../results/PROOF_REPORT.md`

## What to Expect

- **Phase 2**: Takes a few minutes, generates trajectory files
- **Phase 3**: Takes 10-30 minutes depending on your machine, produces models and visualizations
- **Phase 4**: Takes 5-10 minutes, trains decoder and validates shadow
- **Phase 6**: Optional, takes a few minutes
- **Phase 7**: Takes 5-10 minutes, generates proof report

## Troubleshooting

- **Import errors**: Make sure you're in the correct directory and dependencies are installed
- **File not found**: Check that previous phases completed successfully
- **Memory issues**: Reduce `--num-steps` or `--n-components` for smaller runs

For phase-specific issues, check each phase's README in its directory.

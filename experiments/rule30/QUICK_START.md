# Quick Start Guide

End-to-end run of the Shadow Rule 30 pipeline from chaos data to proof.

## Prerequisites
- Python 3.10+
- `pip install numpy scipy scikit-learn matplotlib`
- Commands below assume repo root: `/Users/chetanpatil/Desktop/clean-nova-livnium`

## 1) Generate Geometry (Phase 2)
```bash
cd experiments/rule30/PHASE2/code
python3 verify_phase2_integrity.py --verbose             # algebraic checks
python3 verify_phase2_physics.py --verbose               # physical checks (known caveats)
python3 four_bit_chaos_tracker.py --steps 5000 --verbose # writes chaos14 trajectories
```
Outputs land in `../results/chaos14/` (mirrored in `experiments/rule30/results/chaos14/`).

## 2) Learn Dynamics (Phase 3)
```bash
cd ../../PHASE3
python3 code/extract_pca_trajectories.py --n-components 8 --output-dir results --verbose
python3 code/fit_pc1_dynamics.py        --data-dir results --output-dir results --use-pc2-pc3 --verbose
python3 code/fit_full_dynamics.py       --data-dir results --output-dir results --n-components 8 --verbose
python3 code/shadow_rule30.py           --data-dir results --output-dir results --num-steps 5000 --verbose
python3 code/evaluate_dynamics.py       --data-dir results --output-dir results --verbose
python3 code/visualize_dynamics.py      --data-dir results --output-dir results --verbose
python3 code/generate_report.py         --data-dir results --output docs/PHASE3_RESULTS.md --verbose
```

## 3) Decode + Keep Shadow Alive (Phase 4)
```bash
cd ../PHASE4
bash run_all.sh                        # trains decoder + runs shadow with energy injection
python3 code/validate_shadow.py --results-dir results --verbose
```
Outputs: `results/center_decoder.pkl`, `results/shadow_statistics.json`, `results/shadow_center_column.npy`.

## 4) Optional: Livnium Steering (Phase 6)
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

## 5) Proof Without Livnium (Phase 7)
```bash
cd ../PHASE7/code
python3 run_phase7_proof.py \
  --data-dir ../../PHASE3/results \
  --decoder-dir ../../PHASE4/results \
  --results-dir ../results \
  --num-steps 5000
```
Generates decoder-consistency and no-Livnium density checks in `../results/` and `../results/PROOF_REPORT.md`.

## References
- Phase-level quick starts: `PHASE3/QUICK_START.md`, `PHASE4/QUICK_START.md`.
- See `PHASE2/docs/RUN_NOW.md` for additional Phase 2 details.

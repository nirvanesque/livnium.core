# Restructuring Script Verification Report

**Script**: `restructure_rule30.sh`  
**Scope**: Safe, idempotent repair of the phase layout without overwriting existing work.

## What the Script Does Now
- Creates scaffolding for `PHASE1`–`PHASE7` plus `archive/`.
- Rehomes legacy flat files into their phase folders **only when destinations are empty**.
- Copies chaos trajectories into `PHASE2/results/` if missing (`results/chaos14` and `results/chaos15` sources).
- Leaves all current READMEs, reports, and results untouched.

## Verified Inputs on Disk
- **Phase 1**: code (7) and docs (6) all present under `PHASE1/{code,docs}/`.
- **Phase 2**: code (`four_bit_system.py`, `four_bit_chaos_tracker.py`, `decode_manifold.py`, verifiers, Groebner/recurrence solvers) and docs (`PHASE2_SUMMARY.md`, `PHASE2_EXECUTION_SUMMARY.md`, `RUN_NOW.md`, etc.) present; `PHASE2/results/chaos14/` populated.
- **Phase 3**: code + docs (`PHASE3_PLAN.md`, `IMPLEMENTATION_SUMMARY.md`, `PHASE3_RESULTS.md`); results include `trajectory_pca.npy`, `polynomial_degree_3_dynamics_model.pkl`, `shadow_statistics.json`.
- **Phase 4**: decoder + shadow code (`fit_center_decoder.py`, `shadow_rule30_phase4.py`, `validate_shadow.py`); docs (`PHASE4_REPORT.md`, `ENERGY_INJECTION_FIX.md`, `PHASE4B_PATCHES.md`, `QUICK_START.md`); results (decoder metadata, shadow stats/trajectory).
- **Phase 6**: Livnium module (`code/livnium_force.py`, `shadow_rule30_phase6.py`); results populated.
- **Phase 7**: proof harness (`code/run_phase7_proof.py`), summary/report artifacts (`results/PROOF_REPORT.md`, `results/exp3_decoder_consistency.json`).
- **Root results mirrors**: `../../results/chaos14/` and `../../results/chaos15/` found for optional copy.
- **Archive**: `divergence_v3.py`, `test_divergence_v3_invariant.py`, `reproduce_results.sh`.

## Safety Notes
- Uses `set -euo pipefail` and explicit logging.
- Moves occur only when the source exists **and** the destination does not, preventing accidental overwrite.
- Result copies are skipped when the target directory already exists.
- Script does **not** rewrite README/QUICK_START/report files; current documentation stays intact.

## Outstanding Items
- `PHASE5/` remains an empty scaffold for future Livnium expansion.
- Physical validation caveats for Phase 2 remain documented in `PHASE2/docs/`.

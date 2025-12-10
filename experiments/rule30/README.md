# Rule 30 Center Column Research

Research pipeline that maps Rule 30's center column into continuous geometry and reconstructs it through the "Shadow Rule 30" system.

## Phase Map
- [PHASE1](./PHASE1/) — ✅ 3-bit invariants discovered and exhaustively verified.
- [PHASE2](./PHASE2/) — ✅ 4-bit constraint system + chaos geometry (chaos14/chaos15 data); physical validation caveats noted in docs.
- [PHASE3](./PHASE3/) — 🚀 Motion law learned in PCA space (8D, polynomial degree 3); shadow trajectory + report generated.
- [PHASE4](./PHASE4/) — 🚀 Non-linear decoder + energy injection; shadow matches Rule 30 density (~49%) with 94% decoder accuracy.
- [PHASE5](./PHASE5/) — 📋 Reserved scaffold for future Livnium expansion.
- [PHASE6](./PHASE6/) — ✅ Minimal Livnium steering integrated on top of PCA dynamics.
- [PHASE7](./PHASE7/) — 🔬 Proof in progress: remove Livnium, stress-test initial conditions, and compare decoder outputs.

## How to Run
- Use `QUICK_START.md` for the end-to-end command sequence from Phase 2 through Phase 7.
- Each phase directory contains its own README or quick start with details and troubleshooting.

## Key Artifacts
- `PHASE2/results/chaos14/` and `results/chaos15/` — raw chaos trajectories and plots.
- `PHASE3/results/` — PCA model, dynamics fits, shadow trajectories, and evaluation outputs.
- `PHASE4/results/` — decoder model, energy-injected shadow runs, and statistics.
- `PHASE6/results/` — Livnium-influenced runs and metrics.
- `PHASE7/results/` — proof experiments (no-Livnium) and generated reports.
- `archive/` — legacy scripts retained for reference.

## Notes
- Run scripts from within their phase directories unless noted; adjust relative paths when launching from repo root.
- Python dependencies are phase-specific; see each quick start for installation hints.

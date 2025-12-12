# Nova v3 Architecture (Livnium Core v1.0)

Three clean layers, frozen core physics, and swappable encoders. The default SNLI stack uses a pretrained **quantum** embedding table from `nova/quantum_embed` as its main encoder option.

## Layer 0 · Core Physics (frozen)
- **Location**: `core/`
- **Files**: `vector_state.py`, `physics_laws.py`, `vector_collapse_engine.py`, `basin_field.py`
- **Laws**: alignment = cos(OM, LO); divergence = `0.38 - alignment`; tension = `|divergence|`.
- **Dynamics**: `VectorCollapseEngine` iterates L steps with learned state update + anchor forces (entail/contra/neutral), soft norm cap, and trace logging.
- **Dynamic basins**: optional `BasinField` maintains per-label micro-anchors (route, spawn on high tension/low align, EMA update, prune/merge).
- **Knows nothing about text or labels** beyond numeric labels passed into dynamic mode.

## Layer 1 · Encoders & Heads
- **Encoders (text/)**
  - `text/encoder.py` — legacy embedding + mean pool (kept for reference, not used for SNLI).
  - `quantum_embed/text_encoder_quantum.py` — **pretrained encoder**; loads `quantum_embeddings_final.pt` from `nova/quantum_embed/model_full_physics/`, providing tokenizer + embedding table shaped by Livnium energy (optionally collapsed during pretraining).
  - Legacy geom/sanskrit encoders have been removed.
- **SNLI glue (tasks/snli/)**
- `encoding_snli.py` — builds OM/LO vectors, initial state `h0 = v_p + v_h + noise`, quantum encoder only (legacy/geom/sanskrit removed).
  - `head_snli.py` — classification head using `h_final` plus alignment/opposition/radial and neutral-direction features → logits (E/N/C).

## Layer 2 · Scripts (train/eval/visualize)
- `training/train_snli_vector.py` — end-to-end SNLI training; uses quantum encoder; wires dynamic basins, class weights, sampling, and saves checkpoints.
- `chat/test_snli_vector.py` — evaluation (accuracy + confusion matrix, optional error dump).
- `chat/visualize_snli_geometry.py` — 2D projection of anchors/basins/trajectories.

## Data & Artifacts
- SNLI JSONL under `data/snli/` (train/dev/test).
- Checkpoints under `model/<run>/best_model.pt` containing collapse engine, encoder, head, args, vocab (if applicable), and basin field when used.
- Quantum encoder checkpoint lives in `nova/quantum_embed/model_full_physics/quantum_embeddings_final.pt` (tokenizer + embeddings; may also carry its own collapse/basin state from pretraining).

## Training Flow (SNLI)
1) Load SNLI, drop invalid/ambiguous label pairs; tokenize with the quantum tokenizer.
2) Encode premise/hypothesis → `v_p` (OM), `v_h` (LO); build `h0 = v_p + v_h + ε`.
3) Collapse: `VectorCollapseEngine.collapse(h0)` (static anchors) or `.collapse_dynamic(h0, labels, basin_field)` (per-label basins).
4) Head: `SNLIHead(h_final, v_p, v_h)` → logits.
5) Loss: `CrossEntropyLoss` (optional label smoothing/class weights); optimize encoder + collapse engine + head with Adam.

## Physics Snapshot
```
alignment = cos(om, lo)
divergence = 0.38 - alignment
tension = |divergence|
h_{t+1} = h_t + δ(h_t) - Σ_k strength_k * divergence_k * dir(h_t - anchor_k)
‖h‖ is softly capped (≈10)
```

## Adding a New Task
1) Add an encoding to `tasks/<task>/encoding_<task>.py` that returns `h0, v_a, v_b` (or similar OM/LO vectors).
2) Add a head to `tasks/<task>/head_<task>.py` that maps collapsed state (+ optional geometry features) to logits/outputs.
3) Add a training script under `training/` that loads data, instantiates the encoder/head, and calls the shared collapse engine.
4) Keep Layer 0 untouched; reuse physics and basin machinery.

## File Structure (v3)
```
nova_v3/
├── core/                  # Layer 0: physics + basins
├── text/                  # Layer 1: encoders (quantum bridge; legacy kept for reference)
├── tasks/snli/            # Layer 1: SNLI encoding + head
├── training/              # Layer 2: training scripts
├── chat/                  # Layer 2: eval/visualize
├── utils/                 # helpers (vocab, scaling)
├── data/snli/             # datasets
├── model/                 # checkpoints
├── README.md
└── ARCHITECTURE.md
```
Primary pretrained encoder lives at `nova/quantum_embed/` and is always used for SNLI.

## Principles
- Core physics is frozen; extend via encoders/heads/scripts.
- Encoders are quantum-only for SNLI; legacy kept for reference; geom/sanskrit encoders have been removed.
- Dynamic basins add topology without changing laws; prune/merge keeps them bounded.
- Watchdogs/visualizations read traces; they don’t mutate the core.

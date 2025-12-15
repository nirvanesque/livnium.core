# ECW-BT Master Plan (Official)

## Implementation Status (Updated)

**What Was Built**: Level-0 pipeline with SBERT seeding and pairwise CCD training.

**Key Differences from Original Plan**:
- Uses pre-built pair shards (`pairs_pos_*.bin`) instead of streaming windows
- SBERT seed (`V_seed.npy`) instead of random initialization
- Fusion anchoring prevents catastrophic forgetting (teacher acts as elastic scaffold)
- Pre-generated negatives for performance (eliminates CPU→GPU transfers)
- Validation script (`validate_level0.py`) for acceptance testing
- Optimized for Apple Silicon (MPS) with chunked negative processing

**Pipeline Phases**:
- **Phase 0**: `make_sbert_seed.py` - Create teacher vectors from SBERT
- **Phase 1A**: `build_vocab.py` - Build vocabulary and frequency table
- **Phase 1B**: `build_pairs.py` - Extract co-occurrence pairs to binary shards
- **Phase 2**: `distill_pairwise.py` - Train with CCD physics + fusion anchoring
- **Phase 3**: `validate_level0.py` - Acceptance validation (drift, collapse, neighbors)

**Original Plan**: See below (kept for historical reference)

---

## Scope & Constraints
- Build only ECW-BT Level-0: geometry-only basins with CCD and attractor-barrier hybrid forces.
- No Nova v4, no collapse engine beyond gravity pooling, no IntensionNet, no SNLI, no backprop nets.
- Use canonical laws: `m_w = 1 / log(1 + f_w)`, unit-sphere init, always-attract contexts, barrier repulsion only when `cos > 0.38`, renormalize after each update.

## Target Outputs
- `train_ecw_bt.py`: training entry point for CCD with tunnel + barrier forces.
- `src/physics.py`, `src/data_loader.py`, `src/trainer.py`, `src/collapse_engine.py`, `src/config.py`.
- `data/wikipedia_plain/` shards and `data/mass_table.json`.
- `checkpoints/` `.npy` vector snapshots; `logs/` CSV stats.
- `probe_galaxy.py` and `README.md` describing commands, laws, and probes.

## File/Folder Layout
```
nova/ecw-BT/
├── data/
│   ├── wikipedia_plain/      # text shards from local dump
│   └── mass_table.json       # word -> {frequency, mass}
├── logs/                     # training metrics
├── checkpoints/              # vector states
├── src/
│   ├── config.py
│   ├── physics.py            # force laws + renorm
│   ├── data_loader.py        # streaming targets/contexts + noise
│   ├── trainer.py            # CCD loop
│   └── collapse_engine.py    # gravity pooling / ghost basin
├── train_ecw_bt.py           # CLI entry to train
├── probe_galaxy.py           # quick neighbor/probe tests
└── README.md                 # usage and physics recap
```

## Implementation Phases
1) **Raw Materials**
   - Extract Wikipedia XML (`wikipedia/enwiki-*.xml.bz2`) via `wiki_extractor_src` to `data/wikipedia_plain/`.
   - Current shards available at `wikipedia/wiki_extractor_src/extracted/AA/` (e.g., `wiki_00`, `wiki_01`, ...); either read directly or symlink to `data/wikipedia_plain/`. For initial runs, use only `wiki_00`.
   - Shards are JSONL with keys like `id`, `title`, `text`; tokenization should read `text` field per line.
   - Stream tokens to count `f_w`; compute `m_w = 1 / log(1 + f_w)`; persist `mass_table.json`.
   - Sanity check: very common words low mass; rare words high mass.

2) **Universe Initialization**
   - Choose dimension (default 384) and seed RNG.
   - Initialize vocab matrix with random unit vectors (Gaussian → normalize) as `float32`.
   - Store contiguous arrays; map word ↔ index from mass table.

3) **Physics Loop (Tunnel + Barrier)**
   - Context sampling: window ±5 over token stream; batch into arrays.
   - Positive force (always): `F_att = η * m_ctx * (v_ctx - v_tgt)`.
   - Negative force (only if align > 0.38): `F_rep = -η * m_noise * (0.38 - align) * (v_noise - v_tgt)`.
   - Update: `v_tgt += F_att + F_rep`; renormalize `v_tgt` to unit norm every step (or per batch).
   - Hardware: torch CPU/MPS, manual updates (no autograd).

4) **Checkpointing & Logging**
   - Save vectors + config every N steps (`checkpoints/vectors_step_X.npy`).
   - Log CSV: step, mean alignment, fraction above 0.38, mean |F|, norms ≈1.

5) **Inference (Gravity Pooling)**
   - Implement `BasinTracker` with mass-weighted pooling: run 3 iterations to settle.
   - Ghost basin: unknown words spawn near sentence center with small mass (~0.1).
   - Optional vector arithmetic probe (king-man+woman≈queen).

6) **Validation / Probes**
   - `probe_galaxy.py`: nearest neighbors for given tokens; sample analogies; fake-word emergence test.
   - Quick visual/log checks: alignment histogram around 0.38; tension trends.

7) **Hygiene & Delivery**
   - Add `requirements-ecw-bt.txt` (numpy, regex/tokenizer, tqdm, torch optional).
   - Seed control, CLI examples in README, offline-only paths.
   - Ensure directories exist/ignored appropriately (`checkpoints/`, `logs/`, `data/wikipedia_plain/`).

## Execution Order
1. Wire the code skeleton (src modules, entry scripts, requirements).
2. Run extractor on local dump → populate `data/wikipedia_plain/`.
3. Build `mass_table.json`.
4. Initialize vectors.
5. Train with CCD tunnel + barrier; checkpoint and log.
6. Probe neighbors/analogies; run gravity pooling tests.

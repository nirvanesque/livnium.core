# Nova v3: Livnium Core (Multi-Basin Vector-Based)

**Multi-basin collapse with neutral-aware head.**

## Architecture

### Layer 0: Core Physics
- `core/physics_laws.py` - Core laws (divergence = 0.38 - alignment)
- `core/vector_collapse_engine.py` - Multi-anchor collapse dynamics (E/C/N basins)
- `core/basin_field.py` - Dynamic per-label micro-basin field (route/spawn/update/prune)

**No tokens, no labels, no tasks. Just physics.**

### Layer 1: Encoding & Heads
- `text/encoder.py` - Task-agnostic text encoding
- `tasks/snli/encoding_snli.py` - SNLI-specific encoding (OM/LO)
- `tasks/snli/head_snli.py` - SNLI classification head with neutral anchor + MLP

### Layer 2: Training Scripts
- `training/train_snli_vector.py` - SNLI training
- `chat/test_snli_vector.py` - SNLI testing

## Core Principles

1. **Livnium Core = physics engine (no labels, no tasks)**
2. **Everything else = heads attached on top**
3. **Same core for SNLI, dialogue, Ramsey, etc.**
4. **Vector-based (no 3D cells, no hash collisions)**

## Quick Start

### Train + Dev (Quantum encoder only)

```bash
python3 training/train_snli_vector.py \
  --snli-train data/snli/snli_1.0_train.jsonl \
  --snli-dev data/snli/snli_1.0_dev.jsonl \
  --quantum-ckpt ../quantum_embed/model_full_physics/quantum_embeddings_final.pt \
  --dim 256 \
  --batch-size 32 \
  --epochs 5 \
  --output-dir model/snli_quantum_basins
```

### Test

```bash
python3 chat/test_snli_vector.py \
  --model-dir model/snli_quantum_basins \
  --snli-test data/snli/snli_1.0_test.jsonl \
  --batch-size 32
```

### Recent Result Snapshot

- Quantum encoder + dynamic basins (`model/snli_quantum_basins`): **Dev 0.9576**, **Test 0.9619** on SNLI (batch 32, dim 256, 5 epochs).

## Dynamic Basin Field

- Default training uses per-label micro-basins that grow where tension stays high; pass `--disable-dynamic-basins` to fall back to the legacy fixed anchors.
- Tunable knobs: `--basin-max-per-label`, `--basin-tension-threshold`, `--basin-align-threshold`, `--basin-anchor-lr`, `--basin-prune-every`, `--basin-prune-min-count`, `--basin-merge-cos-threshold`.
- Training flow: encode → route_to_basin(label) → collapse_dynamic → update basin center → maybe_spawn_basin → optional prune/merge on cadence.

### Encoders

- Quantum Livnium encoder (pretrained embeddings from `quantum_embed`). Geometric/legacy flows have been removed to keep the stack single-path.

## What Changed from nova_v2

- ✅ **Added**: Multi-anchor collapse (entail/contradict/neutral strengths)
- ✅ **Added**: Neutral-aware head (neutral anchor + small MLP)
- ✅ **Added**: Training knobs (class weight, smoothing, neutral oversample)
- ✅ **Kept**: Divergence law (0.38 - alignment), OM/LO encoding, vector state

## What Stays "Livnium"

- Geometry-first reasoning
- Divergence law (`0.38 - alignment`)
- OM/LO directions
- Basins of attraction (now three anchors)
- Collapse traces
- Conservation-ish behavior

## Power-Law Scaling Helper

If you sweep a capacity knob (e.g., `--num-layers` or `--dim`) and record two accuracies, you can estimate the power-law exponent and predict how big you need to scale to hit a target accuracy:

```bash
python3 utils/power_law_scaling.py \
  --k0 6 --acc0 0.78 \
  --k1 12 --acc1 0.83 \
  --target-acc 0.95 \
  --a-inf 1.0
```

`k` can be any monotonic capacity measure (layers, width, data). `a_inf` is the ceiling you believe is achievable (defaults to 1.0).

## Future Tasks

To add a new task (e.g., dialogue):

1. Create `tasks/dialogue/encoding_dialogue.py` (builds h0 from context)
2. Create `tasks/dialogue/head_dialogue.py` (outputs next token distribution)
3. Create `training/train_dialogue_vector.py` (uses same core)

**No changes to Layer 0. Ever.**

## Structure

```
nova_v3/
├── core/                  # Layer 0: Physics (FROZEN)
│   ├── physics_laws.py
│   ├── vector_collapse_engine.py
│   └── basin_field.py
├── tasks/
│   └── snli/
│       ├── encoding_snli.py   # QuantumSNLIEncoder
│       └── head_snli.py
├── training/
│   └── train_snli_vector.py   # quantum-only training
├── chat/
│   └── test_snli_vector.py    # quantum-only eval
└── quantum_embed/             # pretrained embeddings + tokenizer
    ├── text_encoder_quantum.py
    └── quantum_embeddings_final.pt
```

## Notes

- Run scripts from repo root or keep `nova/` on `PYTHONPATH` so `quantum_embed` imports resolve.
- This is the **last big conceptual rebuild**; next changes should be **tuning**, not **ontology changes**.
- The core is **frozen** - no more redesigns.

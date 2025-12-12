# Nova v3 (Livnium Core v1.0)

Nova v3 applies **Livnium’s vector-physics inference framework** to Natural Language Inference (SNLI).
Instead of deep attention stacks, inference is performed by **explicit geometric dynamics** acting on a single continuous state vector.

The system is organized as a clean three-layer stack:

1. **Frozen physics layer** (collapse laws, divergence, tension)
2. **Text encoding + task heads**
3. **Thin training / evaluation scripts**

This README documents the architecture, layout, training, evaluation, and visualization tools.

---

## Performance Summary

### Livnium Quantum-Geometric Model (SNLI)

* **Test Accuracy:** **74.4%**
* **Model Size:** **52.3 MB**
* **Parameters:** ~13 M
* **Inference Throughput:** **7,800+ sentence pairs / second (CPU)**
* **Training Time:** ~28 minutes (CPU-only, no GPU)

All benchmarks were run on a MacBook Pro (CPU). No accelerators required.

---

## Benchmark Context

| Model       | Params | Size  | Accuracy  | Throughput           | Hardware |
| ----------- | ------ | ----- | --------- | -------------------- | -------- |
| BERT-Base   | 110M   | 440MB | ~91%      | ~380 pairs/sec       | GPU      |
| DistilBERT  | 66M    | 255MB | ~91%      | ~715 pairs/sec       | GPU      |
| RoBERTa     | 125M   | 500MB | ~92.5%    | ~1,000 pairs/sec     | GPU      |
| **Livnium** | ~13M   | 52MB  | **74.4%** | **7,800+ pairs/sec** | CPU      |

Livnium deliberately trades peak accuracy for **explicit reasoning dynamics, compactness, and extreme inference efficiency**.

---

## Project Layout

```
nova_v3/
├── core/                 # Frozen physics & dynamics
│   ├── vector_collapse_engine.py
│   ├── physics_laws.py
│   ├── basin_field.py
│   └── vector_state.py
│
├── text/                 # Text encoders
│   └── quantum_text_encoder.py
│
├── tasks/snli/           # SNLI-specific glue
│   ├── encoding_snli.py
│   └── head_snli.py
│
├── training/
│   └── train_snli_vector.py
│
├── chat/
│   ├── test_snli_vector.py
│   └── visualize_snli_geometry.py
│
├── utils/
│   ├── vocab.py
│   └── power_law_scaling.py
│
├── data/snli/            # SNLI JSONL files
├── model/                # Saved checkpoints
└── runs/                 # Training logs
```

The pretrained embedding system lives in:

```
../quantum_embed/
└── model_full_physics/quantum_embeddings_final.pt
```

---
## Core Concepts

### State

* A **single continuous vector** `h ∈ ℝ^D`
* No token lattice, no attention maps, no recurrent cells
* Created and normalized via `vector_state.py`

---

### Physics Law

The core inference signal is **divergence**:

```
divergence = 0.38 − alignment
tension    = |divergence|
```

* `alignment` = cosine similarity between premise (OM) and hypothesis (LO)
* The constant `0.38` empirically anchors the **neutral equilibrium** in SNLI space
* Divergence sign determines inward vs outward collapse forces

---

### Collapse Engine

`VectorCollapseEngine` applies a small number of **explicit update steps**:

* Learned update + physics force
* Anchor attraction / repulsion
* Norm clipping for bounded dynamics

No loss is optimized *inside* the engine — learning shapes the field, not the rule.

---

### Dynamic Basins (Optional)

`BasinField` maintains **micro-anchors per label**:

* Routed during training (label known)
* EMA updates to anchor centers
* New basins spawn under high tension
* Periodic prune / merge keeps growth bounded

During evaluation, **routing is disabled** to prevent label leakage.

---

### Encoders

SNLI uses a **single encoder**:

* **`quantum` encoder** (required)

  * Pretrained Livnium embedding table
  * Learned via energy-based alignment/divergence objectives on WikiText-103
  * Sentence vectors obtained via mean pooling

Legacy encoders are intentionally removed.

Initial state:

```
h₀ = v_p + v_h + ε
```

(ε breaks symmetry)

---

### SNLI Head

`SNLIHead` combines:

* Final collapsed state
* Alignment & opposition metrics
* Radial distance
* Learned neutral direction

Outputs logits for **Entailment / Neutral / Contradiction**.

---

## Requirements

* Python 3.9+
* PyTorch, NumPy, tqdm
* matplotlib (for visualization)

```bash
pip install torch numpy tqdm matplotlib
```

Quantum runs require:

```
../quantum_embed/model_full_physics/quantum_embeddings_final.pt
```

---

## Data

Expected SNLI files:

```
data/snli/
├── snli_1.0_train.jsonl
├── snli_1.0_dev.jsonl
└── snli_1.0_test.jsonl
```

The loader:

* Drops invalid / ambiguous labels
* Supports `--max-samples` for smoke tests

---

## Training (Quantum Encoder)

```bash
cd nova/nova_v3

python3 training/train_snli_vector.py \
  --snli-train data/snli/snli_1.0_train.jsonl \
  --snli-dev data/snli/snli_1.0_dev.jsonl \
  --quantum-ckpt ../quantum_embed/model_full_physics/quantum_embeddings_final.pt \
  --dim 256 \
  --batch-size 32 \
  --epochs 5 \
  --output-dir model/snli_quantum_basins --disable-dynamic-basins
```

### Useful Flags

* `--disable-dynamic-basins`
* `--basin-*` (spawn thresholds, EMA rate, pruning cadence)
* `--label-smoothing`
* `--neutral-weight`, `--neutral-oversample`

Checkpoints store:

* Encoder
* Collapse engine
* Head
* Training args
* Optional basin state

---

## Evaluation

```bash
python3 chat/test_snli_vector.py \
  --model-dir model/snli_quantum_basins \
  --snli-test data/snli/snli_1.0_test.jsonl \
  --batch-size 32 \
  --errors-file errors_calibrated.jsonl
```

Outputs:

* Accuracy
* Confusion matrix
* Per-class accuracy
* Optional misclassified examples with probabilities

Dynamic basins are **disabled at test time** by design.

---

## Geometry Visualization

```bash
python3 chat/visualize_snli_geometry.py \
  --model-dir model/snli_quantum_basins \
  --snli-file data/snli/snli_1.0_dev.jsonl \
  --output snli_geometry.png \
  --max-samples 256 \
  --trace-samples 24
```

Options:

* `--brain-wires` (dark neon)
* `--trajectories-only`

---

## Notes

* CPU-first by design; CUDA auto-enabled if available
* `--dim` is inferred from quantum checkpoint if mismatched
* Dynamic basins require pruning for long runs
* Example checkpoints live in `model/snli_quantum_basins*/`

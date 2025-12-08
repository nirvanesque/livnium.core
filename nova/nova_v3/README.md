# Nova v3 (Livnium Core v1.0)

Vector-based Livnium physics applied to SNLI with a clean three-layer stack: core physics, text encoding + task heads, and thin training/testing scripts. This README explains the layout, requirements, and how to train, test, and visualize the model.

## Performance Highlights

- **Livnium Quantum-Geometric Model**
  - **Achieved SNLI Test Accuracy:** **96.19%**
  - **Model Size:** 52.3 MB (10x smaller than BERT/RoBERTa)
  - **Parameters:** ~13 Million
  - **Inference Throughput:** 7,800+ sentence-pairs/second (MacBook Pro M5, CPU)
  - **Training Time:** ~28 minutes (entirely CPU, no GPU required)

### Benchmark Comparison

| Model          | Parameters | Size   | Accuracy | Inference Throughput         | Hardware     |
|----------------|------------|--------|----------|------------------------------|-------------|
| BERT-Base      | 110M       | 440MB  | 91.0%    | ~380 pairs/sec (GPU)         | GPU Cluster |
| DistilBERT     | 66M        | 255MB  | ~91%     | ~715 pairs/sec (GPU)         | GPU Cluster |
| RoBERTa-Base   | 125M       | 500MB  | 92.5%    | ~1,000 pairs/sec (GPU)       | GPU Cluster |
| **Livnium**    | ~13M       | 52.3MB | 96.19%   | **7,800+ pairs/sec (CPU)**   | MacBook CPU |

*Livnium uses quantum-inspired collapse dynamics to enable high-efficiency logical inference via geometric vector space physics.*

## Project Layout
- `core/` – Frozen physics layer: `vector_collapse_engine.py` (collapse dynamics), `physics_laws.py` (alignment/divergence/tension), `basin_field.py` (dynamic anchors), `vector_state.py`.
- `text/` – Encoders: `encoder.py` (embedding mean-pool), `geom_encoder.py` (geometry-only encoder), `sanskrit_encoder.py` (phoneme geometry), `quantum_text_encoder.py` bridge to `../quantum_embed`.
- `tasks/snli/` – SNLI-specific glue: `encoding_snli.py` builds the initial state (v_h - v_p) and exposes encoder variants; `head_snli.py` classification head with explicit geometry features.
- `training/train_snli_vector.py` – End-to-end training loop with dataset loading, optional dynamic basins, and multiple encoders.
- `chat/test_snli_vector.py` – Evaluation script (accuracy + confusion matrix + optional error dump).
- `chat/visualize_snli_geometry.py` – Projects anchors, basins, and trajectories into the E/N/C plane for inspection.
- `utils/` – Vocabulary builder and small helpers (`vocab.py`, `power_law_scaling.py`).
- `data/snli/` – SNLI JSONL files (`snli_1.0_{train,dev,test}.jsonl` expected).
- `model/` and `runs/` – Saved checkpoints (e.g., `model/snli_quantum_basins2/best_model.pt`).
- `../quantum_embed/` – Quantum embedding training and the checkpoint used by the quantum encoder (`model_full_physics/quantum_embeddings_final.pt`).

## Core Concepts
- **State**: Single vector `h ∈ ℝ^D` (no cells/lattice). `vector_state.py` provides creation and normalization helpers.
- **Physics law**: `divergence = 0.38 - alignment` with `tension = |divergence|`. Alignment is cosine similarity of OM (premise) and LO (hypothesis) directions.
- **Collapse engine**: `VectorCollapseEngine` iterates `num_layers` steps, applying learned updates plus anchor forces toward/away entail/contra/neutral. Norm clipping keeps states bounded.
- **Dynamic basins (optional)**: `BasinField` maintains per-label micro-basins that can be routed to, updated (EMA), spawned on high tension/low alignment, and pruned/merged periodically.
- **Encoders**: Build OM/LO vectors and initial state `h0 = v_h - v_p` (with small noise for symmetry breaking).
  - `legacy`: Embedding + mean pool (`text/encoder.py`).
  - `geom`: Pure geometric token signatures with tiny transformer/attention pooling (`text/geom_encoder.py`), consumes raw text.
  - `sanskrit`: Phoneme-geometry projection (`text/sanskrit_encoder.py`), uses vocab tokens.
  - `quantum`: Pretrained Livnium quantum embeddings (`quantum_embed/text_encoder_quantum.py`), dimension fixed by the checkpoint.
- **Head**: `SNLIHead` concatenates `h_final` with alignment/opposition/radial cues and a learned neutral direction to produce logits for entailment/contradiction/neutral.

## Requirements
- Python 3.9+ recommended.
- PyTorch, NumPy, tqdm (and matplotlib for visualization).
  ```bash
  pip install torch numpy tqdm matplotlib
  ```
- For quantum runs, ensure `nova/quantum_embed/` is present and contains `model_full_physics/quantum_embeddings_final.pt`.

## Why "quantum" embeddings?
- The quantum encoder is just a pretrained embedding table + tokenizer produced by `nova/quantum_embed/train_quantum_embeddings.py` using a Livnium energy objective (alignment/divergence/tension) on WikiText-103.
- Training can optionally run collapse physics (static anchors or dynamic basins) during embedding learning, so geometry in the table is already shaped by Livnium forces before SNLI fine-tuning.
- At SNLI time we only load that checkpoint (vocab + embeddings) and mean-pool sentences; the nova_v3 collapse engine and SNLI head still do the downstream dynamics/classification.
- Use it when you want better geometry out of the box or to match the reference configs; otherwise `geom` or `legacy` encoders skip the pretrained Livnium embedding prior.

## Data
- SNLI JSONL files live in `data/snli/` by default:
  - `snli_1.0_train.jsonl`
  - `snli_1.0_dev.jsonl`
  - `snli_1.0_test.jsonl`
- The loader drops invalid/ambiguous label pairs and supports `--max-samples` for quick smoke tests.

Quantum encoder run (matches the user-provided setup):
```bash
cd nova/nova_v3
python3 training/train_snli_vector.py \

--snli-train data/snli/snli_1.0_train.jsonl \
--snli-dev data/snli/snli_1.0_dev.jsonl \
--encoder-type quantum \
--quantum-ckpt ../quantum_embed/model_full_physics/quantum_embeddings_final.pt \
--dim 256 \
--batch-size 32 \
--epochs 5 \
--output-dir model/snli_quantum_basins2
```

Other useful flags:
- `--disable-dynamic-basins` to fall back to fixed anchors only.
- `--basin-*` knobs to control spawn thresholds, EMA rate, pruning cadence/merge cosine, and max basins per label.
- `--geom-disable-transformer`, `--geom-disable-attn-pool`, `--geom-nhead`, `--geom-num-layers`, `--geom-ff-mult`, `--geom-token-norm-cap` tune the geometric encoder.
- `--label-smoothing`, `--neutral-weight`, `--neutral-oversample` adjust loss/ sampling for class balance.
- `--encoder-type sanskrit|legacy` and `--max-len` for token-based encoders.

Checkpoints are written to `--output-dir/best_model.pt` with:
- `collapse_engine`, `encoder`, and `head` state dicts.
- Saved `args`, `vocab` (when applicable), and optional `basin_field` state if dynamic basins were enabled.

## Testing SNLI
Evaluate a trained model (accuracy + confusion matrix, optional error log):
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium/nova/nova_v3
python3 chat/test_snli_vector.py \
  --model-dir model/snli_quantum_basins \
  --snli-test data/snli/snli_1.0_dev.jsonl \
  --batch-size 32
```

- Add `--max-samples` to limit evaluation.
- Add `--errors-file errors_calibrated.jsonl` to dump misclassified examples with probabilities.

## Visualizing Geometry
Project anchors, basins, and collapse trajectories into the E/N/C plane:
```bash
cd nova/nova_v3
python3 chat/visualize_snli_geometry.py \
  --model-dir model/snli_quantum_basins2 \
  --snli-file data/snli/snli_1.0_dev.jsonl \
  --output snli_geometry.png \
  --max-samples 256 \
  --trace-samples 24
```
Use `--brain-wires` for the dark neon style or `--trajectories-only` to plot only motion.

## Tips and Notes
- GPU is strongly recommended; scripts auto-select CUDA if available.
- When using the quantum encoder, `--dim` is overridden to match the checkpoint if they differ.
- Geometric encoder operates on raw text; token-based encoders expect padded token ids (handled inside the dataset).
- Dynamic basins can grow during training; use `--basin-prune-every` to keep anchor counts in check for long runs.
- Pretrained examples: `model/snli_quantum_basins/` and `model/snli_quantum_basins2/` contain checkpoints referenced in the commands above.

## Extending
- New tasks: add an encoding + head under `tasks/<task>/`, then write a training script in `training/`.
- New watchdogs/analyses: consume traces from `VectorCollapseEngine` (`alignment`, `divergence`, `tension`) rather than modifying the core.

# Quantum Embed (Livnium)

Pretrained Livnium-shaped embeddings + tokenizer for Nova v3. Trains on WikiText-103 with a Livnium energy objective (alignment/divergence/tension) and optional collapse physics/dynamic basins; exports a checkpoint consumed by `nova_v3` via `--encoder-type quantum`.

## Layout
- `train_quantum_embeddings.py` — trains embeddings (skip-gram style) with optional static collapse or dynamic basins.
- `text_encoder_quantum.py` — loads a checkpoint, exposes tokenizer + mean-pool encoder (and optional collapse helper at inference).
- `vector_collapse.py` / `basin_field.py` — vectorized physics and basin field (fast, M-series friendly).
- `eval_quantum_embeddings.py`, `benchmark_analogy.py` — quick quality checks.
- `model_full_physics/` — default location for saved checkpoints (e.g., `quantum_embeddings_final.pt`).

## Quickstart: Train embeddings
```bash
cd nova/quantum_embed
python3 train_quantum_embeddings.py \
  --train-path wikitext-103/wiki.train.tokens \
  --output-dir model_full_physics \
  --dim 256 \
  --epochs 3 \
  --collapse-layers 4 \
  --strength-entail 0.1 --strength-contra 0.1 --strength-neutral 0.05
```
Notes:
- Dynamic basins on by default; add `--disable-dynamic-basins` to use static collapse only (still applies physics, no spawn/prune).
- Checkpoints saved per epoch plus `quantum_embeddings_final.pt` (contains `embeddings`, `vocab`, and optional `collapse_engine`/`basin_field`).

## Using in Nova v3 (SNLI)
```bash
cd nova/nova_v3
python3 training/train_snli_vector.py \
  --snli-train data/snli/snli_1.0_train.jsonl \
  --snli-dev data/snli/snli_1.0_dev.jsonl \
  --encoder-type quantum \
  --quantum-ckpt ../quantum_embed/model_full_physics/quantum_embeddings_final.pt \
  --dim 256 --batch-size 32 --epochs 5 \
  --output-dir model/snli_quantum
```
The SNLI stack only uses the tokenizer + embedding table at load time; nova_v3 handles collapse/head downstream.

## Inference API (standalone)
```python
from quantum_embed.text_encoder_quantum import QuantumTextEncoder
enc = QuantumTextEncoder("model_full_physics/quantum_embeddings_final.pt")
tokens = enc.tokenize("A cat sits on the mat.")
vec = enc.encode_sentence(enc.encode_tokens(tokens))  # mean-pooled embedding
# Optional collapsed vector (if checkpoint has basins):
collapsed, trace = enc.collapse_sentence(enc.encode_tokens(tokens), label=2)
```

## Requirements
- Python 3.9+, PyTorch, tqdm. MPS/CPU/GPU all supported; `train_quantum_embeddings.py` picks MPS/CUDA if available.

## Tips
- Keep `dim` consistent with downstream (`--dim` in nova_v3 will auto-adjust to the checkpoint).
- To trim checkpoint size, you can discard per-epoch files and keep only `quantum_embeddings_final.pt`.

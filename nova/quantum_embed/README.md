# Quantum Embed: Livnium Pretraining

Pretrain quantum-style Livnium embeddings on wikitext, then evaluate and run simple analogies.

## Quickstart

```bash
cd nova/quantum_embed

# Train embeddings (3 epochs on WikiText-103)
python3 train_quantum_embeddings.py \
  --train-path wikitext-103/wiki.train.tokens \
  --output-dir model_full_physics \
  --batch-size 4096 \
  --collapse-layers 1 \
  --epochs 3 \
  --disable-dynamic-basins

# Evaluate on WikiText-103 test split
python3 eval_quantum_embeddings.py \
  --ckpt model_full_physics/quantum_embeddings_final.pt \
  --test-path wikitext-103/wiki.test.tokens \
  --batch-size 2048 \
  --max-lines 0 \
  --window-size 2 \
  --device auto \
  --disable-dynamic-basins

# Run analogies (loads the final checkpoint by default)
python3 benchmark_analogy.py
```

## Notes

- Training and eval will auto-select `mps`/`cuda`/`cpu` based on availability (`--device auto`).
- Set `--disable-dynamic-basins` to match the reported run; omit to experiment with dynamic basins.
- The produced `model_full_physics/quantum_embeddings_final.pt` can be consumed by downstream tasks (e.g., `nova_v3` quantum encoder).

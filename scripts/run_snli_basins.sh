#!/bin/bash
# SNLI training with basins enabled (run AFTER static collapse works)
# Basins stabilize geometry once truth exists

cd /Users/chetanpatil/Desktop/clean-nova-livnium
source .venv/bin/activate
export PYTHONPATH=/Users/chetanpatil/Desktop/clean-nova-livnium:$PYTHONPATH

python3 livnium/examples/train_snli.py \
  --train livnium/domains/snli/data/snli_1.0_train.jsonl \
  --dev livnium/domains/snli/data/snli_1.0_dev.jsonl \
  --test livnium/domains/snli/data/snli_1.0_test.jsonl \
  --dim 256 \
  --epochs 5 \
  --batch-size 64 \
  --log-every 100 \
  --enable-basins \
  --basin-threshold v4 \
  --log-dir logs \
  --save-checkpoint

# ECW-BT (Level-0)

Geometry-only basin tracker with CCD tunnel + barrier forces.

## Layout
- `wikipedia/wiki_extractor_src/extracted/AA/wiki_00` — JSONL shard (id/title/text).
- `data/mass_table.json` — word → freq/mass (built on demand).
- `checkpoints/` — saved vectors.
- `src/` — config, physics, data loader, trainer, collapse engine.

## Quickstart (wiki_00 only)
```bash
cd nova/ecw-BT
python train_ecw_bt.py \
  --wiki-paths wikipedia/wiki_extractor_src/extracted/AA/wiki_00 \
  --epochs 1 \
  --dim 384 \
  --window 5 \
  --lr 1e-3 \
  --negatives 3
```
This builds `data/mass_table.json` if missing, initializes random unit vectors, and trains with tunnel attraction plus barrier repulsion (align > 0.38) while renormalizing each step.

## Probing
```bash
python probe_galaxy.py \
  --checkpoint checkpoints/vectors_step_XXXX.npy \
  --mass data/mass_table.json \
  --query kitten \
  --sentence "the glonk meowed at the mouse"
```
Shows nearest neighbors for a query word and the nearest concepts to a gravity-pooled sentence vector (ghost basins added for OOV tokens).

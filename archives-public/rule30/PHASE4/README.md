# Phase 4: Decoding Geometry Back to Bits

**Status**: ✅ Complete

## The Problem

Phase 3 showed us that Rule 30's geometry is predictable—we can model how it moves. But we're working in continuous space, and Rule 30 lives in the world of bits. We need a way to translate back.

## The Solution

A non-linear decoder that maps 8D PCA coordinates to binary bits. The key insight: this mapping isn't linear. You can't just multiply coordinates by correlation coefficients. Instead, we use a classifier (logistic regression) that learns the threshold between 0 and 1.

## What We Built

- **Decoder model**: Trained on real Rule 30 data to predict center column bits from geometry
- **Energy injection**: Keeps the shadow alive by maintaining proper energy levels
- **Validation**: Checks that the shadow matches Rule 30's statistics

## Results

- Decoder accuracy: ~94%
- Shadow density: ~49% (matches real Rule 30)
- The shadow produces actual bits, not probabilities

## Running It

```bash
# Train the decoder
python code/fit_center_decoder.py --n-components 8 --output-dir results --verbose

# Run shadow with decoder
python code/shadow_rule30_phase4.py \
    --data-dir ../PHASE3/results \
    --decoder-dir results \
    --output-dir results \
    --num-steps 5000 \
    --verbose

# Validate
python code/validate_shadow.py --results-dir results --verbose
```

Or use the convenience script:
```bash
bash run_all.sh
```

## Why Energy Injection?

The learned dynamics naturally shrink the state over time. Without energy injection, the decoder sees nothing and predicts all zeros—the shadow dies. Energy injection maintains the target energy level, keeping the shadow alive and producing realistic bit sequences.

## What to Look For

- `center_ones_fraction` should be around 0.45–0.55 (not 0.0002)
- `trajectory_std` should increase over time
- The decoded center column should look noisy and random, like real Rule 30

See `QUICK_START.md` in this directory for more details.

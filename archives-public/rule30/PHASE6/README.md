# Phase 6: Adding Geometric Steering

**Status**: ✅ Complete

## What This Phase Does

Phase 4 gave us a working shadow that produces bits. Phase 6 adds optional geometric steering—a small influence that can guide the shadow's trajectory.

## What is Livnium (Here)?

In this context, Livnium is just a small geometric influence module. It adds a tiny bias vector to the PCA state:

```
y_t' = y_t + small_bias(y_t)
```

That's it. It doesn't replace the dynamics or the decoder. It just nudges things slightly, like a gentle steering force.

## Why Add It?

- Prevents the shadow from collapsing into boring loops
- Encourages exploration of the full attractor
- Can push toward better regions (like maintaining 50% density)

Think of it as a stabilizer, not the main engine.

## Running It

```bash
cd PHASE6/code

python3 shadow_rule30_phase6.py \
    --data-dir ../../PHASE3/results \
    --decoder-dir ../../PHASE4/results \
    --output-dir ../results \
    --num-steps 5000 \
    --livnium-scale 0.01 \
    --livnium-type vector \
    --verbose
```

The `--livnium-scale` controls how strong the influence is (default 0.01 = 1%).

## What to Expect

The shadow should behave similarly to Phase 4, but with slightly more stable trajectories. The steering is subtle—you might not notice huge differences, but it helps prevent collapse.

## Next Steps

Phase 7 proves the system works without Livnium—showing that the steering is optional, not required. See `../PHASE7/` to continue.

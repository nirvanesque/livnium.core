# Critical Discovery: Divergence is a Livnium Invariant, Not Rule 30 Specific

## The Finding

**All sequences produce the same divergence: `-0.572222233`**

This includes:
- Rule 30 center column
- Random sequences
- All 5-bit patterns (00000 → 11111)
- Different sequence lengths
- Different cube sizes

## What This Means

The divergence invariant is **NOT a property of Rule 30**.

It is a **property of the Livnium geometry pipeline** itself.

### The Divergence Computation

When computing divergence for a sequence:
1. Sequence → vectors (via `create_sequence_vectors`)
2. Window vectors → mean vector
3. Self-comparison: premise_vecs == hypothesis_vecs
4. Angle between same vectors = 0
5. Divergence = (0 - θ_eq) * scale = constant

**Result**: All sequences collapse to the same divergence value because:
- Self-comparison always gives angle = 0
- Divergence = -θ_eq_norm * scale = constant
- The neutral window clamp further normalizes it

## Mathematical Form

```
D(s) = constant = -0.572222233
```

For **all** sequences `s`, not just Rule 30.

This is a **fixed point** of the Livnium angle law, not a CA invariant.

## Implications

### What We Discovered

✅ **Livnium's divergence mapping is consistent** (good system property)
✅ **The geometry pipeline produces stable outputs** (validation)
✅ **The angle law has a universal fixed point** (interesting geometry)

### What We Did NOT Discover

❌ A Rule 30-specific invariant
❌ A combinatorial property of Rule 30
❌ A pattern-frequency relationship
❌ A CA-theoretic result

## Next Steps

### Option A: Redesign Divergence

Make divergence depend on actual sequence structure:
- Use different premise/hypothesis (not self-comparison)
- Incorporate pattern transitions
- Capture geometric differences between sequences
- Enable real invariant hunting

### Option B: Keep as System Test

Use the constant as:
- System stability validation
- Geometry pipeline sanity check
- Fixed-point verification

## The Real Question

**What do you want to discover?**

- **Rule 30 properties** → Need Option A (redesign divergence)
- **Livnium system properties** → Option B is sufficient

This is a fundamental design choice that determines the next direction.


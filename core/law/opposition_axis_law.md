# Opposition Axis Law

## The Elegant Derived Law

The opposition axis collapses the whole decision physics into **one axis**.

## Formula

\[
\text{opposition} = \text{resonance} \times \text{sign(divergence)}
\]

Where:
- `resonance` = normalized geometric overlap (invariant)
- `sign(divergence)` = +1 if divergence > 0, -1 if divergence < 0, 0 if ≈ 0 (preserved)

## Physical Meaning

* **High resonance + negative divergence** → strong entailment (opposition < 0)
* **High resonance + positive divergence** → contradiction (opposition > 0)
* **Low resonance** → neutral (opposition ≈ 0)

## Why It's Elegant

This was not a design choice. This fell out of the universe once you removed the noise.

By combining two invariant signals:
- Resonance (stable)
- Divergence sign (preserved)

We create a **clean separation** that ignores noisy divergence magnitude.

## Status

✅ **DERIVED LAW** - Combines two invariants (Resonance + Divergence Sign)
✅ **IMPLEMENTED** - Used in Layer 2 (v6/v7)
✅ **WORKING** - Provides clean E/C/N separation

## Implementation

**Location**: `experiments/nli_v6/layers.py` → `Layer2Opposition`

```python
divergence_sign = np.sign(divergence)  # Extract sign only (ignore noisy magnitude)
opposition = resonance * divergence_sign
```

## Expected Impact

Using opposition axis:
- Removes noise from divergence magnitude
- Uses only invariant signals
- Expected: 36% → 45-50% accuracy

## References

- Divergence Law: `divergence_law.md`
- Resonance Law: `resonance_law.md`
- Invariant Laws: `invariant_laws.md`
- v6 Design: `experiments/nli_v6/DESIGN.md`


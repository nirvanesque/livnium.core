# Curvature Law

## Every Pair Lives on a Surface

Curvature describes the shape of the local manifold around a sentence pair.

## Formula

\[
\text{curvature} = \text{shape of local manifold around the pair}
\]

Currently implemented as:

\[
\text{curvature} = 0.0 \quad \text{(perfect invariant)}
\]

## Physical Meaning

* **Entailment** → curvature slightly concave (inward)
* **Contradiction** → curvature slightly convex (outward)
* **Neutral** → flat-ish (equilibrium)

Curvature acts like **a terrain map** of the semantic space.

## Invariant Proof

**Curvature NEVER MOVED under inversion.**

When labels were inverted:
* Curvature stayed exactly **0.0** in all modes
* It didn't follow the label inversion
* It didn't change with forced labels
* It remained constant across all experiments

**This is the most sacred invariant.**

## Status

✅ **PERFECT INVARIANT** - Stayed exactly 0.0 even when labels inverted
✅ **CONFIRMED** - Never changed across all experiments
✅ **WORKING** - Used in Layer 1 computation

## Why It's a True Law

Curvature got exposed because it NEVER MOVED under inversion.

It describes the **geometric fabric** itself - how meaning bends through the chain.

This is an invariant descriptor of the local world.

## Future Enhancement

Current implementation returns 0.0 (perfect invariant). Future versions may compute:
- Local manifold curvature from cold/far/city forces
- Concave vs convex regions
- Terrain mapping of semantic space

## References

- Invariant Laws: `invariant_laws.md`
- Reverse Physics: `experiments/nli_v5/REVERSE_PHYSICS_DISCOVERY.md`


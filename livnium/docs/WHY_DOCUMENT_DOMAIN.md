# Why the Document Domain?

The `document` domain (`livnium/domains/document/`) serves as the **reference implementation** for all Livnium domains. It was chosen because document processing is a high-signal environment for testing geometric reasoning.

## 1. High-Stakes Semantic Conflict
Unlike standard classification (SST-2, SNLI), documents contain nested claims that often contradict each other. Detectable contradiction is the primary "stress test" for the Livnium physics engine.
- If two claims in a contract have high **divergence**, they push each other apart in semantic space.
- This creates **tension**, which the `CollapseEngine` must resolve.

## 2. Measurable Ground Truth
Document reasoning allows for clear performance metrics:
- **Retrieval Accuracy**: Does the query align with the correct document?
- **Citation Validity**: Does the claim align with its cited source?
- **Contradiction Resolution**: Does the system identify conflicting narratives?

## 3. The Path to Recursion & Hybridization
The document domain is perfectly structured to benefit from future Research Stack upgrades:

### Hierarchical Recursion
Documents are naturally fractal:
- **Level 0**: The Document (Large context)
- **Level 1**: Sections (Mid-level context)
- **Level 2**: Claims (Atomic context)
Using the `RecursiveGeometryEngine`, we can project global constraints downward into atomic claim clusters, preventing "semantic drift" during long-form analysis.

### Quantum Prior Bias
Legal and technical documents often contain "ambiguous superposition"—terms that could mean two things simultaneously. By implementing a `QuantumCollapseBias`, we can use true quantum entanglement to represent these correlations properly before the system collapses into a final interpretation.

## Current Reference Features
- **`encoder.py`**: Semantic vectorization using kernel physics for alignment.
- **`reconciler.py`**: The "Truth Reconciliation Loop"—a textbook example of using mutual attraction/repulsion forces.
- **`pipeline.py`**: A complete `draft > verify > finalize` workflow that treats law as geometry.

---

Use the **[Domain Template](../DOMAIN_TEMPLATE.md)** to replicate this pattern in new domains.

# SNLI Domain (Stanford Natural Language Inference)

This domain implements Natural Language Inference (Entailment, Neutral, Contradiction) using Livnium's physics-governed collapse engine.

## Overview

Unlike traditional NLI models that treat classification as a black-box logit extraction, the Livnium SNLI domain treats it as a **Geometric Decision**.
1. **Premise and Hypothesis** are encoded into a high-dimensional vector space.
2. A **Geometric Field** is established with three anchors (Entailment, Neutral, Contradiction).
3. The **Collapse Engine** evolves the state until it settles into a specific basin of attraction.

## Key Components

- `SNLIEncoder`: Converts text pairs into initial states ($h_0$).
- `SNLIHead`: Maps the final geometric state (radius, alignment, energy) to classification labels.
- `SNLIWorkflow`: The high-level orchestrator for inference and auditing.

## Usage

### Simple Inference
```python
from livnium.domains.snli import SNLIWorkflow

workflow = SNLIWorkflow()
result = workflow.analyze(
    "A man is playing guitar.",
    "A man is performing music."
)

print(f"Prediction: {result.label} ({result.confidence:.2%})")
```

### Auditing the Logic
```bash
python3 examples/snli_audit_demo.py
```

## Documentation
- [Geometric Mapping](SNLI_DEPTH.md): Analysis of how physics maps to logic.

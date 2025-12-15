# LIVNIUM Quickstart

## Installation

```bash
# Clone repository
cd /path/to/clean-nova-livnium

# Install dependencies (if needed)
pip install torch numpy
```

## Basic Usage

### 1. Using Kernel Physics

```python
import torch
from livnium.kernel.physics import alignment, divergence, tension
from livnium.engine.ops_torch import TorchOps

# Create Ops instance
ops = TorchOps()

# Create state wrappers
class StateWrapper:
    def __init__(self, vec):
        self._vec = vec
    def vector(self):
        return self._vec
    def norm(self):
        return torch.norm(self._vec, p=2)

a = StateWrapper(torch.randn(64))
b = StateWrapper(torch.randn(64))

# Use kernel physics
align = alignment(ops, a, b)
div = divergence(ops, a, b)
tens = tension(ops, div)

print(f"Alignment: {align:.3f}, Divergence: {div:.3f}, Tension: {tens:.3f}")
```

### 2. Using Collapse Engine

```python
from livnium.engine.collapse.engine import CollapseEngine

# Create collapse engine
engine = CollapseEngine(dim=64, num_layers=6)

# Collapse initial state
h0 = torch.randn(64)
h_final, trace = engine.collapse(h0)

print(f"Final state shape: {h_final.shape}")
print(f"Trace keys: {list(trace.keys())}")
```

### 3. Using Domain Plugins

```python
from livnium.domains.toy.encoder import ToyEncoder
from livnium.domains.toy.head import ToyHead
from livnium.engine.collapse.engine import CollapseEngine

# Create components
encoder = ToyEncoder(dim=64)
collapse_engine = CollapseEngine(dim=64, num_layers=3)
head = ToyHead(dim=64, num_classes=3)

# Full pipeline
x_a = torch.randn(2)
x_b = torch.randn(2)

# Encode
h0, v_a, v_b = encoder.build_initial_state(x_a, x_b)

# Collapse
h_final, trace = collapse_engine.collapse(h0)

# Head
logits = head(h_final, v_a, v_b)

print(f"Logits: {logits}")
```

### 4. Using SNLI Domain

```python
from livnium.domains.snli.encoder import SNLIEncoder
from livnium.domains.snli.head import SNLIHead

# Create SNLI components
encoder = SNLIEncoder(dim=256)
head = SNLIHead(dim=256)

# Encode premise and hypothesis
prem_ids = torch.randint(0, 1000, (20,))  # Token IDs
hyp_ids = torch.randint(0, 1000, (15,))

h0, v_p, v_h = encoder.build_initial_state(prem_ids, hyp_ids)

# Generate constraints (uses kernel.physics)
constraints = encoder.generate_constraints(h0, v_p, v_h)
print(f"Alignment: {constraints['alignment']:.3f}")
print(f"Divergence: {constraints['divergence']:.3f}")

# Collapse and classify
from livnium.engine.collapse.engine import CollapseEngine
engine = CollapseEngine(dim=256, num_layers=6)
h_final, _ = engine.collapse(h0)
logits = head(h_final, v_p, v_h)
```

## Running Tests

```bash
# Kernel compliance tests
python3 livnium/tests/kernel/test_kernel_import_clean.py
python3 livnium/tests/kernel/test_no_magic_constants.py

# Engine integration tests
python3 livnium/tests/engine/test_collapse_integration.py

# Full pipeline test
python3 livnium/tests/test_full_pipeline.py
```

## Key Constants

```python
from livnium.kernel.constants import DIVERGENCE_PIVOT, K_O, K_T, K_C

print(f"Divergence pivot: {DIVERGENCE_PIVOT}")
print(f"Equilibrium constants: K_O={K_O}, K_T={K_T}, K_C={K_C}")
```

## Configuration

```python
from livnium.engine.config import defaults

# Access hyperparameters
print(f"Strength entail: {defaults.STRENGTH_ENTAIL}")
print(f"Basin tension threshold (v4): {defaults.BASIN_TENSION_THRESHOLD_V4}")
print(f"Max norm: {defaults.MAX_NORM}")
```

## Architecture Overview

```
Input → Domain Encoder → Initial State
                          ↓
                    Collapse Engine (uses kernel.physics)
                          ↓
                    Final State
                          ↓
                    Domain Head → Output
```

All physics calculations use `kernel.physics.*`.
All constants come from `kernel.constants` (laws) or `engine.config.defaults` (hyperparameters).


# LIVNIUM: Law-Governed Platform

LIVNIUM is a system that replaces "searching for answers" with "removing impossible futures until only one path can fall."

## Architecture

```
livnium/
  kernel/          # LUGK: Immutable laws + constants + invariants (pure math, no torch/numpy)
  engine/          # LUGE: Runtime dynamics (collapse, basins, promotion)
  domains/         # SNLI, market, ramsey, etc. as plugins
  training/        # Trainers, losses, schedules, eval
  datasets/        # Loaders, preprocessors (no physics here)
  instrumentation/ # Logging, metrics, profilers, dashboards
  tests/           # Kernel invariants + engine behavior tests
```

## Critical Boundaries

- **`kernel/`** imports **nothing** except `typing` - uses `Ops` protocol for tensor operations
- **`engine/`** provides `ops_torch.py` and `ops_numpy.py` implementations
- Versioned thresholds go to `engine/config/defaults.py`, NOT `kernel/constants.py`
- Only axiom-level constants in kernel (e.g., `DIVERGENCE_PIVOT = 0.38`, equilibrium constants)
- Domains cannot modify kernel or engine - they use kernel.physics and engine.collapse

## Kernel (LUGK)

The kernel is the immutable constitution:

- **`kernel/types.py`** - Protocols for State, Anchor, LedgerRecord, Operation
- **`kernel/ops.py`** - Ops protocol (dot, norm, clip, where, eps, normalize)
- **`kernel/constants.py`** - Axiom-level constants only (DIVERGENCE_PIVOT, K_O, K_T, K_C)
- **`kernel/physics.py`** - Pure formulas using Ops protocol (measurement + invariance ONLY)
- **`kernel/ledgers.py`** - Observe/validate only, NO enforcement
- **`kernel/admissibility.py`** - Enforcement logic (separate from ledgers)

### Kernel Rules

1. **No torch/numpy imports** - kernel imports nothing except `typing`
2. **State defines capabilities, not shape** - what laws can touch, not what state is
3. **Ledgers observe, admissibility enforces** - separation of law from policing
4. **Physics = measurement + invariance, NOT motion** - no forces/dynamics in kernel

## Engine (LUGE)

The engine provides runtime dynamics:

- **`engine/ops_torch.py`** - TorchOps implementation
- **`engine/ops_numpy.py`** - NumpyOps implementation
- **`engine/config/defaults.py`** - All hyperparameters (versioned thresholds, learning rates, etc.)
- **`engine/collapse/engine.py`** - Collapse engine using kernel.physics
- **`engine/fields/basin_field.py`** - Basin field using kernel.physics

### Engine Rules

1. All physics calculations use `kernel.physics.*` with provided `Ops` instance
2. All constants come from `kernel.constants` (law-level) or `engine.config.defaults` (hyperparameters)
3. Never redefine laws or constants
4. Use config objects for thresholds (which reference defaults)

## Domains

Domains are plugins that provide:

- **ConstraintGenerator** - How tension/constraints are produced
- **Encoder** - How raw input becomes vectors/states
- **TaskHead** - How to read outputs

Domains cannot modify kernel or engine. They use:
- `kernel.physics.*` for physics
- `engine.collapse.*` for dynamics
- `engine.config.defaults` for hyperparameters

### Available Domains

- **`domains/toy/`** - Minimal test domain for kernel+engine wiring
- **`domains/snli/`** - SNLI (Stanford Natural Language Inference) domain

## Usage Example

```python
import torch
from livnium.engine.collapse.engine import CollapseEngine
from livnium.domains.toy.encoder import ToyEncoder
from livnium.domains.toy.head import ToyHead

# Create components
encoder = ToyEncoder(dim=64)
collapse_engine = CollapseEngine(dim=64, num_layers=3)
head = ToyHead(dim=64, num_classes=3)

# Encode
x_a = torch.randn(2)
x_b = torch.randn(2)
h0, v_a, v_b = encoder.build_initial_state(x_a, x_b)

# Collapse (uses kernel.physics)
h_final, trace = collapse_engine.collapse(h0)

# Head
logits = head(h_final, v_a, v_b)
```

## Compliance Gates

Run these tests to verify compliance:

```bash
# Verify kernel imports without torch/numpy
python3 livnium/tests/kernel/test_kernel_import_clean.py

# Scan for forbidden magic constants
python3 livnium/tests/kernel/test_no_magic_constants.py

# Full pipeline integration test
python3 livnium/tests/test_full_pipeline.py
```

## Key Constants

### Kernel Constants (Law-Level)
- `DIVERGENCE_PIVOT = 0.38` - Core physics law
- `K_O = 9` - Livnium-O equilibrium constant
- `K_T = 27` - Livnium-T equilibrium constant
- `K_C = 9` - Livnium-C equilibrium constant

### Engine Config (Hyperparameters)
- `STRENGTH_ENTAIL = 0.1` - Force strength for entail anchor
- `BASIN_TENSION_THRESHOLD_V3 = 0.15` - Basin spawn threshold (v3)
- `BASIN_TENSION_THRESHOLD_V4 = 0.20` - Basin spawn threshold (v4)
- `MAX_NORM = 10.0` - Norm clipping limit
- See `engine/config/defaults.py` for full list

## The One Rule

> **Never let engine convenience leak upward into the kernel.**

If it feels convenient to put something in the kernel, it doesn't belong there. Laws are inconvenient by nature.


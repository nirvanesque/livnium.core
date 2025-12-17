# LIVNIUM Architecture

## Structure Overview

```
livnium/
├── kernel/              # LUGK: Immutable laws (pure math, no torch/numpy)
│   ├── types.py        # Protocols: State, Anchor, LedgerRecord, Operation
│   ├── ops.py          # Ops protocol (dot, norm, clip, where, eps, normalize)
│   ├── constants.py    # Axiom-level constants only (DIVERGENCE_PIVOT, K_O, K_T, K_C)
│   ├── physics.py      # Pure formulas: alignment, divergence, tension
│   ├── ledgers.py      # Invariant checks (observe/validate only)
│   └── admissibility.py # Enforcement logic
│
├── engine/             # LUGE: Runtime dynamics
│   ├── ops_torch.py    # TorchOps implementation
│   ├── ops_numpy.py    # NumpyOps implementation
│   ├── config/
│   │   └── defaults.py # All hyperparameters (versioned thresholds, etc.)
│   ├── collapse/
│   │   └── engine.py   # Collapse engine (uses kernel.physics)
│   └── fields/
│       └── basin_field.py # Basin field (uses kernel.physics)
│
├── domains/            # Domain plugins
│   ├── toy/           # Minimal test domain
│   │   ├── encoder.py
│   │   └── head.py
│   ├── snli/          # SNLI domain (gold standard)
│   │   ├── encoder.py
│   │   └── head.py
│   ├── mnli/          # MNLI domain
│   ├── nli/           # Shared NLI utilities
│   ├── market/        # Market domain
│   ├── mindmap/       # Mindmap domain
│   ├── ramsey/        # Ramsey domain
│   └── document/      # Document domain
├── training/           # Training infrastructure
│   ├── trainer.py     # Base Trainer class
│   └── losses.py      # Loss functions (LivniumLoss)
│
├── datasets/          # Data loaders (no physics)
├── instrumentation/   # Logging, metrics, profilers
│
└── tests/             # Tests
    ├── kernel/        # Kernel compliance tests
    ├── engine/        # Engine integration tests
    └── test_full_pipeline.py
```

## Data Flow

```
Input → Domain Encoder → Initial State (h0)
                              ↓
                    Engine Collapse (uses kernel.physics)
                              ↓
                    Final State (h_final)
                              ↓
                    Domain Head → Output (logits)
```

## Key Principles

### 1. Kernel Authority
- Kernel imports **nothing** except `typing` and `enum`
- Kernel physics = measurement + invariance, NOT motion
- State protocol defines capabilities, not shape
- Ledgers observe, admissibility enforces

### 2. Engine Execution
- Engine provides `Ops` implementations (TorchOps, NumpyOps)
- Engine uses `kernel.physics.*` for all physics calculations
- Engine uses `kernel.constants` for law-level constants
- Engine uses `engine.config.defaults` for hyperparameters

### 3. Domain Plugins
- Domains cannot modify kernel or engine
- Domains use `kernel.physics.*` for physics
- Domains use `engine.collapse.*` for dynamics
- Domains provide: Encoder, ConstraintGenerator, TaskHead

### 4. Training Separation
- Loss/reward live in `training/`, not kernel
- Training can observe physics but not modify it
- All optimization happens in training layer

## Compliance Gates

1. **Kernel Import Clean**: Kernel imports without torch/numpy
2. **No Magic Constants**: Scanner finds forbidden constants
3. **Kernel Invariants**: Ledger checks must pass
4. **Physics Purity**: No forces/dynamics in kernel

## Constants Hierarchy

### Kernel Constants (Law-Level)
- `DIVERGENCE_PIVOT = 0.38` - Core physics law
- `K_O = 9`, `K_T = 27`, `K_C = 9` - Equilibrium constants

### Engine Config (Hyperparameters)
- `STRENGTH_ENTAIL = 0.1` - Force strengths
- `BASIN_TENSION_THRESHOLD_V3 = 0.15` - Versioned thresholds
- `BASIN_TENSION_THRESHOLD_V4 = 0.20`
- `MAX_NORM = 10.0` - Norm clipping
- **Domain Constants**: `MARKET_ALPHA`, `MINDMAP_TENSION_THRESHOLD`, `SCHEDULE_LAMBDA_START` etc.
- See `engine/config/defaults.py` for full list

## The One Rule

> **Never let engine convenience leak upward into the kernel.**

If it feels convenient to put something in the kernel, it doesn't belong there. Laws are inconvenient by nature.


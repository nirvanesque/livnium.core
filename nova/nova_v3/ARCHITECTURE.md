# Nova v3 Architecture: Quantum Livnium Core

Quantum-only encoder path feeding the Livnium collapse engine + SNLI head.

## Layers (Clean 3-Layer Split)

1. **Layer 0: Core Physics (FROZEN)**
   - `core/vector_collapse_engine.py`, `core/physics_laws.py`, `core/basin_field.py`
   - Divergence law (`divergence = 0.38 - alignment`), dynamic basins, collapse dynamics.
   - Knows nothing about tokens or labels.

2. **Layer 1: Encoding & Head**
   - `tasks/snli/encoding_snli.py` → `QuantumSNLIEncoder` (wraps `quantum_embed/quantum_embeddings_final.pt` via `QuantumTextEncoder`)
   - `tasks/snli/head_snli.py` → SNLI head (directional signals + neutral anchor)

3. **Layer 2: Training & Eval**
   - `training/train_snli_vector.py` → SNLI training (quantum-only)
   - `chat/test_snli_vector.py` → SNLI testing

## Data Flow (SNLI)

```
SNLI JSONL
  ↓
QuantumTextEncoder.tokenize/encode (pretrained embeddings)
  ↓
QuantumSNLIEncoder.build_initial_state()
  → h0 (hypothesis - premise)
  → v_p (premise OM)
  → v_h (hypothesis LO)
  ↓
VectorCollapseEngine.collapse[_dynamic](h0)
  → h_final (+ trace)
  ↓
SNLIHead(h_final, v_p, v_h)
  → logits (E/N/C)
  ↓
CrossEntropyLoss → backprop
```

Dynamic basins (default) add per-label routing, spawn, and optional prune/merge; disable with `--disable-dynamic-basins`.

## Key Principles

- Physics core stays frozen; only heads/encoders change.
- Single encoder path: pretrained Livnium quantum embeddings (no geom/legacy fallback).
- OM/LO separation + divergence law remain the semantic signal.

## File Structure (nova_v3)

```
nova_v3/
├── core/                 # Layer 0
├── tasks/
│   └── snli/
│       ├── encoding_snli.py  # QuantumSNLIEncoder
│       └── head_snli.py
├── training/
│   └── train_snli_vector.py  # quantum-only training
├── chat/
│   └── test_snli_vector.py   # quantum-only eval
└── quantum_embed/            # pretrained embeddings + tokenizer
    └── quantum_embeddings_final.pt
```

## Notes on Dependencies

- `quantum_embed/text_encoder_quantum.py` must be importable (run from repo root or keep `nova/` on `PYTHONPATH`).
- Checkpoint (`quantum_embeddings_final.pt`) provides vocab, pad/unk indices, and embedding table used by the encoder.

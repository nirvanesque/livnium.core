# Files Moved from experiments/crypto

The following quantum-related files were moved from `experiments/crypto/` to `experiments/quantum-inspired-livnium-core/`:

## Moved Files

1. **`aes128_quantum_topology_mapper.py`** - Quantum-enhanced topology mapper
2. **`test_entanglement_capacity.py`** - Tests entanglement capacity
3. **`test_recursive_qubit_capacity.py`** - Tests qubit capacity
4. **`aes128_round_sweep_experiment_livnium.py`** - Quantum-based round sweep experiment
5. **`QUBIT_ANALYSIS.md`** - Qubit type analysis documentation

## Why Moved

These files are quantum-focused and use the Livnium Core System's quantum layer and recursive geometry engine. They belong in a dedicated quantum experiments directory rather than the general crypto directory.

## Dependencies

These files still import from `experiments.crypto` for:
- `aes128_base.py` - Base AES implementation
- `aes128_*round.py` - Per-round cipher classes

This is intentional - the crypto directory contains the cipher implementations, while this directory contains quantum-enhanced experiments using those ciphers.

## Usage

Files can be run directly:
```bash
python3 experiments/quantum-inspired-livnium-core/test_recursive_qubit_capacity.py
python3 experiments/quantum-inspired-livnium-core/test_entanglement_capacity.py
python3 experiments/quantum-inspired-livnium-core/aes128_quantum_topology_mapper.py
python3 experiments/quantum-inspired-livnium-core/aes128_round_sweep_experiment_livnium.py
```


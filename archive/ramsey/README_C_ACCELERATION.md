# C-Accelerated Ramsey Core

This directory contains a C-accelerated core for Ramsey number search that leverages M5 hardware optimizations.

## Architecture

The C-accelerated core provides:

1. **Bitset-based edge representation**: Uses `uint64_t` words for adjacency matrices
2. **Bitwise clique checking**: AND/POPCOUNT operations instead of Python loops
3. **Batch operations**: Process thousands of omcubes in parallel
4. **M5 CPU optimizations**: Native POPCOUNT, bitwise ops, cache-friendly memory layout

## Building

```bash
cd experiments/ramsey
python setup_ramsey_core.py build_ext --inplace
```

This will compile `ramsey_core.c` into a Python extension module `ramsey_core.so` (or `.dylib` on macOS).

## Integration

The C core plugs seamlessly into the existing Python code:

```python
from ramsey_core_wrapper import get_accelerator

accelerator = get_accelerator()
if accelerator.available:
    # Use C-accelerated checking
    is_valid, clique = accelerator.check_coloring(graph, n, k)
    
    # Or batch check thousands of omcubes
    graphs = [omcube_states[i] for i in range(10000)]
    results = accelerator.batch_check_colorings(graphs, n, k)
else:
    # Falls back to Python/Numba automatically
    is_valid, clique = graph.has_monochromatic_clique(k)
```

## Performance

Expected speedups on M5:
- **Single graph checking**: 5-10x faster than Numba
- **Batch operations**: 20-50x faster (processes 20k omcubes in seconds)

## Next Steps

1. **GPU acceleration**: Batch energy/heuristic calculations on M5 GPU via PyTorch MPS
2. **ANE integration**: Learned heuristics via Core ML
3. **SAT solver backend**: Use state-of-the-art SAT solvers for the raw search

## Notes

- The C core handles the **brutal search** (clique checking, validation)
- Livnium remains the **meta-physics layer** (geometry, mutations, search policy)
- This separation allows us to upgrade the engine without touching the Livnium logic


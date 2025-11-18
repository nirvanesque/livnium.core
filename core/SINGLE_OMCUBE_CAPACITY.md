# Single Omcube Capacity

## What is an Omcube?

An **omcube** is a general container that can hold any classical or quantum state. In the Livnium Core System, each omcube represents one parallel "universe" or configuration being explored.

---

## Capacity by Use Case

### 1. Classical State (RamseyGraph Example)

**In the Ramsey solver**, each omcube holds a `RamseyGraph` object:

| Graph Size | Edges | Empty Graph | Full Coloring | Typical Size |
|------------|-------|-------------|----------------|--------------|
| K_5 | 10 | ~120 bytes | ~440 bytes | ~0.4 KB |
| K_10 | 45 | ~120 bytes | ~1,280 bytes | ~1.3 KB |
| K_20 | 190 | ~120 bytes | ~4,760 bytes | ~4.7 KB |
| K_45 | 990 | ~120 bytes | ~24,000 bytes | ~24 KB |
| K_100 | 4,950 | ~120 bytes | ~120,000 bytes | ~120 KB |
| K_200 | 19,900 | ~120 bytes | ~1,500,000 bytes | ~1.5 MB |

**Structure:**
```python
RamseyGraph:
  - n: int (number of vertices)
  - num_edges: int (total edges)
  - edge_coloring: Dict[(u,v), color]  # Main data
  - edge_index_map: Dict[(u,v), index]  # Lookup table
  - _hash_cache: Optional[int]  # Cached hash
```

**Memory breakdown:**
- Base object: ~56-120 bytes
- Each edge coloring: ~24-76 bytes (tuple + color + dict overhead)
- Total: `base + (num_edges × ~24-76 bytes)`

---

### 2. Quantum State

**If storing a qubit:**
- **Single qubit**: 2 complex amplitudes = **16 bytes** (complex128)
- **Multiple qubits**: 16 bytes × number of qubits
- **Entangled state**: Depends on entanglement structure

**Example:**
- 1 qubit: 16 bytes
- 10 qubits: 160 bytes
- 100 qubits: 1,600 bytes (~1.6 KB)

---

### 3. Memory Capsule (Layer 3)

**Memory capsule structure:**
- Working memory: ~1-5 KB
- Long-term memory: ~1-10 KB
- Metadata: ~100-500 bytes
- **Total: ~1-10 KB** per capsule

---

### 4. Semantic Embeddings (Layer 5)

**Semantic data:**
- Feature vectors: ~1-5 KB
- Symbol-to-meaning graph: ~1-5 KB
- Context propagation data: ~1-2 KB
- **Total: ~1-10 KB** per embedding

---

### 5. Combined State

**An omcube can hold multiple layers simultaneously:**

| Configuration | Typical Size | Maximum Size |
|---------------|--------------|--------------|
| Classical only | 1-100 KB | 1-10 MB |
| Quantum only | 16 bytes - 1 KB | ~100 KB |
| Memory only | 1-10 KB | ~100 KB |
| Semantic only | 1-10 KB | ~100 KB |
| **All layers** | **1-120 KB** | **1-10 MB** |

---

## Practical Limits

### Typical Use Cases

**Ramsey Solver (K_45):**
- Each omcube: ~24-75 KB
- 5,000 omcubes: ~120-375 MB
- 50,000 omcubes: ~1.2-3.75 GB

**Small Quantum System:**
- Each omcube: ~16-160 bytes (1-10 qubits)
- 10,000 omcubes: ~160 KB - 1.6 MB

**Memory System:**
- Each omcube: ~1-10 KB
- 100,000 omcubes: ~100 MB - 1 GB

---

## Maximum Capacity

### Theoretical Maximum

**No hard limit** - scales to system memory!

**Practical limits:**
- **Typical**: 1-100 KB per omcube
- **Large**: 100 KB - 1 MB per omcube
- **Maximum**: 1-10 MB per omcube (for very large states)

### System-Wide Capacity

**For 8GB RAM:**
- Small omcubes (1 KB each): ~8,000,000 omcubes
- Medium omcubes (100 KB each): ~80,000 omcubes
- Large omcubes (1 MB each): ~8,000 omcubes

---

## Memory Breakdown (RamseyGraph Example)

**For K_45 (990 edges):**

```
Base object:            ~56 bytes
Edge coloring dict:    ~200 bytes (overhead)
990 edges × 24 bytes:  ~23,760 bytes
Hash cache:            ~24 bytes
Total:                 ~24,040 bytes (~24 KB)
```

**For K_100 (4,950 edges):**

```
Base object:            ~56 bytes
Edge coloring dict:    ~200 bytes (overhead)
4,950 edges × 24 bytes: ~118,800 bytes
Hash cache:            ~24 bytes
Total:                  ~119,080 bytes (~119 KB)
```

---

## Summary

**One omcube can hold:**

✅ **Classical state**: 1-100 KB (typical), 1-10 MB (max)  
✅ **Quantum state**: 16 bytes - 1 KB (typical), ~100 KB (max)  
✅ **Memory capsule**: 1-10 KB  
✅ **Semantic data**: 1-10 KB  
✅ **Combined**: 1-120 KB (typical), 1-10 MB (maximum)  

**Key Points:**
- No hard limit - scales to system memory
- Memory scales linearly with data size
- Typical: 1-100 KB per omcube
- Maximum: 1-10 MB per omcube (for very large states)
- System can handle millions of omcubes with small states

---

## Examples

**Example 1: Ramsey Solver (K_45)**
- 5,000 omcubes × 24 KB = ~120 MB ✅

**Example 2: Quantum System (10 qubits each)**
- 100,000 omcubes × 160 bytes = ~16 MB ✅

**Example 3: Memory System**
- 1,000,000 omcubes × 5 KB = ~5 GB ✅

**Example 4: Large Classical States (K_200)**
- 1,000 omcubes × 1.5 MB = ~1.5 GB ✅


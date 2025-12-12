# Memory System

Persistent memory cells that store and recall information across episodes.

## What is the Memory System?

**Memory that behaves like energy in a lattice.**

This system provides:
- **Per-cell memory capsules**: Each geometric cell has its own working + long-term memory
- **Global memory lattice**: A brain grid overlaying the geometric structure
- **Geometric coupling**: Memory strength tied to symbolic weight and face exposure
- **Decay rules**: Memories fade naturally based on geometry
- **Associative graph**: Memories link together forming connections
- **Cross-step persistence**: Information survives across episodes

## Contents

- **`memory_cell.py`**: Individual memory cell with MemoryCapsule (working + long-term memory)
- **`memory_lattice.py`**: Lattice of memory cells (1-to-1 overlay on geometric structure)
- **`memory_coupling.py`**: Coupling between memory cells and geometric properties

## Purpose

This module provides:
- **Persistent storage**: Information survives across episodes
- **Associative recall**: Memory cells can be queried by content
- **Geometric coupling**: Memory strength derived from symbolic weight and face exposure
- **Decay**: Old memories fade over time (geometric decay rules)
- **State transitions**: Working memory → long-term memory consolidation

Used by NLI system and other learning applications. The memory system integrates seamlessly with LivniumCoreSystem, using the same coordinate structure.

## Future Directions

Potential expansions for the memory layer:

### High-Impact Next Steps

1. **Autonomous Memory Consolidation (Sleep Cycle)**
   - Nightly sleep phases: pause system, decay active memories, strengthen high-importance ones
   - Compress recent experiences, stabilize important ones, prune noise, merge clusters
   - Like real sleep - makes the system more efficient and accurate

2. **Predictive Memory (Next-State Forecasting)**
   - Each memory capsule learns: "Given what I saw at (x,y,z), what will I likely see next?"
   - Store most common next symbols, transitions, tension trajectories
   - Turns memory layer into a predictive physics engine

3. **Content-Based Retrieval (CBR)**
   - Expand context retrieval into true CBR
   - Retrieve by similar SW, face exposure, geometric similarity, vector-compressed signatures
   - Becomes the Livnium Search Engine

4. **Memory-Driven Geometry Updates**
   - Geometric cells shift SW or polarity based on accumulated memory
   - Memory → geometry feedback loop
   - "I've seen this coordinate behave like this 20 times - strengthen its attractor"

5. **Temporal Memory (Time as 4th Reference Axis)**
   - Add timestamp, decay_half_life, age_weight fields
   - Recency effects, forgetting curve, importance shaped by time
   - Real cognitive time-based memory

### Additional Directions

6. **Episodic vs Semantic Memory Split**
   - Formalize working memory (episodic, time-stamped, fast decay) vs long-term memory (semantic, compressed, class-level)
   - Lattice learns concepts, not just coordinates

7. **Memory Compression (The Haircut v2)**
   - Compress long-term memory into smaller latent vectors
   - Merge similar states, drop redundant associations
   - Keep memory lattice clean as it grows to millions of states

8. **Associative Graph Becomes Neural Map**
   - Weighted edges (frequency, strength), recurrent activation
   - Spreading activation like semantic networks
   - Activation-based retrieval - memory behaves like a biological brain

9. **Cross-Step Memory Influence for NLI**
   - Experience-based NLI: memory nudges reasoning system
   - "man riding bike" → stores relationship → influences "person riding bicycle" classification

10. **Hierarchical Memory (Macros inside cubes)**
    - Stack memory lattices: each 3×3×3 chunk forms macro-memory node
    - Recursive memory architecture matching hierarchical Livnium Core

These expansions would transform Livnium from a system that *stores* memory into a system that *uses* it like a mind.


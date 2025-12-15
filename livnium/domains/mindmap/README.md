# Mindmap Domain

Maps text documents into a graph of ThoughtNodes connected by Livnium physics (alignment, divergence, tension).

## Mental Model

| Livnium concept | Mind-map meaning                      |
| --------------- | ------------------------------------- |
| State           | ThoughtNode (one paragraph / message) |
| Alignment       | Semantic closeness                    |
| Divergence      | Conceptual distance                   |
| Tension         | Conflict / unresolved pressure        |
| Basin           | Recurrent theme                       |
| Ledger          | Idea conservation over time           |

## Architecture

This is a **domain plugin** - it uses `kernel.physics` for measurement, not for dynamics.

- **`ingest.py`** - Converts markdown files into ThoughtNodes (one paragraph = one node)
- **`encoder.py`** - Text → embeddings (supports sentence-transformers or simple fallback)
- **`geometry.py`** - Measures relationships using `kernel.physics` (alignment, divergence, tension)
- **`export.py`** - Exports thought graph to JSON for visualization
- **`basins.py`** - Identifies emergent stable attractors (recurrent themes)
- **`narrator.py`** - Generates fast summaries for basins (heuristic + optional LLM polish)
- **`basin_narrator.py`** - Advanced basin narration with LLM integration

## Usage

### Quick Start: Drop Files and Run

1. **Drop your files** into `livnium/domains/mindmap/sources/`:
   - Text files: `.txt`, `.md`
   - Code files: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.go`, `.rs`, etc.
   - **JSON files**: `.json` (parsed and content extracted)
   - Any text-based file format

2. **Run the demo**:
   ```bash
   python3 -m livnium.domains.mindmap.demo
   ```

3. **View the graph**: Open `livnium/tools/visualize/viewer.html` in a browser

The demo will:
1. Automatically scan `sources/` folder for all supported files
2. Ingest core documentation (README.md, ARCHITECTURE.md, QUICKSTART.md)
3. Embed all text/code blocks using available encoder
4. Measure geometry using kernel.physics (alignment, divergence, tension)
5. Export to `livnium/tools/visualize/mindmap.json`
6. Print statistics (nodes, edges, max alignment, max tension)

### Programmatic Usage

```python
from pathlib import Path
from livnium.domains.mindmap import (
    ingest_file,
    ingest_sources_folder,
    embed_thoughts,
    measure_thought_graph,
    compute_node_masses,
    export_to_json,
    print_statistics,
    get_encoder,
)

# Option 1: Ingest entire sources folder
nodes = ingest_sources_folder()  # Scans sources/ automatically

# Option 2: Ingest specific files
nodes = []
for file_path in [Path("README.md"), Path("script.py"), Path("notes.txt")]:
    nodes.extend(ingest_file(file_path))  # Works with any text/code file

# Embed (try sentence-transformers, fallback to simple)
try:
    encoder = get_encoder(use_sentence_transformers=True)
except ImportError:
    encoder = get_encoder(use_sentence_transformers=False)

embed_thoughts(nodes, encoder)

# Measure geometry
edges = measure_thought_graph(nodes, alignment_threshold=0.4)
compute_node_masses(nodes, edges)

# Print statistics
print_statistics(nodes, edges)

# Export
export_to_json(nodes, edges, Path("mindmap.json"))
```

## Visualization

Open `livnium/tools/visualize/viewer.html` in a browser (after running the demo to generate `mindmap.json`).

The visualizer shows:
- **Node size** = mass (connectivity)
- **Edge thickness** = alignment
- **Dashed edges** = high tension (conflict)

## Key Principles

1. **One paragraph = one ThoughtNode** - Not sentences, not whole files
2. **Measurement, not collapse** - We're surveying the idea-field, not evolving it
3. **Uses kernel.physics exactly as designed** - No new physics, just a new domain
4. **Visualization threshold is NOT a law** - The 0.4 alignment threshold is for visualization, not physics

## Basin Detection & Narration

The mindmap domain includes basin detection to identify recurrent themes and stable attractors in the thought graph. Basins can be narrated using fast heuristics or optional LLM polish. See `NARRATOR.md` and `QUICK_LLM_SETUP.md` for details.

## Future Work

- **Better embeddings** - Integrate quantum embeddings when ready
- **Interactive exploration** - Filter by source, search nodes, etc.


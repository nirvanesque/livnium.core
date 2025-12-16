#!/usr/bin/env python3
"""
Filter mindmap.json to reduce file size for browser visualization.

Keeps only the top N edges by alignment score.
"""

import json
from pathlib import Path

def filter_mindmap(input_path: Path, output_path: Path, max_edges: int = 50000):
    """Filter mindmap.json to keep only top edges by alignment."""
    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes = data.get('nodes', [])
    edges = data.get('edges', [])
    
    print(f"Original: {len(nodes)} nodes, {len(edges)} edges")
    
    if len(edges) <= max_edges:
        print(f"File already has {len(edges)} edges (≤ {max_edges}), no filtering needed.")
        return
    
    # Sort edges by alignment (descending) and keep top N
    print(f"Sorting edges by alignment...")
    sorted_edges = sorted(edges, key=lambda e: e.get('alignment', 0), reverse=True)
    filtered_edges = sorted_edges[:max_edges]
    
    print(f"Filtered: {len(nodes)} nodes, {len(filtered_edges)} edges")
    print(f"  → Kept top {max_edges} edges by alignment")
    if filtered_edges:
        print(f"  → Alignment range: {filtered_edges[-1]['alignment']:.3f} - {filtered_edges[0]['alignment']:.3f}")
    
    # Write filtered data
    output = {
        'nodes': nodes,
        'edges': filtered_edges
    }
    
    print(f"Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Check file sizes
    input_size = input_path.stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)
    print(f"File size: {input_size:.1f} MB → {output_size:.1f} MB ({output_size/input_size*100:.1f}%)")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    input_path = script_dir / "mindmap.json"
    output_path = script_dir / "mindmap.json"
    
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        exit(1)
    
    # Filter to 50K edges (good balance between detail and performance)
    filter_mindmap(input_path, output_path, max_edges=50000)
    print("\nDone! Refresh your browser to see the filtered graph.")


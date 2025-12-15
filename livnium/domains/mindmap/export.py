"""
Mindmap Export: Thought Graph → JSON

Exports the measured thought graph to JSON for visualization.
This is the externalized mind - structure the physics earned.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from .ingest import ThoughtNode


def export_to_json(
    nodes: List[ThoughtNode],
    edges: List[Dict[str, Any]],
    output_path: Path,
    max_edges: int = None,
    basins: List[Dict[str, Any]] = None
) -> None:
    """
    Export thought graph to JSON.
    
    Format:
    {
        "nodes": [
            {
                "id": "arch_12",
                "text": "...",
                "mass": 0.73,
                "source": "path/to/file.md"
            }
        ],
        "edges": [
            {
                "source": "arch_12",
                "target": "readme_5",
                "alignment": 0.81,
                "divergence": -0.43,
                "tension": 0.43
            }
        ],
        "basins": [
            {
                "id": "basin_0",
                "anchor_id": "arch_12",
                "node_ids": ["arch_12", "readme_5", ...],
                "summary": "Constitution / Laws",
                "avg_alignment": 0.85,
                "avg_tension": 0.12,
                "stability": 0.75,
                "size": 45
            }
        ]
    }
    
    Args:
        nodes: List of ThoughtNodes
        edges: List of edge dictionaries (from measure_thought_graph)
        output_path: Path to write JSON file
        max_edges: Optional maximum number of edges to export (keeps highest alignment edges)
        basins: Optional list of basin dictionaries (from identify_basins)
    """
    # Filter edges if max_edges is specified
    edges_to_export = edges
    if max_edges is not None and len(edges) > max_edges:
        # Sort by alignment (descending) and take top N
        edges_to_export = sorted(edges, key=lambda e: e["alignment"], reverse=True)[:max_edges]
        print(f"  → Limiting to top {max_edges} edges by alignment (from {len(edges)} total)")
    
    # Build nodes list
    nodes_data = []
    for node in nodes:
        nodes_data.append({
            "id": node.id,
            "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,  # Truncate for readability
            "mass": float(node.mass),
            "source": node.source
        })
    
    # Build output structure
    output = {
        "nodes": nodes_data,
        "edges": edges_to_export
    }
    
    # Add basins if provided
    if basins:
        # Convert basins to JSON-serializable format
        basins_data = []
        for basin in basins:
            basin_data = {
                "id": basin["id"],
                "anchor_id": basin["anchor_id"],
                "node_ids": basin["node_ids"],
                "summary": basin.get("summary", ""),
                "narrative": basin.get("narrative", ""),  # Full narrative statement
                "avg_alignment": float(basin["avg_alignment"]),
                "avg_tension": float(basin["avg_tension"]),
                "stability": float(basin["stability"]),
                "size": basin["size"]
            }
            # Add fast summary if available
            if "fast_summary" in basin:
                basin_data["fast_summary"] = basin["fast_summary"]
            basins_data.append(basin_data)
        output["basins"] = basins_data
        print(f"  → Included {len(basins_data)} basins")
    
    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Exported {len(nodes_data)} nodes and {len(edges_to_export)} edges to {output_path}")


def print_statistics(nodes: List[ThoughtNode], edges: List[Dict[str, Any]]) -> None:
    """
    Print statistics about the thought graph.
    
    Args:
        nodes: List of ThoughtNodes
        edges: List of edge dictionaries
    """
    if not edges:
        print("No edges found (all alignments below threshold)")
        return
    
    alignments = [e["alignment"] for e in edges]
    tensions = [e["tension"] for e in edges]
    
    print("\n=== Thought Graph Statistics ===")
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")
    print(f"Max alignment: {max(alignments):.3f}")
    print(f"Min alignment: {min(alignments):.3f}")
    print(f"Mean alignment: {sum(alignments) / len(alignments):.3f}")
    print(f"Max tension: {max(tensions):.3f}")
    print(f"Mean tension: {sum(tensions) / len(tensions):.3f}")
    print("=" * 32)


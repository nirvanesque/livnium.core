"""
Basin Narrator: Read-Only Observer for Basin Interpretation

This module is a READ-ONLY observer that converts basins into tentative
human-readable statements. It does NOT modify kernel, physics, or geometry.

Hard constraints:
- Read-only (observer only)
- Non-authoritative (uses "seems to", "appears to", "may be about")
- Deterministic (no LLM required)
- One basin at a time
- Does not affect basin formation or graph structure

Output format:
```
Hypothesis:
This region appears to center on [core idea].
It shows [tension/alignment pattern].

Signals:
- [metric observations]
```

Placement: Called only by viewer/export layer, never by geometry/physics.
"""

from typing import List, Dict, Any
from ..ingestion.ingest import ThoughtNode


def narrate_basin(
    basin: Dict[str, Any],
    node_map: Dict[str, ThoughtNode],
    edges: List[Dict[str, Any]],
    top_n: int = 20
) -> str:
    """
    Narrate a basin as a read-only observer.
    
    Produces tentative, non-authoritative statements about what the basin
    appears to represent, based on node texts and metrics.
    
    Args:
        basin: Basin dictionary (from identify_basins)
        node_map: Map of node_id -> ThoughtNode
        edges: List of all edges (to examine internal basin edges)
        top_n: Number of top nodes by mass to consider
        
    Returns:
        Formatted string with Hypothesis and Signals sections
    """
    node_ids = basin["node_ids"]
    
    # Get nodes in this basin
    basin_nodes = []
    for node_id in node_ids:
        node = node_map.get(node_id)
        if node:
            basin_nodes.append(node)
    
    if not basin_nodes:
        return "Hypothesis:\nThis region appears empty.\n\nSignals:\n- No nodes found"
    
    # Sort by mass (gravitational center)
    basin_nodes.sort(key=lambda n: n.mass, reverse=True)
    top_nodes = basin_nodes[:top_n]
    
    # Find edges between top nodes (internal basin structure)
    top_node_ids = {n.id for n in top_nodes}
    basin_edges = [
        e for e in edges
        if e["source"] in top_node_ids and e["target"] in top_node_ids
    ]
    
    # Analyze internal tension
    if basin_edges:
        tensions = [e.get("tension", 0) for e in basin_edges]
        alignments = [e.get("alignment", 0) for e in basin_edges]
        avg_tension = sum(tensions) / len(tensions)
        avg_alignment = sum(alignments) / len(alignments)
        max_tension = max(tensions) if tensions else 0
        min_alignment = min(alignments) if alignments else 0
    else:
        avg_tension = basin.get("avg_tension", 0)
        avg_alignment = basin.get("avg_alignment", 0)
        max_tension = avg_tension
        min_alignment = avg_alignment
    
    # Extract core texts from top nodes
    core_texts = []
    for node in top_nodes[:10]:  # Top 10 by mass
        text = node.text.strip()
        # Extract first meaningful sentence
        sentences = text.split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            # Filter meaningful sentences
            if len(first_sentence) > 15 and len(first_sentence) < 200:
                # Remove markdown headers
                first_sentence = first_sentence.replace('#', '').strip()
                if first_sentence:
                    core_texts.append(first_sentence)
    
    # Build Hypothesis section (non-authoritative)
    hypothesis_parts = []
    
    # Core idea (from anchor/center node)
    if core_texts:
        anchor_text = core_texts[0]
        # Truncate if too long
        if len(anchor_text) > 100:
            anchor_text = anchor_text[:97] + "..."
        hypothesis_parts.append(f"This region appears to center on {anchor_text.lower()}")
    else:
        hypothesis_parts.append("This region appears to center on an unclear concept")
    
    # Tension/alignment pattern (tentative observation)
    if basin_edges:
        if avg_tension < 0.15 and avg_alignment > 0.7:
            hypothesis_parts.append("It shows low internal tension and high alignment")
        elif avg_tension < 0.25:
            hypothesis_parts.append("It shows moderate internal tension")
        elif max_tension > 0.4:
            hypothesis_parts.append("It shows significant internal tension between elements")
        else:
            hypothesis_parts.append("It shows mixed alignment patterns")
    else:
        hypothesis_parts.append("It shows limited internal connectivity")
    
    # Build Signals section (metric observations)
    signals = []
    
    # Stability signal
    stability = basin.get("stability", 0)
    if stability > 0.7:
        signals.append("High stability (stable attractor)")
    elif stability > 0.4:
        signals.append("Moderate stability")
    else:
        signals.append("Lower stability")
    
    # Alignment signal
    if avg_alignment > 0.7:
        signals.append("High internal alignment")
    elif avg_alignment > 0.5:
        signals.append("Moderate internal alignment")
    else:
        signals.append("Lower internal alignment")
    
    # Tension signal
    if avg_tension < 0.15:
        signals.append("Low internal tension")
    elif avg_tension < 0.3:
        signals.append("Moderate internal tension")
    else:
        signals.append("Higher internal tension")
    
    # Size signal
    size = basin.get("size", 0)
    if size > 50:
        signals.append(f"Large basin ({size} nodes)")
    elif size > 20:
        signals.append(f"Medium basin ({size} nodes)")
    else:
        signals.append(f"Small basin ({size} nodes)")
    
    # Format output
    hypothesis = ". ".join(hypothesis_parts) + "."
    signals_text = "\n".join(f"- {s}" for s in signals)
    
    return f"Hypothesis:\n{hypothesis}\n\nSignals:\n{signals_text}"


def narrate_all_basins(
    basins: List[Dict[str, Any]],
    node_map: Dict[str, ThoughtNode],
    edges: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Narrate all basins, adding narrative statements to each.
    
    This is a read-only operation that does not modify basins or graph structure.
    
    Args:
        basins: List of basin dictionaries
        node_map: Map of node_id -> ThoughtNode
        edges: List of all edges
        
    Returns:
        List of basins with added 'narrative' field
    """
    narrated = []
    for basin in basins:
        narrative = narrate_basin(basin, node_map, edges)
        basin_with_narrative = basin.copy()
        basin_with_narrative["narrative"] = narrative
        narrated.append(basin_with_narrative)
    
    return narrated

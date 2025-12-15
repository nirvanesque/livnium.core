"""
Mindmap Basins: Emergent Stable Attractors

Basins are NOT clusters, labels, or hierarchies.
They are regions of meaning where many thoughts naturally fall together
under the laws - stable attractors formed by repeated alignment.

A basin is: "A stable attractor formed by repeated alignment under the same laws."

This module detects basins by:
1. Finding anchor nodes (highly connected, central nodes)
2. Grouping nodes that align strongly with anchors
3. Computing basin stability (low average tension)
4. Summarizing each basin from its central nodes
"""

from typing import List, Dict, Any, Set, Tuple
import torch
from .ingest import ThoughtNode


def identify_basins(
    thoughts: List[ThoughtNode],
    edges: List[Dict[str, Any]],
    min_basin_size: int = 5,
    alignment_threshold: float = 0.7,
    max_basins: int = 20
) -> List[Dict[str, Any]]:
    """
    Identify basins (emergent stable attractors) in the thought graph.
    
    Basins are detected by:
    1. Finding anchor nodes (high connectivity + centrality)
    2. Grouping nodes that align strongly with anchors
    3. Filtering by stability (low average tension)
    
    This is NOT clustering - basins emerge from physics, not algorithms.
    
    Args:
        thoughts: List of ThoughtNodes
        edges: List of edge dictionaries (from measure_thought_graph)
        min_basin_size: Minimum number of nodes in a basin
        alignment_threshold: Minimum alignment to anchor to be in basin
        max_basins: Maximum number of basins to return
        
    Returns:
        List of basin dictionaries, each with:
        - id: Basin identifier
        - anchor_id: ID of the anchor node
        - node_ids: List of node IDs in this basin
        - summary: Human-readable summary (from summarize_basin)
        - avg_alignment: Average alignment within basin
        - avg_tension: Average tension within basin
        - stability: Stability score (higher = more stable)
    """
    if not edges or not thoughts:
        return []
    
    # Build node map and edge maps
    node_map = {node.id: node for node in thoughts}
    node_edges = {node.id: [] for node in thoughts}
    
    for edge in edges:
        node_edges[edge["source"]].append(edge)
        node_edges[edge["target"]].append(edge)
    
    # Compute node centrality (sum of alignments)
    centrality = {}
    for node_id in node_map.keys():
        centrality[node_id] = sum(
            e["alignment"] for e in node_edges[node_id]
        )
    
    # Find anchor nodes (high centrality + high connectivity)
    # Sort by centrality * connectivity
    anchor_scores = {
        node_id: centrality[node_id] * len(node_edges[node_id])
        for node_id in node_map.keys()
        if len(node_edges[node_id]) > 0
    }
    
    # Sort anchors by score (descending)
    sorted_anchors = sorted(
        anchor_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Build alignment graph for fast lookup
    alignment_graph = {}
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        align = edge["alignment"]
        
        if source not in alignment_graph:
            alignment_graph[source] = {}
        if target not in alignment_graph:
            alignment_graph[target] = {}
        
        alignment_graph[source][target] = align
        alignment_graph[target][source] = align
    
    # Build basins from anchors
    basins = []
    assigned_nodes: Set[str] = set()
    
    for anchor_id, anchor_score in sorted_anchors[:max_basins * 2]:  # Check more anchors than needed
        if anchor_id in assigned_nodes:
            continue
        
        # Find nodes that align strongly with this anchor
        basin_nodes = [anchor_id]
        anchor_alignments = alignment_graph.get(anchor_id, {})
        
        for node_id, align in anchor_alignments.items():
            if node_id in assigned_nodes:
                continue
            if align >= alignment_threshold:
                basin_nodes.append(node_id)
        
        # Filter by minimum size
        if len(basin_nodes) < min_basin_size:
            continue
        
        # Compute basin metrics
        basin_edges = []
        basin_alignments = []
        basin_tensions = []
        
        for i, node_a in enumerate(basin_nodes):
            for node_b in basin_nodes[i+1:]:
                align = alignment_graph.get(node_a, {}).get(node_b, 0.0)
                if align > 0:
                    basin_alignments.append(align)
                    # Find edge to get tension
                    for edge in edges:
                        if ((edge["source"] == node_a and edge["target"] == node_b) or
                            (edge["source"] == node_b and edge["target"] == node_a)):
                            basin_tensions.append(edge["tension"])
                            break
        
        avg_alignment = sum(basin_alignments) / len(basin_alignments) if basin_alignments else 0.0
        avg_tension = sum(basin_tensions) / len(basin_tensions) if basin_tensions else 0.0
        
        # Stability = high alignment, low tension
        stability = avg_alignment * (1.0 - min(avg_tension, 1.0))
        
        # Create basin
        basin = {
            "id": f"basin_{len(basins)}",
            "anchor_id": anchor_id,
            "node_ids": basin_nodes,
            "avg_alignment": avg_alignment,
            "avg_tension": avg_tension,
            "stability": stability,
            "size": len(basin_nodes)
        }
        
        basins.append(basin)
        
        # Mark nodes as assigned (prevent overlap)
        assigned_nodes.update(basin_nodes)
        
        if len(basins) >= max_basins:
            break
    
    # Sort basins by stability (most stable first)
    basins.sort(key=lambda b: b["stability"], reverse=True)
    
    # Generate summaries for each basin
    for basin in basins:
        basin["summary"] = summarize_basin(basin, node_map)
    
    return basins


def summarize_basin(basin: Dict[str, Any], node_map: Dict[str, ThoughtNode]) -> str:
    """
    Summarize a basin by stitching together central node texts.
    
    This is "basin narration" - converting nodes into a statement
    without inventing new meaning. Simply: "What do these nodes align toward?"
    
    Strategy:
    1. Get anchor node text (most central)
    2. Get top 3-5 most aligned nodes
    3. Stitch into 1-2 sentences
    
    Args:
        basin: Basin dictionary (from identify_basins)
        node_map: Map of node_id -> ThoughtNode
        
    Returns:
        Human-readable summary string
    """
    node_ids = basin["node_ids"]
    anchor_id = basin["anchor_id"]
    
    # Get anchor text
    anchor_node = node_map.get(anchor_id)
    if not anchor_node:
        return "Unknown basin"
    
    anchor_text = anchor_node.text.strip()
    
    # Get other node texts (limit to most relevant)
    other_texts = []
    for node_id in node_ids:
        if node_id == anchor_id:
            continue
        node = node_map.get(node_id)
        if node:
            text = node.text.strip()
            # Clean and truncate
            text = text.split('.')[0]  # First sentence
            if len(text) > 100:
                text = text[:97] + "..."
            if text and text not in other_texts:
                other_texts.append(text)
                if len(other_texts) >= 4:  # Limit to 4 additional nodes
                    break
    
    # Build summary
    # Strategy: Anchor text + "related to" + other concepts
    if not other_texts:
        # Just use anchor text, cleaned
        summary = anchor_text.split('.')[0]
        if len(summary) > 150:
            summary = summary[:147] + "..."
        return summary
    
    # Combine: anchor + "relates to" + others
    anchor_clean = anchor_text.split('.')[0]
    if len(anchor_clean) > 80:
        anchor_clean = anchor_clean[:77] + "..."
    
    # Try to create a coherent summary
    if len(other_texts) == 1:
        summary = f"{anchor_clean}. Related: {other_texts[0]}"
    elif len(other_texts) == 2:
        summary = f"{anchor_clean}. Related: {other_texts[0]}, {other_texts[1]}"
    else:
        # More than 2 - create a list
        others_str = ", ".join(other_texts[:3])
        summary = f"{anchor_clean}. Related concepts: {others_str}"
    
    # Clean up
    summary = summary.replace('\n', ' ').replace('  ', ' ').strip()
    if len(summary) > 200:
        summary = summary[:197] + "..."
    
    return summary


def compute_basin_statistics(basins: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics about detected basins.
    
    Args:
        basins: List of basin dictionaries
        
    Returns:
        Dictionary with statistics
    """
    if not basins:
        return {
            "num_basins": 0,
            "total_nodes": 0,
            "avg_basin_size": 0,
            "avg_stability": 0,
            "avg_alignment": 0,
            "avg_tension": 0
        }
    
    total_nodes = sum(b["size"] for b in basins)
    
    return {
        "num_basins": len(basins),
        "total_nodes": total_nodes,
        "avg_basin_size": total_nodes / len(basins),
        "avg_stability": sum(b["stability"] for b in basins) / len(basins),
        "avg_alignment": sum(b["avg_alignment"] for b in basins) / len(basins),
        "avg_tension": sum(b["avg_tension"] for b in basins) / len(basins),
        "max_stability": max(b["stability"] for b in basins),
        "min_stability": min(b["stability"] for b in basins)
    }

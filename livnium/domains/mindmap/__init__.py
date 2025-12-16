"""
Mindmap Domain: Text → Thought Graph

Maps text documents (README, ARCHITECTURE, etc.) into a graph of ThoughtNodes
connected by Livnium physics (alignment, divergence, tension).

This is a domain plugin - it uses kernel.physics for measurement,
not for dynamics. The visualizer shows the idea-field as-is.
"""

from .ingestion.ingest import (
    ThoughtNode,
    ingest_markdown,
    ingest_file,
    ingest_sources_folder,
    scan_sources_folder,
    embed_thoughts,
)
from .graph.geometry import measure_thought_graph, compute_node_masses
from .utils.export import export_to_json, print_statistics
from .ingestion.encoder import get_encoder
from .graph.basins import (
    identify_basins,
    compute_basin_statistics,
)
from .narrative.basin_narrator import narrate_basin, narrate_all_basins
from .narrative.narrator import summarize_basin as fast_summarize_basin, format_basin_summary

__all__ = [
    "ThoughtNode",
    "ingest_markdown",
    "ingest_file",
    "ingest_sources_folder",
    "scan_sources_folder",
    "embed_thoughts",
    "measure_thought_graph",
    "compute_node_masses",
    "export_to_json",
    "print_statistics",
    "get_encoder",
    "identify_basins",
    "summarize_basin",
    "compute_basin_statistics",
    "narrate_basin",
    "narrate_all_basins",
    "fast_summarize_basin",
    "format_basin_summary",
]


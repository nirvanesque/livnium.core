"""
Mindmap Demo: Ingest → Embed → Measure → Export

Processes files from sources/ folder and core docs, exports the thought graph.
"""

from pathlib import Path
from livnium.domains.mindmap import (
    ingest_markdown,
    ingest_sources_folder,
    embed_thoughts,
    measure_thought_graph,
    compute_node_masses,
    export_to_json,
    print_statistics,
    get_encoder,
    identify_basins,
    compute_basin_statistics,
    narrate_all_basins,
    fast_summarize_basin,
    format_basin_summary,
)


def main():
    """Run the mindmap demo."""
    repo_root = Path(__file__).parent.parent.parent.parent
    all_nodes = []
    
    # 1. Ingest files from sources/ folder (user's dumped files)
    sources_dir = Path(__file__).parent / "sources"
    print("Scanning sources/ folder...")
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        print("  (Install tqdm for progress bars: pip install tqdm)")
    
    sources_files = ingest_sources_folder(sources_dir)
    if sources_files:
        all_nodes.extend(sources_files)
        print(f"  → Found {len(sources_files)} nodes from sources/ folder")
    else:
        print("  → No files found in sources/ folder (add .txt, .md, or code files)")
    
    # 2. Ingest core documentation files (optional)
    markdown_files = [
        repo_root / "livnium" / "README.md",
        repo_root / "livnium" / "ARCHITECTURE.md",
        repo_root / "livnium" / "QUICKSTART.md",
    ]
    
    existing_files = [f for f in markdown_files if f.exists()]
    if existing_files:
        print(f"\nIngesting {len(existing_files)} core documentation files...")
        for file_path in existing_files:
            print(f"  → {file_path.name}...")
            nodes = ingest_markdown(file_path)
            all_nodes.extend(nodes)
            print(f"    {len(nodes)} nodes")
    
    if not all_nodes:
        print("\nNo files found! Add files to:")
        print(f"  {sources_dir}")
        print("\nSupported: .txt, .md, .py, .js, .ts, .java, .cpp, .go, .rs, etc.")
        return
    
    print(f"\nTotal: {len(all_nodes)} ThoughtNodes")
    
    # Embed (try sentence-transformers, fallback to simple)
    print("\nEmbedding thoughts...")
    try:
        encoder = get_encoder(use_sentence_transformers=True)
        print("  → Using sentence-transformers")
    except ImportError:
        encoder = get_encoder(use_sentence_transformers=False)
        print("  → Using simple encoder (install sentence-transformers for better results)")
    
    embed_thoughts(all_nodes, encoder, show_progress=True)
    embedded_count = len([n for n in all_nodes if n.vector is not None])
    print(f"\n  → Embedded {embedded_count} nodes")
    
    # Measure geometry
    print("\nMeasuring thought graph...")
    edges = measure_thought_graph(all_nodes, alignment_threshold=0.4, show_progress=True)
    print(f"\n  → Found {len(edges)} edges")
    
    # Compute masses
    compute_node_masses(all_nodes, edges)
    
    # Print statistics
    print_statistics(all_nodes, edges)
    
    # Detect basins (emergent stable attractors)
    print("\nDetecting basins...")
    basins = identify_basins(
        all_nodes,
        edges,
        min_basin_size=5,
        alignment_threshold=0.7,
        max_basins=20
    )
    
    if basins:
        # Build node map for narration
        node_map = {node.id: node for node in all_nodes}
        
        # Narrate basins (convert to statements)
        print("\nNarrating basins...")
        basins = narrate_all_basins(basins, node_map, edges)
        
        # Add fast summaries (for viewer tooltips)
        print("Generating fast summaries...")
        for basin in basins:
            try:
                fast_summary = fast_summarize_basin(
                    basin["id"],
                    all_nodes,
                    edges,
                    basin["node_ids"]
                )
                basin["fast_summary"] = fast_summary
            except Exception as e:
                print(f"  ⚠ Warning: Could not generate fast summary for {basin['id']}: {e}")
        
        basin_stats = compute_basin_statistics(basins)
        print(f"\n=== Basin Statistics ===")
        print(f"Number of basins: {basin_stats['num_basins']}")
        print(f"Total nodes in basins: {basin_stats['total_nodes']}")
        print(f"Average basin size: {basin_stats['avg_basin_size']:.1f}")
        print(f"Average stability: {basin_stats['avg_stability']:.3f}")
        print(f"Average alignment: {basin_stats['avg_alignment']:.3f}")
        print(f"Average tension: {basin_stats['avg_tension']:.3f}")
        print("\nTop basins with narratives:")
        for i, basin in enumerate(basins[:5], 1):
            print(f"\n  {i}. {basin['summary'][:60]}...")
            print(f"     Size: {basin['size']}, Stability: {basin['stability']:.3f}")
            if 'narrative' in basin:
                print(f"     Narrative: {basin['narrative'][:120]}...")
        print("=" * 32)
    else:
        print("  → No basins detected (try lowering alignment_threshold or min_basin_size)")
    
    # Export (limit to 100K edges max for browser performance)
    output_path = repo_root / "livnium" / "tools" / "visualize" / "mindmap.json"
    export_to_json(all_nodes, edges, output_path, max_edges=100000, basins=basins)
    
    print(f"\nDone! View the graph at: {output_path}")


if __name__ == "__main__":
    main()


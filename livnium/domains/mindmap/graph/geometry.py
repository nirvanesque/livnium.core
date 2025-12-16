"""
Mindmap Geometry: Measure Thought Graph Using Kernel Physics

Uses kernel.physics (alignment, divergence, tension) to measure
relationships between ThoughtNodes. This is measurement, not collapse.
"""

from typing import List, Dict, Any
import torch
from livnium.kernel.physics import alignment, divergence, tension
from livnium.engine.ops_torch import TorchOps
from ..ingestion.ingest import ThoughtNode


class ThoughtState:
    """
    State wrapper for ThoughtNode to work with kernel.physics.
    
    This implements the State protocol so kernel.physics can measure it.
    """
    def __init__(self, node: ThoughtNode):
        self.node = node
        if node.vector is None:
            raise ValueError(f"ThoughtNode {node.id} has no vector - embed first")
    
    def vector(self):
        """Return the vector representation."""
        return self.node.vector
    
    def norm(self):
        """Return the norm of the state vector."""
        return torch.norm(self.node.vector, p=2)


def measure_thought_graph(
    thoughts: List[ThoughtNode],
    alignment_threshold: float = 0.4,
    show_progress: bool = True,
    use_batching: bool = True,
    batch_size: int = 1000
) -> List[Dict[str, Any]]:
    """
    Measure relationships between ThoughtNodes using kernel physics.
    
    This is pure measurement - surveying the idea-field.
    No collapse, no dynamics, just observation.
    
    Uses vectorized batching for speed when use_batching=True.
    
    Args:
        thoughts: List of ThoughtNodes (must have vectors)
        alignment_threshold: Minimum alignment to include edge (visualization threshold, NOT a law)
        show_progress: Whether to show progress bar
        use_batching: Use vectorized batch operations (much faster for large graphs)
        batch_size: Batch size for vectorized operations
        
    Returns:
        List of edge dictionaries with physics measurements
    """
    import torch
    
    # Filter to nodes with vectors
    valid_thoughts = [t for t in thoughts if t.vector is not None]
    
    if len(valid_thoughts) < 2:
        return []
    
    edges = []
    
    if use_batching and len(valid_thoughts) > 100:
        # Vectorized batch approach - much faster for large graphs
        # Stack all vectors into a matrix
        vectors = torch.stack([t.vector for t in valid_thoughts])  # [N, dim]
        
        # Normalize all vectors at once
        vectors_norm = torch.nn.functional.normalize(vectors, p=2, dim=1)  # [N, dim]
        
        # Compute all pairwise alignments using matrix multiplication
        # alignment_matrix[i,j] = cosine similarity between node i and node j
        alignment_matrix = torch.mm(vectors_norm, vectors_norm.t())  # [N, N]
        
        # Get upper triangle (avoid duplicates and self-comparisons)
        N = len(valid_thoughts)
        triu_indices = torch.triu_indices(N, N, offset=1)  # Get (i,j) pairs where j > i
        
        # Extract alignments for all pairs
        alignments = alignment_matrix[triu_indices[0], triu_indices[1]]  # [num_pairs]
        
        # Compute divergence and tension for all pairs at once
        from livnium.kernel.constants import DIVERGENCE_PIVOT
        divergences = DIVERGENCE_PIVOT - alignments
        tensions = torch.abs(divergences)
        
        # Filter by threshold
        mask = alignments > alignment_threshold
        valid_indices = torch.where(mask)[0]
        
        if show_progress:
            try:
                from tqdm import tqdm
                valid_indices = tqdm(valid_indices, desc="Filtering edges", unit="edges", total=len(valid_indices))
            except ImportError:
                pass
        
        # Build edge list
        # Detach tensors to avoid gradient warnings
        alignments_detached = alignments.detach()
        divergences_detached = divergences.detach()
        tensions_detached = tensions.detach()
        
        for idx in valid_indices:
            i = int(triu_indices[0][idx])
            j = int(triu_indices[1][idx])
            
            edges.append({
                "source": valid_thoughts[i].id,
                "target": valid_thoughts[j].id,
                "alignment": float(alignments_detached[idx]),
                "divergence": float(divergences_detached[idx]),
                "tension": float(tensions_detached[idx])
            })
    else:
        # Original pairwise approach (for small graphs or when batching disabled)
        ops = TorchOps()
        
        try:
            from tqdm import tqdm
            outer_iterator = tqdm(enumerate(valid_thoughts), total=len(valid_thoughts), desc="Measuring", unit="nodes") if show_progress else enumerate(valid_thoughts)
        except ImportError:
            outer_iterator = enumerate(valid_thoughts)
        
        for i, a_node in outer_iterator:
            a_state = ThoughtState(a_node)
            
            for j, b_node in enumerate(valid_thoughts):
                if j <= i:
                    continue
                
                b_state = ThoughtState(b_node)
                
                # Use kernel physics exactly as designed
                align = alignment(ops, a_state, b_state)
                div = divergence(ops, a_state, b_state)
                tens = tension(ops, div)
                
                # Only include edges above threshold (for visualization)
                if align > alignment_threshold:
                    edges.append({
                        "source": a_node.id,
                        "target": b_node.id,
                        "alignment": float(align),
                        "divergence": float(div),
                        "tension": float(tens)
                    })
    
    return edges


def compute_node_masses(thoughts: List[ThoughtNode], edges: List[Dict[str, Any]]) -> None:
    """
    Compute mass for each ThoughtNode based on connectivity.
    
    Mass = sum of alignments to other nodes (normalized).
    This is a simple heuristic - could be refined later.
    
    Args:
        thoughts: List of ThoughtNodes
        edges: List of edges (from measure_thought_graph)
    """
    # Build alignment sums per node
    alignment_sums = {node.id: 0.0 for node in thoughts}
    
    for edge in edges:
        alignment_sums[edge["source"]] += edge["alignment"]
        alignment_sums[edge["target"]] += edge["alignment"]
    
    # Normalize to [0, 1] range
    if alignment_sums:
        max_sum = max(alignment_sums.values())
        if max_sum > 0:
            for node in thoughts:
                node.mass = alignment_sums[node.id] / max_sum


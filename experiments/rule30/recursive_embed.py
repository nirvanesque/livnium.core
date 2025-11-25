"""
Recursive Geometry Embedding for Rule 30

Embeds Rule 30 sequences into recursive Livnium geometry for multi-scale analysis.
This enables detection of fractal patterns and self-similarity at different scales.
"""

from typing import List, Tuple, Dict
from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig
from core.recursive.recursive_geometry_engine import RecursiveGeometryEngine, GeometryLevel
from experiments.rule30.geometry_embed import Rule30Path, embed_into_cube


def embed_recursive(
    center_seq: List[int],
    base_cube_size: int = 3,
    max_depth: int = 3
) -> Tuple[RecursiveGeometryEngine, Dict[int, Rule30Path]]:
    """
    Embed Rule 30 sequence into recursive geometry at multiple scales.
    
    Creates a recursive hierarchy where:
    - Level 0: Full sequence embedded in base cube
    - Level 1+: Subsequences embedded in child cubes
    
    This enables multi-scale analysis of Rule 30's fractal structure.
    
    Args:
        center_seq: Binary sequence from Rule 30 center column
        base_cube_size: Size of base omcube (3, 5, etc.)
        max_depth: Maximum recursion depth
        
    Returns:
        Tuple of (RecursiveGeometryEngine, dict mapping level_id to Rule30Path)
    """
    # Create base geometry
    config = LivniumCoreConfig(
        lattice_size=base_cube_size,
        enable_symbolic_weight=True,
        enable_face_exposure=True,
        enable_global_observer=True
    )
    
    base_system = LivniumCoreSystem(config)
    
    # Embed full sequence at level 0
    _, base_path = embed_into_cube(center_seq, base_cube_size)
    
    # Create recursive engine
    engine = RecursiveGeometryEngine(
        base_geometry=base_system,
        max_depth=max_depth
    )
    
    # Store paths at each level
    paths_by_level: Dict[int, Rule30Path] = {0: base_path}
    
    # Embed subsequences at deeper levels
    # Each level analyzes a different scale of the sequence
    # Recursive engine stores levels hierarchically, so we need to traverse children
    def traverse_levels(parent_level, current_depth):
        """Recursively traverse levels and embed subsequences."""
        if current_depth > max_depth:
            return
        
        # Process children of this level
        for coords, child_level in parent_level.children.items():
            level_id = child_level.level_id
            
            # Extract subsequence for this scale
            # Scale down: take every 2^level_id-th element
            scale_factor = 2 ** level_id
            subsequence = center_seq[::scale_factor]
            
            if subsequence and len(subsequence) > 0:
                # Get level geometry
                child_cube_size = child_level.geometry.config.lattice_size
                
                # Create path for this scale
                child_path = Rule30Path(subsequence, child_cube_size)
                paths_by_level[level_id] = child_path
                
                # Embed into level geometry (simplified - embed into first available cell)
                # In a full implementation, we'd embed into all child geometries
                for coord, phi in zip(child_path.get_coordinates(), child_path.get_states()):
                    boundary = (child_cube_size - 1) // 2
                    x, y, z = coord
                    if -boundary <= x <= boundary and -boundary <= y <= boundary and -boundary <= z <= boundary:
                        cell = child_level.geometry.get_cell(coord)
                        if cell:
                            cell.phi_value = phi
            
            # Recursively process grandchildren
            traverse_levels(child_level, current_depth + 1)
    
    # Start traversal from level 0
    base_level = engine.levels[0]
    traverse_levels(base_level, current_depth=1)
    
    return engine, paths_by_level


def analyze_recursive_patterns(
    engine: RecursiveGeometryEngine,
    paths_by_level: Dict[int, Rule30Path]
) -> Dict[int, Dict[str, float]]:
    """
    Analyze Rule 30 patterns at multiple recursive scales.
    
    Computes diagnostics at each level to detect self-similarity.
    
    Args:
        engine: RecursiveGeometryEngine instance
        paths_by_level: Dict mapping level_id to Rule30Path
        
    Returns:
        Dict mapping level_id to diagnostic summary
    """
    from experiments.rule30.diagnostics import compute_all_diagnostics
    
    results = {}
    
    for level_id, path in paths_by_level.items():
        diagnostics = compute_all_diagnostics(path)
        
        # Compute summary statistics
        import numpy as np
        results[level_id] = {
            'sequence_length': len(path.sequence),
            'divergence_mean': float(np.mean(diagnostics['divergence'])),
            'divergence_std': float(np.std(diagnostics['divergence'])),
            'tension_mean': float(np.mean(diagnostics['tension'])),
            'tension_std': float(np.std(diagnostics['tension'])),
            'basin_depth_mean': float(np.mean(diagnostics['basin_depth'])),
            'basin_depth_std': float(np.std(diagnostics['basin_depth']))
        }
    
    return results


"""
3D Map Visualization for Ramsey Solver

Shows the entire system state in 3D:
- Lattice cells (3D positions)
- Edge mappings (which edges map to which cells)
- SW values (color-coded by magnitude)
- Violations (highlighted)
- Constraint relationships
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from core.classical.livnium_core_system import LivniumCoreSystem
    from core.config import LivniumCoreConfig
    from experiments.ramsey.ramsey_encoder import RamseyEncoder
    from experiments.ramsey.ramsey_tension import count_monochromatic_k4, count_monochromatic_k3
except ImportError:
    # Try relative imports
    try:
        from ..core.classical.livnium_core_system import LivniumCoreSystem
        from ..core.config import LivniumCoreConfig
        from .ramsey_encoder import RamseyEncoder
        from .ramsey_tension import count_monochromatic_k4, count_monochromatic_k3
    except ImportError:
        print("Warning: Could not import core modules. Visualization may be limited.")
        LivniumCoreSystem = None
        LivniumCoreConfig = None
        RamseyEncoder = None
        count_monochromatic_k4 = None
        count_monochromatic_k3 = None


def create_3d_map(
    system: LivniumCoreSystem,
    encoder: RamseyEncoder,
    coloring: Dict[Tuple[int, int], int],
    vertices: List[int],
    constraint_type: str = "k4",
    show_edges: bool = True,
    show_violations: bool = True,
    show_sw_values: bool = True,
    show_constraints: bool = False,
    title: str = "Ramsey 3D Map"
) -> None:
    """
    Create a comprehensive 3D visualization of the Ramsey solver state.
    
    Args:
        system: The Livnium core system
        encoder: The Ramsey encoder
        coloring: Current edge coloring
        vertices: List of vertex indices
        constraint_type: "k3" or "k4"
        show_edges: Show edge mappings
        show_violations: Highlight violated constraints
        show_sw_values: Color-code by SW values
        show_constraints: Show constraint relationships (can be slow)
        title: Plot title
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create 3D subplot
    ax = fig.add_subplot(111, projection='3d')
    
    # Get lattice size
    n = system.config.lattice_size
    
    # Collect all cell positions and their properties
    cell_positions = []
    cell_sw_values = []
    cell_colors = []
    cell_sizes = []
    cell_labels = []
    
    # Get violation information
    violations = []
    if show_violations:
        if constraint_type == "k3":
            violations = _get_violated_k3s(coloring, vertices)
        else:
            violations = _get_violated_k4s(coloring, vertices)
    
    # Map edges to violation counts
    edge_violation_map = {}
    for violation in violations:
        if constraint_type == "k3":
            # K3 violation = triangle, get its 3 edges
            edges = [(min(violation[i], violation[j]), max(violation[i], violation[j])) 
                     for i in range(3) for j in range(i+1, 3)]
        else:
            # K4 violation = 4 vertices, get its 6 edges
            edges = [(min(violation[i], violation[j]), max(violation[i], violation[j])) 
                     for i in range(4) for j in range(i+1, 4)]
        for edge in edges:
            edge_violation_map[edge] = edge_violation_map.get(edge, 0) + 1
    
    # Process all cells
    for x in range(n):
        for y in range(n):
            for z in range(n):
                cell = system.get_cell((x, y, z))
                if cell is None:
                    continue
                
                cell_positions.append((x, y, z))
                sw = cell.symbolic_weight
                cell_sw_values.append(sw)
                
                # Find if this cell maps to an edge
                edge_for_cell = None
                for edge, coord in encoder.edge_to_coords.items():
                    if coord == (x, y, z):
                        edge_for_cell = edge
                        break
                
                # Determine color based on what we're showing
                if show_violations and edge_for_cell and edge_for_cell in edge_violation_map:
                    # Violated edge - RED
                    violation_count = edge_violation_map[edge_for_cell]
                    cell_colors.append((1.0, 0.0, 0.0, min(1.0, 0.3 + violation_count * 0.1)))
                    cell_sizes.append(100 + violation_count * 50)
                    cell_labels.append(f"Edge {edge_for_cell}\nViolations: {violation_count}\nSW: {sw:.2f}")
                elif show_sw_values:
                    # Color by SW value
                    # Normalize SW to [-1, 1] range for color mapping
                    sw_normalized = np.clip(sw / 20.0, -1, 1)
                    if sw_normalized > 0:
                        # Positive SW = blue (color 1)
                        cell_colors.append((0.0, 0.0, 1.0, 0.5 + abs(sw_normalized) * 0.5))
                    else:
                        # Negative SW = red (color 0)
                        cell_colors.append((1.0, 0.0, 0.0, 0.5 + abs(sw_normalized) * 0.5))
                    cell_sizes.append(50)
                    if edge_for_cell:
                        cell_labels.append(f"Edge {edge_for_cell}\nSW: {sw:.2f}\nColor: {coloring.get(edge_for_cell, '?')}")
                    else:
                        cell_labels.append(f"SW: {sw:.2f}")
                else:
                    # Default: gray
                    cell_colors.append((0.5, 0.5, 0.5, 0.3))
                    cell_sizes.append(30)
                    if edge_for_cell:
                        cell_labels.append(f"Edge {edge_for_cell}")
                    else:
                        cell_labels.append("")
    
    # Convert to numpy arrays
    positions = np.array(cell_positions)
    colors = np.array(cell_colors)
    sizes = np.array(cell_sizes)
    
    # Plot cells
    if len(positions) > 0:
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=colors,
            s=sizes,
            alpha=0.6,
            edgecolors='black',
            linewidths=0.5
        )
    
    # Draw edge connections (if showing edges)
    if show_edges:
        for edge, coord in encoder.edge_to_coords.items():
            x, y, z = coord
            color_val = coloring.get(edge, 0)
            edge_color = 'blue' if color_val == 1 else 'red'
            
            # Draw a small marker for the edge
            ax.scatter([x], [y], [z], c=edge_color, s=200, marker='s', 
                      edgecolors='black', linewidths=2, alpha=0.8)
    
    # Draw constraint relationships (if requested - can be slow)
    if show_constraints:
        _draw_constraints(ax, encoder, violations, constraint_type, coloring)
    
    # Set labels and title
    ax.set_xlabel('X (Lattice)')
    ax.set_ylabel('Y (Lattice)')
    ax.set_zlabel('Z (Lattice)')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                          positions[:, 1].max() - positions[:, 1].min(),
                          positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                  markersize=10, label='Color 1 (Blue)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                  markersize=10, label='Color 0 (Red)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                  markersize=10, label='Violated Edge', alpha=0.8),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.show()


def _get_violated_k3s(coloring: Dict[Tuple[int, int], int], vertices: List[int]) -> List[Tuple[int, int, int]]:
    """Get list of violated K3 triangles."""
    violations = []
    n = len(vertices)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                edge1 = (min(i, j), max(i, j))
                edge2 = (min(i, k), max(i, k))
                edge3 = (min(j, k), max(j, k))
                
                color1 = coloring.get(edge1, 0)
                color2 = coloring.get(edge2, 0)
                color3 = coloring.get(edge3, 0)
                
                # Monochromatic triangle
                if color1 == color2 == color3:
                    violations.append((i, j, k))
    return violations


def _get_violated_k4s(coloring: Dict[Tuple[int, int], int], vertices: List[int]) -> List[Tuple[int, int, int, int]]:
    """Get list of violated K4 cliques."""
    violations = []
    n = len(vertices)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    # Get all 6 edges
                    edges = [
                        (min(i, j), max(i, j)),
                        (min(i, k), max(i, k)),
                        (min(i, l), max(i, l)),
                        (min(j, k), max(j, k)),
                        (min(j, l), max(j, l)),
                        (min(k, l), max(k, l))
                    ]
                    
                    colors = [coloring.get(edge, 0) for edge in edges]
                    
                    # Monochromatic K4
                    if len(set(colors)) == 1:
                        violations.append((i, j, k, l))
    return violations


def _draw_constraints(ax, encoder, violations, constraint_type, coloring):
    """Draw constraint relationships (can be slow for large graphs)."""
    # For each violation, draw lines connecting the involved edges
    for violation in violations[:50]:  # Limit to first 50 for performance
        if constraint_type == "k3":
            # Triangle: 3 vertices, 3 edges
            v1, v2, v3 = violation
            edges = [
                (min(v1, v2), max(v1, v2)),
                (min(v1, v3), max(v1, v3)),
                (min(v2, v3), max(v2, v3))
            ]
        else:
            # K4: 4 vertices, 6 edges
            v1, v2, v3, v4 = violation
            edges = [
                (min(v1, v2), max(v1, v2)),
                (min(v1, v3), max(v1, v3)),
                (min(v1, v4), max(v1, v4)),
                (min(v2, v3), max(v2, v3)),
                (min(v2, v4), max(v2, v4)),
                (min(v3, v4), max(v3, v4))
            ]
        
        # Get coordinates for these edges
        coords = []
        for edge in edges:
            if edge in encoder.edge_to_coords:
                coords.append(encoder.edge_to_coords[edge])
        
        # Draw lines connecting violated edges
        if len(coords) >= 2:
            coords_array = np.array(coords)
            for i in range(len(coords_array)):
                for j in range(i+1, len(coords_array)):
                    ax.plot(
                        [coords_array[i][0], coords_array[j][0]],
                        [coords_array[i][1], coords_array[j][1]],
                        [coords_array[i][2], coords_array[j][2]],
                        'r--', alpha=0.3, linewidth=1
                    )


def visualize_ramsey_state(
    system,
    encoder,
    vertices: List[int],
    constraint_type: str = "k4",
    title: Optional[str] = None
) -> None:
    """
    Convenience function to visualize current state.
    
    Args:
        system: The Livnium core system
        encoder: The Ramsey encoder
        vertices: List of vertex indices
        constraint_type: "k3" or "k4"
        title: Optional title for the plot
    """
    # Decode current coloring
    coloring = encoder.decode_coloring()
    
    # Count violations
    if constraint_type == "k3":
        num_violations = count_monochromatic_k3(coloring, vertices)
    else:
        num_violations = count_monochromatic_k4(coloring, vertices)
    
    # Create title
    if title is None:
        n_vertices = len(vertices)
        total_constraints = len(encoder.k3_subsets) if constraint_type == "k3" else len(encoder.k4_subsets)
        percent_satisfied = (1.0 - num_violations / total_constraints) * 100 if total_constraints > 0 else 0
        title = f"Ramsey K_{n_vertices} ({constraint_type.upper()}) - {percent_satisfied:.2f}% Satisfied ({num_violations} violations)"
    
    # Create visualization
    create_3d_map(
        system=system,
        encoder=encoder,
        coloring=coloring,
        vertices=vertices,
        constraint_type=constraint_type,
        show_edges=True,
        show_violations=True,
        show_sw_values=True,
        show_constraints=False,  # Can be slow
        title=title
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    if LivniumCoreSystem is None or LivniumCoreConfig is None:
        print("Error: Could not import required modules. Please run from project root.")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="3D Map Visualization for Ramsey Solver")
    parser.add_argument("--n", type=int, default=5, help="Number of vertices")
    parser.add_argument("--constraint", type=str, default="k4", choices=["k3", "k4"], help="Constraint type")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to load")
    
    args = parser.parse_args()
    
    # Create system
    n_vertices = args.n
    n_edges = n_vertices * (n_vertices - 1) // 2
    n_lattice = max(3, int((n_edges) ** (1/3)) + 1)
    if n_lattice % 2 == 0:
        n_lattice += 1
    
    config = LivniumCoreConfig(lattice_size=n_lattice, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    # Create encoder
    encoder = RamseyEncoder(system, n_vertices)
    
    # Encode constraints
    if args.constraint == "k3":
        encoder.encode_k3_constraints()
    else:
        encoder.encode_k4_constraints()
    
    # Load from checkpoint if provided
    if args.checkpoint:
        try:
            from experiments.ramsey.ramsey_checkpoint import load_checkpoint, restore_system_state
            checkpoint = load_checkpoint(args.checkpoint)
            if checkpoint:
                restore_system_state(system, checkpoint['sw_values'])
                print(f"Loaded checkpoint from step {checkpoint['step']}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    
    # Visualize
    vertices = list(range(n_vertices))
    visualize_ramsey_state(system, encoder, vertices, args.constraint)


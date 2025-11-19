"""
Ramsey Visualizer: Live Visualization of System State

Shows the "inside" of the Ramsey solver:
- Graph coloring (edge colors)
- SW values (symbolic weights)
- Violation counts per edge
- Progress percentage
- Real-time updates
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx
from collections import defaultdict

# Handle imports (works both as module and standalone)
try:
    from .ramsey_encoder import RamseyEncoder
    from .ramsey_local_feedback_patch import compute_edge_violation_counts
except ImportError:
    try:
        from ramsey_encoder import RamseyEncoder
        from ramsey_local_feedback_patch import compute_edge_violation_counts
    except ImportError:
        # If running from project root
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from ramsey_encoder import RamseyEncoder
        from ramsey_local_feedback_patch import compute_edge_violation_counts

Edge = Tuple[int, int]


class RamseyVisualizer:
    """
    Live visualizer for Ramsey solver state.
    
    Shows:
    - Graph with edge colors (red/blue)
    - SW values (symbolic weights) as edge thickness
    - Violation counts as edge heatmap
    - Progress percentage
    """
    
    def __init__(self, n_vertices: int, constraint_type: str = "k4"):
        self.n_vertices = n_vertices
        self.constraint_type = constraint_type
        self.fig = None
        self.ax_graph = None
        self.ax_sw = None
        self.ax_violations = None
        self.ax_progress = None
        
        # Build graph layout
        self.G = nx.complete_graph(n_vertices)
        self.pos = nx.spring_layout(self.G, k=2, iterations=50, seed=42)
        
        # History for progress plot
        self.step_history = []
        self.percent_history = []
        self.violations_history = []
        
    def setup_plot(self):
        """Setup the matplotlib figure with subplots."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(f'Ramsey K_{self.n_vertices} Solver - Live View', fontsize=16, fontweight='bold')
        
        # Graph visualization (top left)
        self.ax_graph = plt.subplot(2, 2, 1)
        self.ax_graph.set_title('Graph Coloring (Red=0, Blue=1)', fontsize=12)
        self.ax_graph.axis('off')
        
        # SW values (top right)
        self.ax_sw = plt.subplot(2, 2, 2)
        self.ax_sw.set_title('Symbolic Weights (SW)', fontsize=12)
        self.ax_sw.set_xlabel('Edge Index')
        self.ax_sw.set_ylabel('SW Value')
        
        # Violation counts (bottom left)
        self.ax_violations = plt.subplot(2, 2, 3)
        self.ax_violations.set_title('Violation Counts per Edge', fontsize=12)
        self.ax_violations.set_xlabel('Edge Index')
        self.ax_violations.set_ylabel('Number of Violated Constraints')
        
        # Progress over time (bottom right)
        self.ax_progress = plt.subplot(2, 2, 4)
        self.ax_progress.set_title('Progress Over Time', fontsize=12)
        self.ax_progress.set_xlabel('Step')
        self.ax_progress.set_ylabel('% Satisfied')
        self.ax_progress.set_ylim([0, 100])
        self.ax_progress.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.ion()  # Interactive mode
        plt.show()
    
    def update(
        self,
        system,
        encoder: RamseyEncoder,
        coloring: Dict[Edge, int],
        step: int,
        violations: int,
        best_violations: int
    ):
        """
        Update visualization with current state.
        
        Args:
            system: LivniumCoreSystem
            encoder: RamseyEncoder
            coloring: Current edge coloring
            step: Current step number
            violations: Current violation count
            best_violations: Best violation count so far
        """
        if self.fig is None:
            self.setup_plot()
        
        # Calculate percentage
        if self.constraint_type == "k3":
            total_constraints = len(encoder.k3_subsets)
        else:
            total_constraints = len(encoder.k4_subsets)
        
        if total_constraints > 0:
            satisfied = total_constraints - violations
            percent_satisfied = (satisfied / total_constraints) * 100.0
        else:
            percent_satisfied = 0.0
        
        # Update history
        self.step_history.append(step)
        self.percent_history.append(percent_satisfied)
        self.violations_history.append(violations)
        
        # Keep history reasonable size
        if len(self.step_history) > 1000:
            self.step_history = self.step_history[-1000:]
            self.percent_history = self.percent_history[-1000:]
            self.violations_history = self.violations_history[-1000:]
        
        # 1. Graph visualization
        self.ax_graph.clear()
        self.ax_graph.set_title(f'Graph Coloring (Step {step}, {percent_satisfied:.2f}% satisfied)', fontsize=12)
        self.ax_graph.axis('off')
        
        # Draw edges with colors
        edge_colors = []
        edge_widths = []
        for edge in self.G.edges():
            # Normalize edge to (i, j) with i < j
            e = (min(edge), max(edge))
            if e in coloring:
                color_val = coloring[e]
                edge_colors.append('red' if color_val == 0 else 'blue')
            else:
                edge_colors.append('gray')
            
            # Get SW value for edge width
            if e in encoder.edge_to_coords:
                coord = encoder.edge_to_coords[e]
                cell = system.get_cell(coord)
                if cell:
                    sw = abs(cell.symbolic_weight)
                    edge_widths.append(max(0.5, min(sw / 5.0, 3.0)))  # Scale to reasonable width
                else:
                    edge_widths.append(0.5)
            else:
                edge_widths.append(0.5)
        
        nx.draw_networkx_edges(
            self.G, self.pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.6,
            ax=self.ax_graph
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.G, self.pos,
            node_color='lightgray',
            node_size=300,
            ax=self.ax_graph
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.G, self.pos,
            font_size=8,
            ax=self.ax_graph
        )
        
        # Legend
        red_patch = mpatches.Patch(color='red', label='Color 0')
        blue_patch = mpatches.Patch(color='blue', label='Color 1')
        self.ax_graph.legend(handles=[red_patch, blue_patch], loc='upper right')
        
        # 2. SW values bar chart
        self.ax_sw.clear()
        self.ax_sw.set_title('Symbolic Weights (SW)', fontsize=12)
        self.ax_sw.set_xlabel('Edge Index')
        self.ax_sw.set_ylabel('SW Value')
        
        sw_values = []
        edge_indices = []
        for i, edge in enumerate(sorted(encoder.edges)):
            if edge in encoder.edge_to_coords:
                coord = encoder.edge_to_coords[edge]
                cell = system.get_cell(coord)
                if cell:
                    sw_values.append(cell.symbolic_weight)
                    edge_indices.append(i)
        
        if sw_values:
            colors_sw = ['red' if sw < 0 else 'blue' for sw in sw_values]
            self.ax_sw.bar(edge_indices, sw_values, color=colors_sw, alpha=0.7)
            self.ax_sw.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            self.ax_sw.set_xlim([-1, len(edge_indices)])
        
        # 3. Violation counts
        self.ax_violations.clear()
        self.ax_violations.set_title('Violation Counts per Edge (Hot Patches)', fontsize=12)
        self.ax_violations.set_xlabel('Edge Index')
        self.ax_violations.set_ylabel('Number of Violated Constraints')
        
        edge_violations = compute_edge_violation_counts(coloring, encoder.vertices, self.constraint_type)
        violation_values = []
        violation_indices = []
        for i, edge in enumerate(sorted(encoder.edges)):
            if edge in edge_violations:
                violation_values.append(edge_violations[edge])
                violation_indices.append(i)
        
        if violation_values:
            # Color by violation count (red = many violations, green = few)
            max_violations = max(violation_values) if violation_values else 1
            colors_viol = [
                plt.cm.Reds(v / max_violations) if max_violations > 0 else 'green'
                for v in violation_values
            ]
            self.ax_violations.bar(violation_indices, violation_values, color=colors_viol, alpha=0.7)
            self.ax_violations.set_xlim([-1, len(violation_indices)])
            if max_violations > 0:
                self.ax_violations.set_ylim([0, max_violations * 1.1])
        
        # 4. Progress over time
        self.ax_progress.clear()
        self.ax_progress.set_title(f'Progress Over Time (Best: {best_violations} violations)', fontsize=12)
        self.ax_progress.set_xlabel('Step')
        self.ax_progress.set_ylabel('% Satisfied')
        self.ax_progress.set_ylim([0, 100])
        self.ax_progress.grid(True, alpha=0.3)
        
        if len(self.step_history) > 1:
            self.ax_progress.plot(self.step_history, self.percent_history, 'b-', alpha=0.7, linewidth=2)
            self.ax_progress.axhline(y=100, color='green', linestyle='--', linewidth=1, label='Perfect')
            self.ax_progress.legend(loc='lower right')
        
        # Update display
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)  # Small pause for interactivity
    
    def save_frame(self, filename: str):
        """Save current frame to file."""
        if self.fig is not None:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
    
    def close(self):
        """Close the visualization."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


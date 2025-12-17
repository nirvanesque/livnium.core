"""
Livnium Visualizer: Turning Physics into Insight
"""

import torch
import numpy as np
from typing import List, Optional

class PhysicsVisualizer:
    """
    Console-based visualization for semantic forces.
    """
    
    @staticmethod
    def print_force_map(claim_ids: List[str], correlations: torch.Tensor):
        """
        Prints a 2D grid representing the force field between claims.
        
        correlations: [N, N] matrix of alignment/correlation.
        """
        N = len(claim_ids)
        print("\n[Force Map] (+) Attraction | (-) Repulsion")
        
        # Header
        header = " " * 12
        for cid in claim_ids:
            header += f"{cid[:8]:>10}"
        print(header)
        print("-" * len(header))
        
        # Rows
        for i in range(N):
            row = f"{claim_ids[i][:10]:<12}"
            for j in range(N):
                val = correlations[i, j].item()
                if i == j:
                    char = "  .  "
                elif val > 0.8:
                    char = f" ++{val:.2f}" # Strong Attraction
                elif val > 0.1:
                    char = f" +{val:.2f}"  # Weak Attraction
                elif val < -0.4:
                    char = f" --{val:.2f}" # Strong Repulsion
                elif val < -0.1:
                    char = f" {val:.2f}"   # Weak Repulsion
                else:
                    char = "  0  "
                row += f"{char:>10}"
            print(row)

    @staticmethod
    def export_markdown_audit(claim_ids: List[str], final_align: torch.Tensor, summary: dict) -> str:
        """Generates a markdown report for the reasoning audit."""
        report = "# Livnium Reasoning Audit\n\n"
        report += "## Execution Summary\n"
        report += f"- **Initial Tension**: {summary['initial_tension']:.4f}\n"
        report += f"- **Final Tension**: {summary['final_tension']:.4f}\n"
        report += f"- **Reduction**: {summary['reduction_pct']:.2f}%\n"
        report += f"- **States**: {'MOKSHA (Stable)' if summary['is_stable'] else 'SEARCHING (Unstable)'}\n\n"
        
        report += "## Narrative Basin Map\n"
        # Table of alignments
        report += "| | " + " | ".join(claim_ids) + " |\n"
        report += "|---|" + "---| " * len(claim_ids) + "|\n"
        
        for i, cid in enumerate(claim_ids):
            vals = [f"{final_align[i, j].item():.2f}" for j in range(len(claim_ids))]
            report += f"| **{cid}** | " + " | ".join(vals) + " |\n"
            
        return report
            
    @staticmethod
    def print_tension_curve(history: List[float]):
        """Prints an ASCII bar chart of tension reduction."""
        if not history: return
        
        print("\n[Tension Curve] Reduction over time")
        max_t = max(history)
        width = 40
        
        for i, t in enumerate(history):
            if i % (len(history) // 10 or 1) == 0 or i == len(history) - 1:
                bar_len = int((t / max_t) * width) if max_t > 0 else 0
                bar = "█" * bar_len
                print(f"Step {i:2d}: {t:.4f} {bar}")

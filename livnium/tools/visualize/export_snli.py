"""
Export SNLI Model for Visualization

Exports basin anchors, collapse traces, and model geometry for visualization.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse

from livnium.domains.snli.encoder import SNLIEncoder
from livnium.domains.snli.head import SNLIHead
from livnium.engine.collapse.engine import CollapseEngine


class SNLIModel(torch.nn.Module):
    """Complete model for SNLI domain."""
    
    def __init__(self, dim: int = 256, num_layers: int = 5, vocab_size: int = 2000, 
                 enable_basins: bool = False, basin_threshold: str = "v4"):
        super().__init__()
        self.encoder = SNLIEncoder(dim=dim, vocab_size=vocab_size)
        self.collapse_engine = CollapseEngine(
            dim=dim, 
            num_layers=num_layers,
            enable_basins=enable_basins,
            basin_threshold=basin_threshold
        )
        self.head = SNLIHead(dim=dim)
    
    def forward(self, batch):
        """Forward pass."""
        prem_ids = batch["premise_ids"]
        hyp_ids = batch["hypothesis_ids"]
        labels = batch.get("label")
        
        h0, v_p, v_h = self.encoder.build_initial_state(prem_ids, hyp_ids)
        h_final, trace = self.collapse_engine.collapse(h0, labels=labels)
        logits = self.head(h_final, v_p, v_h)
        
        return logits, trace


def export_basin_anchors(model: SNLIModel, output_path: Path):
    """
    Export basin anchors for visualization.
    
    Args:
        model: Trained SNLI model
        output_path: Path to save JSON
    """
    # Always export static anchors, even if basins are disabled
    basin_field = model.collapse_engine.basin_field if model.collapse_engine.enable_basins else None
    
    # Collect all basin anchors
    nodes = []
    edges = []
    
    # Static anchors (always present)
    static_anchors = {
        "E": model.collapse_engine.anchor_entail.detach().cpu().numpy().tolist(),
        "N": model.collapse_engine.anchor_neutral.detach().cpu().numpy().tolist(),
        "C": model.collapse_engine.anchor_contra.detach().cpu().numpy().tolist(),
    }
    
    for label, anchor_vec in static_anchors.items():
        # Compute norm as "mass" for visualization
        mass = float(np.linalg.norm(anchor_vec))
        nodes.append({
            "id": f"static_{label}",
            "text": f"Static Anchor: {label}",
            "label": f"Static {label}",
            "type": "static_anchor",
            "mass": mass,
            "source": "SNLI Model",
            "vector": anchor_vec,
            "count": 0,
        })
    
    # Basin anchors (if basins are enabled and exist)
    total_basins = 0
    if basin_field is not None:
        for label in ["E", "N", "C"]:
            anchors = basin_field.anchors.get(label, [])
            total_basins += len(anchors)
            for idx, anchor in enumerate(anchors):
                center = anchor.center.detach().cpu().numpy().tolist()
            # Compute mass from count (more visits = higher mass)
            mass = min(1.0, anchor.count / 100.0)  # Normalize count to [0, 1]
            
            nodes.append({
                "id": f"basin_{label}_{idx}",
                "text": f"Basin {label}-{idx} (count: {anchor.count})",
                "label": f"Basin {label}-{idx}",
                "type": "basin_anchor",
                "basin_label": label,
                "mass": mass,
                "source": "SNLI Basins",
                "vector": center,
                "count": anchor.count,
                "last_used": anchor.last_used_step,
            })
            
            # Compute alignment between basin and static anchor
            static_vec = np.array(static_anchors[label])
            basin_vec = np.array(center)
            static_norm = static_vec / (np.linalg.norm(static_vec) + 1e-8)
            basin_norm = basin_vec / (np.linalg.norm(basin_vec) + 1e-8)
            alignment = float(np.dot(static_norm, basin_norm))
            
            # Connect basin to static anchor
            edges.append({
                "source": f"static_{label}",
                "target": f"basin_{label}_{idx}",
                "type": "basin_connection",
                "alignment": alignment,
                "divergence": 0.0,  # Could compute if needed
                "tension": 0.0,
            })
    
    # Connect static anchors (compute real alignments)
    static_vecs = {label: np.array(static_anchors[label]) for label in static_anchors}
    for label1 in ["E", "N", "C"]:
        for label2 in ["E", "N", "C"]:
            if label1 >= label2:  # Avoid duplicates
                continue
            vec1 = static_vecs[label1] / (np.linalg.norm(static_vecs[label1]) + 1e-8)
            vec2 = static_vecs[label2] / (np.linalg.norm(static_vecs[label2]) + 1e-8)
            alignment = float(np.dot(vec1, vec2))
            
            edges.append({
                "source": f"static_{label1}",
                "target": f"static_{label2}",
                "type": "static_connection",
                "alignment": alignment,
                "divergence": 0.0,
                "tension": 0.0,
            })
    
    output = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "model_dim": model.collapse_engine.dim,
            "num_layers": model.collapse_engine.num_layers,
            "basins_enabled": model.collapse_engine.enable_basins,
            "basin_threshold": basin_field.tension_threshold if basin_field else None,
            "total_basins": total_basins,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Exported {len(nodes)} nodes, {len(edges)} edges to {output_path}")
    print(f"  Static anchors: 3")
    print(f"  Basin anchors: {total_basins}")
    if model.collapse_engine.enable_basins and total_basins == 0:
        print(f"  ⚠️  Note: Basins are enabled but empty. This checkpoint was likely saved from a static collapse run.")
        print(f"     Train with --enable-basins to generate basin anchors.")


def export_collapse_traces(model: SNLIModel, dataset, num_samples: int = 100, output_path: Path = None):
    """
    Export collapse traces for visualization.
    
    Args:
        model: Trained SNLI model
        dataset: SNLI dataset
        num_samples: Number of samples to trace
        output_path: Path to save JSON
    """
    model.eval()
    traces = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            batch = {
                "premise_ids": sample["premise_ids"].unsqueeze(0),
                "hypothesis_ids": sample["hypothesis_ids"].unsqueeze(0),
                "label": sample["label"],
            }
            
            logits, trace = model(batch)
            
            # Extract trace data
            trace_data = {
                "sample_id": i,
                "label": int(sample["label"]),
                "prediction": int(logits.argmax().item()),
                "trajectory": [],
            }
            
            # Collect trajectory (simplified - just final state for now)
            # Could expand to show full step-by-step trajectory
            trace_data["trajectory"] = {
                "initial_state": None,  # Would need to capture h0
                "final_state": None,    # Would need to capture h_final
            }
            
            traces.append(trace_data)
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump({"traces": traces}, f, indent=2)
        print(f"Exported {len(traces)} collapse traces to {output_path}")
    
    return traces


def main():
    parser = argparse.ArgumentParser(description="Export SNLI model for visualization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="livnium/tools/visualize/snli_geometry.json", help="Output JSON path")
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--layers", type=int, default=5, help="Number of layers")
    parser.add_argument("--vocab-size", type=int, default=2000, help="Vocabulary size")
    parser.add_argument("--enable-basins", action="store_true", help="Model has basins enabled")
    parser.add_argument("--basin-threshold", type=str, default="v4", choices=["v3", "v4"])
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    model = SNLIModel(
        dim=args.dim,
        num_layers=args.layers,
        vocab_size=args.vocab_size,
        enable_basins=args.enable_basins,
        basin_threshold=args.basin_threshold
    )
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    print("Model loaded successfully")
    
    # Export basin anchors
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_basin_anchors(model, output_path)
    
    print(f"\nVisualization data exported to {output_path}")
    print(f"Open viewer.html in a browser to visualize")


if __name__ == "__main__":
    main()


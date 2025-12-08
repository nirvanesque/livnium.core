"""
Visualize the SNLI collapse geometry in the E/N/C plane.

Loads a trained checkpoint and projects:
- anchor vectors
- dynamic basin centers (if present)
- initial/final states for a sample of SNLI examples
- optional collapse trajectories

The projection plane is built from the three anchor directions:
u1 = normalize(e - n)
u2 = normalize((c - n) - proj_u1(c - n))
For any state h, x = (h_n dot u1), y = (h_n dot u2) where h_n is h normalized.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add nova_v3 and repo root (/nova) to import quantum_embed
nova_v3_root = Path(__file__).parent.parent
repo_root = nova_v3_root.parent
sys.path.insert(0, str(nova_v3_root))
sys.path.insert(0, str(repo_root))

from core import VectorCollapseEngine, BasinField
from core.physics_laws import divergence_from_alignment
from tasks.snli import (
    SNLIEncoder,
    GeometricSNLIEncoder,
    SanskritSNLIEncoder,
    QuantumSNLIEncoder,
)
from quantum_embed.text_encoder_quantum import QuantumTextEncoder
from training.train_snli_vector import SNLIDataset, load_snli_data


def build_projection_basis(collapse_engine: VectorCollapseEngine) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build orthonormal basis (u1, u2) spanning the E/N/C plane.
    """
    e = F.normalize(collapse_engine.anchor_entail.detach(), dim=0)
    c = F.normalize(collapse_engine.anchor_contra.detach(), dim=0)
    n = F.normalize(collapse_engine.anchor_neutral.detach(), dim=0)

    u1 = F.normalize(e - n, dim=0)
    c_rel = c - n
    c_rel_proj_u1 = (c_rel * u1).sum() * u1
    u2 = F.normalize(c_rel - c_rel_proj_u1, dim=0)
    return u1, u2


def project_vectors(vectors: torch.Tensor, u1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
    """
    Project vectors (B, dim) into 2D coordinates using the E/N/C plane.
    """
    h_n = F.normalize(vectors, dim=-1)
    x = torch.matmul(h_n, u1)
    y = torch.matmul(h_n, u2)
    return torch.stack([x, y], dim=-1)


@torch.no_grad()
def collapse_with_states_static(
    collapse_engine: VectorCollapseEngine, h0: torch.Tensor
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Run static collapse and return normalized states per layer.
    """
    h = h0.clone()
    squeeze = False
    if h.dim() == 1:
        h = h.unsqueeze(0)
        squeeze = True

    e_dir = F.normalize(collapse_engine.anchor_entail, dim=0)
    c_dir = F.normalize(collapse_engine.anchor_contra, dim=0)
    n_dir = F.normalize(collapse_engine.anchor_neutral, dim=0)

    states = [F.normalize(h, dim=-1)]
    for _ in range(collapse_engine.num_layers):
        h_n = F.normalize(h, dim=-1)
        a_e = (h_n * e_dir).sum(dim=-1)
        a_c = (h_n * c_dir).sum(dim=-1)
        a_n = (h_n * n_dir).sum(dim=-1)
        d_e = divergence_from_alignment(a_e)
        d_c = divergence_from_alignment(a_c)
        d_n = divergence_from_alignment(a_n)

        delta = collapse_engine.update(h)
        e_vec = F.normalize(h - e_dir.unsqueeze(0), dim=-1)
        c_vec = F.normalize(h - c_dir.unsqueeze(0), dim=-1)
        n_vec = F.normalize(h - n_dir.unsqueeze(0), dim=-1)
        h = (
            h
            + delta
            - collapse_engine.strength_entail * d_e.unsqueeze(-1) * e_vec
            - collapse_engine.strength_contra * d_c.unsqueeze(-1) * c_vec
            - collapse_engine.strength_neutral * d_n.unsqueeze(-1) * n_vec
        )
        h_norm = h.norm(p=2, dim=-1, keepdim=True)
        h = torch.where(h_norm > 10.0, h * (10.0 / (h_norm + 1e-8)), h)
        states.append(F.normalize(h, dim=-1))

    if squeeze:
        h = h.squeeze(0)
    return h, states


@torch.no_grad()
def collapse_with_states_dynamic(
    collapse_engine: VectorCollapseEngine,
    h0: torch.Tensor,
    labels: torch.Tensor,
    basin_field: BasinField,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Run dynamic collapse with basin routing and return normalized states per layer.
    """
    from core.basin_field import route_to_basin  # lazy import to avoid cycles

    h = h0
    squeeze = False
    if h.dim() == 1:
        h = h.unsqueeze(0)
        labels = labels.unsqueeze(0)
        squeeze = True
    h = h.clone()
    labels = labels.to(h.device)
    basin_field.to(h.device)

    label_to_char = {0: "E", 1: "C", 2: "N"}
    label_strength = {
        0: collapse_engine.strength_entail,
        1: collapse_engine.strength_contra,
        2: collapse_engine.strength_neutral,
    }

    anchors = []
    for i in range(h.size(0)):
        y_char = label_to_char.get(int(labels[i].item()))
        anchor, align_val, div_val, tens_val = route_to_basin(
            basin_field, h[i], y_char, step=0
        )
        anchors.append(anchor)

    anchor_dirs = torch.stack([a.center for a in anchors]).to(h.device)
    strengths = torch.tensor([label_strength[int(l.item())] for l in labels], device=h.device)

    states = [F.normalize(h, dim=-1)]
    for _ in range(collapse_engine.num_layers):
        h_n = F.normalize(h, dim=-1)
        align = (h_n * anchor_dirs).sum(dim=-1)
        div = divergence_from_alignment(align)

        delta = collapse_engine.update(h)
        anchor_vec = F.normalize(h - anchor_dirs, dim=-1)
        h = h + delta - strengths.unsqueeze(-1) * div.unsqueeze(-1) * anchor_vec
        h_norm = h.norm(p=2, dim=-1, keepdim=True)
        h = torch.where(h_norm > 10.0, h * (10.0 / (h_norm + 1e-8)), h)
        states.append(F.normalize(h, dim=-1))

    if squeeze:
        h = h.squeeze(0)
    return h, states


def plot_geometry(
    projection: Dict,
    output_path: Path,
    trajectories_only: bool = False,
    brain_wires: bool = False,
    axis_limit: float = None,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting; pip install matplotlib") from exc

    # Styling presets
    if brain_wires:
        colors = {"E": "#7dd3fc", "C": "#f472b6", "N": "#fcd34d"}
        bg = "#0b1221"
        grid = "#334155"
        fg = "#e5e7eb"
        traj_alpha = 0.5
        traj_width = 1.8
        anchor_size = 220
        basin_alpha = 0.25
        basin_size = 28
    else:
        colors = {"E": "#2563eb", "C": "#dc2626", "N": "#f59e0b"}
        bg = "#ffffff"
        grid = "#cccccc"
        fg = "#111111"
        traj_alpha = 0.25
        traj_width = 1.2
        anchor_size = 160
        basin_alpha = 1.0
        basin_size = 40

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axhline(0, color=grid, linewidth=1, linestyle="--", alpha=0.7)
    ax.axvline(0, color=grid, linewidth=1, linestyle="--", alpha=0.7)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(fg)

    if not trajectories_only:
        # Final states
        for label in ["E", "C", "N"]:
            pts = [(x, y) for x, y, l in projection["final_states"] if l == label]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, s=14, alpha=0.3, label=f"{label} final", color=colors[label])

        # Initial states (lighter)
        for label in ["E", "C", "N"]:
            pts = [(x, y) for x, y, l in projection["initial_states"] if l == label]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, s=10, alpha=0.1 if brain_wires else 0.15, color=colors[label], marker="x")

    # Basin centers
    for label, pts in projection["basins"].items():
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(
                xs,
                ys,
                s=basin_size,
                color=colors[label],
                alpha=basin_alpha,
                marker="s",
                edgecolors="#0f172a" if brain_wires else "k",
                linewidths=0.4,
                label=f"{label} basins",
            )

    # Anchors
    for label, coords in projection["anchors"].items():
        ax.scatter(
            [coords[0]],
            [coords[1]],
            s=anchor_size,
            color=colors[label],
            marker="*",
            edgecolors="#0f172a" if brain_wires else "k",
            linewidths=0.8,
            label=f"{label} anchor",
            zorder=5,
        )
        ax.text(coords[0], coords[1], f" {label}", fontsize=11 if brain_wires else 10, weight="bold", color=colors[label])

    # Trajectories
    for traj in projection["trajectories"]:
        xs, ys = zip(*traj["points"])
        ax.plot(xs, ys, color=colors[traj["label"]], alpha=traj_alpha, linewidth=traj_width)

    ax.set_title("SNLI collapse geometry (E/N/C plane)", color=fg)
    ax.set_xlabel("neutral -> entail (u1)", color=fg)
    ax.set_ylabel("neutral -> contra (u2)", color=fg)
    if axis_limit is not None:
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
    legend = ax.legend(loc="best", fontsize=8)
    if legend:
        legend.get_frame().set_edgecolor(fg)
        legend.get_frame().set_facecolor("#111827" if brain_wires else "#ffffff")
        for text in legend.get_texts():
            text.set_color(fg)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Project Livnium SNLI collapse into the E/N/C plane.")
    parser.add_argument("--model-dir", type=str, default=str(nova_v3_root / "model" / "snli_quantum_basins2"),
                        help="Directory containing best_model.pt")
    parser.add_argument("--snli-file", type=str, default=str(nova_v3_root / "data" / "snli" / "snli_1.0_dev.jsonl"),
                        help="SNLI JSONL file to sample from")
    parser.add_argument("--max-samples", type=int, default=256, help="Number of SNLI samples to visualize")
    parser.add_argument("--trace-samples", type=int, default=24, help="How many samples to keep full trajectories for")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--max-len", type=int, default=128, help="Max token length for vocab/quantum encoders")
    parser.add_argument("--output", type=str, default="snli_geometry.png", help="Output image path")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot and only print summary counts")
    parser.add_argument("--trajectories-only", action="store_true", help="Only plot anchors/basins/trajectories (omit initial/final scatters)")
    parser.add_argument("--brain-wires", action="store_true", help="Use a dark neon style to emphasize trajectories (concept art)")
    parser.add_argument("--axis-limit", type=float, default=None, help="If set, clamp axes to [-L, L] to zoom in")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)
    ckpt_path = model_dir / "best_model.pt"
    print(f"Loading checkpoint from {ckpt_path}")

    # weights_only must be False so Vocabulary object (if present) can be loaded
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    vocab = checkpoint.get("vocab")
    model_args = checkpoint["args"]
    use_dynamic_basins = checkpoint.get("use_dynamic_basins", False)
    basin_state = checkpoint.get("basin_field", None)

    collapse_engine = VectorCollapseEngine(
        dim=model_args.dim,
        num_layers=model_args.num_layers,
        strength_entail=getattr(model_args, "strength_entail", 0.1),
        strength_contra=getattr(model_args, "strength_contra", 0.1),
        strength_neutral=getattr(model_args, "strength_neutral", 0.05),
        basin_tension_threshold=getattr(model_args, "basin_tension_threshold", 0.15),
        basin_align_threshold=getattr(model_args, "basin_align_threshold", 0.6),
        basin_anchor_lr=getattr(model_args, "basin_anchor_lr", 0.05),
        basin_prune_min_count=getattr(model_args, "basin_prune_min_count", 10),
        basin_prune_merge_cos=getattr(model_args, "basin_merge_cos_threshold", 0.97),
    ).to(device)

    encoder_type = getattr(model_args, "encoder_type", "legacy")
    encode_fn = None
    quantum_tokenizer = None
    if encoder_type == "quantum":
        quantum_ckpt = getattr(model_args, "quantum_ckpt", None)
        if not quantum_ckpt:
            raise ValueError("encoder_type=quantum requires quantum_ckpt in saved args")
        # Resolve relative ckpt paths against model_dir first, then repo root
        quantum_ckpt_path = Path(quantum_ckpt)
        if not quantum_ckpt_path.is_absolute():
            candidates = [
                model_dir / quantum_ckpt_path,
                nova_v3_root / quantum_ckpt_path,
                repo_root / quantum_ckpt_path,
            ]
            for cand in candidates:
                if cand.exists():
                    quantum_ckpt_path = cand
                    break
        if not quantum_ckpt_path.exists():
            tried = ", ".join(str(c) for c in candidates)
            raise FileNotFoundError(
                f"Cannot find quantum_ckpt at {quantum_ckpt} (tried: {tried}). "
                "Pass an absolute path or adjust the checkpoint args."
            )

        quantum_tokenizer = QuantumTextEncoder(str(quantum_ckpt_path))

        def quantum_encode(text: str, max_len: int = args.max_len):
            tokens = quantum_tokenizer.tokenize(text)
            ids = [quantum_tokenizer.word2idx.get(t, quantum_tokenizer.unk_idx) for t in tokens]
            ids = ids[:max_len]
            if len(ids) < max_len:
                ids.extend([quantum_tokenizer.pad_idx] * (max_len - len(ids)))
            return ids

        encode_fn = quantum_encode

    if encoder_type == "geom":
        encoder = GeometricSNLIEncoder(
            dim=model_args.dim,
            norm_target=None,
            use_transformer=not getattr(model_args, "geom_disable_transformer", False),
            nhead=getattr(model_args, "geom_nhead", 4),
            num_layers=getattr(model_args, "geom_num_layers", 1),
            ff_mult=getattr(model_args, "geom_ff_mult", 2),
            dropout=getattr(model_args, "geom_dropout", 0.1),
            use_attention_pooling=not getattr(model_args, "geom_disable_attn_pool", False),
            token_norm_cap=(
                getattr(model_args, "geom_token_norm_cap", 3.0)
                if getattr(model_args, "geom_token_norm_cap", 3.0) > 0
                else None
            ),
        ).to(device)
        # SNLIDataset still expects an encode_fn; provide a harmless placeholder
        encode_fn = encode_fn or (lambda text, max_len: [0] * max_len)
    elif encoder_type == "sanskrit":
        if vocab is None:
            raise ValueError("Sanskrit encoder requires a saved vocabulary")
        encoder = SanskritSNLIEncoder(
            vocab_size=len(vocab),
            dim=model_args.dim,
            pad_idx=vocab.pad_idx,
            id_to_token=vocab.id_to_token_list() if hasattr(vocab, "id_to_token_list") else None,
        ).to(device)
    elif encoder_type == "quantum":
        encoder = QuantumSNLIEncoder(
            ckpt_path=str(quantum_ckpt_path),
        ).to(device)
    else:
        if vocab is None:
            raise ValueError("Legacy encoder requires a saved vocabulary")
        encoder = SNLIEncoder(
            vocab_size=len(vocab),
            dim=model_args.dim,
            pad_idx=vocab.pad_idx,
        ).to(device)

    collapse_engine.load_state_dict(checkpoint["collapse_engine"])
    encoder.load_state_dict(checkpoint["encoder"])
    collapse_engine.eval()
    encoder.eval()

    basin_field = None
    if use_dynamic_basins and basin_state is not None:
        basin_field = BasinField(max_basins_per_label=basin_state.get("max_basins_per_label", 64))
        basin_field.load_state_dict(basin_state)
        basin_field.to(device)
        print(
            "Loaded dynamic basins:",
            {k: len(v) for k, v in basin_field.anchors.items()},
        )

    samples = load_snli_data(Path(args.snli_file), max_samples=args.max_samples)
    dataset = SNLIDataset(samples, vocab, max_len=args.max_len, encode_fn=encode_fn)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    u1, u2 = build_projection_basis(collapse_engine)
    label_to_char = {0: "E", 1: "C", 2: "N"}

    projection = {
        "anchors": {},
        "basins": {"E": [], "C": [], "N": []},
        "initial_states": [],
        "final_states": [],
        "trajectories": [],
    }

    anchors = {
        "E": collapse_engine.anchor_entail.detach(),
        "C": collapse_engine.anchor_contra.detach(),
        "N": collapse_engine.anchor_neutral.detach(),
    }
    for label, vec in anchors.items():
        coords = project_vectors(vec.unsqueeze(0).to(device), u1, u2)[0]
        projection["anchors"][label] = (float(coords[0].cpu()), float(coords[1].cpu()))

    if basin_field is not None:
        for label, anchors_list in basin_field.anchors.items():
            if anchors_list:
                centers = torch.stack([a.center for a in anchors_list]).to(device)
                coords = project_vectors(centers, u1, u2).cpu().tolist()
                projection["basins"][label] = [(c[0], c[1]) for c in coords]

    traces_kept = 0
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(device)
            if isinstance(encoder, GeometricSNLIEncoder):
                h0, _, _ = encoder.build_initial_state(
                    batch["premise"],
                    batch["hypothesis"],
                    device=device,
                )
            else:
                prem_ids = batch["prem_ids"].to(device)
                hyp_ids = batch["hyp_ids"].to(device)
                h0, _, _ = encoder.build_initial_state(prem_ids, hyp_ids)

            if use_dynamic_basins and basin_field is not None:
                h_final, states = collapse_with_states_dynamic(
                    collapse_engine, h0, labels, basin_field
                )
            else:
                h_final, states = collapse_with_states_static(collapse_engine, h0)

            h0_proj = project_vectors(h0, u1, u2).cpu()
            h_final_proj = project_vectors(h_final, u1, u2).cpu()

            for i in range(h_final_proj.size(0)):
                label_char = label_to_char[int(labels[i].item())]
                projection["initial_states"].append(
                    (float(h0_proj[i, 0]), float(h0_proj[i, 1]), label_char)
                )
                projection["final_states"].append(
                    (float(h_final_proj[i, 0]), float(h_final_proj[i, 1]), label_char)
                )
                if traces_kept < args.trace_samples:
                    traj_proj = project_vectors(torch.stack([s[i] for s in states]), u1, u2).cpu().tolist()
                    projection["trajectories"].append(
                        {
                            "label": label_char,
                            "points": [(p[0], p[1]) for p in traj_proj],
                        }
                    )
                    traces_kept += 1

    print(
        f"Collected {len(projection['final_states'])} samples "
        f"({len(projection['trajectories'])} with full trajectories)."
    )
    if not args.no_plot:
        plot_geometry(
            projection,
            Path(args.output),
            trajectories_only=args.trajectories_only,
            brain_wires=args.brain_wires,
            axis_limit=args.axis_limit,
        )
    else:
        print("Plotting disabled (--no-plot).")


if __name__ == "__main__":
    main()

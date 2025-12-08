"""Probe geometry of a trained SNLI model without label-conditioned collapse.

Run from inside this model directory:

    python3 probe_geometry.py \
      --snli-file ../../data/snli/snli_1.0_dev.jsonl \
      --max-samples 5000 \
      --batch-size 64

Outputs aggregate stats for:
- pre-collapse alignment cos(v_p, v_h) grouped by gold label
- post-collapse alignment of h_final to each static anchor (E/N/C), no label routing

Notes:
- Uses static collapse only (no dynamic basins) to avoid label leakage.
- Works for legacy/geom/sanskrit/quantum encoders; quantum uses the checkpoint tokenizer.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Make nova_v3 and repo root importable
HERE = Path(__file__).resolve().parent
# .../nova/nova_v3/model/snli_quantum_basins -> parents[0]=model, [1]=nova_v3, [2]=nova, [3]=repo root
NOVA_V3_ROOT = HERE.parents[1]
NOVA_ROOT = HERE.parents[2]
REPO_ROOT = HERE.parents[3]
PROJECT_ROOT = REPO_ROOT.parent

# Force search order so we load nova_v3.* packages first
desired_paths = [str(NOVA_V3_ROOT), str(NOVA_ROOT), str(REPO_ROOT), str(PROJECT_ROOT)]
sys.path = desired_paths + [p for p in sys.path if p not in desired_paths]

from nova_v3.core import VectorCollapseEngine, BasinField  # noqa: E402
from nova_v3.tasks.snli import (  # noqa: E402
    SNLIEncoder,
    GeometricSNLIEncoder,
    SanskritSNLIEncoder,
    QuantumSNLIEncoder,
)
from quantum_embed.text_encoder_quantum import QuantumTextEncoder  # noqa: E402
from nova_v3.training.train_snli_vector import SNLIDataset, load_snli_data  # noqa: E402


LABEL_NAMES = {0: "entailment", 1: "contradiction", 2: "neutral"}


def percentile(arr: List[float], p: float) -> float:
    if not arr:
        return float("nan")
    return float(np.percentile(np.array(arr, dtype=float), p))


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {k: float("nan") for k in ["count", "mean", "std", "p10", "p50", "p90"]}
    arr = np.array(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": percentile(values, 10),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
    }


def resolve_quantum_ckpt_path(quantum_ckpt: Optional[str]) -> Path:
    if not quantum_ckpt:
        raise ValueError("encoder_type=quantum requires quantum_ckpt in the saved args")

    ckpt_path = Path(quantum_ckpt)
    search_roots = [Path.cwd(), HERE, HERE.parent, NOVA_V3_ROOT, NOVA_ROOT, REPO_ROOT]
    candidates: List[Path] = []

    if ckpt_path.is_absolute():
        candidates.append(ckpt_path)
    else:
        candidates.extend((root / ckpt_path).resolve() for root in search_roots)

    tried: List[Path] = []
    for cand in candidates:
        if cand in tried:
            continue
        tried.append(cand)
        if cand.is_file():
            return cand

    tried_str = "; ".join(str(p) for p in tried)
    raise FileNotFoundError(f"Quantum checkpoint not found at {quantum_ckpt}. Tried: {tried_str}")


def build_encode_fn(encoder_type: str, quantum_ckpt: Optional[str], max_len: int):
    if encoder_type != "quantum":
        return None, None
    ckpt_path = resolve_quantum_ckpt_path(quantum_ckpt)
    tokenizer = QuantumTextEncoder(str(ckpt_path))

    def quantum_encode(text: str, max_len: int = max_len):
        tokens = tokenizer.tokenize(text)
        ids = [tokenizer.word2idx.get(t, tokenizer.unk_idx) for t in tokens]
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids.extend([tokenizer.pad_idx] * (max_len - len(ids)))
        return ids

    return quantum_encode, tokenizer


def load_models(device: torch.device, model_dir: Path):
    ckpt = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    vocab = ckpt.get("vocab")
    args = ckpt["args"]
    encoder_type = getattr(args, "encoder_type", "legacy")
    use_dynamic_basins = ckpt.get("use_dynamic_basins", False)
    basin_state = ckpt.get("basin_field", None)
    basin_field = None
    if use_dynamic_basins and basin_state is not None:
        basin_field = BasinField(max_basins_per_label=basin_state.get("max_basins_per_label", 64))
        basin_field.load_state_dict(basin_state)
        basin_field.to(device)

    # Collapse engine (we will use static collapse only)
    collapse_engine = VectorCollapseEngine(
        dim=args.dim,
        num_layers=args.num_layers,
        strength_entail=getattr(args, "strength_entail", 0.1),
        strength_contra=getattr(args, "strength_contra", 0.1),
        strength_neutral=getattr(args, "strength_neutral", 0.05),
        basin_tension_threshold=getattr(args, "basin_tension_threshold", 0.15),
        basin_align_threshold=getattr(args, "basin_align_threshold", 0.6),
        basin_anchor_lr=getattr(args, "basin_anchor_lr", 0.05),
        basin_prune_min_count=getattr(args, "basin_prune_min_count", 10),
        basin_prune_merge_cos=getattr(args, "basin_merge_cos_threshold", 0.97),
    ).to(device)
    collapse_engine.load_state_dict(ckpt["collapse_engine"])

    quantum_ckpt_path: Optional[Path] = None
    if encoder_type == "quantum":
        quantum_ckpt_path = resolve_quantum_ckpt_path(getattr(args, "quantum_ckpt", None))

    encode_fn, quantum_tokenizer = build_encode_fn(
        encoder_type,
        quantum_ckpt_path if quantum_ckpt_path is not None else getattr(args, "quantum_ckpt", None),
        args.max_len,
    )
    vocab_id_to_token = vocab.id_to_token_list() if hasattr(vocab, "id_to_token_list") else None

    if encoder_type == "geom":
        encoder = GeometricSNLIEncoder(
            dim=args.dim,
            norm_target=None,
            use_transformer=not getattr(args, "geom_disable_transformer", False),
            nhead=getattr(args, "geom_nhead", 4),
            num_layers=getattr(args, "geom_num_layers", 1),
            ff_mult=getattr(args, "geom_ff_mult", 2),
            dropout=getattr(args, "geom_dropout", 0.1),
            use_attention_pooling=not getattr(args, "geom_disable_attn_pool", False),
            token_norm_cap=(getattr(args, "geom_token_norm_cap", 3.0) if getattr(args, "geom_token_norm_cap", 3.0) > 0 else None),
        ).to(device)
        encoder.load_state_dict(ckpt["encoder"])
    elif encoder_type == "sanskrit":
        if vocab is None:
            raise ValueError("Sanskrit encoder requires a saved vocabulary")
        encoder = SanskritSNLIEncoder(
            vocab_size=len(vocab),
            dim=args.dim,
            pad_idx=vocab.pad_idx,
            id_to_token=vocab_id_to_token,
        ).to(device)
        encoder.load_state_dict(ckpt["encoder"])
    elif encoder_type == "quantum":
        encoder = QuantumSNLIEncoder(ckpt_path=str(quantum_ckpt_path)).to(device)
        encoder.load_state_dict(ckpt["encoder"])
    else:
        if vocab is None:
            raise ValueError("Legacy encoder requires a saved vocabulary")
        encoder = SNLIEncoder(
            vocab_size=len(vocab),
            dim=args.dim,
            pad_idx=vocab.pad_idx,
        ).to(device)
        encoder.load_state_dict(ckpt["encoder"])

    return ckpt, args, vocab, encoder, collapse_engine, encode_fn, basin_field


def main():
    parser = argparse.ArgumentParser(description="Probe geometry of SNLI model without label-conditioned collapse")
    parser.add_argument("--snli-file", required=True, help="Path to SNLI JSONL file (dev/test)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit on samples")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu|mps")
    parser.add_argument(
        "--use-label-routing",
        action="store_true",
        help="Use label-routed dynamic basins (cheating: passes gold labels to collapse)",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    model_dir = HERE
    ckpt, model_args, vocab, encoder, collapse_engine, encode_fn, basin_field = load_models(device, model_dir)

    if args.use_label_routing and basin_field is None:
        print("Requested label routing, but checkpoint has no dynamic basin state; falling back to static collapse.", file=sys.stderr)
        args.use_label_routing = False
    if args.use_label_routing:
        print("Running with label-routed collapse (dynamic basins, gold labels provided). This is a cheating eval.")
    else:
        print("Running with static collapse (no label routing).")

    # Data
    samples = load_snli_data(Path(args.snli_file), max_samples=args.max_samples)
    dataset = SNLIDataset(samples, vocab, max_len=getattr(model_args, "max_len", 128), encode_fn=encode_fn)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    collapse_engine.eval()
    encoder.eval()

    pre_align: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    post_align_ent: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    post_align_neu: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    post_align_con: Dict[int, List[float]] = {0: [], 1: [], 2: []}

    with torch.no_grad():
        e_dir = F.normalize(collapse_engine.anchor_entail, dim=0)
        c_dir = F.normalize(collapse_engine.anchor_contra, dim=0)
        n_dir = F.normalize(collapse_engine.anchor_neutral, dim=0)

        for batch in loader:
            labels = batch["label"].to(device)

            if isinstance(encoder, GeometricSNLIEncoder):
                h0, v_p, v_h = encoder.build_initial_state(batch["premise"], batch["hypothesis"], device=device)
            else:
                prem_ids = batch["prem_ids"].to(device)
                hyp_ids = batch["hyp_ids"].to(device)
                h0, v_p, v_h = encoder.build_initial_state(prem_ids, hyp_ids)

            # Pre-collapse alignment
            v_p_n = F.normalize(v_p, dim=-1)
            v_h_n = F.normalize(v_h, dim=-1)
            align0 = (v_p_n * v_h_n).sum(dim=-1)

            # Collapse: either static (honest) or label-routed dynamic (cheating)
            if args.use_label_routing:
                h_final, _ = collapse_engine.collapse_dynamic(
                    h0,
                    labels,
                    basin_field,
                    global_step=0,
                    spawn_new=False,
                    prune_every=0,
                    update_anchors=False,
                )
            else:
                h_final, _ = collapse_engine.collapse(h0)
            h_final_n = F.normalize(h_final, dim=-1)

            align_ent = (h_final_n * e_dir).sum(dim=-1)
            align_con = (h_final_n * c_dir).sum(dim=-1)
            align_neu = (h_final_n * n_dir).sum(dim=-1)

            for i in range(labels.size(0)):
                lab = int(labels[i].item())
                pre_align[lab].append(float(align0[i].item()))
                post_align_ent[lab].append(float(align_ent[i].item()))
                post_align_con[lab].append(float(align_con[i].item()))
                post_align_neu[lab].append(float(align_neu[i].item()))

    # Report
    print("\n=== Pre-collapse alignment cos(v_p, v_h) ===")
    for lab, name in LABEL_NAMES.items():
        stats = summarize(pre_align[lab])
        print(f"{name:14s} count={stats['count']:6d} mean={stats['mean']:+.4f} std={stats['std']:.4f} p10={stats['p10']:+.4f} p50={stats['p50']:+.4f} p90={stats['p90']:+.4f}")

    mode_desc = "label-routed collapse (dynamic basins, gold labels passed)" if args.use_label_routing else "static collapse"
    print(f"\n=== Post-collapse alignment of h_final to anchors ({mode_desc}) ===")
    for lab, name in LABEL_NAMES.items():
        se = summarize(post_align_ent[lab])
        sn = summarize(post_align_neu[lab])
        sc = summarize(post_align_con[lab])
        print(f"{name:14s} → entail  mean={se['mean']:+.4f} p50={se['p50']:+.4f}; neutral mean={sn['mean']:+.4f} p50={sn['p50']:+.4f}; contra mean={sc['mean']:+.4f} p50={sc['p50']:+.4f}")


if __name__ == "__main__":
    main()

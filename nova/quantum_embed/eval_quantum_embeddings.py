"""
Evaluate trained quantum embeddings on a tokenized corpus.

Computes average Livnium energy loss over Skip-Gram pairs.
Default: disables dynamic basins during eval to avoid mutating anchor state.
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from train_quantum_embeddings import (
    Vocab,
    build_vocab_and_sequences,
    SkipGramDataset,
    sample_negative,
    livnium_energy_loss,
    QuantumEmbeddingModel,
)
from vector_collapse import VectorCollapseEngine
from basin_field import BasinField


def load_vocab_from_ckpt(vocab_data) -> Vocab:
    vocab = Vocab(max_size=len(vocab_data["idx2word"]))
    vocab.idx2word = vocab_data["idx2word"]
    vocab.word2idx = {w: i for i, w in enumerate(vocab.idx2word)}
    vocab.freqs = {}
    vocab.special_tokens = ["<pad>", "<unk>"]
    return vocab


def detect_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            print("🚀 Device Activated: mps")
            return torch.device("mps")
        if torch.cuda.is_available():
            print("🚀 Device Activated: cuda")
            return torch.device("cuda")
        print("⚠️ Running on CPU")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_collapse_from_ckpt(ckpt, dim: int, device: torch.device):
    use_dynamic = bool(ckpt.get("use_dynamic_basins", False))
    collapse_engine = None
    basin_field = None
    if use_dynamic and ckpt.get("collapse_engine") is not None and ckpt.get("basin_field") is not None:
        cfg = ckpt.get("collapse_config", {})
        collapse_engine = VectorCollapseEngine(
            dim=dim,
            num_layers=cfg.get("num_layers", 4),
            strength_entail=cfg.get("strength_entail", 0.1),
            strength_contra=cfg.get("strength_contra", 0.1),
            strength_neutral=cfg.get("strength_neutral", 0.05),
            basin_tension_threshold=cfg.get("basin_tension_threshold", 0.15),
            basin_align_threshold=cfg.get("basin_align_threshold", 0.6),
            basin_anchor_lr=cfg.get("basin_anchor_lr", 0.05),
            basin_prune_min_count=cfg.get("basin_prune_min_count", 10),
            basin_prune_merge_cos=cfg.get("basin_merge_cos_threshold", 0.97),
        )
        collapse_engine.load_state_dict(ckpt["collapse_engine"])
        collapse_engine.to(device)

        max_b = ckpt["basin_field"].get("active", torch.zeros(3, 1)).size(1)
        basin_field = BasinField(dim=dim, max_basins_per_label=max_b)
        basin_field.load_state_dict(ckpt["basin_field"])
        basin_field.to(device)
    return use_dynamic, collapse_engine, basin_field


def evaluate(
    ckpt_path: Path,
    test_path: Path,
    device: torch.device,
    batch_size: int,
    max_lines: int,
    disable_dynamic_basins: bool,
    window_size: int,
):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    vocab = load_vocab_from_ckpt(ckpt["vocab"])
    dim = ckpt["dim"]

    model = QuantumEmbeddingModel(vocab_size=len(vocab), dim=dim, pad_idx=vocab.pad_idx).to(device)
    model.emb.weight.data.copy_(ckpt["embeddings"].to(device))
    model.eval()

    use_dynamic_saved, collapse_engine, basin_field = load_collapse_from_ckpt(ckpt, dim, device)
    use_dynamic = use_dynamic_saved and not disable_dynamic_basins and collapse_engine is not None and basin_field is not None
    if use_dynamic:
        collapse_engine.eval()

    # Build test pairs using saved vocab (no new tokens)
    _, sequences = build_vocab_and_sequences(test_path, max_lines=max_lines, max_size=len(vocab))
    dataset = SkipGramDataset(sequences, window_size=window_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    total_loss = 0.0
    total_pairs = 0
    with torch.no_grad():
        for centers, contexts in loader:
            centers = centers.to(device, non_blocking=True)
            contexts = contexts.to(device, non_blocking=True)
            negatives = sample_negative(
                batch_size=centers.size(0),
                vocab_size=len(vocab),
                pad_idx=vocab.pad_idx,
                device=device,
            )
            loss = livnium_energy_loss(
                model,
                centers,
                contexts,
                negatives,
                collapse_engine=collapse_engine,
                basin_field=basin_field,
                use_dynamic_basins=use_dynamic,
                global_step=0,
                spawn_new=False,
                prune_every=0,
            )
            total_loss += loss.item() * centers.size(0)
            total_pairs += centers.size(0)

    avg_loss = total_loss / max(total_pairs, 1)
    print(f"Test avg loss: {avg_loss:.4f} over {total_pairs} pairs (dynamic_basins={'on' if use_dynamic else 'off'})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantum embeddings on a tokenized corpus.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (e.g., model_full_physics/quantum_embeddings_final.pt)")
    parser.add_argument("--test-path", type=str, required=True, help="Path to wiki.test.tokens or similar")
    parser.add_argument("--batch-size", type=int, default=2048, help="Eval batch size")
    parser.add_argument("--max-lines", type=int, default=0, help="Limit test lines (0 = all)")
    parser.add_argument("--window-size", type=int, default=2, help="Context window for Skip-Gram pairs")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--disable-dynamic-basins", action="store_true", help="Force dynamic basins off during eval")
    args = parser.parse_args()

    device = detect_device(args.device)
    evaluate(
        ckpt_path=Path(args.ckpt),
        test_path=Path(args.test_path),
        device=device,
        batch_size=args.batch_size,
        max_lines=args.max_lines,
        disable_dynamic_basins=args.disable_dynamic_basins,
        window_size=args.window_size,
    )


if __name__ == "__main__":
    main()

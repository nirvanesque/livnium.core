#!/usr/bin/env python
"""
Entry point to train ECW-BT Level-0 on wiki shards (JSONL).
Uses tunnel attraction + barrier repulsion and unit-sphere renorm per update.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from src import config as cfg
from src import data_loader, physics, trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ECW-BT Level-0")
    parser.add_argument(
        "--wiki-paths",
        nargs="+",
        default=cfg.default_wiki_paths(),
        help="JSONL shard paths (default: wiki_00 only)",
    )
    parser.add_argument("--mass-table", default="data/mass_table.json", help="mass table JSON output/input")
    parser.add_argument("--dim", type=int, default=cfg.ECWConfig.dim, help="embedding dimension")
    parser.add_argument("--window", type=int, default=cfg.ECWConfig.window, help="context window radius")
    parser.add_argument("--lr", type=float, default=cfg.ECWConfig.lr, help="learning rate")
    parser.add_argument("--negatives", type=int, default=cfg.ECWConfig.negatives, help="noise samples per target")
    parser.add_argument("--epochs", type=int, default=cfg.ECWConfig.epochs, help="epochs over shards")
    parser.add_argument("--device", default="auto", help="cpu/mps/cuda/auto")
    parser.add_argument("--min-freq", type=int, default=1, help="minimum token frequency to keep")
    parser.add_argument("--align-barrier", type=float, default=0.38, help="alignment threshold for barrier")
    parser.add_argument("--batch-size", type=int, default=cfg.ECWConfig.batch_size, help="Batch size for vectorized trainer")
    parser.add_argument("--catalyst", type=float, default=cfg.ECWConfig.catalyst, help="Resonance catalyst amplification (default 0 disables)")
    parser.add_argument(
        "--rebuild-mass",
        action="store_true",
        default=False,
        help="(disabled) mass table rebuild; reuse existing mass_table.json",
    )
    parser.add_argument("--subsample", type=float, default=None, help="Mikolov-style subsampling threshold (e.g., 1e-5)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Stop after this many token updates (for speed/debug)")
    parser.add_argument("--scan-total", action="store_true", help="Pre-scan shards to set an accurate progress bar total")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    ecw_cfg = cfg.ECWConfig(
        dim=args.dim,
        window=args.window,
        lr=args.lr,
        negatives=args.negatives,
        epochs=args.epochs,
        align_barrier=args.align_barrier,
        batch_size=args.batch_size,
        catalyst=args.catalyst,
        wiki_shards=args.wiki_paths,
        mass_table_path=Path(args.mass_table),
    )

    device = physics.pick_device(args.device)
    physics.set_seed(ecw_cfg.seed)
    print(f"[config] device={device}, dim={ecw_cfg.dim}, window={ecw_cfg.window}, lr={ecw_cfg.lr}")
    print(f"[data] shards={ecw_cfg.wiki_shards}")

    # Mass table
    mass_path = Path(args.mass_table)
    if args.rebuild_mass:
        raise SystemExit("[mass] rebuild disabled; reuse existing mass_table.json")
    if not mass_path.exists():
        raise SystemExit(f"[mass] missing {mass_path}. Rebuild has been disabled; create manually if needed.")
    vocab, freqs, masses, word_to_idx = data_loader.load_mass_table(mass_path)
    print(f"[mass] vocab={len(vocab)}; example: {vocab[:5]}")

    # Trainer
    tr = trainer.Trainer(ecw_cfg, vocab, masses, device=device)
    global_tokens = int(freqs.sum())
    print(f"[mass] global tokens in table: {global_tokens}")
    p_keep = None
    if args.subsample:
        freq_norm = freqs / global_tokens
        t = args.subsample
        # p_keep = (sqrt(f/t) + 1) * (t/f) ; clip to 1
        p_keep = (np.sqrt(freq_norm / t) + 1.0) * (t / freq_norm)
        p_keep = np.clip(p_keep, 0.0, 1.0).astype(np.float32)
        kept_est = int((p_keep * freqs).sum())
        print(f"[subsample] t={t} -> estimated kept tokens: {kept_est}")

    total_tokens = None
    if args.scan_total:
        print("[data] scanning shards to estimate total tokens...")
        total_tokens = data_loader.count_dataset_tokens(ecw_cfg.wiki_shards, word_to_idx, p_keep=p_keep)
        print(f"[data] scan total tokens: {total_tokens}")
    tr.train(
        ecw_cfg.wiki_shards,
        word_to_idx,
        total_tokens=total_tokens,
        p_keep=p_keep,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Phase-1A: Build a vocab and frequency table from Wikipedia JSONL shards.

Outputs:
- vocab.txt  (one token per line; id = line number)
- freq.npy   (int64 frequency aligned with vocab)

Optional:
- mass_table.json (ECW-BT legacy format used by train_ecw_bt.py tooling)

Example:
    python build_vocab.py \\
        --wiki-paths wikipedia/wiki_extractor_src/extracted/AA/wiki_00 \\
        --out-dir data \\
        --max-vocab 50000 \\
        --write-mass-table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src import data_loader


def parse_args():
    ap = argparse.ArgumentParser(description="Build ECW-BT vocab.txt + freq.npy from wiki JSONL shards")
    ap.add_argument(
        "--wiki-paths",
        nargs="+",
        required=True,
        help="One or more JSONL shard paths (wiki extractor output).",
    )
    ap.add_argument("--out-dir", default="data", help="Output directory")
    ap.add_argument("--max-vocab", type=int, default=200000, help="Top-N vocab cutoff (by frequency)")
    ap.add_argument("--min-freq", type=int, default=1, help="Minimum token frequency to keep")
    ap.add_argument("--write-mass-table", action="store_true", help="Also write data/mass_table.json")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[vocab] counting frequencies over {len(args.wiki_paths)} shard(s)...")
    counter = data_loader.count_frequencies(args.wiki_paths, data_loader.default_tokenizer, show_progress=True)

    vocab_items = [(w, f) for w, f in counter.items() if f >= args.min_freq]
    vocab_items.sort(key=lambda x: (-x[1], x[0]))
    if args.max_vocab and args.max_vocab > 0:
        vocab_items = vocab_items[: args.max_vocab]

    vocab = [w for w, _ in vocab_items]
    freqs = np.asarray([int(f) for _, f in vocab_items], dtype=np.int64)

    (out_dir / "vocab.txt").write_text("\n".join(vocab) + "\n")
    np.save(out_dir / "freq.npy", freqs)
    print(f"[save] vocab={len(vocab)} -> {out_dir/'vocab.txt'}")
    print(f"[save] freq -> {out_dir/'freq.npy'} (sum={int(freqs.sum())})")

    if args.write_mass_table:
        # Keep legacy tooling working.
        payload = data_loader.build_mass_table(
            args.wiki_paths,
            output_path=out_dir / "mass_table.json",
            min_freq=args.min_freq,
            tokenize=data_loader.default_tokenizer,
        )
        # Overwrite ordering to match vocab.txt/freq.npy top-N cut.
        payload["vocab"] = vocab
        payload["freq"] = freqs.tolist()
        payload["meta"]["max_vocab"] = args.max_vocab
        (out_dir / "mass_table.json").write_text(json.dumps(payload))
        print(f"[save] mass table -> {out_dir/'mass_table.json'}")


if __name__ == "__main__":
    main()



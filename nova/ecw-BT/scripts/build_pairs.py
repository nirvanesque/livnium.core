#!/usr/bin/env python
"""
Phase-1B: Build positive co-occurrence pairs (target, context) from Wikipedia JSONL shards.

Writes int32 pairs to shard files:
  out_dir/pairs_pos_000.bin, pairs_pos_001.bin, ...

Binary format: raw little-endian int32, shaped (-1, 2):
  [i0, j0, i1, j1, ...]

Example:
    python build_pairs.py \\
        --wiki-paths wikipedia/wiki_extractor_src/extracted/AA/wiki_00 \\
        --vocab data/vocab.txt \\
        --out-dir data/pairs \\
        --window 5 \\
        --pairs-per-shard 5000000 \\
        --symmetric
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from src import data_loader


def parse_args():
    ap = argparse.ArgumentParser(description="Build pair shards from wiki JSONL")
    ap.add_argument("--wiki-paths", nargs="+", required=True, help="JSONL shard paths")
    ap.add_argument("--vocab", required=True, help="Path to vocab.txt (id -> token)")
    ap.add_argument("--out-dir", default="data/pairs", help="Output directory for shard binaries")
    ap.add_argument("--window", type=int, default=5, help="Sliding window radius")
    ap.add_argument("--subsample", type=float, default=None, help="Mikolov subsample t (e.g., 1e-5). None disables.")
    ap.add_argument("--max-tokens", type=int, default=None, help="Stop after processing this many kept tokens (debug)")
    ap.add_argument("--pairs-per-shard", type=int, default=10_000_000, help="Pairs per shard before rolling")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--symmetric", action="store_true", help="Also write reversed (context,target) pairs")
    return ap.parse_args()


def load_vocab(path: str | Path):
    vocab = Path(path).read_text().splitlines()
    w2i = {w: i for i, w in enumerate(vocab) if w}
    return vocab, w2i


def subsample_keep_probs(freq: np.ndarray, t: float) -> np.ndarray:
    # word2vec: p_keep = (sqrt(f/t) + 1) * (t/f), clipped to 1
    total = float(freq.sum())
    f = freq.astype(np.float64) / max(total, 1.0)
    p = (np.sqrt(f / t) + 1.0) * (t / (f + 1e-12))
    return np.clip(p, 0.0, 1.0).astype(np.float32)


def write_shard(out_path: Path, pairs_buf: np.ndarray) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_buf.astype(np.int32, copy=False).tofile(out_path)
    return int(pairs_buf.shape[0])


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    vocab, w2i = load_vocab(args.vocab)
    vocab_size = len(vocab)
    print(f"[vocab] size={vocab_size} from {args.vocab}")

    p_keep = None
    if args.subsample is not None:
        # Build frequencies by streaming once. This is intentionally simple & deterministic.
        # For huge runs: prefer running build_vocab.py first and reusing its freq.npy.
        print("[subsample] counting frequencies for keep-probs...")
        counter = data_loader.count_frequencies(args.wiki_paths, data_loader.default_tokenizer, show_progress=True)
        freq = np.zeros(vocab_size, dtype=np.int64)
        for w, c in counter.items():
            idx = w2i.get(w)
            if idx is not None:
                freq[idx] = int(c)
        p_keep = subsample_keep_probs(freq, args.subsample)
        kept_est = int((p_keep * freq).sum())
        print(f"[subsample] t={args.subsample} estimated kept tokens={kept_est}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "pairs_meta.json"
    meta = {
        "wiki_paths": list(args.wiki_paths),
        "vocab": str(args.vocab),
        "window": int(args.window),
        "subsample": None if args.subsample is None else float(args.subsample),
        "pairs_per_shard": int(args.pairs_per_shard),
        "symmetric": bool(args.symmetric),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    shard_idx = 0
    pairs_in_shard = 0
    total_pairs = 0
    total_tokens = 0

    buf = []
    buf_pairs_target = args.pairs_per_shard

    def flush():
        nonlocal shard_idx, pairs_in_shard, total_pairs, buf
        if not buf:
            return
        pairs = np.concatenate(buf, axis=0)
        path = out_dir / f"pairs_pos_{shard_idx:03d}.bin"
        wrote = write_shard(path, pairs)
        total_pairs += wrote
        pairs_in_shard = 0
        shard_idx += 1
        buf = []
        print(f"[save] {path} pairs={wrote} total_pairs={total_pairs}")

    pbar = tqdm(unit="tok", smoothing=0.1, mininterval=1.0)
    pbar.set_description("pairs")

    for text in data_loader.stream_jsonl_texts(args.wiki_paths, progress=False):
        toks = data_loader.default_tokenizer(text)
        if not toks:
            continue

        # map to ids, optionally subsample
        ids = []
        if p_keep is None:
            for tok in toks:
                idx = w2i.get(tok)
                if idx is not None:
                    ids.append(idx)
        else:
            # vectorize keep decisions
            idxs = [w2i.get(tok) for tok in toks]
            idxs = [i for i in idxs if i is not None]
            if not idxs:
                continue
            idxs = np.asarray(idxs, dtype=np.int64)
            keep = rng.random(idxs.shape[0]) <= p_keep[idxs]
            ids = idxs[keep].tolist()

        if len(ids) < 2:
            continue

        total_tokens += len(ids)
        pbar.update(len(ids))
        if args.max_tokens is not None and total_tokens >= args.max_tokens:
            break

        arr = np.asarray(ids, dtype=np.int32)
        n = arr.shape[0]
        # generate pairs with numpy slicing per offset (small loop over offsets, no token loops)
        pairs_list = []
        for o in range(1, args.window + 1):
            a = arr[:-o]
            b = arr[o:]
            if a.size == 0:
                break
            pairs_list.append(np.stack([a, b], axis=1))
            if args.symmetric:
                pairs_list.append(np.stack([b, a], axis=1))
        if not pairs_list:
            continue
        pairs = np.concatenate(pairs_list, axis=0)

        buf.append(pairs)
        pairs_in_shard += pairs.shape[0]
        if pairs_in_shard >= buf_pairs_target:
            flush()

    pbar.close()
    flush()

    print(f"[done] total_tokens={total_tokens} total_pairs={total_pairs} shards={shard_idx}")


if __name__ == "__main__":
    main()



#!/usr/bin/env python
"""
Apply rotated/whitened/refined embeddings to ECW-BT and save as a new checkpoint.

Steps:
1. Load rotated vectors (.npy).
2. Load mass_table.json to get vocab size/order.
3. Verify shape matches; compute diagnostics (mean norm, per-dim variance).
4. Save a new ECW-BT-compatible checkpoint (.npy).
5. Print neighbors for a few probe words.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description="Swap ECW-BT vectors with rotated embeddings and save a new checkpoint.")
    ap.add_argument("--rotated", required=True, help="Path to rotated vectors (.npy)")
    ap.add_argument("--mass-table", required=True, help="Path to mass_table.json")
    ap.add_argument("--output", required=True, help="Path to save new checkpoint (.npy)")
    ap.add_argument(
        "--probes",
        nargs="*",
        default=["king", "queen", "man", "woman", "dog", "cat", "apple", "orange"],
        help="Probe words for neighbor diagnostics",
    )
    ap.add_argument("--topk", type=int, default=10, help="Neighbors to display per probe")
    return ap.parse_args()


def load_vocab(mass_table: Path) -> Tuple[List[str], dict]:
    obj = json.loads(mass_table.read_text())
    vocab = obj["vocab"]
    w2i = {w: i for i, w in enumerate(vocab)}
    return vocab, w2i


def diagnostics(vecs: np.ndarray, vocab: List[str], w2i: dict, probes: List[str], topk: int):
    norms = np.linalg.norm(vecs, axis=1)
    print(f"[diag] mean norm: {norms.mean():.4f}  std norm: {norms.std():.4f}")
    var_dim = vecs.var(axis=0)
    print(f"[diag] per-dim variance: mean={var_dim.mean():.6f} std={var_dim.std():.6f}")

    vecs_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    for word in probes:
        if word not in w2i:
            print(f"[probe] '{word}' not in vocab")
            continue
        idx = w2i[word]
        qv = vecs_norm[idx]
        sims = vecs_norm @ qv
        sims[idx] = -np.inf
        top_idx = np.argpartition(-sims, topk)[:topk]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        print(f"[probe] {word}:")
        for j in top_idx:
            print(f"  {vocab[j]:20s} cos={sims[j]:.4f}")


def main():
    args = parse_args()
    rotated_path = Path(args.rotated)
    mass_path = Path(args.mass_table)
    out_path = Path(args.output)

    print(f"[load] rotated vectors from {rotated_path}")
    V_rot = np.load(rotated_path)
    print(f"[load] mass table from {mass_path}")
    vocab, w2i = load_vocab(mass_path)

    if V_rot.shape[0] != len(vocab):
        raise SystemExit(f"Shape mismatch: vectors rows {V_rot.shape[0]} != vocab size {len(vocab)}")

    diagnostics(V_rot, vocab, w2i, args.probes, args.topk)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, V_rot.astype(np.float32))
    print(f"[save] new checkpoint -> {out_path}")


if __name__ == "__main__":
    main()

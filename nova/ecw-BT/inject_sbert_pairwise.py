#!/usr/bin/env python
"""
Post-process ECW-BT embeddings by injecting SBERT pairwise similarity constraints.
Fully vectorized; supports CPU/MPS/CUDA when torch is available, else falls back to NumPy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None


def parse_args():
    ap = argparse.ArgumentParser(description="Inject SBERT pairwise curvature into ECW-BT embeddings")
    ap.add_argument("--checkpoint", required=True, help="Path to ECW-BT vectors (.npy)")
    ap.add_argument("--mass-table", required=True, help="Path to mass_table.json")
    ap.add_argument("--output", required=True, help="Path to save updated vectors (.npy)")
    ap.add_argument("--pairs", type=int, default=300000, help="Number of random word pairs")
    ap.add_argument("--iterations", type=int, default=3, help="Number of refinement iterations")
    ap.add_argument("--lr", type=float, default=0.05, help="Learning rate for pairwise updates")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    return ap.parse_args()


def pick_device():
    if torch is None:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sample_pairs(vocab_size: int, num_pairs: int, rng: np.random.Generator):
    i = rng.integers(0, vocab_size, size=num_pairs, dtype=np.int64)
    j = rng.integers(0, vocab_size, size=num_pairs, dtype=np.int64)
    # avoid i == j
    mask_same = i == j
    while mask_same.any():
        j[mask_same] = rng.integers(0, vocab_size, size=mask_same.sum(), dtype=np.int64)
        mask_same = i == j
    return i, j


def encode_sbert(words):
    model = SentenceTransformer("all-mpnet-base-v2")
    emb = model.encode(words, batch_size=512, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ckpt = Path(args.checkpoint)
    mass = Path(args.mass_table)
    out = Path(args.output)

    print(f"[load] vectors from {ckpt}")
    V = np.load(ckpt).astype(np.float32)
    vocab = json.loads(mass.read_text())["vocab"]
    if V.shape[0] != len(vocab):
        raise SystemExit(f"Mismatch: vectors rows {V.shape[0]} != vocab size {len(vocab)}")
    vocab_size, dim = V.shape

    # sample pairs
    i_idx, j_idx = sample_pairs(vocab_size, args.pairs, rng)
    pair_words = list({vocab[k] for k in np.concatenate([i_idx, j_idx])})
    print(f"[pairs] sampled {args.pairs} pairs; unique words to encode: {len(pair_words)}")

    # SBERT encode
    sb_vecs = encode_sbert(pair_words)
    w2sb = {w: sb_vecs[k] for k, w in enumerate(pair_words)}

    # Build SBERT vectors aligned to pairs
    sb_i = np.stack([w2sb[vocab[k]] for k in i_idx]).astype(np.float32)
    sb_j = np.stack([w2sb[vocab[k]] for k in j_idx]).astype(np.float32)

    use_torch = torch is not None
    device = pick_device() if use_torch else "cpu"
    print(f"[device] using {device}")

    if use_torch:
        V_t = torch.tensor(V, device=device)
        i_t = torch.tensor(i_idx, device=device, dtype=torch.long)
        j_t = torch.tensor(j_idx, device=device, dtype=torch.long)
        sb_i_t = torch.tensor(sb_i, device=device)
        sb_j_t = torch.tensor(sb_j, device=device)
        for it in tqdm(range(args.iterations), desc="refine (torch)"):
            vi = V_t[i_t]
            vj = V_t[j_t]
            cos_ecw = (vi * vj).sum(dim=1)
            # normalize SBERT vectors already normalized; ensure
            cos_sb = (sb_i_t * sb_j_t).sum(dim=1)
            delta = cos_sb - cos_ecw  # (P,)
            # updates
            vi_new = vi + args.lr * delta.unsqueeze(1) * (vj - cos_ecw.unsqueeze(1) * vi)
            vj_new = vj + args.lr * delta.unsqueeze(1) * (vi - cos_ecw.unsqueeze(1) * vj)
            # scatter back
            V_t[i_t] = vi_new
            V_t[j_t] = vj_new
            # normalize all rows
            V_t = F.normalize(V_t, p=2, dim=1)
            if (it + 1) % max(1, args.iterations // 3) == 0:
                print(f"[iter {it+1}/{args.iterations}] mean delta={delta.abs().mean().item():.6f}")
        V_new = V_t.cpu().numpy()
    else:
        V_new = V.copy()
        for it in tqdm(range(args.iterations), desc="refine (numpy)"):
            vi = V_new[i_idx]
            vj = V_new[j_idx]
            cos_ecw = np.einsum("ij,ij->i", vi, vj)
            cos_sb = np.einsum("ij,ij->i", sb_i, sb_j)
            delta = cos_sb - cos_ecw  # (P,)
            vi_new = vi + args.lr * delta[:, None] * (vj - cos_ecw[:, None] * vi)
            vj_new = vj + args.lr * delta[:, None] * (vi - cos_ecw[:, None] * vj)
            V_new[i_idx] = vi_new
            V_new[j_idx] = vj_new
            norms = np.linalg.norm(V_new, axis=1, keepdims=True) + 1e-8
            V_new = V_new / norms
            if (it + 1) % max(1, args.iterations // 3) == 0:
                print(f"[iter {it+1}/{args.iterations}] mean delta={np.abs(delta).mean():.6f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, V_new.astype(np.float32))
    print(f"[save] updated vectors -> {out}")


if __name__ == "__main__":
    main()

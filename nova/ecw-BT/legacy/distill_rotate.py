#!/usr/bin/env python
"""
Align ECW-BT embeddings to SBERT via whitening + Procrustes rotation, with optional residual MLP.

Pipeline:
1) Load ECW vectors and vocab subset.
2) Mean-center ECW; compute PCA on subset and whiten full space.
3) Encode subset with SBERT (all-mpnet-base-v2); project SBERT to ECW dim via PCA if needed.
4) Procrustes rotation on whitened ECW vs SBERT subset.
5) Optional residual MLP fine-tune (small, fast).
6) Apply transform to all vectors and save.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import torch
    from torch import nn
except Exception:
    torch = None


def parse_args():
    ap = argparse.ArgumentParser(description="Rotate ECW-BT embeddings to align with SBERT")
    ap.add_argument("--checkpoint", required=True, help="Path to ECW-BT vectors (.npy)")
    ap.add_argument("--mass-table", required=True, help="Path to mass_table.json")
    ap.add_argument("--output", required=True, help="Path to save rotated vectors (.npy)")
    ap.add_argument("--limit", type=int, default=200000, help="Number of vocab words to use for alignment")
    ap.add_argument("--pca-eps", type=float, default=1e-6, help="Epsilon for whitening stability")
    ap.add_argument("--mlp-steps", type=int, default=0, help="Residual MLP fine-tune steps (0 disables)")
    ap.add_argument("--mlp-lr", type=float, default=1e-3, help="Learning rate for residual MLP")
    ap.add_argument("--mlp-batch", type=int, default=4096, help="Batch size for residual MLP training")
    return ap.parse_args()


def load_ecw(checkpoint: Path, mass_table: Path, limit: int) -> Tuple[np.ndarray, List[str], List[int]]:
    vectors = np.load(checkpoint)
    vocab = json.loads(mass_table.read_text())["vocab"]
    if limit and limit > 0:
        vocab = vocab[:limit]
    keep_words = []
    keep_idx = []
    for i, w in enumerate(vocab):
        if not w:
            continue
        keep_words.append(w)
        keep_idx.append(i)
    return vectors.astype(np.float32), keep_words, keep_idx


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return mat / norms


def compute_rotation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def pca_whiten(full_vectors: np.ndarray, subset_vectors: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = full_vectors.mean(axis=0, keepdims=True)
    X_center = full_vectors - mean
    X_sub = subset_vectors - mean
    U, S, Vt = np.linalg.svd(X_sub, full_matrices=False)
    S_inv = 1.0 / (S + eps)
    T = Vt.T * S_inv  # (d, d)
    X_white_full = X_center @ T
    X_white_sub = X_sub @ T
    return X_white_full, X_white_sub


def project_sbert(B: np.ndarray, target_dim: int) -> np.ndarray:
    if B.shape[1] == target_dim:
        return B
    Bc = B - B.mean(axis=0, keepdims=True)
    Ub, Sb, Vtb = np.linalg.svd(Bc, full_matrices=False)
    Vproj = Vtb[:target_dim, :].T  # (orig_dim, target_dim)
    return Bc @ Vproj


def diagnostics(words: List[str], w2i: dict, vecs_old: np.ndarray, vecs_new: np.ndarray):
    pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("paris", "france"),
        ("apple", "orange"),
        ("cat", "dog"),
    ]
    print("\n[diag] cosine before vs after:")
    for a, b in pairs:
        if a not in w2i or b not in w2i:
            continue
        ia, ib = w2i[a], w2i[b]
        v1_old = vecs_old[ia]
        v2_old = vecs_old[ib]
        v1_new = vecs_new[ia]
        v2_new = vecs_new[ib]
        cos_old = float(np.dot(v1_old, v2_old) / (np.linalg.norm(v1_old) * np.linalg.norm(v2_old) + 1e-8))
        cos_new = float(np.dot(v1_new, v2_new) / (np.linalg.norm(v1_new) * np.linalg.norm(v2_new) + 1e-8))
        print(f"  {a:>8s} ~ {b:<8s}: before={cos_old: .4f} after={cos_new: .4f}")


def train_residual_mlp(V_rot: np.ndarray, A_idx: List[int], B_target: np.ndarray, steps: int, lr: float, batch_size: int) -> np.ndarray:
    if torch is None or steps <= 0:
        return V_rot
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    d = V_rot.shape[1]
    model = nn.Sequential(
        nn.Linear(d, d * 2),
        nn.ReLU(),
        nn.Linear(d * 2, d),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    A_rot = torch.tensor(V_rot[A_idx], device=device)
    B_t = torch.tensor(B_target, device=device)
    n = A_rot.shape[0]
    bs = min(batch_size, n)
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), device=device)
        a_batch = A_rot[idx]
        b_batch = B_t[idx]
        pred = a_batch + model(a_batch)
        loss = torch.mean((pred - b_batch) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"[mlp] step {step+1}/{steps} loss={loss.item():.6f}")
    with torch.no_grad():
        V_full = torch.tensor(V_rot, device=device)
        V_refined = (V_full + model(V_full)).cpu().numpy()
    return V_refined


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint)
    mass = Path(args.mass_table)
    out = Path(args.output)

    print(f"[load] vectors from {ckpt}")
    V, vocab_subset, idx_subset = load_ecw(ckpt, mass, args.limit)
    print(f"[load] vocab subset size: {len(vocab_subset)} (limit={args.limit})")

    print("[encode] loading SBERT all-mpnet-base-v2 ...")
    model = SentenceTransformer("all-mpnet-base-v2")
    b_emb = model.encode(vocab_subset, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
    B = np.array(b_emb, dtype=np.float32)

    # Pre-align ECW: mean center, PCA, whiten
    A_subset = V[idx_subset]
    V_white, A_white = pca_whiten(V, A_subset, eps=args.pca_eps)
    d_ecw = V_white.shape[1]

    # Project SBERT to ECW dim if needed
    B_proj = project_sbert(B, d_ecw)
    B_proj = normalize_rows(B_proj)
    A_white = normalize_rows(A_white)

    # Ensure same sample size
    n = min(A_white.shape[0], B_proj.shape[0])
    A_white = A_white[:n]
    B_proj = B_proj[:n]
    print(f"[align] using {n} pairs for Procrustes")

    R = compute_rotation(A_white, B_proj)
    print("[align] rotation matrix computed")

    V_rot = V_white @ R

    if args.mlp_steps > 0:
        print(f"[mlp] training residual MLP ({args.mlp_steps} steps)")
        V_final = train_residual_mlp(V_rot, idx_subset[:n], B_proj, args.mlp_steps, args.mlp_lr, args.mlp_batch)
    else:
        V_final = V_rot

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, V_final.astype(np.float32))
    print(f"[save] rotated vectors -> {out}")

    w2i_full = {w: i for i, w in enumerate(json.loads(mass.read_text())["vocab"])}
    diagnostics(vocab_subset, w2i_full, V, V_final)


if __name__ == "__main__":
    main()

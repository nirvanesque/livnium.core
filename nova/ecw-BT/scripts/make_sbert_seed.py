#!/usr/bin/env python
"""
Phase-0: Create SBERT-seeded word vectors aligned to vocab.txt.

Loads vocab.txt, encodes each word with SBERT (sentence-transformers/all-mpnet-base-v2),
projects to target dimension if needed, normalizes, saves as V_seed.npy.

This becomes the immutable "teacher" geometry for Level-0 training.

Example:
    python make_sbert_seed.py \\
        --vocab data/vocab.txt \\
        --output data/V_seed.npy \\
        --dim 256 \\
        --model sentence-transformers/all-mpnet-base-v2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import torch
except Exception:
    torch = None


def parse_args():
    ap = argparse.ArgumentParser(description="Create SBERT-seeded vectors for ECW-BT Level-0")
    ap.add_argument("--vocab", required=True, help="Path to vocab.txt")
    ap.add_argument("--output", default="data/V_seed.npy", help="Output path for seed vectors")
    ap.add_argument("--dim", type=int, default=256, help="Target dimension (will project if SBERT dim != this)")
    ap.add_argument("--batch-size", type=int, default=512, help="SBERT encoding batch size")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2", help="SBERT model name (pinned for reproducibility)")
    return ap.parse_args()


def project_to_dim(vectors: np.ndarray, target_dim: int) -> np.ndarray:
    """Project vectors to target_dim via PCA (keep top components)."""
    if vectors.shape[1] == target_dim:
        return vectors
    # Mean-center
    mean = vectors.mean(axis=0, keepdims=True)
    centered = vectors - mean
    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Project to top target_dim components
    V_proj = Vt[:target_dim, :].T  # (orig_dim, target_dim)
    projected = centered @ V_proj
    return projected


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    """L2-normalize each row."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return mat / norms


def main():
    args = parse_args()
    vocab_path = Path(args.vocab)
    out_path = Path(args.output)

    print(f"[load] vocab from {vocab_path}")
    vocab = [line.strip() for line in vocab_path.read_text().splitlines() if line.strip()]
    vocab_size = len(vocab)
    print(f"[vocab] size={vocab_size}")

    print(f"[sbert] loading {args.model} (pinned for reproducibility)...")
    model = SentenceTransformer(args.model)
    sbert_dim = model.get_sentence_embedding_dimension()
    print(f"[sbert] dim={sbert_dim} target={args.dim}")

    print(f"[encode] encoding {vocab_size} words (batch={args.batch_size})...")
    embeddings = model.encode(
        vocab,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)
    print(f"[encode] shape={embeddings.shape}")

    if embeddings.shape[1] != args.dim:
        print(f"[project] {embeddings.shape[1]} -> {args.dim} via PCA...")
        embeddings = project_to_dim(embeddings, args.dim)
        embeddings = normalize_rows(embeddings)
        print(f"[project] done, shape={embeddings.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    print(f"[save] V_seed.npy -> {out_path} shape={embeddings.shape} dtype={embeddings.dtype}")
    print(f"[done] seed vectors ready for Level-0 training")


if __name__ == "__main__":
    main()


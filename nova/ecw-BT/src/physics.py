"""
Physics laws for ECW-BT (Level-0) - Batched & Vectorized.
Equilibrium attraction toward align_barrier shell and barrier repulsion.
- Smooth bounded shaping on divergence (tanh).
- Optional resonance catalyst to amplify aligned pairs.
"""

from __future__ import annotations

import random
import numpy as np

try:
    import torch
except Exception:
    torch = None

# Divergence shaping factor for tanh; higher saturates sooner
ALIGN_SHAPE = 2.0


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch is None:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def random_unit_vectors(count: int, dim: int, device: str):
    """
    Sample random unit vectors (Gaussian -> normalize).
    """
    if torch is None:
        vecs = np.random.randn(count, dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        return vecs / norms
    vecs = torch.randn(count, dim, device=device)
    vecs = vecs / (vecs.norm(dim=1, keepdim=True) + 1e-8)
    return vecs


def renorm_rows(mat):
    """
    Normalize rows of a matrix to unit length.
    """
    if torch is None or isinstance(mat, np.ndarray):
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        return mat / norms
    norms = mat.norm(dim=1, keepdim=True) + 1e-8
    mat /= norms
    return mat


def attraction_force(
    v_tgt,
    v_ctx,
    m_ctx,
    lr: float,
    align_barrier: float = 0.38,
    resonance_amp: float = 0.0,
):
    """
    Batched equilibrium attraction toward align_barrier shell.
    v_tgt: (B, D)
    v_ctx: (B, K, D)
    m_ctx: (B, K)
    """
    if torch is not None and isinstance(v_tgt, torch.Tensor):
        sims = (v_ctx * v_tgt.unsqueeze(1)).sum(dim=2)  # (B, K)
        d_raw = align_barrier - sims
        d_shaped = torch.tanh(d_raw * ALIGN_SHAPE)
        resonance = torch.relu(sims)  # only boost when aligned
        catalyst = 1.0 + resonance_amp * resonance
        factors = (d_shaped * m_ctx * catalyst) * lr  # (B, K)
        diff = v_ctx - v_tgt.unsqueeze(1)  # (B, K, D)
        force = (factors.unsqueeze(2) * diff).sum(dim=1)  # (B, D)
        return v_tgt + force

    sims = np.einsum("bkd,bd->bk", v_ctx, v_tgt)
    d_raw = align_barrier - sims
    d_shaped = np.tanh(d_raw * ALIGN_SHAPE)
    resonance = np.maximum(sims, 0.0)
    catalyst = 1.0 + resonance_amp * resonance
    factors = (d_shaped * m_ctx * catalyst) * lr
    diff = v_ctx - v_tgt[:, None, :]
    force = (factors[:, :, None] * diff).sum(axis=1)
    return v_tgt + force


def barrier_force(v_tgt, v_noise, m_noise, lr: float, align_barrier: float):
    """
    Batched barrier repulsion.
    v_tgt: (B, D)
    v_noise: (B, N, D)
    m_noise: (B, N)
    """
    if torch is not None and isinstance(v_tgt, torch.Tensor):
        align = (v_noise * v_tgt.unsqueeze(1)).sum(dim=2)  # (B, N)
        d_raw = align - align_barrier
        d_shaped = torch.tanh(d_raw * ALIGN_SHAPE)
        mask = (align > align_barrier).float()
        factors = d_shaped * m_noise * lr * mask  # (B, N)
        direction = v_tgt.unsqueeze(1) - v_noise  # (B, N, D)
        rep = (factors.unsqueeze(2) * direction).sum(dim=1)  # (B, D)
        return v_tgt + rep

    align = np.einsum("bnd,bd->bn", v_noise, v_tgt)
    d_raw = align - align_barrier
    d_shaped = np.tanh(d_raw * ALIGN_SHAPE)
    mask = (align > align_barrier).astype(np.float32)
    factors = d_shaped * m_noise * lr * mask
    direction = v_tgt[:, None, :] - v_noise
    rep = (factors[:, :, None] * direction).sum(axis=1)
    return v_tgt + rep


def batched_update(
    vectors,
    masses,
    target_idx,
    ctx_idx,
    ctx_mask,
    noise_idx,
    lr: float,
    align_barrier: float,
    resonance_amp: float = 0.0,
):
    """
    Fully batched update:
    - vectors: storage of shape (V, D)
    - masses: storage of shape (V,)
    - target_idx: (B,)
    - ctx_idx: (B, K)
    - ctx_mask: (B, K)
    - noise_idx: (B, N) or None
    Returns updated target vectors (B, D) normalized.
    """
    if torch is not None and isinstance(vectors, torch.Tensor):
        b_t = torch.as_tensor(target_idx, device=vectors.device, dtype=torch.long)
        b_ctx = torch.as_tensor(ctx_idx, device=vectors.device, dtype=torch.long)
        b_mask = torch.as_tensor(ctx_mask, device=vectors.device, dtype=torch.float32)
        v_t = vectors[b_t]  # (B, D)
        v_c = vectors[b_ctx]  # (B, K, D)
        m_c = masses[b_ctx] * b_mask  # (B, K)

        sims = (v_c * v_t.unsqueeze(1)).sum(dim=2)  # (B, K)
        d_raw = align_barrier - sims
        d_shaped = torch.tanh(d_raw * ALIGN_SHAPE)
        resonance = torch.relu(sims)
        catalyst = 1.0 + resonance_amp * resonance
        factors = (d_shaped * m_c * catalyst) * lr  # (B, K)
        diff = v_c - v_t.unsqueeze(1)
        att = (factors.unsqueeze(2) * diff).sum(dim=1)  # (B, D)

        rep = 0.0
        if noise_idx is not None:
            b_noise = torch.as_tensor(noise_idx, device=vectors.device, dtype=torch.long)
            v_n = vectors[b_noise]  # (B, N, D)
            m_n = masses[b_noise]  # (B, N)
            align_n = (v_n * v_t.unsqueeze(1)).sum(dim=2)  # (B, N)
            d_raw_n = align_n - align_barrier
            d_shaped_n = torch.tanh(d_raw_n * ALIGN_SHAPE)
            mask = (align_n > align_barrier).float()
            factors_n = d_shaped_n * m_n * lr * mask  # (B, N)
            direction = v_t.unsqueeze(1) - v_n
            rep = (factors_n.unsqueeze(2) * direction).sum(dim=1)  # (B, D)

        v_new = v_t + att + rep
        v_norm = torch.norm(v_new, dim=1, keepdim=True) + 1e-8
        v_new = v_new / v_norm
        return v_new

    # NumPy path
    b_t = np.asarray(target_idx, dtype=np.int64)
    b_ctx = np.asarray(ctx_idx, dtype=np.int64)
    b_mask = np.asarray(ctx_mask, dtype=np.float32)
    v_t = vectors[b_t]  # (B, D)
    v_c = vectors[b_ctx]  # (B, K, D)
    m_c = masses[b_ctx] * b_mask  # (B, K)
    sims = np.einsum("bkd,bd->bk", v_c, v_t)
    d_raw = align_barrier - sims
    d_shaped = np.tanh(d_raw * ALIGN_SHAPE)
    resonance = np.maximum(sims, 0.0)
    catalyst = 1.0 + resonance_amp * resonance
    factors = (d_shaped * m_c * catalyst) * lr
    diff = v_c - v_t[:, None, :]
    att = (factors[:, :, None] * diff).sum(axis=1)

    rep = 0.0
    if noise_idx is not None:
        b_noise = np.asarray(noise_idx, dtype=np.int64)
        v_n = vectors[b_noise]
        m_n = masses[b_noise]
        align_n = np.einsum("bnd,bd->bn", v_n, v_t)
        d_raw_n = align_n - align_barrier
        d_shaped_n = np.tanh(d_raw_n * ALIGN_SHAPE)
        mask = (align_n > align_barrier).astype(np.float32)
        factors_n = d_shaped_n * m_n * lr * mask
        direction = v_t[:, None, :] - v_n
        rep = (factors_n[:, :, None] * direction).sum(axis=1)

    v_new = v_t + att + rep
    v_norm = np.linalg.norm(v_new, axis=1, keepdims=True) + 1e-8
    v_new = v_new / v_norm
    return v_new

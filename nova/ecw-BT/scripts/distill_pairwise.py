#!/usr/bin/env python
"""
Phase-2: Batched CCD pairwise trainer for ECW-BT Level-0.

Reads pair shards (pairs_pos_*.bin), applies CCD force law with negatives,
uses index_add_ scatter updates (MPS-friendly), applies fusion anchoring
to prevent catastrophic forgetting, and tracks geometry drift metrics.

This is the core training loop from the Level-0 plan.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None


def parse_args():
    ap = argparse.ArgumentParser(description="Train ECW-BT Level-0 with batched CCD pairwise updates")
    ap.add_argument("--seed", required=True, help="Path to V_seed.npy (teacher, immutable)")
    ap.add_argument("--pairs", required=True, nargs="+", help="Pair shard paths (pairs_pos_*.bin)")
    ap.add_argument("--vocab", required=True, help="Path to vocab.txt")
    ap.add_argument("--freq", required=True, help="Path to freq.npy (for negative sampling)")
    ap.add_argument("--output", default="data/ecw_bt_vectors.npy", help="Output vectors path")
    ap.add_argument("--dim", type=int, default=256, help="Embedding dimension")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--negatives", type=int, default=5, help="Negatives per positive")
    ap.add_argument("--clip-norm", type=float, default=0.1, help="Gradient clipping norm")
    ap.add_argument("--force-cap", type=float, default=0.3, help="Force cap (or use tanh with tau)")
    ap.add_argument("--use-tanh", action="store_true", help="Use tanh force shaping instead of clip")
    ap.add_argument("--tau", type=float, default=0.05, help="Tanh temperature (if --use-tanh)")
    ap.add_argument("--align-barrier", type=float, default=0.38, help="CCD divergence pivot")
    ap.add_argument("--fusion-lambda", type=float, default=0.05, help="Fusion weight (lambda schedule: epoch1=0.05, epoch2=0.10, epoch3+=0.15)")
    ap.add_argument("--epochs", type=int, default=3, help="Epochs over pair shards")
    ap.add_argument("--pairs-per-step", type=int, default=1000000, help="Pairs per training step (larger = fewer scatters, faster)")
    ap.add_argument("--device", default="auto", help="cpu/mps/cuda/auto")
    ap.add_argument("--checkpoint-interval", type=int, default=20000000, help="Save checkpoint every N pairs (default: 20M to reduce I/O)")
    ap.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    ap.add_argument("--probe-words", nargs="+", default=["king", "queen", "paris", "france", "cat", "dog"], help="Words to track drift")
    ap.add_argument("--throttle", type=float, default=0.0, help="Sleep seconds between batches (default: 0, auto if --auto-optimize)")
    ap.add_argument("--cpu-threads", type=int, default=None, help="Limit PyTorch CPU threads (default: auto-detect, use 1-2 for lighter load)")
    ap.add_argument("--auto-optimize", action="store_true", default=False, help="Auto-adjust batch size and throttle (DISABLED by default - hurts performance)")
    ap.add_argument("--max-batch-size", type=int, default=2000000, help="Maximum batch size (auto-reduces if OOM)")
    ap.add_argument("--verbose", action="store_true", default=False, help="Enable verbose debug output")
    return ap.parse_args()


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


def load_vocab_freq(vocab_path: Path, freq_path: Path) -> tuple[list[str], np.ndarray]:
    """Load vocabulary and frequency arrays."""
    vocab = [line.strip() for line in vocab_path.read_text().splitlines() if line.strip()]
    freq = np.load(freq_path).astype(np.int64)
    if len(vocab) != len(freq):
        raise ValueError(f"Vocab size {len(vocab)} != freq size {len(freq)}")
    return vocab, freq


def load_pair_shard(path: Path) -> np.ndarray:
    """Load int32 pairs from binary file, reshape to (-1, 2)."""
    data = np.fromfile(path, dtype=np.int32)
    if data.size % 2 != 0:
        raise ValueError(f"Odd number of int32s in {path}")
    return data.reshape(-1, 2)


def sample_negatives_freq(vocab_size: int, num_neg: int, probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Sample negatives from precomputed freq^0.75 distribution.
    
    Uses word2vec-style unigram^0.75 sampling for negative sampling.
    """
    return rng.choice(vocab_size, size=num_neg, p=probs, replace=True).astype(np.int64)


def compute_ccd_force(cos, align_barrier: float, force_cap: float, use_tanh: bool, tau: float):
    """Compute bounded force from divergence. Works with numpy or torch."""
    if torch is not None and isinstance(cos, torch.Tensor):
        div = align_barrier - cos
        if use_tanh:
            force = torch.tanh(div / tau)
        else:
            force = torch.clamp(div, -force_cap, force_cap)
    else:
        div = align_barrier - cos
        if use_tanh:
            force = np.tanh(div / tau)
        else:
            force = np.clip(div, -force_cap, force_cap)
    return force


def normalize_rows(mat):
    """L2-normalize each row. Uses src.physics.renorm_rows if available."""
    if torch is not None and isinstance(mat, torch.Tensor):
        return F.normalize(mat, p=2, dim=1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return mat / norms


def compute_drift_metrics(V_student, V_seed, probe_indices, vocab):
    """Track geometry drift: mean cosine between student and seed for probe words."""
    if len(probe_indices) == 0:
        return {}
    probe_student = V_student[probe_indices]
    probe_seed = V_seed[probe_indices]
    if torch is not None and isinstance(probe_student, torch.Tensor):
        cosines = (probe_student * probe_seed).sum(dim=1).cpu().numpy()
    else:
        cosines = np.einsum("ij,ij->i", probe_student, probe_seed)
    mean_drift = float(cosines.mean())
    return {"mean_teacher_sim": mean_drift, "probe_words": [vocab[i] for i in probe_indices]}


def get_system_load() -> tuple[float, float]:
    """
    Get current system load (CPU, memory).
    
    Returns:
        Tuple of (cpu_percent, mem_percent). Returns (0.0, 0.0) if psutil unavailable.
    """
    if psutil is None:
        return 0.0, 0.0
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        return cpu, mem
    except Exception:
        return 0.0, 0.0


def auto_adjust_resources(args, current_batch_size: int, step_times: list[float]) -> tuple[int, float]:
    """
    Auto-adjust batch size and throttle based on system load and performance.
    
    Note: Auto-optimize is disabled by default as it reduces batch size and hurts performance.
    Only use if system is struggling and you can't manually tune.
    
    Returns:
        Tuple of (new_batch_size, new_throttle)
    """
    if not args.auto_optimize:
        return args.pairs_per_step, args.throttle
    
    # Check system load
    cpu_load, mem_load = get_system_load()
    
    # Check if we're getting slower (stalling)
    if len(step_times) >= 3:
        recent_avg = sum(step_times[-3:]) / 3
        if recent_avg > 60.0:  # > 60s per step = too slow
            new_batch = max(100000, int(current_batch_size * 0.7))
            print(f"[auto] Reducing batch size {current_batch_size} -> {new_batch} (slow steps)")
            return new_batch, max(args.throttle, 0.02)
    
    # High CPU load (>80%) = add throttle
    if cpu_load > 80.0:
        new_throttle = max(args.throttle, 0.01)
        if new_throttle != args.throttle:
            print(f"[auto] High CPU load ({cpu_load:.1f}%), adding throttle {new_throttle:.3f}s")
        return current_batch_size, new_throttle
    
    # High memory (>85%) = reduce batch
    if mem_load > 85.0:
        new_batch = max(100000, int(current_batch_size * 0.8))
        if new_batch != current_batch_size:
            print(f"[auto] High memory ({mem_load:.1f}%), reducing batch {current_batch_size} -> {new_batch}")
        return new_batch, args.throttle
    
    # System is fine - can we increase batch?
    if cpu_load < 50.0 and mem_load < 70.0 and len(step_times) >= 5:
        recent_avg = sum(step_times[-5:]) / 5
        if recent_avg < 10.0:  # Fast steps, can handle more
            new_batch = min(args.max_batch_size, int(current_batch_size * 1.1))
            if new_batch != current_batch_size:
                print(f"[auto] System idle, increasing batch {current_batch_size} -> {new_batch}")
            return new_batch, max(0.0, args.throttle - 0.005)  # Reduce throttle if possible
    
    return current_batch_size, args.throttle


def main():
    args = parse_args()
    device = pick_device(args.device)
    rng = np.random.default_rng(42)
    
    # Auto-detect optimal CPU threads
    if args.cpu_threads is None:
        cpu_count = os.cpu_count() or 4
        # Use 50-75% of cores, but at least 2, max 8
        cpu_threads = max(2, min(8, int(cpu_count * 0.75)))
    else:
        cpu_threads = args.cpu_threads
    
    if torch is not None:
        torch.set_num_threads(cpu_threads)
        print(f"[config] PyTorch CPU threads: {cpu_threads}")
    
    if args.auto_optimize:
        print(f"[auto] Auto-optimization enabled (will adjust batch size and throttle dynamically)")
    if args.throttle > 0:
        print(f"[throttle] Fixed throttle: {args.throttle}s between batches")

    print(f"[config] device={device} dim={args.dim} lr={args.lr} negatives={args.negatives}")
    print(f"[config] align_barrier={args.align_barrier} fusion_lambda={args.fusion_lambda}")

    # Load vocab + freq
    vocab, freq = load_vocab_freq(Path(args.vocab), Path(args.freq))
    vocab_size = len(vocab)
    print(f"[vocab] size={vocab_size}")

    # Load seed (teacher, immutable)
    V_seed = np.load(args.seed).astype(np.float32)
    if V_seed.shape[0] != vocab_size:
        raise ValueError(f"Seed rows {V_seed.shape[0]} != vocab size {vocab_size}")
    if V_seed.shape[1] != args.dim:
        raise ValueError(f"Seed dim {V_seed.shape[1]} != target dim {args.dim}")
    V_seed = normalize_rows(V_seed)
    print(f"[seed] loaded {V_seed.shape}")

    # Initialize student as copy of seed
    V_student = V_seed.copy()
    use_torch = torch is not None and device != "cpu"
    if use_torch:
        V_student = torch.tensor(V_student, device=device, dtype=torch.float32)
        V_seed_t = torch.tensor(V_seed, device=device, dtype=torch.float32)
    else:
        V_seed_t = V_seed
    
    # Pre-compute negative sampling distribution (freq^0.75)
    neg_probs = np.power(freq.astype(np.float64), 0.75)
    neg_probs = neg_probs / neg_probs.sum()

    # Probe indices for drift tracking
    w2i = {w: i for i, w in enumerate(vocab)}
    probe_indices = [w2i.get(w) for w in args.probe_words if w2i.get(w) is not None]
    probe_indices = np.asarray(probe_indices, dtype=np.int64)

    # Load pair shards
    pair_paths = sorted([Path(p) for p in args.pairs])
    print(f"[pairs] {len(pair_paths)} shard(s)")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    total_pairs_processed = 0
    step = 0

    for epoch in range(args.epochs):
        # Fusion lambda schedule
        if epoch == 0:
            lam = 0.05
        elif epoch == 1:
            lam = 0.10
        else:
            lam = 0.15
        if args.fusion_lambda > 0:
            lam = args.fusion_lambda  # override if user set it
        print(f"\n[epoch {epoch+1}/{args.epochs}] fusion_lambda={lam}")

        epoch_pairs = 0
        pbar = tqdm(
            total=len(pair_paths),
            desc=f"epoch {epoch+1}",
            unit="shard",
            mininterval=1.0,
            smoothing=0.1
        )
        
        # Track step times only if auto-optimize is enabled (disabled by default)
        step_times = [] if args.auto_optimize else None

        for shard_idx, shard_path in enumerate(pair_paths):
            if args.verbose and shard_idx == 0:
                print(f"[debug] loading first shard: {shard_path.name}")
            pairs = load_pair_shard(shard_path)
            num_pairs = pairs.shape[0]
            
            # FIX: Pre-generate all negatives for this shard (removes hot-loop sampling overhead)
            # This eliminates CPU→GPU transfers and RNG calls in the hot loop
            if args.negatives > 0:
                total_negs_needed = num_pairs * args.negatives
                if args.verbose and shard_idx == 0:
                    print(f"[negatives] Pre-generating {total_negs_needed:,} negatives for shard {shard_idx} (one-time cost)...")
                neg_table = sample_negatives_freq(vocab_size, total_negs_needed, neg_probs, rng)
                neg_table = neg_table.reshape(num_pairs, args.negatives)  # (num_pairs, negatives)
                if use_torch:
                    # Convert once to torch tensor on GPU (no per-step transfers)
                    neg_table_t = torch.from_numpy(neg_table).to(device=device, dtype=torch.long)
                    if args.verbose and shard_idx == 0:
                        print(f"[negatives] Pre-generated and transferred to {device}, ready for slicing")
                else:
                    neg_table_t = neg_table
                    if args.verbose and shard_idx == 0:
                        print(f"[negatives] Pre-generated, ready for slicing")
            else:
                neg_table_t = None
            
            cursor = 0
            step_in_shard = 0

            while cursor < num_pairs:
                step_start_time = time.time()
                
                # Use fixed batch size (auto-optimize disabled by default - it hurts performance)
                batch_size = min(args.pairs_per_step, num_pairs - cursor)
                batch_start_in_shard = cursor  # Track position in shard for negative indexing
                batch_pairs = pairs[cursor : cursor + batch_size]
                cursor += batch_size
                step_in_shard += 1
                
                if args.verbose and shard_idx == 0 and step_in_shard == 1:
                    print(f"[debug] processing first batch: {batch_size} pairs")

                i_idx = batch_pairs[:, 0]
                j_idx = batch_pairs[:, 1]

                # Gather vectors - CRITICAL: convert numpy indices to torch tensors for MPS
                if use_torch:
                    i_idx_t = torch.from_numpy(i_idx).to(device=device, dtype=torch.long)
                    j_idx_t = torch.from_numpy(j_idx).to(device=device, dtype=torch.long)
                    vi = V_student[i_idx_t]  # (B, d)
                    vj = V_student[j_idx_t]  # (B, d)
                else:
                    vi = V_student[i_idx]
                    vj = V_student[j_idx]

                # Positive cosine
                if use_torch:
                    cos_pos = (vi * vj).sum(dim=1)  # (B,)
                else:
                    cos_pos = np.einsum("ij,ij->i", vi, vj)

                # Compute positive force
                force_pos = compute_ccd_force(cos_pos, args.align_barrier, args.force_cap, args.use_tanh, args.tau)
                # Positives pull: s = -1 (entailment-like pulls inward)
                delta_i_pos = -args.lr * force_pos[:, None] * (vj - vi)
                delta_j_pos = -args.lr * force_pos[:, None] * (vi - vj)

                # Negatives - FIXED: use pre-generated table, but chunk processing to avoid OOM
                if args.negatives > 0 and neg_table_t is not None:
                    # Chunk size: max 200k pairs at a time to avoid 38GB allocation
                    neg_chunk_size = min(200000, batch_size)
                    delta_i_neg_chunks = []
                    
                    for neg_chunk_start in range(0, batch_size, neg_chunk_size):
                        neg_chunk_end = min(neg_chunk_start + neg_chunk_size, batch_size)
                        neg_chunk_batch_size = neg_chunk_end - neg_chunk_start
                        
                        # Slice pre-generated negatives (already on GPU, no transfer)
                        neg_chunk_t = neg_table_t[batch_start_in_shard + neg_chunk_start:batch_start_in_shard + neg_chunk_end]  # (chunk_B, N)
                        vi_chunk = vi[neg_chunk_start:neg_chunk_end]  # (chunk_B, d)
                        
                        if use_torch:
                            vk = V_student[neg_chunk_t]  # (chunk_B, N, d)
                            vi_expanded = vi_chunk.unsqueeze(1)  # (chunk_B, 1, d)
                            cos_neg = (vi_expanded * vk).sum(dim=2)  # (chunk_B, N)
                            force_neg = compute_ccd_force(cos_neg, args.align_barrier, args.force_cap, args.use_tanh, args.tau)
                            force_neg = force_neg[:, :, None]  # (chunk_B, N, 1)
                            direction = vi_expanded - vk
                            delta_i_neg_chunk = args.lr * (force_neg * direction).sum(dim=1)  # (chunk_B, d)
                            delta_i_neg_chunks.append(delta_i_neg_chunk)
                        else:
                            vk = V_student[neg_chunk_t]
                            cos_neg = np.einsum("bnd,bd->bn", vk, vi_chunk)
                            force_neg = compute_ccd_force(cos_neg, args.align_barrier, args.force_cap, args.use_tanh, args.tau)
                            force_neg = force_neg[:, :, None]
                            direction = vi_chunk[:, None, :] - vk
                            delta_i_neg_chunks.append(args.lr * (force_neg * direction).sum(axis=1))
                    
                    if use_torch:
                        delta_i_neg = torch.cat(delta_i_neg_chunks, dim=0)
                    else:
                        delta_i_neg = np.concatenate(delta_i_neg_chunks, axis=0)
                else:
                    delta_i_neg = torch.zeros_like(vi) if use_torch else np.zeros_like(vi)

                # Aggregate deltas
                delta_i = delta_i_pos + delta_i_neg
                delta_j = delta_j_pos

                # Clip
                if torch is not None and isinstance(delta_i, torch.Tensor):
                    norm_i = delta_i.norm(dim=1, keepdim=True)
                    norm_j = delta_j.norm(dim=1, keepdim=True)
                    delta_i = delta_i * torch.clamp(args.clip_norm / (norm_i + 1e-8), max=1.0)
                    delta_j = delta_j * torch.clamp(args.clip_norm / (norm_j + 1e-8), max=1.0)
                else:
                    norm_i = np.linalg.norm(delta_i, axis=1, keepdims=True)
                    norm_j = np.linalg.norm(delta_j, axis=1, keepdims=True)
                    delta_i = delta_i * np.clip(args.clip_norm / (norm_i + 1e-8), 0.0, 1.0)
                    delta_j = delta_j * np.clip(args.clip_norm / (norm_j + 1e-8), 0.0, 1.0)

                # Scatter updates (index_add_) - renorm only updated rows (fast)
                if use_torch:
                    V_student.index_add_(0, i_idx_t, delta_i)
                    V_student.index_add_(0, j_idx_t, delta_j)
                    # Renorm updated rows every step (fast because only ~400k unique indices per 1M batch)
                    # FIX: Move unique() to CPU to avoid MPS hang with large tensors
                    all_updated_cpu = torch.cat([i_idx_t, j_idx_t]).cpu().unique()
                    all_updated = all_updated_cpu.to(device=device, dtype=torch.long)
                    V_student[all_updated] = F.normalize(V_student[all_updated], p=2, dim=1)
                else:
                    # NumPy: use bincount for fast accumulation
                    dV = np.zeros_like(V_student)
                    np.add.at(dV, i_idx, delta_i)
                    np.add.at(dV, j_idx, delta_j)
                    V_student += dV
                    # Renorm updated rows
                    updated_indices = np.unique(np.concatenate([i_idx, j_idx]))
                    norms = np.linalg.norm(V_student[updated_indices], axis=1, keepdims=True) + 1e-8
                    V_student[updated_indices] = V_student[updated_indices] / norms

                step += 1
                total_pairs_processed += batch_size
                epoch_pairs += batch_size
                
                # Track step time (only if auto-optimize enabled)
                step_time = time.time() - step_start_time
                if step_times is not None:
                    step_times.append(step_time)
                    if len(step_times) > 10:
                        step_times.pop(0)  # Keep last 10 steps
                
                if args.verbose and shard_idx == 0 and step_in_shard == 1:
                    print(f"[debug] first batch done, total_pairs={total_pairs_processed}, step_time={step_time:.2f}s")
                
                # Auto-optimize: adjust throttle based on system load (disabled by default - hurts performance)
                if args.auto_optimize and step_times is not None and step % 5 == 0:
                    _, current_throttle = auto_adjust_resources(args, args.pairs_per_step, step_times)
                    if current_throttle > args.throttle:
                        time.sleep(current_throttle - args.throttle)
                
                # Fixed throttle if set
                if args.throttle > 0:
                    time.sleep(args.throttle)

                # Checkpoint (less frequently to avoid CPU sync overhead)
                if total_pairs_processed % args.checkpoint_interval == 0:
                    if use_torch:
                        V_save = V_student.cpu().numpy()  # Sync to CPU only when checkpointing
                    else:
                        V_save = V_student
                    ckpt_path = checkpoint_dir / f"vectors_step_{total_pairs_processed}.npy"
                    np.save(ckpt_path, V_save.astype(np.float32))
                    print(f"\n[checkpoint] {ckpt_path}")
                    # Renorm after checkpoint (ensures clean state)
                    if use_torch:
                        V_student = F.normalize(V_student, p=2, dim=1)
                    else:
                        norms = np.linalg.norm(V_student, axis=1, keepdims=True) + 1e-8
                        V_student = V_student / norms

                # Fusion anchoring (only at end of shard)
                if cursor >= num_pairs:
                    if use_torch:
                        V_student = lam * V_student + (1 - lam) * V_seed_t
                        V_student = F.normalize(V_student, p=2, dim=1)  # Full vocab renorm after fusion
                    else:
                        V_student = lam * V_student + (1 - lam) * V_seed
                        norms = np.linalg.norm(V_student, axis=1, keepdims=True) + 1e-8
                        V_student = V_student / norms

            # Final renorm of all updated rows at end of shard (ensures stability)
            if use_torch:
                V_student = F.normalize(V_student, p=2, dim=1)
            else:
                norms = np.linalg.norm(V_student, axis=1, keepdims=True) + 1e-8
                V_student = V_student / norms
            
            pbar.update(1)
            elapsed = pbar.format_dict.get('elapsed', 0) or 1
            pairs_per_sec = epoch_pairs / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                "pairs": f"{epoch_pairs:,}",
                "rate": f"{pairs_per_sec:,.0f} pairs/s"
            })

        pbar.close()

        # End-of-epoch: fusion + drift metrics
        if use_torch:
            V_student = lam * V_student + (1 - lam) * V_seed_t
            V_student = normalize_rows(V_student)
            V_eval = V_student.cpu().numpy()
        else:
            V_student = lam * V_student + (1 - lam) * V_seed
            V_student = normalize_rows(V_student)
            V_eval = V_student

        if len(probe_indices) > 0:
            drift = compute_drift_metrics(V_eval, V_seed, probe_indices, vocab)
            print(f"[drift] mean_teacher_sim={drift.get('mean_teacher_sim', 0.0):.4f}")

    # Final save
    if torch is not None and isinstance(V_student, torch.Tensor):
        V_final = V_student.cpu().numpy()
    else:
        V_final = V_student
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, V_final.astype(np.float32))
    print(f"\n[save] final vectors -> {out_path}")
    print(f"[done] total_pairs={total_pairs_processed}")


if __name__ == "__main__":
    main()


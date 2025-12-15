#!/usr/bin/env python
"""
Phase-3: Level-0 Acceptance Validation Script

Runs automatic acceptance tests after training to ensure vectors are ready for production.
Logs metrics to validate_level0.log for reproducibility.

Checks:
- Teacher drift (mean cosine similarity with seed)
- Collapse detection (random pair cosine should NOT be near 1.0)
- Nearest neighbor sanity
- Analogy accuracy

Example:
    python validate_level0.py \\
        --checkpoint data/ecw_bt_vectors.npy \\
        --seed data/V_seed.npy \\
        --mass-table data/mass_table.json \\
        --vocab data/vocab.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.collapse_engine import BasinTracker


def parse_args():
    ap = argparse.ArgumentParser(description="Validate Level-0 vectors meet acceptance criteria")
    ap.add_argument("--checkpoint", required=True, help="Path to trained vectors (.npy)")
    ap.add_argument("--seed", required=True, help="Path to V_seed.npy (teacher)")
    ap.add_argument("--mass-table", required=True, help="Path to mass_table.json")
    ap.add_argument("--vocab", required=True, help="Path to vocab.txt")
    ap.add_argument("--output-log", default="validate_level0.log", help="Output log file")
    return ap.parse_args()


def compute_teacher_drift(V_student, V_seed, probe_indices):
    """Compute mean cosine similarity between student and seed for probe words."""
    if len(probe_indices) == 0:
        return 0.0, []
    probe_student = V_student[probe_indices]
    probe_seed = V_seed[probe_indices]
    cosines = np.einsum("ij,ij->i", probe_student, probe_seed)
    return float(cosines.mean()), cosines.tolist()


def compute_collapse_metric(V_student, num_samples=1000):
    """Check for collapse: mean cosine of random pairs (should NOT be near 1.0)."""
    vocab_size = V_student.shape[0]
    rng = np.random.default_rng(42)
    i_idx = rng.integers(0, vocab_size, size=num_samples)
    j_idx = rng.integers(0, vocab_size, size=num_samples)
    # Avoid same word
    mask = i_idx != j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    if len(i_idx) == 0:
        return 0.0
    vi = V_student[i_idx]
    vj = V_student[j_idx]
    cosines = np.einsum("ij,ij->i", vi, vj)
    return float(cosines.mean())


def test_nearest_neighbors(tracker, test_words):
    """Test nearest neighbors for sanity."""
    results = {}
    for word in test_words:
        neighbors = tracker.nearest(word.lower(), k=10)
        if neighbors:
            results[word] = neighbors
        else:
            results[word] = "NOT_IN_VOCAB"
    return results


def main():
    args = parse_args()
    
    print(f"[validate] Loading vectors from {args.checkpoint}")
    V_student = np.load(args.checkpoint).astype(np.float32)
    V_seed = np.load(args.seed).astype(np.float32)
    
    if V_student.shape != V_seed.shape:
        raise ValueError(f"Shape mismatch: student {V_student.shape} != seed {V_seed.shape}")
    
    vocab = [line.strip() for line in Path(args.vocab).read_text().splitlines() if line.strip()]
    if len(vocab) != V_student.shape[0]:
        raise ValueError(f"Vocab size {len(vocab)} != vectors {V_student.shape[0]}")
    
    print(f"[validate] Vocab size: {len(vocab)}, dim: {V_student.shape[1]}")
    
    # Load tracker for neighbor tests
    tracker = BasinTracker.from_checkpoint(args.checkpoint, args.mass_table)
    
    # Probe words for drift tracking
    probe_words = ["king", "queen", "paris", "france", "cat", "dog", "kitten", "puppy", "man", "woman"]
    w2i = {w: i for i, w in enumerate(vocab)}
    # Filter to only include words that exist in vocabulary
    valid_probe_words = [w for w in probe_words if w in w2i]
    probe_indices = np.array([w2i[w] for w in valid_probe_words], dtype=np.int64)
    
    print(f"[validate] Computing metrics...")
    
    # 1. Teacher drift
    mean_drift, drift_details = compute_teacher_drift(V_student, V_seed, probe_indices)
    
    # 2. Collapse check
    collapse_metric = compute_collapse_metric(V_student)
    
    # 3. Nearest neighbor sanity
    test_words = ["kitten", "king", "paris", "cat"]
    neighbor_results = test_nearest_neighbors(tracker, test_words)
    
    # 4. Analogy test (basic)
    analogy_tests = [
        ("man", "woman", "king", "queen"),
        ("paris", "france", "berlin", "germany"),
    ]
    analogy_results = {}
    for a, b, c, expected in analogy_tests:
        if all(w in w2i for w in [a, b, c, expected]):
            va = V_student[w2i[a]]
            vb = V_student[w2i[b]]
            vc = V_student[w2i[c]]
            target = vb - va + vc
            target /= np.linalg.norm(target) + 1e-8
            sims = V_student @ target
            sims[w2i[a]] = sims[w2i[b]] = sims[w2i[c]] = -np.inf
            pred_idx = int(np.argmax(sims))
            pred = vocab[pred_idx]
            analogy_results[f"{a}:{b}::{c}:{expected}"] = {
                "predicted": pred,
                "correct": pred == expected,
                "score": float(sims[pred_idx])
            }
        else:
            analogy_results[f"{a}:{b}::{c}:{expected}"] = "missing_words"
    
    # Compile report
    report = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(args.checkpoint),
        "seed": str(args.seed),
        "vocab_size": len(vocab),
        "dim": int(V_student.shape[1]),
        "metrics": {
            "teacher_drift_mean": mean_drift,
            "teacher_drift_probe_words": valid_probe_words,
            "collapse_metric": collapse_metric,
            "nearest_neighbors": neighbor_results,
            "analogies": analogy_results,
        },
        "acceptance": {
            "teacher_drift_ok": mean_drift > 0.5,  # Should retain >50% similarity
            "no_collapse": collapse_metric < 0.8,  # Random pairs should NOT be near 1.0
            "neighbors_sane": all(
                isinstance(neighbors, list) and len(neighbors) > 0
                for neighbors in neighbor_results.values()
            ),
        }
    }
    
    # Overall acceptance
    report["accepted"] = all(report["acceptance"].values())
    
    # Write log
    log_path = Path(args.output_log)
    with log_path.open("w") as f:
        f.write(f"=== Level-0 Validation Report ===\n")
        f.write(f"Timestamp: {report['timestamp']}\n")
        f.write(f"Checkpoint: {report['checkpoint']}\n")
        f.write(f"Seed: {report['seed']}\n")
        f.write(f"\n=== Metrics ===\n")
        f.write(f"Teacher Drift (mean cosine): {mean_drift:.4f}\n")
        f.write(f"  → {'✓ OK' if report['acceptance']['teacher_drift_ok'] else '✗ FAIL'} (should be >0.5)\n")
        f.write(f"\nCollapse Metric (random pair cosine): {collapse_metric:.4f}\n")
        f.write(f"  → {'✓ OK' if report['acceptance']['no_collapse'] else '✗ FAIL'} (should be <0.8)\n")
        f.write(f"\n=== Nearest Neighbors ===\n")
        for word, neighbors in neighbor_results.items():
            f.write(f"{word}:\n")
            if isinstance(neighbors, list):
                for n_word, sim in neighbors[:5]:
                    f.write(f"  {n_word}: {sim:.4f}\n")
            else:
                f.write(f"  {neighbors}\n")
        f.write(f"\n=== Analogies ===\n")
        for test, result in analogy_results.items():
            if isinstance(result, dict):
                f.write(f"{test}: {result['predicted']} ({'✓' if result['correct'] else '✗'}, score={result['score']:.4f})\n")
            else:
                f.write(f"{test}: {result}\n")
        f.write(f"\n=== Acceptance ===\n")
        f.write(f"Overall: {'✓ ACCEPTED' if report['accepted'] else '✗ REJECTED'}\n")
        f.write(f"\n=== Full JSON ===\n")
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n=== Validation Summary ===")
    print(f"Teacher Drift: {mean_drift:.4f} {'✓' if report['acceptance']['teacher_drift_ok'] else '✗'}")
    print(f"Collapse Check: {collapse_metric:.4f} {'✓' if report['acceptance']['no_collapse'] else '✗'}")
    print(f"Neighbors: {'✓' if report['acceptance']['neighbors_sane'] else '✗'}")
    print(f"\nOverall: {'✓ ACCEPTED - Ready for production' if report['accepted'] else '✗ REJECTED - Needs retraining'}")
    print(f"\nFull report: {log_path}")
    
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())


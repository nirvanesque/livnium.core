#!/usr/bin/env python
"""
CLI tool to probe the ECW-BT Galaxy.
Usage: python probe_galaxy.py --checkpoint checkpoints/vectors_step_10000.npy --mass data/mass_table.json --query kitten
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.collapse_engine import BasinTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Probe ECW-BT embeddings")
    parser.add_argument("--checkpoint", required=True, help="Path to vectors.npy")
    parser.add_argument("--mass", default="data/mass_table.json", help="Path to mass_table.json")
    parser.add_argument("--query", type=str, help="Single word to find neighbors for")
    parser.add_argument("--sentence", type=str, help="Sentence to collapse and find nearest neighbors")
    parser.add_argument("--topk", type=int, default=10, help="Neighbors to show")
    parser.add_argument("--resonance", type=float, default=0.0, help="Auto-resonance strength for collapse")
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(args.checkpoint).exists():
        raise SystemExit(f"Missing checkpoint file: {args.checkpoint}")

    print(f"[Probe] Loading Galaxy from {args.checkpoint}...")
    tracker = BasinTracker.from_checkpoint(args.checkpoint, args.mass)
    print(f"[Probe] Vocab size: {len(tracker.vocab)}")

    if args.query:
        print(f"\n--- Neighbors for '{args.query}' ---")
        neighbors = tracker.nearest(args.query.lower(), k=args.topk)
        if not neighbors:
            print("Word not in vocab.")
        for w, sim in neighbors:
            print(f"{w:<15} {sim:.4f}")

    if args.sentence:
        print(f"\n--- Collapse Vector for: '{args.sentence}' ---")
        s_vec = tracker.sentence_vector(args.sentence, iterations=5, resonance_strength=args.resonance or 1.0)
        if getattr(tracker, "last_reward_log", None):
            log = tracker.last_reward_log
            print("\n[Economic Physics Telemetry]")
            print(f"  Alignment Raw: {log.get('alignment_raw', 0.0):.4f}")
            print(f"  Reward Signal: {log.get('reward_signal', 0.0):.4f} (Cubic)")
            print(f"  Drift Cost:    {log.get('drift_cost', 0.0):.4f}")
            print(f"  NET SCORE:     {log.get('net_score', 0.0):.4f}")
            print("-" * 30)
        sims = tracker.vectors @ s_vec
        topk = sims.argsort()[::-1][: args.topk]
        print("Nearest Concepts to Result:")
        for i in topk:
            w = tracker.vocab[i]
            sim = sims[i]
            print(f"{w:<15} {sim:.4f}")


if __name__ == "__main__":
    main()

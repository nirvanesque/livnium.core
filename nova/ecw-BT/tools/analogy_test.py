#!/usr/bin/env python
"""
Analogy tester for ECW-BT embeddings.

Usage:
    python analogy_test.py \
        --checkpoint checkpoints/vectors_step_XXXXX.npy \
        --mass-table data/mass_table.json
"""

from __future__ import annotations
import argparse
import json
import numpy as np
from pathlib import Path


DEFAULT_TESTS = [
    ("man", "woman", "king", "queen"),
    ("brother", "sister", "father", "mother"),
    ("paris", "france", "berlin", "germany"),
    ("tokyo", "japan", "beijing", "china"),
    ("running", "ran", "walking", "walked"),
    ("sing", "sang", "drive", "drove"),
]


def load_vectors(checkpoint: Path, mass_table: Path):
    print(f"[load] vectors from {checkpoint}")
    vecs = np.load(checkpoint)

    obj = json.loads(mass_table.read_text())
    vocab = obj["vocab"]

    if len(vocab) != vecs.shape[0]:
        raise ValueError(
            f"Vocab size {len(vocab)} != vectors rows {vecs.shape[0]}"
        )

    w2i = {w: i for i, w in enumerate(vocab)}
    return w2i, vecs


def analogy(w2i, vecs, a, b, c, expected):
    # Check if all words exist
    missing = [w for w in (a, b, c, expected) if w not in w2i]
    if missing:
        return None, f"missing words: {', '.join(missing)}"

    va = vecs[w2i[a]]
    vb = vecs[w2i[b]]
    vc = vecs[w2i[c]]

    # classic vector analogy:   b - a + c
    target = vb - va + vc
    target /= np.linalg.norm(target) + 1e-8

    # cosine with all vectors
    sims = vecs @ target
    # mask out source words
    sims[w2i[a]] = sims[w2i[b]] = sims[w2i[c]] = -np.inf

    idx = int(np.argmax(sims))
    pred = list(w2i.keys())[idx]

    return (pred == expected, pred, float(sims[idx])), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mass-table", default="data/mass_table.json")
    ap.add_argument("--tests", help="file with custom analogies a b c d per line")
    args = ap.parse_args()

    w2i, vecs = load_vectors(Path(args.checkpoint), Path(args.mass_table))

    # Load test cases
    if args.tests:
        tests = []
        for line in Path(args.tests).read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 4:
                tests.append(tuple(parts))
    else:
        tests = DEFAULT_TESTS

    print(f"[info] vocab={len(w2i)} dim={vecs.shape[1]} tests={len(tests)}")

    correct = 0
    total = 0

    for a, b, c, d in tests:
        total += 1
        res, err = analogy(w2i, vecs, a, b, c, d)

        if err:
            print(f"⚠️  {err} in {a}:{b}::{c}:{d}")
            continue

        ok, pred, score = res

        if ok:
            correct += 1
            print(f"✅ {a}:{b} :: {c}:{d} (pred={pred}, cos={score:.4f})")
        else:
            print(f"❌ {a}:{b} :: {c}:{d} → {pred} (cos={score:.4f})")

    print(f"\n[result] accuracy {correct}/{total} = {(100*correct/total):.2f}%")


if __name__ == "__main__":
    main()

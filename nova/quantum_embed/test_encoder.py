"""
Quick smoke test for quantum text encoder.

Usage:
    python3 test_encoder.py --ckpt model_full_physics/quantum_embeddings_final.pt \
        --tokens-file wikitext-103/wiki.test.tokens --num-lines 4
"""

import argparse
from pathlib import Path
from typing import List

import torch

from text_encoder_quantum import QuantumTextEncoder


def load_sample_lines(path: Path, num_lines: int) -> List[str]:
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("="):
                continue
            lines.append(line)
            if len(lines) >= num_lines:
                break
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the quantum text encoder.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to quantum_embeddings_final.pt")
    parser.add_argument(
        "--tokens-file",
        type=str,
        default="wikitext-103/wiki.test.tokens",
        help="File with sample text lines",
    )
    parser.add_argument("--num-lines", type=int, default=4, help="How many lines to encode")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    tokens_path = Path(args.tokens_file)

    enc = QuantumTextEncoder(str(ckpt_path))
    print(f"Loaded encoder dim={enc.dim}, vocab={len(enc.word2idx)}")

    samples = load_sample_lines(tokens_path, args.num_lines)
    if not samples:
        raise SystemExit("No sample lines loaded; check tokens file.")
    print("Sample lines:")
    for i, line in enumerate(samples, 1):
        print(f"  [{i}] {line[:120]}{'...' if len(line) > 120 else ''}")

    ids = []
    for ln in samples:
        toks = enc.tokenize(ln)
        ids.append([enc.word2idx.get(t, enc.unk_idx) for t in toks])

    max_len = max(len(x) for x in ids)
    ids = [x + [enc.pad_idx] * (max_len - len(x)) for x in ids]

    batch = torch.tensor(ids, dtype=torch.long)
    vecs = enc.encode_sentence(batch)

    norms = [v.norm().item() for v in vecs]
    print(f"Encoded shape: {vecs.shape}")
    print("Vector norms:", norms)


if __name__ == "__main__":
    main()

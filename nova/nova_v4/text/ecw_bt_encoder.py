"""
ECW-BT encoder for nova_v4 using rotated/pairwise-refined embeddings.

Loads a checkpoint (default: nova_v4/checkpoints/vectors_pairwise.npy) and the ECW-BT mass table
(default: ../ecw-BT/data/mass_table.json) and provides a mean-pool sentence encoder.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def default_paths() -> Tuple[Path, Path]:
    """
    Returns (checkpoint_path, mass_table_path) defaults.
    """
    root = Path(__file__).resolve().parents[1]  # nova_v4/
    ckpt = root / "checkpoints" / "vectors_pairwise.npy"
    mass = root.parents[1] / "ecw-BT" / "data" / "mass_table.json"
    return ckpt, mass


def load_vocab(mass_table_path: Path) -> Tuple[List[str], Dict[str, int]]:
    obj = json.loads(mass_table_path.read_text())
    vocab = obj["vocab"]
    w2i = {w: i for i, w in enumerate(vocab)}
    return vocab, w2i


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in re.finditer(r"[A-Za-z']+", text)]


class ECWBTEncoder(nn.Module):
    """
    Mean-pool encoder backed by ECW-BT embeddings.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        mass_table_path: Optional[Path] = None,
        pad_idx: int = 0,
    ):
        super().__init__()
        if checkpoint_path is None or mass_table_path is None:
            ckpt_default, mass_default = default_paths()
            checkpoint_path = checkpoint_path or ckpt_default
            mass_table_path = mass_table_path or mass_default

        checkpoint_path = Path(checkpoint_path)
        mass_table_path = Path(mass_table_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint at {checkpoint_path}")
        if not mass_table_path.exists():
            raise FileNotFoundError(f"Missing mass table at {mass_table_path}")

        vectors = np.load(checkpoint_path).astype(np.float32)
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

        vocab, w2i = load_vocab(mass_table_path)
        if vectors.shape[0] != len(vocab):
            raise ValueError(f"Vocab size {len(vocab)} != vectors rows {vectors.shape[0]}")

        self.vocab = vocab
        self.word_to_idx = w2i
        self.pad_idx = pad_idx
        self.dim = vectors.shape[1]
        weight = torch.tensor(vectors, dtype=torch.float32)
        self.embed = nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=self.pad_idx)

    def encode_tokens(self, tokens: List[str]) -> torch.Tensor:
        ids = [self.word_to_idx.get(t, self.pad_idx) for t in tokens]
        if not ids:
            ids = [self.pad_idx]
        ids_t = torch.tensor(ids, dtype=torch.long, device=self.embed.weight.device)
        return self.forward(ids_t)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embed(token_ids)
        if token_ids.dim() == 1:
            mask = (token_ids != self.pad_idx).float().unsqueeze(-1)
            summed = (emb * mask).sum(dim=0)
            count = mask.sum(dim=0).clamp(min=1.0)
            return summed / count
        else:
            mask = (token_ids != self.pad_idx).float().unsqueeze(-1)
            summed = (emb * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1.0)
            return summed / count


__all__ = ["ECWBTEncoder", "tokenize", "load_vocab", "default_paths"]

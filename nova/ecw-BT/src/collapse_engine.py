"""
Gravity pooling / inference for ECW-BT (Level-2: Economic Physics).
Includes Energy Cost and Alignment Rewards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .data_loader import default_tokenizer, load_mass_table


class BasinTracker:
    def __init__(self, vectors: np.ndarray, vocab: List[str], masses: np.ndarray):
        self.vectors = vectors.astype(np.float32)
        self.vocab = vocab
        self.masses = masses.astype(np.float32)
        self.word_to_idx: Dict[str, int] = {w: i for i, w in enumerate(vocab)}
        self.last_reward_log = {}

    @classmethod
    def from_checkpoint(cls, vector_path: str | Path, mass_table_path: str | Path):
        vectors = np.load(vector_path)
        vocab, _, masses, _ = load_mass_table(mass_table_path)
        return cls(vectors, vocab, masses)

    def _ghost_basin(self, center: np.ndarray) -> Tuple[np.ndarray, float]:
        noise = np.random.randn(*center.shape).astype(np.float32)
        ghost = center + 0.01 * noise
        norm = np.linalg.norm(ghost) + 1e-8
        ghost = ghost / norm
        return ghost, 0.1  # ghost mass

    def _calculate_auto_resonance(self, current_vec: np.ndarray, anchor_vec: np.ndarray) -> float:
        """
        Squared cosine similarity between current state and anchor.
        """
        sim = float(np.dot(current_vec, anchor_vec))
        sim = max(0.0, min(1.0, sim))
        return sim ** 2

    def _calculate_reward(self, current_vec: np.ndarray, anchor_vec: np.ndarray, start_vec: np.ndarray) -> float:
        """
        Economic Physics:
        Reward = (Alignment Strength)^3 - (Movement Cost)
        """
        alignment = float(np.dot(current_vec, anchor_vec))
        alignment = max(0.0, alignment)
        reward_signal = alignment ** 3
        drift = np.linalg.norm(current_vec - start_vec)
        cost = 0.05 * drift
        total_score = reward_signal - cost
        self.last_reward_log = {
            "alignment_raw": alignment,
            "reward_signal": reward_signal,
            "drift_cost": cost,
            "net_score": total_score,
        }
        return total_score

    def sentence_vector(self, text: str, iterations: int = 5, resonance_strength: float = 0.0) -> np.ndarray:
        tokens = default_tokenizer(text)
        vecs = []
        masses = []
        for tok in tokens:
            idx = self.word_to_idx.get(tok)
            if idx is None:
                continue
            vecs.append(self.vectors[idx])
            masses.append(self.masses[idx])

        if not vecs:
            ghost, _ = self._ghost_basin(np.random.randn(self.vectors.shape[1]).astype(np.float32))
            return ghost

        vecs = np.stack(vecs)
        masses = np.array(masses, dtype=np.float32)

        anchor = (masses[:, None] * vecs).sum(axis=0) / (masses.sum() + 1e-8)
        anchor = anchor / (np.linalg.norm(anchor) + 1e-8)

        center = anchor.copy()
        start_pos = anchor.copy()

        for _ in range(iterations):
            ghost, ghost_mass = self._ghost_basin(center)
            stacked = np.concatenate([vecs, ghost[None, :]], axis=0)
            mass_all = np.concatenate([masses, np.array([ghost_mass], dtype=np.float32)], axis=0)
            center = (mass_all[:, None] * stacked).sum(axis=0) / (mass_all.sum() + 1e-8)
            center = center / (np.linalg.norm(center) + 1e-8)

            if resonance_strength > 0.0:
                _ = self._calculate_reward(center, anchor, start_pos)
                res = self._calculate_auto_resonance(center, anchor)
                force_dir = anchor - center
                warp = force_dir * (resonance_strength * res)
                center = center + warp
                center = center / (np.linalg.norm(center) + 1e-8)

        return center

    def nearest(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        idx = self.word_to_idx.get(query)
        if idx is None:
            return []
        qv = self.vectors[idx]
        sims = self.vectors @ qv
        topk = sims.argsort()[::-1][: k + 1]
        out = []
        for i in topk:
            if i == idx:
                continue
            out.append((self.vocab[i], float(sims[i])))
            if len(out) >= k:
                break
        return out

"""
Trainer Utilities for Physics Embed

Vocab and Dataset classes for Skip-Gram training.
"""

from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset

class Vocab:
    def __init__(self, max_size: int = 50000, min_freq: int = 1):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []
        self.freqs: Dict[str, int] = {}
        self.special_tokens = ["<pad>", "<unk>"]
        for tok in self.special_tokens:
            self._add(tok)

    def _add(self, token: str):
        if token not in self.word2idx:
            idx = len(self.idx2word)
            self.word2idx[token] = idx
            self.idx2word.append(token)

    def add_tokens_from_line(self, line: str):
        for tok in line.strip().split():
            if not tok:
                continue
            self.freqs[tok] = self.freqs.get(tok, 0) + 1

    def build(self):
        sorted_items = sorted(self.freqs.items(), key=lambda x: -x[1])
        for tok, freq in sorted_items:
            if freq < self.min_freq:
                continue
            if tok in self.word2idx:
                continue
            if len(self.idx2word) >= self.max_size:
                break
            self._add(tok)

    @property
    def pad_idx(self) -> int:
        return self.word2idx["<pad>"]

    @property
    def unk_idx(self) -> int:
        return self.word2idx["<unk>"]

    def __len__(self) -> int:
        return len(self.idx2word)

    def encode_line(self, line: str) -> List[int]:
        return [self.word2idx.get(tok, self.unk_idx) for tok in line.strip().split() if tok]


class SkipGramDataset(Dataset):
    def __init__(self, sequences: List[List[int]], window_size: int = 2):
        self.pairs: List[Tuple[int, int]] = []
        # We process silently to avoid spamming logs in imports
        for seq in sequences:
            for i, c in enumerate(seq):
                if c == 0:
                    continue
                left = max(0, i - window_size)
                right = min(len(seq), i + window_size + 1)
                for j in range(left, right):
                    if j == i:
                        continue
                    self.pairs.append((c, seq[j]))
    
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        return self.pairs[idx]

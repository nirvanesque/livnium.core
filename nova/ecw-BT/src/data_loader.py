"""
Data utilities for ECW-BT using JSONL wiki extractor shards.
- Tokenize `text` field per line.
- Build frequency/mass tables.
- Stream token id sequences for training.
"""

from __future__ import annotations

import math
import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Generator, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

try:
    import regex as re
except Exception:
    import re


TokenizeFn = Callable[[str], List[str]]


def default_tokenizer(text: str) -> List[str]:
    """
    Simple word tokenizer: lowercase alpha+apostrophe sequences.
    """
    return [m.group(0).lower() for m in re.finditer(r"[A-Za-z']+", text)]


def stream_jsonl_texts(paths: Iterable[str], progress: bool = False) -> Generator[str, None, None]:
    for path in paths:
        with Path(path).open() as f:
            iterator = f
            if progress:
                iterator = tqdm(f, desc=f"mass:{Path(path).name}", unit="line", leave=False)
            for line in iterator:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = obj.get("text", "")
                if text:
                    yield text


def count_frequencies(paths: Iterable[str], tokenize: TokenizeFn = default_tokenizer, show_progress: bool = False) -> Counter:
    counter = Counter()
    for text in stream_jsonl_texts(paths, progress=show_progress):
        counter.update(tokenize(text))
    return counter


def build_mass_table(
    paths: Iterable[str],
    output_path: str | Path,
    min_freq: int = 1,
    tokenize: TokenizeFn = default_tokenizer,
) -> dict:
    """
    Build and persist mass table JSON with aligned arrays:
    {vocab: [...], freq: [...], mass: [...], meta: {...}}
    """
    counter = count_frequencies(paths, tokenize, show_progress=True)
    vocab_items = [(w, f) for w, f in counter.items() if f >= min_freq]
    # Deterministic order: desc freq, then alpha
    vocab_items.sort(key=lambda x: (-x[1], x[0]))
    vocab = [w for w, _ in vocab_items]
    freqs = [int(f) for _, f in vocab_items]
    masses = [1.0 / math.log(1.0 + f) for f in freqs]

    payload = {
        "vocab": vocab,
        "freq": freqs,
        "mass": masses,
        "meta": {
            "min_freq": min_freq,
            "paths": list(paths),
            "tokenizer": "regex_word",
        },
    }
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    return payload


def load_mass_table(path: str | Path) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, int]]:
    obj = json.loads(Path(path).read_text())
    vocab = obj["vocab"]
    freqs = np.array(obj["freq"], dtype=np.int64)
    masses = np.array(obj["mass"], dtype=np.float32)
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    return vocab, freqs, masses, word_to_idx


def stream_token_ids(
    paths: Iterable[str],
    word_to_idx: Dict[str, int],
    tokenize: TokenizeFn = default_tokenizer,
    p_keep: np.ndarray | None = None,
) -> Generator[List[int], None, None]:
    """
    Yield token-id lists per document/text line; skips out-of-vocab tokens.
    If p_keep is provided (length=vocab), tokens are kept with probability p_keep[idx].
    """
    for text in stream_jsonl_texts(paths):
        ids = []
        for tok in tokenize(text):
            idx = word_to_idx.get(tok)
            if idx is not None:
                if p_keep is not None:
                    if random.random() > p_keep[idx]:
                        continue
                ids.append(idx)
        if ids:
            yield ids


def count_dataset_tokens(
    paths: Iterable[str],
    word_to_idx: Dict[str, int],
    tokenize: TokenizeFn = default_tokenizer,
    p_keep: np.ndarray | None = None,
) -> int:
    """
    Fast pass to count total tokens yielded by stream_token_ids.
    """
    total = 0
    for ids in stream_token_ids(paths, word_to_idx, tokenize, p_keep=p_keep):
        total += len(ids)
    return total

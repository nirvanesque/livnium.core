"""
Configuration defaults for ECW-BT Level-0.
Adjust values via CLI flags in train_ecw_bt.py.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ECWConfig:
    # Geometry / physics
    dim: int = 384
    window: int = 5
    lr: float = 1e-3
    negatives: int = 3
    epochs: int = 1
    align_barrier: float = 0.38
    batch_size: int = 4096  # tuned for batched MPS/CPU throughput
    catalyst: float = 0.0  # resonance-based acceleration (0 disables)

    # Paths
    wiki_shards: list[str] = None  # set by CLI; defaults to wiki_00 shard
    mass_table_path: Path = Path("data/mass_table.json")
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")

    # Training
    device: str = "auto"  # "auto", "cpu", "mps", "cuda"
    seed: int = 42
    min_freq: int = 1  # keep all tokens for now


def default_wiki_paths() -> list[str]:
    """
    Return default shard list (uses wiki_00 only).
    """
    return ["wikipedia/wiki_extractor_src/extracted/AA/wiki_00"]

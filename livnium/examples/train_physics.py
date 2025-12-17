"""
Physics Embed Training Script (Livnium Core)

Trains word embeddings on text data using Livnium energy objectives.
Uses core Engine and BasinField components.
"""

import os
import argparse
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from livnium.domains.physics_embed.encoder import PhysicsEmbeddingModel
from livnium.domains.physics_embed.trainer_utils import Vocab, SkipGramDataset
from livnium.engine.collapse.engine import CollapseEngine
from livnium.engine.fields.basin_field import BasinField
from livnium.engine.config import defaults


def build_vocab_and_sequences(path: str, max_lines: int, max_size: int) -> Tuple[Vocab, List[List[int]]]:
    vocab = Vocab(max_size=max_size, min_freq=2)
    print(f"[build_vocab] scanning {path}...")
    lines: List[str] = []
    
    # Check if file exists, if not, generate dummy data for testing
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Generating dummy data for testing.")
        dummy_text = "the quick brown fox jumps over the lazy dog . " * 100
        for _ in range(100):
            lines.append(dummy_text)
            vocab.add_tokens_from_line(dummy_text)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                lines.append(line)
                vocab.add_tokens_from_line(line)
                
    vocab.build()
    print(f"[build_vocab] vocab size: {len(vocab)}")
    sequences: List[List[int]] = [vocab.encode_line(line) for line in lines]
    print(f"[build_vocab] sequences: {len(sequences)}")
    return vocab, sequences


def sample_negative(batch_size: int, vocab_size: int, pad_idx: int, device: torch.device) -> torch.Tensor:
    neg = torch.randint(low=0, high=vocab_size, size=(batch_size,), device=device)
    neg = torch.where(neg == pad_idx, (neg + 1) % vocab_size, neg)
    return neg


def livnium_energy_loss(
    model: PhysicsEmbeddingModel,
    centers: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    d_margin: float = defaults.D_MARGIN,
    neg_weight: float = defaults.NEG_WEIGHT,
    norm_reg_weight: float = defaults.NORM_REG_WEIGHT,
    collapse_engine: Optional[CollapseEngine] = None,
    basin_field: Optional[BasinField] = None,
    use_dynamic_basins: bool = False,
    global_step: int = 0,
    spawn_new: bool = True,
    prune_every: int = 0,
) -> torch.Tensor:
    
    # 1. Look up raw embeddings
    v_c = model(centers)
    v_p = model(positives)
    v_n = model(negatives)

    # 2. Apply Collapse Physics (If engine exists)
    v_c_for_neg = v_c
    
    if collapse_engine is not None:
        # Core CollapseEngine operates differently than the simplified VectorCollapseEngine
        # We need to adapt the call signature.
        
        # Prepare labels for basin field if dynamic
        # 0=Entail(Pos), 1=Neutral, 2=Contra(Neg) - Arbitrary mapping for energy learning
        # Ideally, we want 'positives' to attract (Entail) and 'negatives' to repel (Contra)
        
        if use_dynamic_basins and basin_field is not None:
            # We must assign labels for routing/spawning
            # Center-Positive pair -> Entailment (0)
            # Center-Negative pair -> Contradiction (2)
            
            # Collapse Centers (with Positive context intent)
            labels_pos = torch.zeros(centers.size(0), dtype=torch.long, device=centers.device) # 0 = Entail
            v_c_pos, _ = collapse_engine.collapse(v_c, labels=labels_pos)
            
            # Collapse Positives (with Entailment intent)
            v_p, _ = collapse_engine.collapse(v_p, labels=labels_pos)
            
            # Collapse Centers (with Negative context intent)
            labels_neg = torch.full((centers.size(0),), 2, dtype=torch.long, device=centers.device) # 2 = Contra
            v_c_neg, _ = collapse_engine.collapse(v_c.detach(), labels=labels_neg)
            
            # Collapse Negatives (with Contra intent)
            v_n, _ = collapse_engine.collapse(v_n, labels=labels_neg)
            
            v_c = v_c_pos
            v_c_for_neg = v_c_neg
            
        else:
            # Static Collapse (no labels needed for routing, just default physics)
            v_c, _ = collapse_engine.collapse(v_c)
            v_p, _ = collapse_engine.collapse(v_p)
            v_n, _ = collapse_engine.collapse(v_n)
            v_c_for_neg = v_c

    # 3. Compute Energy
    v_c_n = F.normalize(v_c, dim=-1)
    v_p_n = F.normalize(v_p, dim=-1)
    v_n_n = F.normalize(v_n, dim=-1)
    v_c_neg_n = F.normalize(v_c_for_neg, dim=-1)

    a_pos = (v_c_n * v_p_n).sum(dim=-1)
    a_neg = (v_c_neg_n * v_n_n).sum(dim=-1)

    from livnium.kernel import constants as k_const
    d_pos = k_const.DIVERGENCE_PIVOT - a_pos
    d_neg = k_const.DIVERGENCE_PIVOT - a_neg

    E_pos = d_pos.pow(2)
    diff = torch.clamp(d_margin - d_neg, min=0.0)
    E_neg = diff.pow(2)

    norm = v_c.norm(dim=-1) + v_p.norm(dim=-1) + v_n.norm(dim=-1)
    norm_reg = norm.mean()

    loss = E_pos.mean() + neg_weight * E_neg.mean() + norm_reg_weight * norm_reg
    return loss


def train(
    train_path: str, output_dir: str, dim: int = 256, max_vocab: int = 50000,
    max_lines: int = 200000, window_size: int = 2, batch_size: int = 1024,
    epochs: int = 3, lr: float = 3e-4, device: str = "auto",
    disable_dynamic_basins: bool = False, collapse_layers: int = 4,
    dry_run: bool = False
):
    os.makedirs(output_dir, exist_ok=True)
    if device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Device Activated: {device}")
    else:
        device = torch.device(device)

    # Dry run adjustments
    if dry_run:
        print("[Dry Run] Reducing limits for quick test...")
        max_lines = 1000
        max_vocab = 1000
        epochs = 1
        batch_size = 32

    vocab, sequences = build_vocab_and_sequences(train_path, max_lines=max_lines, max_size=max_vocab)
    dataset = SkipGramDataset(sequences, window_size=window_size)
    
    # Persistent workers can cause issues in simple scripts/dry runs sometimes, keeping simple
    num_workers = 0 if dry_run else 4
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True
    )

    model = PhysicsEmbeddingModel(vocab_size=len(vocab), dim=dim, pad_idx=vocab.pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    use_dynamic_basins = not disable_dynamic_basins
    
    # Initialize Engine
    collapse_engine = None
    basin_field = None
    
    if collapse_layers > 0:
        # Use Core Collapse Engine
        # NOTE: Core Engine creates its own basin_field if enable_basins=True
        collapse_engine = CollapseEngine(
            dim=dim, 
            num_layers=collapse_layers,
            enable_basins=use_dynamic_basins,
            basin_threshold="v4"
        ).to(device)
        
        if use_dynamic_basins:
            basin_field = collapse_engine.basin_field
            print("[init] Dynamic Basins Enabled in Engine")
        
        print(f"[init] Collapse Engine Active (Layers: {collapse_layers})")

    print(f"[train] device={device}, vocab={len(vocab)}, dim={dim}, batches={len(loader)}")

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=True, unit="batch")

        for step, (centers, contexts) in enumerate(pbar):
            centers = centers.to(device, non_blocking=True)
            contexts = contexts.to(device, non_blocking=True)
            negatives = sample_negative(centers.size(0), len(vocab), vocab.pad_idx, device)

            optimizer.zero_grad()
            loss = livnium_energy_loss(
                model, centers, contexts, negatives,
                collapse_engine=collapse_engine,
                basin_field=basin_field,
                use_dynamic_basins=use_dynamic_basins,
                global_step=global_step,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += centers.size(0)
            pbar.set_postfix(loss=f"{total_loss / (step + 1):.4f}", step=global_step)
            
            if dry_run and step >= 5:
                # Early exit for dry run
                print("[Dry Run] Stopping early loop")
                break

        # Save checkpoint (simplified)
        ckpt_path = os.path.join(output_dir, f"physics_embed_epoch{epoch}.pt")
        torch.save({
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "vocab_size": len(vocab)
        }, ckpt_path)
        print(f"[train] saved {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Physics Embed Trainer (Livnium Core)")
    parser.add_argument("--train-path", type=str, default="data/wikitext-103/wiki.train.tokens", help="Path to training text")
    parser.add_argument("--output-dir", type=str, default="models/physics_embed", help="Output directory")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Run a quick smoke test")
    
    args = parser.parse_args()
    train(**vars(args))

if __name__ == "__main__":
    from livnium.kernel.constants import DIVERGENCE_PIVOT # Ensure kernel init
    # Monkey patch defaults to ensure they are available if needed by imported modules
    defaults.DIVERGENCE_PIVOT = DIVERGENCE_PIVOT
    
    main()

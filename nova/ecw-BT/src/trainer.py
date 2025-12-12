"""
Training loop for ECW-BT (High-Performance Vectorized).
Uses sliding windows, preallocated buffers, and batched physics.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from . import data_loader, physics

try:
    import torch
except Exception:
    torch = None


class Trainer:
    def __init__(self, config, vocab, masses, device: str):
        self.config = config
        self.device = device
        self.vocab_size = len(vocab)

        if torch is None:
            self.masses = masses.astype(np.float32)
            self.vectors = physics.random_unit_vectors(self.vocab_size, config.dim, "cpu")
        else:
            self.masses = torch.tensor(masses, device=device)
            self.vectors = physics.random_unit_vectors(self.vocab_size, config.dim, device)

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # Buffers
        self.bs = config.batch_size
        self.win = config.window
        self.ctx_width = 2 * self.win + 1
        self.buf_tgt = np.empty(self.bs, dtype=np.int64)
        self.buf_ctx = np.empty((self.bs, self.ctx_width), dtype=np.int64)
        self.buf_mask = np.empty((self.bs, self.ctx_width), dtype=np.float32)
        self.ptr = 0

    def save_checkpoint(self, step: int):
        path = self.checkpoint_dir / f"vectors_step_{step}.npy"
        vecs = self.vectors.detach().cpu().numpy() if torch is not None else self.vectors
        np.save(path, vecs)
        meta_path = self.checkpoint_dir / f"config_step_{step}.json"
        meta_path.write_text(str(asdict(self.config)))

    def _flush_buffer(self):
        if self.ptr == 0:
            return 0.0

        batch_tgt = self.buf_tgt[: self.ptr]
        batch_ctx = self.buf_ctx[: self.ptr]
        batch_mask = self.buf_mask[: self.ptr]

        if torch is not None:
            if self.config.negatives > 0:
                b_noise = torch.randint(
                    0,
                    self.vocab_size,
                    (self.ptr, self.config.negatives),
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                b_noise = None

            v_new = physics.batched_update(
                self.vectors,
                self.masses,
                batch_tgt,
                batch_ctx,
                batch_mask,
                b_noise,
                self.config.lr,
                self.config.align_barrier,
                resonance_amp=self.config.catalyst,
            )
            b_tgt = torch.as_tensor(batch_tgt, device=self.device, dtype=torch.long)
            self.vectors[b_tgt] = v_new
            norm_mean = torch.norm(v_new, dim=1).mean().item()
        else:
            b_noise = None
            if self.config.negatives > 0:
                b_noise = np.random.randint(0, self.vocab_size, (self.ptr, self.config.negatives))
            v_new = physics.batched_update(
                self.vectors,
                self.masses,
                batch_tgt,
                batch_ctx,
                batch_mask,
                b_noise,
                self.config.lr,
                self.config.align_barrier,
                resonance_amp=self.config.catalyst,
            )
            self.vectors[batch_tgt] = v_new
            norm_mean = float(np.linalg.norm(v_new, axis=1).mean())

        self.ptr = 0
        return norm_mean

    def train(
        self,
        paths,
        word_to_idx,
        tokenize=data_loader.default_tokenizer,
        total_tokens: int | None = None,
        p_keep=None,
        max_tokens: int | None = None,
    ):
        total_steps = 0
        total_for_bar = total_tokens
        if max_tokens is not None:
            if total_for_bar is None:
                total_for_bar = max_tokens
            else:
                total_for_bar = min(total_for_bar, max_tokens)

        pbar = tqdm(total=total_for_bar, unit="tok", smoothing=0.1, mininterval=1.0)
        pbar.set_description("Training (Buffered)")

        for epoch in range(self.config.epochs):
            if epoch > 0:
                pbar.reset(total=total_for_bar)
            t0 = time.time()
            for seq in data_loader.stream_token_ids(paths, word_to_idx, tokenize, p_keep=p_keep):
                n = len(seq)
                if n < 2:
                    continue
                arr = np.array(seq, dtype=np.int64)
                padded = np.pad(arr, (self.win, self.win), mode="constant", constant_values=0)
                windows = sliding_window_view(padded, window_shape=self.ctx_width)
                masks = (windows != 0).astype(np.float32)
                masks[:, self.win] = 0.0  # center is target, not context
                targets = arr

                num_items = len(targets)
                cursor = 0
                while cursor < num_items:
                    space = self.bs - self.ptr
                    take = min(space, num_items - cursor)
                    self.buf_tgt[self.ptr : self.ptr + take] = targets[cursor : cursor + take]
                    self.buf_ctx[self.ptr : self.ptr + take] = windows[cursor : cursor + take]
                    self.buf_mask[self.ptr : self.ptr + take] = masks[cursor : cursor + take]
                    self.ptr += take
                    cursor += take

                    if self.ptr >= self.bs:
                        norm = self._flush_buffer()
                        total_steps += self.bs
                        pbar.update(self.bs)
                        if total_steps % 50000 < self.bs:
                            pbar.set_postfix({"norm": f"{norm:.3f}"})
                        if max_tokens and total_steps >= max_tokens:
                            pbar.close()
                            self.save_checkpoint(total_steps)
                            elapsed = time.time() - t0
                            print(f"Epoch {epoch+1} stopped at {total_steps} tokens in {elapsed/60:.2f} min")
                            return

            if self.ptr > 0:
                count = self.ptr
                norm = self._flush_buffer()
                total_steps += count
                pbar.update(count)
                pbar.set_postfix({"norm": f"{norm:.3f}"})

            elapsed = time.time() - t0
            self.save_checkpoint(total_steps)
            print(f"Epoch {epoch+1} done in {elapsed/60:.2f} min")

        pbar.close()

"""
SNLI Encoding: Premise/Hypothesis → Initial State (quantum or ECW-BT)
"""

import torch
import torch.nn as nn
from typing import Tuple

from text import QuantumTextEncoder, ECWBTEncoder


class QuantumSNLIEncoder(nn.Module):
    """
    SNLI encoder that uses the pretrained Livnium quantum embeddings.
    """

    def __init__(self, ckpt_path: str):
        super().__init__()
        self.text_encoder = QuantumTextEncoder(ckpt_path, use_gravity=True)
        self.dim = self.text_encoder.dim
        self.pad_idx = self.text_encoder.pad_idx
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, 2 * self.dim),
            nn.GELU(),
            nn.Linear(2 * self.dim, self.dim),
        )

    def encode_sentence(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder.encode_sentence(token_ids)

    def build_initial_state(
        self, prem_ids: torch.Tensor, hyp_ids: torch.Tensor, add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_p = self.encode_sentence(prem_ids)
        v_h = self.encode_sentence(hyp_ids)
        s_p = self.mlp(v_p)
        s_h = self.mlp(v_h)
        h0 = s_p + s_h
        if add_noise:
            h0 = h0 + 0.01 * torch.randn_like(h0)
        return h0, v_p, v_h


class ECWBSNLIEncoder(nn.Module):
    """
    SNLI encoder using ECW-BT embeddings (rotated/pairwise-refined).
    """

    def __init__(self, ckpt_path: str = None, mass_table_path: str = None):
        super().__init__()
        self.text_encoder = ECWBTEncoder(
            checkpoint_path=ckpt_path,
            mass_table_path=mass_table_path,
        )
        self.dim = self.text_encoder.dim
        self.pad_idx = self.text_encoder.pad_idx
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, 2 * self.dim),
            nn.GELU(),
            nn.Linear(2 * self.dim, self.dim),
        )

    def encode_sentence(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(token_ids)

    def build_initial_state(
        self, prem_ids: torch.Tensor, hyp_ids: torch.Tensor, add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_p = self.encode_sentence(prem_ids)
        v_h = self.encode_sentence(hyp_ids)
        s_p = self.mlp(v_p)
        s_h = self.mlp(v_h)
        h0 = s_p + s_h
        if add_noise:
            h0 = h0 + 0.01 * torch.randn_like(h0)
        return h0, v_p, v_h

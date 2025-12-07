"""
SNLI Encoding: Premise/Hypothesis → Initial State (Quantum Only)

Builds initial state h0 from premise and hypothesis using pretrained Livnium
quantum embeddings.
"""

import torch
import torch.nn as nn
from typing import Tuple

from quantum_embed.text_encoder_quantum import QuantumTextEncoder


class QuantumSNLIEncoder(nn.Module):
    """
    SNLI encoder that uses the pretrained Livnium quantum embeddings.
    """

    def __init__(self, ckpt_path: str):
        super().__init__()
        self.text_encoder = QuantumTextEncoder(ckpt_path)
        self.dim = self.text_encoder.dim
        self.pad_idx = self.text_encoder.pad_idx

    def encode_sentence(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder.encode_sentence(token_ids)

    def build_initial_state(
        self, prem_ids: torch.Tensor, hyp_ids: torch.Tensor, add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_p = self.encode_sentence(prem_ids)
        v_h = self.encode_sentence(hyp_ids)
        h0 = v_h - v_p
        if add_noise:
            h0 = h0 + 0.01 * torch.randn_like(h0)
        return h0, v_p, v_h

"""
Shadow cognition layer: IntensionNet + LessonLogger.

These components operate AFTER physics collapse. They must not alter core physics.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class LessonLogger:
    """Buffer for (state, lesson_vector) pairs."""

    def __init__(self):
        self._states: List[torch.Tensor] = []
        self._lessons: List[torch.Tensor] = []

    def log(self, state: torch.Tensor, target_anchor: torch.Tensor) -> None:
        """Record a lesson: lesson_vector = target_anchor - state."""
        with torch.no_grad():
            self._states.append(state.detach())
            self._lessons.append((target_anchor - state).detach())

    def flush(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return stacked (states, lessons) and clear buffer."""
        if not self._states:
            return None
        states = torch.stack(self._states, dim=0)
        lessons = torch.stack(self._lessons, dim=0)
        self._states.clear()
        self._lessons.clear()
        return states, lessons


class IntensionNet(nn.Module):
    """
    Small MLP that predicts a correction vector I(h) for a given state h.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def apply_intension(h: torch.Tensor, net: IntensionNet, alpha: float) -> torch.Tensor:
    """
    Apply shadow correction: h_corrected = h + alpha * I(h).
    """
    return h + alpha * net(h)

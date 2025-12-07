"""SNLI task components (quantum encoder only)."""

from .head_snli import SNLIHead
from .encoding_snli import QuantumSNLIEncoder

__all__ = ['SNLIHead', 'QuantumSNLIEncoder']

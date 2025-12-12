"""
SNLI Task Head

SNLI-specific encoding and classification head.
"""

from .head_snli import SNLIHead
from .encoding_snli import QuantumSNLIEncoder, ECWBSNLIEncoder


class _RemovedEncoder:
    def __init__(self, *args, **kwargs):
        raise ImportError("This encoder has been removed; use QuantumSNLIEncoder instead.")


# Legacy names kept for import compatibility but raise on use.
SNLIEncoder = GeometricSNLIEncoder = SanskritSNLIEncoder = _RemovedEncoder

__all__ = ['SNLIHead', 'QuantumSNLIEncoder', 'ECWBSNLIEncoder', 'SNLIEncoder', 'GeometricSNLIEncoder', 'SanskritSNLIEncoder']

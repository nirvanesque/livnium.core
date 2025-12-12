"""
Text Encoding Layer

Task-agnostic text encoding.
"""

from .encoder import TextEncoder
from .quantum_text_encoder import QuantumTextEncoder
from .ecw_bt_encoder import ECWBTEncoder, tokenize as ecw_bt_tokenize

# Optional/legacy encoders removed from tree; provide stubs so imports fail loudly if used.
try:
    from .geom_encoder import GeometricTextEncoder, tokenize
except ImportError:
    class GeometricTextEncoder:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("GeometricTextEncoder unavailable: geom_encoder.py removed from repository")

    def tokenize(*args, **kwargs):  # type: ignore
        raise ImportError("tokenize unavailable: geom_encoder.py removed from repository")

try:
    from .sanskrit_encoder import TextEncoder as SanskritTextEncoder
except ImportError:
    class SanskritTextEncoder:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("SanskritTextEncoder unavailable: sanskrit_encoder.py removed from repository")

__all__ = [
    'TextEncoder',
    'GeometricTextEncoder',
    'SanskritTextEncoder',
    'QuantumTextEncoder',
    'ECWBTEncoder',
    'tokenize',
    'ecw_bt_tokenize',
]

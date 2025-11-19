"""
Natural Language Inference (NLI) using Livnium Core System

This module implements NLI using the geometric and quantum capabilities
of the Livnium Core System.
"""

from .native_chain import Omchain, WordOmcube
from .native_chain_encoder import NativeChainNLIEncoder, NativeEncodedPair
from .inference_detectors import EntailmentDetector, ContradictionDetector, NLIClassifier
from .omcube import OmcubeNLIClassifier, OmcubeCollapseResult, CrossOmcubeCoupling, GeometricFeedback
from .nli_memory import NLIMemory, MemoryPattern

__all__ = [
    'Omchain',
    'WordOmcube',
    'NativeChainNLIEncoder',
    'NativeEncodedPair',
    'EntailmentDetector',
    'ContradictionDetector',
    'NLIClassifier',
    'OmcubeNLIClassifier',
    'OmcubeCollapseResult',
    'CrossOmcubeCoupling',
    'GeometricFeedback',
    'NLIMemory',
    'MemoryPattern',
]


"""
SNLI Domain: Gold Standard Plugin

SNLI (Stanford Natural Language Inference) domain implementation.
Uses kernel.physics for all physics calculations.
Uses engine.collapse for dynamics.
"""
from .encoder import SNLIEncoder
from .head import SNLIHead
from .workflow import SNLIWorkflow

__all__ = ["SNLIEncoder", "SNLIHead", "SNLIWorkflow"]

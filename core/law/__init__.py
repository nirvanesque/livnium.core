"""
Law Extraction Module: Auto-Discovery of Physical Laws

This module extracts physical laws from Livnium Core by observing:
- Invariants (values that remain constant)
- Conserved quantities (sum/mean/energy)
- Functional relationships (f(x) = y)

This enables Livnium to discover its own laws instead of having them hardcoded.
"""

from .law_extractor import LivniumLawExtractor

__all__ = [
    'LivniumLawExtractor',
]


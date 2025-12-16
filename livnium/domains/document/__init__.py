"""
Document Workflow Domain: Real-World Document Processing

This domain provides a minimal reference implementation for document workflows:
- Retrieval: Finding relevant documents/sections
- Citation validity: Verifying citations are valid and consistent
- Contradiction checks: Detecting contradictions within documents

This is designed to be closer to real workflows than SNLI/toy demos,
making it suitable for integration with AI Lawyer-style document pipelines.
"""

from .encoder import DocumentEncoder
from .head import DocumentHead

__all__ = ["DocumentEncoder", "DocumentHead"]


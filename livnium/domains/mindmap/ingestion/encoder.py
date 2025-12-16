"""
Mindmap Encoder: Text → Embeddings

Simple encoder wrapper that can use sentence-transformers or fallback.
This is domain-level encoding - happens outside the kernel.
"""

import torch
import torch.nn as nn
from typing import Optional


class SimpleTextEncoder:
    """
    Simple fallback encoder using basic torch operations.
    
    This is a minimal encoder for testing. For production,
    use sentence-transformers or your quantum embed.
    """
    def __init__(self, dim: int = 256, vocab_size: int = 50000):
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self._vocab = {}
        self._next_id = 0
        self.vocab_size = vocab_size
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Simple tokenization - just word IDs."""
        words = text.lower().split()
        token_ids = []
        for word in words:
            if word not in self._vocab:
                if self._next_id >= self.vocab_size:
                    # Hash to existing vocab if overflow
                    self._vocab[word] = hash(word) % self.vocab_size
                else:
                    self._vocab[word] = self._next_id
                    self._next_id += 1
            token_ids.append(self._vocab[word])
        return torch.tensor(token_ids, dtype=torch.long)
    
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text to vector.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector [dim]
        """
        token_ids = self._tokenize(text)
        if len(token_ids) == 0:
            return torch.zeros(self.dim)
        # Mean pooling
        emb = self.embedding(token_ids)  # [L, dim]
        return emb.mean(dim=0)  # [dim]


class SentenceTransformerEncoder:
    """
    Wrapper for sentence-transformers if available.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text to vector.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector [dim]
        """
        embedding = self.model.encode(text, convert_to_tensor=True)
        return embedding


def get_encoder(use_sentence_transformers: bool = False, **kwargs):
    """
    Get an encoder instance.
    
    Args:
        use_sentence_transformers: Whether to use sentence-transformers (if available)
        **kwargs: Additional arguments for encoder
        
    Returns:
        Encoder instance with encode(text: str) -> torch.Tensor method
    """
    if use_sentence_transformers:
        try:
            return SentenceTransformerEncoder(**kwargs)
        except ImportError:
            print("Warning: sentence-transformers not available, using simple encoder")
            return SimpleTextEncoder(**kwargs)
    else:
        return SimpleTextEncoder(**kwargs)


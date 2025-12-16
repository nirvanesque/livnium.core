"""
Document Workflow Encoder: Real-World Document Processing

Encodes documents, citations, and claims into state vectors for constraint checking.
Uses kernel physics for:
- Retrieval relevance (alignment between query and document)
- Citation validity (alignment between claim and citation)
- Contradiction detection (divergence between claims)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a document with text and metadata."""
    text: str
    doc_id: str
    metadata: Optional[Dict] = None


@dataclass
class Citation:
    """Represents a citation reference."""
    claim_text: str
    cited_doc_id: str
    citation_text: Optional[str] = None
    position: Optional[int] = None


@dataclass
class Claim:
    """Represents a claim in a document."""
    text: str
    claim_id: str
    position: int
    citations: List[Citation] = None


class DocumentEncoder(nn.Module):
    """
    Encoder for document workflow domain.
    
    Handles:
    - Encoding documents for retrieval
    - Encoding claims and citations for validity checking
    - Encoding claims for contradiction detection
    """
    
    def __init__(
        self,
        text_encoder: Optional[nn.Module] = None,
        dim: int = 256,
        vocab_size: int = 2000
    ):
        """
        Initialize document encoder.
        
        Args:
            text_encoder: Optional text encoder (e.g., sentence transformer)
            dim: Dimension of state vectors
            vocab_size: Vocabulary size for fallback embedding
        """
        super().__init__()
        self.dim = dim
        self.text_encoder = text_encoder
        
        # Fallback embedding if no text encoder provided
        if text_encoder is None:
            self.embedding = nn.Embedding(vocab_size, dim)
            self.proj = nn.Linear(dim, dim)
        else:
            self.embedding = None
            self.proj = None
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text to vector.
        
        Args:
            text: Text to encode
            
        Returns:
            Encoded vector [dim]
        """
        if self.text_encoder is not None:
            # Use provided text encoder
            return self.text_encoder.encode(text)
        else:
            # Simple fallback: tokenize and mean pool
            # In practice, you'd use a proper tokenizer
            # This is a placeholder for demonstration
            tokens = self._simple_tokenize(text)
            if len(tokens) == 0:
                tokens = [0]  # padding token
            token_ids = torch.tensor(tokens[:100], dtype=torch.long)  # truncate
            emb = self.embedding(token_ids)  # [L, dim]
            v = emb.mean(dim=0)  # [dim]
            v = self.proj(v) if self.proj else v
            return v
    
    def _simple_tokenize(self, text: str) -> List[int]:
        """
        Simple tokenization (placeholder).
        
        In practice, use a proper tokenizer.
        """
        # Placeholder: hash-based tokenization
        words = text.lower().split()
        tokens = []
        for word in words[:100]:  # truncate
            token_id = hash(word) % (self.embedding.num_embeddings if self.embedding else 2000)
            tokens.append(token_id)
        return tokens if tokens else [0]
    
    def encode_document(self, document: Document) -> torch.Tensor:
        """
        Encode a document for retrieval.
        
        Args:
            document: Document to encode
            
        Returns:
            Document vector [dim]
        """
        return self.encode_text(document.text)
    
    def encode_claim(self, claim: Claim) -> torch.Tensor:
        """
        Encode a claim for contradiction checking.
        
        Args:
            claim: Claim to encode
            
        Returns:
            Claim vector [dim]
        """
        return self.encode_text(claim.text)
    
    def encode_citation(self, citation: Citation) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a citation for validity checking.
        
        Returns both claim and citation vectors.
        
        Args:
            citation: Citation to encode
            
        Returns:
            Tuple of (claim_vector, citation_vector) both [dim]
        """
        claim_vec = self.encode_text(citation.claim_text)
        citation_vec = self.encode_text(citation.citation_text or citation.cited_doc_id)
        return claim_vec, citation_vec
    
    def generate_constraints(
        self,
        state: torch.Tensor,
        claim: Optional[Claim] = None,
        other_claims: Optional[List[Claim]] = None,
        citation: Optional[Citation] = None,
        query: Optional[str] = None,
        document: Optional[Document] = None
    ) -> Dict:
        """
        Generate constraints from state and context.
        
        Uses kernel.physics for:
        - Retrieval relevance (query-document alignment)
        - Citation validity (claim-citation alignment)
        - Contradiction detection (claim-claim divergence)
        
        Args:
            state: Current state vector
            claim: Current claim (for contradiction/citation checks)
            other_claims: Other claims to check against
            citation: Citation to validate
            query: Query text (for retrieval)
            document: Document (for retrieval)
            
        Returns:
            Dictionary of constraints
        """
        from livnium.kernel.physics import alignment, divergence, tension
        from livnium.engine.ops_torch import TorchOps
        
        ops = TorchOps()
        
        # Create state wrapper for kernel physics
        class StateWrapper:
            def __init__(self, vec):
                self._vec = vec
            def vector(self):
                return self._vec
            def norm(self):
                return torch.norm(self._vec, p=2)
        
        constraints = {
            "state": state,
            "norm": torch.norm(state, p=2),
        }
        
        # Retrieval relevance: query-document alignment
        if query is not None and document is not None:
            query_vec = self.encode_text(query)
            doc_vec = self.encode_document(document)
            
            query_state = StateWrapper(query_vec)
            doc_state = StateWrapper(doc_vec)
            
            align = alignment(ops, query_state, doc_state)
            div = divergence(ops, query_state, doc_state)
            tens = tension(ops, div)
            
            constraints["retrieval"] = {
                "alignment": align,
                "divergence": div,
                "tension": tens,
                "relevance": align  # Higher alignment = more relevant
            }
        
        # Citation validity: claim-citation alignment
        if citation is not None:
            claim_vec, citation_vec = self.encode_citation(citation)
            
            claim_state = StateWrapper(claim_vec)
            citation_state = StateWrapper(citation_vec)
            
            align = alignment(ops, claim_state, citation_state)
            div = divergence(ops, claim_state, citation_state)
            tens = tension(ops, div)
            
            constraints["citation"] = {
                "alignment": align,
                "divergence": div,
                "tension": tens,
                "is_valid": align > 0.5  # Threshold for validity
            }
        
        # Contradiction detection: claim-claim divergence
        if claim is not None and other_claims:
            contradictions = []
            claim_vec = self.encode_claim(claim)
            claim_state = StateWrapper(claim_vec)
            
            for other_claim in other_claims:
                other_vec = self.encode_claim(other_claim)
                other_state = StateWrapper(other_vec)
                
                align = alignment(ops, claim_state, other_state)
                div = divergence(ops, claim_state, other_state)
                tens = tension(ops, div)
                
                # High divergence = potential contradiction
                is_contradiction = div > 0.5  # Threshold
                
                contradictions.append({
                    "other_claim_id": other_claim.claim_id,
                    "alignment": align,
                    "divergence": div,
                    "tension": tens,
                    "is_contradiction": is_contradiction
                })
            
            constraints["contradictions"] = contradictions
        
        return constraints
    
    def build_initial_state(
        self,
        claim: Claim,
        document: Optional[Document] = None,
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build initial state from claim and optional document.
        
        Args:
            claim: Claim to encode
            document: Optional document context
            add_noise: Whether to add symmetry-breaking noise
            
        Returns:
            Tuple of (initial_state, claim_vector, document_vector)
        """
        v_claim = self.encode_claim(claim)
        v_doc = self.encode_document(document) if document else torch.zeros_like(v_claim)
        
        h0 = v_claim + v_doc
        if add_noise:
            from livnium.engine.config import defaults
            h0 = h0 + defaults.EPS_NOISE * torch.randn_like(h0)
        
        return h0, v_claim, v_doc


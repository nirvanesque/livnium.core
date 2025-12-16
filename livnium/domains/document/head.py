"""
Document Workflow Head: Output Interpretation

Interprets collapsed state to produce:
- Retrieval scores (relevance)
- Citation validity scores
- Contradiction detection scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DocumentHead(nn.Module):
    """
    Head for document workflow domain.
    
    Produces outputs for:
    - Retrieval: relevance score [0, 1]
    - Citation validity: validity score [0, 1]
    - Contradiction: contradiction score [0, 1]
    """
    
    def __init__(self, dim: int):
        """
        Initialize document head.
        
        Args:
            dim: Dimension of input state vector
        """
        super().__init__()
        self.dim = dim
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(dim + 3, dim),  # +3 for alignment, divergence, tension
            nn.ReLU(),
            nn.Linear(dim, dim // 2),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.retrieval_head = nn.Sequential(
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.citation_head = nn.Sequential(
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.contradiction_head = nn.Sequential(
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        h_final: torch.Tensor,
        v_claim: torch.Tensor,
        v_doc: torch.Tensor,
        task: str = "retrieval"
    ) -> torch.Tensor:
        """
        Forward pass: state → task-specific output.
        
        Args:
            h_final: Collapsed state vector [batch, dim] or [dim]
            v_claim: Claim vector [batch, dim] or [dim]
            v_doc: Document vector [batch, dim] or [dim]
            task: Task type ("retrieval", "citation", "contradiction")
            
        Returns:
            Task-specific output score [batch, 1] or [1]
        """
        # Ensure batch dimension
        squeeze = False
        if h_final.dim() == 1:
            h_final = h_final.unsqueeze(0)
            v_claim = v_claim.unsqueeze(0)
            v_doc = v_doc.unsqueeze(0)
            squeeze = True
        
        # Normalize vectors
        v_claim_n = F.normalize(v_claim, dim=-1)
        v_doc_n = F.normalize(v_doc, dim=-1)
        
        # Compute physics signals using kernel
        from livnium.kernel.physics import alignment, divergence, tension
        from livnium.engine.ops_torch import TorchOps
        
        ops = TorchOps()
        
        # Create state wrappers
        class StateWrapper:
            def __init__(self, vec):
                self._vec = vec
            def vector(self):
                return self._vec
            def norm(self):
                return torch.norm(self._vec, p=2)
        
        # Compute alignment, divergence, tension
        align_values = []
        div_values = []
        tens_values = []
        
        for i in range(v_claim_n.shape[0]):
            claim_state = StateWrapper(v_claim_n[i])
            doc_state = StateWrapper(v_doc_n[i])
            
            align_val = alignment(ops, claim_state, doc_state)
            div_val = divergence(ops, claim_state, doc_state)
            tens_val = tension(ops, div_val)
            
            align_values.append(torch.tensor(align_val, device=v_claim_n.device, dtype=v_claim_n.dtype))
            div_values.append(torch.tensor(div_val, device=v_claim_n.device, dtype=v_claim_n.dtype))
            tens_values.append(torch.tensor(tens_val, device=v_claim_n.device, dtype=v_claim_n.dtype))
        
        align = torch.stack(align_values).unsqueeze(-1)  # [B, 1]
        div = torch.stack(div_values).unsqueeze(-1)  # [B, 1]
        tens = torch.stack(tens_values).unsqueeze(-1)  # [B, 1]
        
        # Concatenate features
        features = torch.cat([h_final, align, div, tens], dim=-1)  # [B, dim+3]
        
        # Extract shared features
        shared_features = self.feature_extractor(features)  # [B, dim//2]
        
        # Task-specific output
        if task == "retrieval":
            output = self.retrieval_head(shared_features)
        elif task == "citation":
            output = self.citation_head(shared_features)
        elif task == "contradiction":
            output = self.contradiction_head(shared_features)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        if squeeze:
            output = output.squeeze(0)
        
        return output


"""
SNLI Workflow: High-Level Orchestrator

Integrates Encoder, CollapseEngine, and Head with Instrumentation.
"""

import torch
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from livnium.domains.snli.encoder import SNLIEncoder
from livnium.domains.snli.head import SNLIHead
from livnium.engine.collapse.engine import CollapseEngine
from livnium.instrumentation.ledger import TensionLedger

@dataclass
class SNLIAnalysisResult:
    """Consolidated result of SNLI inference."""
    label: str
    confidence: float
    alignment: float
    tension: float
    audit_summary: Optional[Dict[str, Any]] = None

class SNLIWorkflow:
    """
    Standard orchestrator for SNLI inference.
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 6,
        vocab_size: int = 2000,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.encoder = SNLIEncoder(dim=dim, vocab_size=vocab_size).to(self.device)
        self.engine = CollapseEngine(dim=dim, num_layers=num_layers).to(self.device)
        self.head = SNLIHead(dim=dim).to(self.device)
        
        self.id_to_label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        self.vocab = {"<pad>": 0, "<unk>": 1} # Simple internal vocab for demo

    def _tokenize(self, text: str) -> torch.Tensor:
        """Fallback tokenizer for demo."""
        # In a real setup, we'd load the vocab from trained model
        tokens = []
        for word in text.lower().split():
            # Use hash as a deterministic mockup if word not in vocab
            if word not in self.vocab:
                self.vocab[word] = (hash(word) % 1998) + 2
            tokens.append(self.vocab[word])
        
        # Pad to 50
        if len(tokens) < 50:
            tokens = tokens + [0] * (50 - len(tokens))
        else:
            tokens = tokens[:50]
            
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

    def analyze(
        self,
        premise: str,
        hypothesis: str,
        use_instrumentation: bool = True
    ) -> SNLIAnalysisResult:
        """
        Run full inference pipeline for a sentence pair.
        """
        # 1. Setup Instrumentation
        ledger = None
        if use_instrumentation:
            ledger = TensionLedger(["premise", "hypothesis"])
            
        # 2. Encode
        p_ids = self._tokenize(premise)
        h_ids = self._tokenize(hypothesis)
        
        h0, v_p, v_h = self.encoder.build_initial_state(p_ids, h_ids)
        constraints = self.encoder.generate_constraints(h0, v_p, v_h)
        
        # 3. Collapse
        h_final, trace = self.engine.collapse(h0, auditor=ledger)
        
        # 4. Classify
        logits = self.head(h_final, v_p, v_h, auditor=ledger)
        probs = torch.softmax(logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)
        
        return SNLIAnalysisResult(
            label=self.id_to_label[pred.item()],
            confidence=conf.item(),
            alignment=constraints["alignment"].item() if isinstance(constraints["alignment"], torch.Tensor) else constraints["alignment"],
            tension=constraints["tension"].item() if isinstance(constraints["tension"], torch.Tensor) else constraints["tension"],
            audit_summary=ledger.get_summary() if ledger else None
        )

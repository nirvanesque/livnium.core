"""
Document Pipeline: Draft > Verify Constraints > Finalize Workflow

Provides a complete pipeline for document processing workflows:
1. Draft: Create initial document/claim
2. Verify Constraints: Check retrieval, citations, contradictions
3. Finalize: Accept or reject based on constraints

Designed for integration with AI Lawyer-style document pipelines.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import torch

from livnium.domains.document.encoder import (
    DocumentEncoder, Document, Claim, Citation
)
from livnium.engine.collapse.engine import CollapseEngine
from livnium.domains.document.head import DocumentHead
from livnium.kernel.constraints import ConstraintChecker
from livnium.kernel.ledgers import Ledger
from livnium.integration.constraint_verifier import ConstraintVerifier, VerificationResult


class PipelineStage(Enum):
    """Stages of the document pipeline."""
    DRAFT = "draft"
    VERIFY = "verify"
    FINALIZE = "finalize"


@dataclass
class VerificationReport:
    """Report from constraint verification."""
    retrieval_score: float
    citation_validity: Dict[str, float]  # claim_id -> validity score
    contradictions: List[Dict[str, Any]]  # List of contradiction detections
    is_valid: bool
    explanation: str


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    stage: PipelineStage
    is_accepted: bool
    verification_report: Optional[VerificationReport]
    constraint_violations: List[Dict[str, Any]]
    explanation: str
    final_state: Optional[torch.Tensor] = None


class DocumentPipeline:
    """
    Document processing pipeline with constraint verification.
    
    Workflow:
    1. Draft: Create document with claims and citations
    2. Verify: Check retrieval, citation validity, contradictions
    3. Finalize: Accept or reject based on constraints
    """
    
    def __init__(
        self,
        encoder: DocumentEncoder,
        collapse_engine: CollapseEngine,
        head: DocumentHead,
        constraint_verifier: Optional[ConstraintVerifier] = None
    ):
        """
        Initialize document pipeline.
        
        Args:
            encoder: Document encoder
            collapse_engine: Collapse engine for state evolution
            head: Document head for output interpretation
            constraint_verifier: Optional constraint verifier (creates default if None)
        """
        self.encoder = encoder
        self.collapse_engine = collapse_engine
        self.head = head
        
        if constraint_verifier is None:
            constraint_verifier = ConstraintVerifier()
        self.constraint_verifier = constraint_verifier
    
    def draft(
        self,
        claims: List[Claim],
        document: Optional[Document] = None
    ) -> Dict[str, Any]:
        """
        Draft stage: Create initial document state.
        
        Args:
            claims: List of claims in the document
            document: Optional document context
            
        Returns:
            Dictionary with initial state and metadata
        """
        # Encode first claim as initial state
        if not claims:
            raise ValueError("At least one claim is required")
        
        initial_claim = claims[0]
        h0, v_claim, v_doc = self.encoder.build_initial_state(
            initial_claim, document, add_noise=True
        )
        
        return {
            "initial_state": h0,
            "claim_vector": v_claim,
            "document_vector": v_doc,
            "claims": claims,
            "document": document
        }
    
    def verify(
        self,
        draft_result: Dict[str, Any],
        query: Optional[str] = None,
        check_retrieval: bool = True,
        check_citations: bool = True,
        check_contradictions: bool = True
    ) -> VerificationReport:
        """
        Verify stage: Check constraints on draft.
        
        Args:
            draft_result: Result from draft() stage
            query: Optional query for retrieval check
            check_retrieval: Whether to check retrieval relevance
            check_citations: Whether to check citation validity
            check_contradictions: Whether to check for contradictions
            
        Returns:
            VerificationReport with all constraint checks
        """
        claims = draft_result["claims"]
        document = draft_result.get("document")
        h0 = draft_result["initial_state"]
        
        # Collapse state
        h_final, trace = self.collapse_engine.collapse(h0)
        
        # Initialize results
        retrieval_score = 0.0
        citation_validity = {}
        contradictions = []
        
        # Check retrieval if requested
        if check_retrieval and query is not None and document is not None:
            v_claim = draft_result["claim_vector"]
            v_doc = draft_result["document_vector"]
            retrieval_score = self.head(
                h_final, v_claim, v_doc, task="retrieval"
            ).item()
        
        # Check citations if requested
        if check_citations:
            for claim in claims:
                if claim.citations:
                    for citation in claim.citations:
                        v_claim, v_citation = self.encoder.encode_citation(citation)
                        # Use citation head to check validity
                        validity = self.head(
                            h_final, v_claim, v_citation, task="citation"
                        ).item()
                        citation_key = f"{claim.claim_id}:{citation.cited_doc_id}"
                        citation_validity[citation_key] = validity
        
        # Check contradictions if requested
        if check_contradictions and len(claims) > 1:
            for i, claim in enumerate(claims):
                other_claims = [c for j, c in enumerate(claims) if j != i]
                v_claim = self.encoder.encode_claim(claim)
                v_doc = draft_result["document_vector"]
                
                # Check contradiction with each other claim
                for other_claim in other_claims:
                    v_other = self.encoder.encode_claim(other_claim)
                    contradiction_score = self.head(
                        h_final, v_claim, v_other, task="contradiction"
                    ).item()
                    
                    if contradiction_score > 0.5:  # Threshold
                        contradictions.append({
                            "claim_id": claim.claim_id,
                            "other_claim_id": other_claim.claim_id,
                            "contradiction_score": contradiction_score
                        })
        
        # Determine overall validity
        is_valid = True
        explanation_parts = []
        
        if check_retrieval and retrieval_score < 0.5:
            is_valid = False
            explanation_parts.append(f"Retrieval relevance too low: {retrieval_score:.3f}")
        
        if check_citations:
            invalid_citations = [
                k for k, v in citation_validity.items() if v < 0.5
            ]
            if invalid_citations:
                is_valid = False
                explanation_parts.append(f"Invalid citations: {', '.join(invalid_citations)}")
        
        if check_contradictions and contradictions:
            is_valid = False
            contradiction_pairs = [
                f"{c['claim_id']} vs {c['other_claim_id']}"
                for c in contradictions
            ]
            explanation_parts.append(f"Contradictions detected: {', '.join(contradiction_pairs)}")
        
        explanation = "; ".join(explanation_parts) if explanation_parts else "All constraints satisfied"
        
        return VerificationReport(
            retrieval_score=retrieval_score,
            citation_validity=citation_validity,
            contradictions=contradictions,
            is_valid=is_valid,
            explanation=explanation
        )
    
    def finalize(
        self,
        draft_result: Dict[str, Any],
        verification_report: VerificationReport,
        accept_threshold: float = 0.7
    ) -> PipelineResult:
        """
        Finalize stage: Accept or reject based on verification.
        
        Args:
            draft_result: Result from draft() stage
            verification_report: Report from verify() stage
            accept_threshold: Threshold for acceptance (0-1)
            
        Returns:
            PipelineResult with acceptance decision
        """
        # Collapse final state
        h0 = draft_result["initial_state"]
        h_final, trace = self.collapse_engine.collapse(h0)
        
        # Determine acceptance
        is_accepted = verification_report.is_valid
        
        # Additional threshold check
        if verification_report.retrieval_score < accept_threshold:
            is_accepted = False
        
        # Build constraint violations list
        violations = []
        if not verification_report.is_valid:
            violations.append({
                "type": "verification_failed",
                "explanation": verification_report.explanation
            })
        
        explanation = (
            f"Document {'accepted' if is_accepted else 'rejected'}: "
            f"{verification_report.explanation}"
        )
        
        return PipelineResult(
            stage=PipelineStage.FINALIZE,
            is_accepted=is_accepted,
            verification_report=verification_report,
            constraint_violations=violations,
            explanation=explanation,
            final_state=h_final
        )
    
    def run(
        self,
        claims: List[Claim],
        document: Optional[Document] = None,
        query: Optional[str] = None,
        accept_threshold: float = 0.7
    ) -> PipelineResult:
        """
        Run complete pipeline: draft > verify > finalize.
        
        Args:
            claims: List of claims in the document
            document: Optional document context
            query: Optional query for retrieval check
            accept_threshold: Threshold for acceptance
            
        Returns:
            PipelineResult with final decision
        """
        # Draft
        draft_result = self.draft(claims, document)
        
        # Verify
        verification_report = self.verify(draft_result, query=query)
        
        # Finalize
        result = self.finalize(draft_result, verification_report, accept_threshold)
        
        return result


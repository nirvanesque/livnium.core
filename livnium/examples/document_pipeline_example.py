"""
Example: Document Pipeline with Constraint Verification

Demonstrates the draft > verify constraints > finalize workflow
for document processing with transparent refusal paths.
"""

import torch
from livnium.domains.document.encoder import DocumentEncoder, Document, Claim, Citation
from livnium.engine.collapse.engine import CollapseEngine
from livnium.domains.document.head import DocumentHead
from livnium.integration.pipeline import DocumentPipeline


def example_basic_pipeline():
    """Basic example of document pipeline."""
    # Initialize components
    dim = 256
    encoder = DocumentEncoder(dim=dim)
    collapse_engine = CollapseEngine(dim=dim, num_layers=3)
    head = DocumentHead(dim=dim)
    
    # Create pipeline
    pipeline = DocumentPipeline(encoder, collapse_engine, head)
    
    # Create sample document and claims
    document = Document(
        text="This is a legal document about contract law.",
        doc_id="doc1"
    )
    
    claim1 = Claim(
        text="The contract is valid under Section 2-201.",
        claim_id="claim1",
        position=0,
        citations=[
            Citation(
                claim_text="The contract is valid under Section 2-201.",
                cited_doc_id="statute1",
                citation_text="Section 2-201 of the Uniform Commercial Code"
            )
        ]
    )
    
    claim2 = Claim(
        text="The contract requires written confirmation.",
        claim_id="claim2",
        position=1,
        citations=[
            Citation(
                claim_text="The contract requires written confirmation.",
                cited_doc_id="statute1",
                citation_text="Section 2-201 requires written confirmation"
            )
        ]
    )
    
    # Run pipeline
    result = pipeline.run(
        claims=[claim1, claim2],
        document=document,
        query="contract validity requirements"
    )
    
    print(f"Pipeline Result:")
    print(f"  Accepted: {result.is_accepted}")
    print(f"  Explanation: {result.explanation}")
    print(f"  Violations: {len(result.constraint_violations)}")
    
    if result.verification_report:
        print(f"  Retrieval Score: {result.verification_report.retrieval_score:.3f}")
        print(f"  Citation Validity: {result.verification_report.citation_validity}")
        print(f"  Contradictions: {len(result.verification_report.contradictions)}")


def example_constraint_explanation():
    """Example of getting constraint explanations."""
    from livnium.kernel.constraints import ConstraintChecker
    from livnium.kernel.ledgers import Ledger
    from livnium.kernel.types import Operation
    
    # Create constraint checker
    ledger = Ledger()
    checker = ConstraintChecker(ledger)
    
    # Create sample states (simplified for example)
    class SimpleState:
        def __init__(self, vec):
            self._vec = vec
        def vector(self):
            return self._vec
        def norm(self):
            return torch.norm(self._vec, p=2)
    
    state_before = SimpleState(torch.randn(64))
    state_after = SimpleState(torch.randn(64))
    
    # Check transition
    check = checker.check_transition(state_before, state_after, Operation.COLLAPSE)
    
    print(f"Transition Check:")
    print(f"  Admissible: {check.is_admissible}")
    print(f"  Explanation: {check.explain()}")
    
    # Example constants (using values from defaults or explicit names)
    from livnium.engine.config import defaults
    EXAMPLE_ENERGY_COST = defaults.MAX_NORM
    EXAMPLE_AVAILABLE_ENERGY = defaults.MAX_NORM / 2
    
    # Check promotion with insufficient energy
    check_promotion = checker.check_promotion(
        state=state_before,
        depth=2,
        energy_cost=EXAMPLE_ENERGY_COST,
        available_energy=EXAMPLE_AVAILABLE_ENERGY
    )
    
    print(f"\nPromotion Check:")
    print(f"  Admissible: {check_promotion.is_admissible}")
    print(f"  Explanation: {check_promotion.explain()}")


def example_integration_api():
    """Example of using the integration API."""
    from livnium.integration.constraint_verifier import ConstraintVerifier
    from livnium.kernel.types import Operation
    
    # Create verifier
    verifier = ConstraintVerifier()
    
    # Create sample states
    class SimpleState:
        def __init__(self, vec):
            self._vec = vec
        def vector(self):
            return self._vec
        def norm(self):
            return torch.norm(self._vec, p=2)
    
    state_before = SimpleState(torch.randn(64))
    state_after = SimpleState(torch.randn(64))
    
    # Verify transition
    result = verifier.verify_transition(state_before, state_after, Operation.COLLAPSE)
    
    print(f"Verification Result:")
    print(f"  Valid: {result.is_valid}")
    print(f"  Explanation: {result.explanation}")
    print(f"  Violations: {result.violations}")
    
    # This provides a transparent refusal path - the agent knows exactly why
    # the action is inadmissible


if __name__ == "__main__":
    print("=== Basic Pipeline Example ===")
    example_basic_pipeline()
    
    print("\n=== Constraint Explanation Example ===")
    example_constraint_explanation()
    
    print("\n=== Integration API Example ===")
    example_integration_api()


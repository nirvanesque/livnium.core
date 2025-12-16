#!/usr/bin/env python3
"""
Test Script for Integration Features

Tests the new constraint query system, document domain, and integration API.
Run with: python3 test_integration_features.py
"""

import torch
import sys

def test_constraint_checker():
    """Test 1: Constraint checker with explanations"""
    print("=" * 60)
    print("TEST 1: Constraint Checker")
    print("=" * 60)
    
    try:
        from livnium.kernel.constraints import ConstraintChecker
        from livnium.kernel.ledgers import Ledger
        from livnium.kernel.types import Operation
        
        # Create constraint checker
        ledger = Ledger()
        checker = ConstraintChecker(ledger)
        
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
        
        # Test 1a: Valid transition
        print("\n1a. Checking valid transition...")
        check = checker.check_transition(state_before, state_after, Operation.COLLAPSE)
        print(f"   Admissible: {check.is_admissible}")
        print(f"   Explanation: {check.explain()}")
        assert check.is_admissible, "Valid transition should be admissible"
        
        # Test 1b: Promotion with insufficient energy
        print("\n1b. Checking promotion with insufficient energy...")
        check_promotion = checker.check_promotion(
            state=state_before,
            depth=2,
            energy_cost=10.0,
            available_energy=5.0
        )
        print(f"   Admissible: {check_promotion.is_admissible}")
        print(f"   Explanation: {check_promotion.explain()}")
        assert not check_promotion.is_admissible, "Insufficient energy should make promotion inadmissible"
        assert "Insufficient energy" in check_promotion.explain(), "Explanation should mention energy"
        
        # Test 1c: Invalid depth
        print("\n1c. Checking promotion with invalid depth...")
        check_depth = checker.check_promotion(
            state=state_before,
            depth=-1,
            energy_cost=1.0,
            available_energy=10.0
        )
        print(f"   Admissible: {check_depth.is_admissible}")
        print(f"   Explanation: {check_depth.explain()}")
        assert not check_depth.is_admissible, "Negative depth should be inadmissible"
        
        print("\n✓ Constraint checker tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Constraint checker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_constraint_verifier():
    """Test 2: High-level constraint verifier API"""
    print("\n" + "=" * 60)
    print("TEST 2: Constraint Verifier API")
    print("=" * 60)
    
    try:
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
        
        # Test verification
        print("\n2a. Verifying transition...")
        result = verifier.verify_transition(state_before, state_after, Operation.COLLAPSE)
        print(f"   Valid: {result.is_valid}")
        print(f"   Explanation: {result.explanation}")
        print(f"   Violations: {len(result.violations)}")
        
        # Test serialization
        result_dict = result.to_dict()
        assert "is_valid" in result_dict
        assert "explanation" in result_dict
        assert "violations" in result_dict
        
        print("\n✓ Constraint verifier tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Constraint verifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_encoder():
    """Test 3: Document encoder"""
    print("\n" + "=" * 60)
    print("TEST 3: Document Encoder")
    print("=" * 60)
    
    try:
        from livnium.domains.document.encoder import DocumentEncoder, Document, Claim, Citation
        
        # Create encoder
        encoder = DocumentEncoder(dim=128)
        
        # Create sample document
        document = Document(
            text="This is a legal document about contract law and validity requirements.",
            doc_id="doc1"
        )
        
        # Create claims
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
            position=1
        )
        
        # Test encoding
        print("\n3a. Encoding document...")
        doc_vec = encoder.encode_document(document)
        print(f"   Document vector shape: {doc_vec.shape}")
        assert doc_vec.shape == (128,), f"Expected shape (128,), got {doc_vec.shape}"
        
        print("\n3b. Encoding claim...")
        claim_vec = encoder.encode_claim(claim1)
        print(f"   Claim vector shape: {claim_vec.shape}")
        assert claim_vec.shape == (128,), f"Expected shape (128,), got {claim_vec.shape}"
        
        print("\n3c. Encoding citation...")
        claim_v, citation_v = encoder.encode_citation(claim1.citations[0])
        print(f"   Claim vector shape: {claim_v.shape}")
        print(f"   Citation vector shape: {citation_v.shape}")
        
        print("\n3d. Generating constraints...")
        state = torch.randn(128)
        constraints = encoder.generate_constraints(
            state=state,
            claim=claim1,
            other_claims=[claim2],
            citation=claim1.citations[0],
            query="contract validity",
            document=document
        )
        print(f"   Constraints keys: {list(constraints.keys())}")
        assert "state" in constraints
        assert "norm" in constraints
        
        print("\n✓ Document encoder tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Document encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_pipeline():
    """Test 4: Document pipeline workflow"""
    print("\n" + "=" * 60)
    print("TEST 4: Document Pipeline")
    print("=" * 60)
    
    try:
        from livnium.domains.document.encoder import DocumentEncoder, Document, Claim, Citation
        from livnium.engine.collapse.engine import CollapseEngine
        from livnium.domains.document.head import DocumentHead
        from livnium.integration.pipeline import DocumentPipeline
        
        # Initialize components
        dim = 128
        encoder = DocumentEncoder(dim=dim)
        collapse_engine = CollapseEngine(dim=dim, num_layers=2)  # Smaller for testing
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
        
        # Test draft stage
        print("\n4a. Testing draft stage...")
        draft_result = pipeline.draft([claim1, claim2], document)
        print(f"   Initial state shape: {draft_result['initial_state'].shape}")
        assert "initial_state" in draft_result
        assert "claims" in draft_result
        
        # Test verify stage
        print("\n4b. Testing verify stage...")
        verification = pipeline.verify(draft_result, query="contract validity requirements")
        print(f"   Retrieval score: {verification.retrieval_score:.3f}")
        print(f"   Citation validity: {len(verification.citation_validity)} citations checked")
        print(f"   Contradictions: {len(verification.contradictions)}")
        print(f"   Is valid: {verification.is_valid}")
        print(f"   Explanation: {verification.explanation}")
        
        # Test finalize stage
        print("\n4c. Testing finalize stage...")
        result = pipeline.finalize(draft_result, verification, accept_threshold=0.5)
        print(f"   Accepted: {result.is_accepted}")
        print(f"   Explanation: {result.explanation}")
        assert result.stage.value == "finalize"
        
        # Test full pipeline
        print("\n4d. Testing full pipeline run...")
        full_result = pipeline.run(
            claims=[claim1, claim2],
            document=document,
            query="contract validity requirements",
            accept_threshold=0.5
        )
        print(f"   Final result - Accepted: {full_result.is_accepted}")
        print(f"   Final result - Explanation: {full_result.explanation}")
        
        print("\n✓ Document pipeline tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Document pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("LIVNIUM Integration Features Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Constraint Checker", test_constraint_checker()))
    results.append(("Constraint Verifier", test_constraint_verifier()))
    results.append(("Document Encoder", test_document_encoder()))
    results.append(("Document Pipeline", test_document_pipeline()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())


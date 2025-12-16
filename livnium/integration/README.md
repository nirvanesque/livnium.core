# LIVNIUM Integration API

This module provides first-class constraint query and explanation capabilities, making LIVNIUM suitable for integration with external tooling like AI Lawyer-style document pipelines.

## Key Features

### 1. Transparent Refusal Paths

Instead of silent failures or weird behavior, LIVNIUM now provides clear explanations for why actions are inadmissible:

```python
from livnium.integration.constraint_verifier import ConstraintVerifier
from livnium.kernel.types import Operation

verifier = ConstraintVerifier()
result = verifier.verify_transition(state_before, state_after, Operation.COLLAPSE)

if not result.is_valid:
    print(result.explanation)
    # Output: "Action is inadmissible because:
    #          - State before collapse operation is invalid (missing vector or norm capability)"
```

### 2. Document Workflow Domain

A minimal reference domain closer to real workflows than SNLI/toy demos:

- **Retrieval**: Finding relevant documents/sections using alignment
- **Citation Validity**: Verifying citations are valid and consistent
- **Contradiction Checks**: Detecting contradictions within documents

### 3. Draft > Verify > Finalize Pipeline

Complete workflow for document processing:

```python
from livnium.integration.pipeline import DocumentPipeline
from livnium.domains.document.encoder import Document, Claim, Citation

# Create pipeline
pipeline = DocumentPipeline(encoder, collapse_engine, head)

# Run complete workflow
result = pipeline.run(
    claims=[claim1, claim2],
    document=document,
    query="contract validity requirements"
)

if result.is_accepted:
    print("Document passed all constraint checks")
else:
    print(f"Document rejected: {result.explanation}")
```

## Architecture

### Constraint System (`kernel/constraints.py`)

First-class constraint query system that:
- Exposes kernel invariants as queryable constraints
- Provides explanations for constraint violations
- Maintains separation: observation (ledgers) vs enforcement (admissibility)

### Document Domain (`domains/document/`)

Real-world document processing domain with:
- `DocumentEncoder`: Encodes documents, claims, and citations
- `DocumentHead`: Produces retrieval, citation validity, and contradiction scores
- Uses kernel physics for all constraint calculations

### Integration API (`integration/`)

High-level APIs for external systems:
- `ConstraintVerifier`: Simple API for constraint checking
- `DocumentPipeline`: Complete draft > verify > finalize workflow

## Usage Examples

### Basic Constraint Checking

```python
from livnium.kernel.constraints import ConstraintChecker
from livnium.kernel.ledgers import Ledger

checker = ConstraintChecker(Ledger())
check = checker.check_transition(state_before, state_after, Operation.COLLAPSE)

print(check.explain())
```

### Document Pipeline

```python
from livnium.integration.pipeline import DocumentPipeline

# Initialize
pipeline = DocumentPipeline(encoder, collapse_engine, head)

# Draft
draft_result = pipeline.draft(claims, document)

# Verify
verification = pipeline.verify(draft_result, query="search query")

# Finalize (with acceptance threshold)
result = pipeline.finalize(draft_result, verification, accept_threshold=0.7)

if not result.is_accepted:
    print(f"Rejected: {result.explanation}")
    # Explanation includes both verification failures and threshold failures
    # Example: "Document rejected: All constraints satisfied; 
    #          Retrieval score 0.492 below threshold 0.700"
```

**Note**: The pipeline separates **verification** (constraint checks) from **acceptance thresholds**. A document can pass verification but still be rejected if it doesn't meet acceptance thresholds. The explanation clearly indicates both types of failures with quantitative information.

### Integration with External Systems

The constraint system is designed to integrate naturally with AI Lawyer-style document pipelines:

1. **Draft**: Agent creates initial document/claim
2. **Verify Constraints**: LIVNIUM checks retrieval, citations, contradictions
3. **Finalize**: System accepts or rejects based on constraint violations

The transparent refusal paths mean agents get clear explanations:
- "This action is inadmissible because: citation validity too low (0.3 < 0.5)"
- "This action is inadmissible because: contradiction detected between claim1 and claim2"

## Benefits

1. **Transparency**: Agents understand exactly why actions fail
2. **Real-World Relevance**: Document domain is closer to actual workflows
3. **Integration Ready**: Clean APIs for external tooling
4. **Robustness**: Constraint checking handles messy inputs better than SNLI/toy demos

## See Also

- `examples/document_pipeline_example.py` - Complete usage examples
- `kernel/constraints.py` - Constraint query system implementation
- `domains/document/` - Document workflow domain


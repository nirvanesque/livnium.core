# Integration Features Summary

This document summarizes the new integration features added to LIVNIUM based on the integration idea.

## What Was Implemented

### 1. First-Class Constraint Query System (`kernel/constraints.py`)

**Problem**: Kernel invariants were only checked internally, leading to silent failures or weird behavior when actions were inadmissible.

**Solution**: Exposed kernel invariants as queryable constraints that agents can:
- Query to check if an action is admissible
- Get explanations for why an action is inadmissible
- Understand what constraints are active

**Key Components**:
- `ConstraintChecker`: Main interface for constraint checking
- `ConstraintViolation`: Represents a violation with explanation
- `ConstraintCheck`: Result of checking constraints with summary

**Example**:
```python
from livnium.kernel.constraints import ConstraintChecker

checker = ConstraintChecker(ledger)
check = checker.check_transition(state_before, state_after, Operation.COLLAPSE)

if not check.is_admissible:
    print(check.explain())
    # "Action is inadmissible because:
    #  - State before collapse operation is invalid..."
```

### 2. Document Workflow Domain (`domains/document/`)

**Problem**: SNLI/toy demos don't always prove robustness under messy inputs from real workflows.

**Solution**: Created a minimal reference domain closer to real workflows with:
- **Retrieval**: Finding relevant documents/sections using alignment
- **Citation Validity**: Verifying citations are valid and consistent
- **Contradiction Checks**: Detecting contradictions within documents

**Key Components**:
- `DocumentEncoder`: Encodes documents, claims, and citations
- `DocumentHead`: Produces retrieval, citation validity, and contradiction scores
- Uses kernel physics for all constraint calculations

**Example**:
```python
from livnium.domains.document.encoder import Document, Claim, Citation

encoder = DocumentEncoder(dim=256)
constraints = encoder.generate_constraints(
    state=state,
    claim=claim,
    other_claims=other_claims,
    citation=citation,
    query="search query",
    document=document
)
```

### 3. Integration API (`integration/`)

**Problem**: No clean API for external tooling to integrate with LIVNIUM.

**Solution**: Created high-level APIs for external systems:
- `ConstraintVerifier`: Simple API for constraint checking
- `DocumentPipeline`: Complete `draft > verify constraints > finalize` workflow

**Key Features**:
- Transparent refusal paths with clear explanations
- Natural integration with AI Lawyer-style document pipelines
- Handles messy inputs better than SNLI/toy demos

**Example**:
```python
from livnium.integration.pipeline import DocumentPipeline

pipeline = DocumentPipeline(encoder, collapse_engine, head)
result = pipeline.run(claims, document, query="search query")

if not result.is_accepted:
    print(f"Rejected: {result.explanation}")
    # "Document rejected: Invalid citations: claim1:statute1; 
    #  Contradictions detected: claim1 vs claim2"
```

## Architecture

### Separation of Concerns

The implementation maintains LIVNIUM's core architecture:
- **Kernel** (`kernel/constraints.py`): Pure constraint checking logic, no torch/numpy
- **Domains** (`domains/document/`): Domain-specific encoding and constraint generation
- **Integration** (`integration/`): High-level APIs for external systems

### Transparent Refusal Paths

Instead of silent failures, agents get clear explanations:
- "This action is inadmissible because: citation validity too low (0.3 < 0.5)"
- "This action is inadmissible because: contradiction detected between claim1 and claim2"
- "This action is inadmissible because: insufficient energy (required 10.0, available 5.0)"

## Benefits

1. **Transparency**: Agents understand exactly why actions fail
2. **Real-World Relevance**: Document domain is closer to actual workflows
3. **Integration Ready**: Clean APIs for external tooling
4. **Robustness**: Constraint checking handles messy inputs better

## Usage

See:
- `examples/document_pipeline_example.py` - Complete usage examples
- `integration/README.md` - Detailed integration documentation
- `livnium/README.md` - Updated main README with integration section

## Files Created

- `livnium/kernel/constraints.py` - Constraint query system
- `livnium/domains/document/__init__.py` - Document domain
- `livnium/domains/document/encoder.py` - Document encoder
- `livnium/domains/document/head.py` - Document head
- `livnium/integration/__init__.py` - Integration module
- `livnium/integration/constraint_verifier.py` - Constraint verifier API
- `livnium/integration/pipeline.py` - Document pipeline
- `livnium/integration/README.md` - Integration documentation
- `livnium/examples/document_pipeline_example.py` - Usage examples
- `INTEGRATION_FEATURES.md` - This summary

## Next Steps

The system is now ready for:
1. Integration with AI Lawyer-style document pipelines
2. Testing with real-world document workflows
3. Extension with additional constraint types
4. Performance optimization for production use


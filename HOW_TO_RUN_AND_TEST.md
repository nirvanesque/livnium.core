# How to Run and Test Integration Features

This document explains how to run and test the new integration features that were just created.

## Quick Start

### 1. Run the Test Suite (Recommended First Step)

```bash
python3 test_integration_features.py
```

This comprehensive test suite verifies:
- ✓ Constraint Checker with transparent explanations
- ✓ Constraint Verifier API
- ✓ Document Encoder (retrieval, citations, contradictions)
- ✓ Document Pipeline (draft > verify > finalize workflow)

**Expected output**: All 4 tests should pass with detailed output showing constraint explanations.

### 2. Run the Examples

```bash
./run_integration_examples.sh
```

Or manually:
```bash
PYTHONPATH=. python3 livnium/examples/document_pipeline_example.py
```

This demonstrates:
- Basic document pipeline workflow
- Constraint explanation examples
- Integration API usage

## What Each Test Does

### Test 1: Constraint Checker
**File**: `livnium/kernel/constraints.py`

Tests the core constraint query system that provides transparent refusal paths:

```python
# Example: Checking why a promotion is inadmissible
check = checker.check_promotion(
    state=state,
    depth=2,
    energy_cost=10.0,
    available_energy=5.0  # Not enough!
)

print(check.explain())
# Output: "Action is inadmissible because:
#          - Insufficient energy: required 10.0000, available 5.0000"
```

**What it verifies**:
- Valid transitions are marked as admissible
- Invalid operations get clear explanations
- Energy constraints are properly checked
- Depth constraints are properly checked

### Test 2: Constraint Verifier API
**File**: `livnium/integration/constraint_verifier.py`

Tests the high-level API for external systems:

```python
verifier = ConstraintVerifier()
result = verifier.verify_transition(state_before, state_after, Operation.COLLAPSE)

if not result.is_valid:
    print(result.explanation)  # Clear explanation
    print(result.violations)   # Detailed violation list
```

**What it verifies**:
- API provides clean interface for external systems
- Results can be serialized (to_dict())
- Explanations are human-readable

### Test 3: Document Encoder
**File**: `livnium/domains/document/encoder.py`

Tests the document workflow domain encoder:

```python
encoder = DocumentEncoder(dim=128)
constraints = encoder.generate_constraints(
    state=state,
    claim=claim,
    other_claims=other_claims,
    citation=citation,
    query="search query",
    document=document
)
```

**What it verifies**:
- Documents can be encoded to vectors
- Claims can be encoded
- Citations can be encoded
- Constraints are generated (retrieval, citation validity, contradictions)

### Test 4: Document Pipeline
**File**: `livnium/integration/pipeline.py`

Tests the complete `draft > verify constraints > finalize` workflow:

```python
pipeline = DocumentPipeline(encoder, collapse_engine, head)
result = pipeline.run(claims, document, query="search query")

if not result.is_accepted:
    print(f"Rejected: {result.explanation}")
```

**What it verifies**:
- Draft stage creates initial state
- Verify stage checks all constraints
- Finalize stage makes acceptance decision
- Full pipeline workflow works end-to-end

## Understanding the Output

### Successful Test Output

```
============================================================
TEST 1: Constraint Checker
============================================================

1a. Checking valid transition...
   Admissible: True
   Explanation: Action is admissible. All constraints satisfied.

1b. Checking promotion with insufficient energy...
   Admissible: False
   Explanation: Action is inadmissible because:
- Insufficient energy: required 10.0000, available 5.0000

✓ Constraint checker tests passed!
```

### Example Pipeline Output

```
=== Basic Pipeline Example ===
Pipeline Result:
  Accepted: False
  Explanation: Document rejected: Retrieval relevance too low: 0.492; 
               Contradictions detected: claim1 vs claim2, claim2 vs claim1; 
               Retrieval score 0.492 below threshold 0.700
  Violations: 1
  Retrieval Score: 0.492
  Citation Validity: {'claim1:statute1': 0.508, 'claim2:statute1': 0.508}
  Contradictions: 2
```

This shows:
- The document was **rejected** with **clear, quantitative explanations**:
  - Verification failures (contradictions detected)
  - Threshold failures (retrieval score below acceptance threshold)
- The system provides **transparent refusal paths** - you know exactly why it was rejected
- **Retrieval score** shows relevance (0.492 = 49.2% relevant)
- **Citation validity** shows which citations passed/failed
- **Contradictions** list shows which claims contradict each other
- **Threshold information** is included when documents fail acceptance thresholds even if verification passes

## Key Features Demonstrated

### 1. Transparent Refusal Paths

Instead of silent failures, you get clear, quantitative explanations:
- "Insufficient energy: required 10.0, available 5.0"
- "Invalid citations: claim1:statute1"
- "Contradictions detected: claim1 vs claim2"
- "Retrieval score 0.492 below threshold 0.700" (when threshold checks fail)

**Important**: The system separates **verification** (constraint checks) from **acceptance thresholds**. A document can pass verification but still be rejected if it doesn't meet acceptance thresholds (e.g., retrieval score too low). The explanation clearly indicates both types of failures.

### 2. Real-World Document Processing

The document domain handles:
- **Retrieval**: Finding relevant documents (alignment-based)
- **Citation Validity**: Checking if citations support claims
- **Contradiction Detection**: Finding conflicting claims

### 3. Integration-Ready APIs

Clean APIs for external systems:
- `ConstraintVerifier`: Simple constraint checking
- `DocumentPipeline`: Complete workflow management

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'livnium'`

**Solution**: Make sure you're running from the repo root and set PYTHONPATH:
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
PYTHONPATH=. python3 test_integration_features.py
```

### Missing Dependencies

**Problem**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### CUDA Errors

**Problem**: CUDA-related errors on systems without GPU

**Solution**: The code should auto-fallback to CPU. If not, force CPU:
```python
import torch
torch.set_default_tensor_type('torch.FloatTensor')
```

## Files Created

All new files are in the repository:

- **Core Constraint System**:
  - `livnium/kernel/constraints.py` - Constraint query system

- **Document Domain**:
  - `livnium/domains/document/__init__.py`
  - `livnium/domains/document/encoder.py`
  - `livnium/domains/document/head.py`

- **Integration APIs**:
  - `livnium/integration/__init__.py`
  - `livnium/integration/constraint_verifier.py`
  - `livnium/integration/pipeline.py`
  - `livnium/integration/README.md`

- **Examples & Tests**:
  - `livnium/examples/document_pipeline_example.py`
  - `test_integration_features.py`
  - `run_integration_examples.sh`

- **Documentation**:
  - `QUICK_START_INTEGRATION.md`
  - `HOW_TO_RUN_AND_TEST.md` (this file)
  - `INTEGRATION_FEATURES.md`

## Next Steps

1. **Explore the code**: Read through the example files to understand usage
2. **Modify examples**: Try changing the test data to see how constraints work
3. **Integrate**: Use `ConstraintVerifier` and `DocumentPipeline` in your own workflows
4. **Extend**: Add new constraint types or domain-specific checks

## Summary

The integration features provide:
- ✅ **Transparent refusal paths** - Clear explanations for constraint violations
- ✅ **Real-world domain** - Document processing closer to actual workflows
- ✅ **Integration APIs** - Clean interfaces for external systems
- ✅ **Complete workflow** - Draft > verify > finalize pipeline

All features are tested and working. Run `python3 test_integration_features.py` to verify!


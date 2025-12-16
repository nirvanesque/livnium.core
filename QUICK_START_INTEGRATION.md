# Quick Start: Integration Features

This guide shows you how to run and test the new integration features.

## Prerequisites

Make sure you have the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- `torch` (PyTorch)
- `numpy`
- `tqdm`

## Running Tests

### Option 1: Run the Complete Test Suite

```bash
python3 test_integration_features.py
```

This runs all integration feature tests:
- ✓ Constraint Checker
- ✓ Constraint Verifier API
- ✓ Document Encoder
- ✓ Document Pipeline

### Option 2: Run Individual Examples

**Method 1: Using the run script (recommended)**
```bash
./run_integration_examples.sh
```

**Method 2: Manual run with PYTHONPATH**
```bash
PYTHONPATH=. python3 livnium/examples/document_pipeline_example.py
```

**Method 3: From Python with sys.path**
```python
import sys
sys.path.insert(0, '.')
from livnium.examples.document_pipeline_example import *
```

This demonstrates:
- Basic document pipeline workflow
- Constraint explanation examples
- Integration API usage

## What Gets Tested

### 1. Constraint Checker (`kernel/constraints.py`)

Tests transparent refusal paths - getting explanations for why actions are inadmissible:

```python
from livnium.kernel.constraints import ConstraintChecker
from livnium.kernel.ledgers import Ledger
from livnium.kernel.types import Operation

checker = ConstraintChecker(Ledger())
check = checker.check_promotion(
    state=state,
    depth=2,
    energy_cost=10.0,
    available_energy=5.0  # Insufficient!
)

print(check.explain())
# "Action is inadmissible because:
#  - Insufficient energy: required 10.0000, available 5.0000"
```

### 2. Constraint Verifier API (`integration/constraint_verifier.py`)

High-level API for external systems:

```python
from livnium.integration.constraint_verifier import ConstraintVerifier

verifier = ConstraintVerifier()
result = verifier.verify_transition(state_before, state_after, Operation.COLLAPSE)

if not result.is_valid:
    print(result.explanation)
```

### 3. Document Encoder (`domains/document/encoder.py`)

Tests document encoding with retrieval, citation validity, and contradiction checks:

```python
from livnium.domains.document.encoder import DocumentEncoder, Document, Claim, Citation

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

### 4. Document Pipeline (`integration/pipeline.py`)

Tests the complete `draft > verify constraints > finalize` workflow:

```python
from livnium.integration.pipeline import DocumentPipeline

pipeline = DocumentPipeline(encoder, collapse_engine, head)
result = pipeline.run(claims, document, query="search query")

if not result.is_accepted:
    print(f"Rejected: {result.explanation}")
```

## Expected Output

When you run `test_integration_features.py`, you should see:

```
============================================================
LIVNIUM Integration Features Test Suite
============================================================
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

[... more tests ...]

============================================================
TEST SUMMARY
============================================================
  Constraint Checker: ✓ PASSED
  Constraint Verifier: ✓ PASSED
  Document Encoder: ✓ PASSED
  Document Pipeline: ✓ PASSED

Total: 4/4 tests passed

🎉 All tests passed!
```

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the repository root:

```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 test_integration_features.py
```

### Missing Dependencies

If you get `ModuleNotFoundError`, install dependencies:

```bash
pip install torch numpy tqdm
```

### CUDA Errors

If you get CUDA-related errors, the code should fall back to CPU automatically. If not, you can force CPU:

```python
import torch
torch.set_default_tensor_type('torch.FloatTensor')
```

## Next Steps

1. **Explore the examples**: See `livnium/examples/document_pipeline_example.py`
2. **Read the documentation**: See `livnium/integration/README.md`
3. **Integrate with your system**: Use `ConstraintVerifier` and `DocumentPipeline` in your workflows

## Files to Explore

- `test_integration_features.py` - Complete test suite
- `livnium/examples/document_pipeline_example.py` - Usage examples
- `livnium/integration/README.md` - Detailed integration docs
- `livnium/kernel/constraints.py` - Constraint query system
- `livnium/domains/document/` - Document workflow domain
- `livnium/integration/` - Integration APIs


# Domain Implementation Template

This document defines the contract for implementing new Livnium domains. Use `domains/document/` as the reference implementation.

## Required Components

### 1. Encoder (`encoder.py`)
**Purpose**: Convert raw problem input → state vectors

**Must provide:**
- `encode_*()` methods for domain-specific objects
- `generate_constraints()` using `kernel.physics.*`
- `build_initial_state()` with noise injection

**Contract:**
```python
class DomainEncoder(nn.Module):
    def encode_X(self, x: DomainObject) -> torch.Tensor:
        """Convert domain object to vector [dim]"""
        pass
    
    def generate_constraints(self, state, context) -> Dict:
        """Use kernel.physics to compute alignment/divergence/tension"""
        from livnium.kernel.physics import alignment, divergence, tension
        # ... use these, not custom metrics
        pass
```

### 2. Head (`head.py`)
**Purpose**: Interpret collapsed state → task outputs

**Must provide:**
- Task-specific output layers
- Use `kernel.physics` for feature extraction
- Return calibrated scores [0, 1] or logits

**Contract:**
```python
class DomainHead(nn.Module):
    def forward(self, h_final, context, task="default") -> torch.Tensor:
        """Map final state to task output"""
        # Compute physics features using kernel
        # Feed through task-specific head
        pass
```

### 3. Reconciler (Optional, `reconciler.py`)
**Purpose**: Physics-based reasoning over domain objects

**Must provide:**
- Iterative mutual physics (attraction/repulsion)
- Observable metrics (tension history, clusters)
- Convergence detection

**Contract:**
```python
class DomainReconciler:
    def reconcile(self, objects: List[DomainObject]) -> ReconciliationResult:
        """Run collapse loop until convergence"""
        # 1. Encode objects
        # 2. Compute mutual forces
        # 3. Track tension reduction
        # 4. Return clusters/contradictions
        pass
```

## Required Metrics

Every domain must expose:
- **Alignment**: Cosine similarity to anchors/references
- **Divergence**: `DIVERGENCE_PIVOT - alignment`
- **Tension**: Magnitude of divergence
- **Convergence**: Reduction in global tension over iterations

## Physics Usage Rules

✅ **DO:**
- Import from `livnium.kernel.physics` only
- Use `TorchOps()` instance for all tensor operations
- Track force interactions and state evolution
- Expose observable metrics for debugging

❌ **DON'T:**
- Implement custom similarity metrics
- Bypass kernel physics
- Mutate state during inference (use `allow_create` flags)
- Assume specific batch dimensions

## Integration Checklist

- [ ] Encoder uses `kernel.physics.*` for constraints
- [ ] Head uses `kernel.physics.*` for features
- [ ] Reconciler (if present) tracks tension history
- [ ] Demo script shows measurable outcomes
- [ ] Documentation explains physics interpretation
- [ ] No magic constants (use `engine.config.defaults`)

## Reference Implementation

See `domains/document/`:
- `encoder.py` - Claim/Citation/Document encoding
- `head.py` - Retrieval/Citation/Contradiction outputs
- `reconciler.py` - Contradiction collapse loop
- `CONTRADICTION_COLLAPSE.md` - Theory and results

## Testing Pattern

Minimum viable test:
```python
def test_domain_physics_integration():
    encoder = DomainEncoder()
    h0 = encoder.build_initial_state(...)
    
    # Verify physics integration
    constraints = encoder.generate_constraints(h0, context)
    assert "alignment" in constraints
    assert "divergence" in constraints
    assert "tension" in constraints
```

## Next Steps After Implementation

1. Run domain against `test_no_magic_constants.py`
2. Verify all physics uses `kernel.*`
3. Create demo showing measurable outcomes
4. Document tension reduction and convergence

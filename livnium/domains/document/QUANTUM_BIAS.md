# Quantum Collapse Bias

## Overview
This feature implements **Phase 3 of Domain Maturity**: Hybrid Physics.
It allows the experimental **Quantum Layer** to influence the production **Collapse Engine** via a one-way bias hook.

## Theory: "The Whispering Prior"
Standard Livnium reconciliation uses flat mutual exclusion forces.
With Quantum Bias, we initialize a `QuantumRegister` alongside the document claims.

1. **Entanglement**: We explicitly entangle qubits representing specific claims (e.g., A and B).
2. **Measurement Statistics**: We compute the correlation/covariance of the quantum state.
3. **Bias Force**: This correlation is injected into the collapse loop as an attractive or repulsive "ghost force."

If Claim A and Claim B are entangled in a Bell State $(|00\rangle + |11\rangle)/\sqrt{2}$, they have perfect correlation. The bias force helps pull them into the same basin *even if their semantic vectors are slightly contradictory*.

## One-Way Influence Rule
The influence flows strictly: **Research Stack → Production Stack**.
- The `CollapseEngine` does NOT know about qubits.
- It simply accepts a generic `bias_vector` from the `HybridHook`.
- This ensures the production engine remains clean and mathematically simple.

## Usage

```python
from livnium.domains.document.quantum_bias import QuantumEntanglementBias

# Create bias connecting Claim A and Claim B
bias = QuantumEntanglementBias(
    claim_pairs=[("CLAIM_A", "CLAIM_B")]
)

# Inject into reconciler
reconciler.reconcile(
    claims, 
    hybrid_config=HybridConfig(enabled=True, hook=bias)
)
```

## Scientific Goal
To test if structured quantum priors can accelerate convergence or resolve ambiguities that flat physics cannot.

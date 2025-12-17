"""
Quantum Bias Demo: The Whispering Prior

Demonstrates Phase 3 (Hybrid Physics) by running an A/B test:
1. Classical Run: Standard reconciliation of conflicting claims.
2. Quantum Run: Same claims, but we entangle the conflicting claims in a Bell State.

Hypothesis: The quantum prior should accelerate convergence or force a common basin
despite semantic conflict.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from typing import List

from livnium.domains.document.encoder import DocumentEncoder, Claim
from livnium.domains.document.reconciler import ContradictionReconciler
from livnium.domains.document.quantum_bias import QuantumEntanglementBias
from livnium.engine.hooks.hybrid import HybridConfig

def run_quantum_ab_test():
    print("=" * 60)
    print("Quantum Bias Demo: The Whispering Prior")
    print("=" * 60)
    
    # 1. Setup Data: Two ambiguous claims that could go either way
    claims = [
        Claim(
            claim_id="CLAIM_A",
            text="The deployment window is set for late summer.",
            position=0,
            citations=[]
        ),
        Claim(
            claim_id="CLAIM_B",
            text="Launch tends to happen in the third quarter.",
            position=1,
            citations=[]
        ),
        # Neutral bystander to verify isolation
        Claim(
            claim_id="CLAIM_C",
            text="The weather is nice today.",
            position=2,
            citations=[]
        )
    ]
    
    encoder = DocumentEncoder(dim=64)
    reconciler = ContradictionReconciler(encoder, iterations=20)
    
    print("\nScenario: Two conflicting claims (A & B) + one neutral (C).")
    
    # 2. Run A: Classical (Control)
    print("\n[Run A] Classical Reconciliation (No Quantum Bias)...")
    res_a = reconciler.reconcile(claims)
    
    print(f"  - Final Tension: {res_a.global_tension_history[-1]:.4f}")
    print(f"  - Clusters: {res_a.clusters}")
    
    # 3. Run B: Quantum (Experimental)
    print("\n[Run B] Quantum-Biased Reconciliation...")
    print("  -> Creating Bell Pair (|00> + |11>) between CLAIM_A and CLAIM_B")
    
    q_bias = QuantumEntanglementBias(
        claim_ids=[c.claim_id for c in claims],
        entangled_pairs=[("CLAIM_A", "CLAIM_B")]
    )
    
    config = HybridConfig(
        enabled=True,
        bias_weight=2.0, # Strong bias for demo visibility
        hook=q_bias
    )
    
    res_b = reconciler.reconcile(
        claims, 
        hybrid_config=config
    )
    
    print(f"  - Final Tension: {res_b.global_tension_history[-1]:.4f}")
    print(f"  - Clusters: {res_b.clusters}")
    
    # 4. Analysis
    print("\n" + "=" * 60)
    print("A/B Test Analysis")
    print("=" * 60)
    
    tension_diff = res_a.global_tension_history[-1] - res_b.global_tension_history[-1]
    
    did_merge_classical = len([c for c in res_a.clusters if "CLAIM_A" in c and "CLAIM_B" in c]) > 0
    did_merge_quantum = len([c for c in res_b.clusters if "CLAIM_A" in c and "CLAIM_B" in c]) > 0
    
    print(f"Did A/B merge in Classical? {did_merge_classical}")
    print(f"Did A/B merge in Quantum?   {did_merge_quantum}")
    print(f"Tension Difference (A - B): {tension_diff:.4f}")
    
    if not did_merge_classical and did_merge_quantum:
        print("\n✓ SUCCESS: Quantum Entanglement forced a reconciliation!")
        print("  The 'Whispering Prior' successfully overcame semantic repulsion.")
    elif did_merge_classical:
        print("\n⚠ INCONCLUSIVE: Claims merged even without quantum bias.")
    else:
        print("\n⚠ FAILED: Quantum bias was too weak to overcome repulsion.")

if __name__ == "__main__":
    run_quantum_ab_test()

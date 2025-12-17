"""
Quantum Bias Implementation: The Whispering Prior

Implements the Hybrid Hook interface to inject quantum entanglement
biases into the classical reconciliation loop.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any

from livnium.engine.hooks.hybrid import CollapseBias
from livnium.quantum.core.quantum_register import TrueQuantumRegister
from livnium.quantum.core.quantum_gates import QuantumGates
from livnium.engine.config.defaults import QUANTUM_BIAS_STRENGTH

class QuantumEntanglementBias(CollapseBias):
    """
    Injects entanglement correlations as attractive forces.
    
    If two claims are entangled in a Quantum Register, they exert
    a "ghost force" pulling them toward the same narrative basin,
    even if their semantic vectors are initially divergent.
    """
    
    def __init__(self, 
                 claim_ids: List[str],
                 entangled_pairs: List[Tuple[str, str]]):
        """
        Initialize quantum bias.
        
        Args:
            claim_ids: Ordered list of claim IDs (matches tensor rows)
            entangled_pairs: List of (id1, id2) pairs to entangle
        """
        self.claim_map = {cid: i for i, cid in enumerate(claim_ids)}
        self.num_claims = len(claim_ids)
        self.entangled_pairs = entangled_pairs
        
        # Initialize True Quantum Register (Tensor Product)
        # We assign one qubit per claim (simplified mapping)
        qubit_indices = list(range(self.num_claims))
        self.qr = TrueQuantumRegister(qubit_indices)
        
        # Prepare Entangled States (Bell States)
        self._prepare_entanglement()
        
        # Cache correlations so we don't re-measure every step
        # In a real dynamic system, this might evolve
        self.correlation_matrix = self._measure_correlations()
        
    def _prepare_entanglement(self):
        """Create Bell states (|00> + |11>)/sqrt(2) for paired claims."""
        used_qubits = set()
        
        for cid1, cid2 in self.entangled_pairs:
            if cid1 not in self.claim_map or cid2 not in self.claim_map:
                continue
                
            q1 = self.claim_map[cid1]
            q2 = self.claim_map[cid2]
            
            if q1 in used_qubits or q2 in used_qubits:
                continue # Simplify: disentangle measuring conflicts
                
            # Create Bell Pair: H(q1) -> CNOT(q1, q2)
            self.qr.apply_gate(QuantumGates.hadamard(), q1)
            self.qr.apply_cnot(q1, q2)
            
            used_qubits.add(q1)
            used_qubits.add(q2)
            
    def _measure_correlations(self) -> torch.Tensor:
        """
        Compute pairwise correlations from the quantum state.
        Returns matrix [N, N] where value is alignment strength [0, 1].
        """
        corr = torch.zeros((self.num_claims, self.num_claims))
        
        # This is expensive (O(2^N)), so we do it once during init
        # For each pair, calculate probability of matching outcomes (00 or 11)
        
        full_state = self.qr.state
        dim = 2 ** self.num_claims
        
        for i in range(self.num_claims):
            for j in range(i + 1, self.num_claims):
                prob_match = 0.0
                
                for state_idx in range(dim):
                    amp = abs(full_state[state_idx]) ** 2
                    if amp < 1e-10: continue
                    
                    # Check bits
                    bit_i = (state_idx >> (self.num_claims - 1 - i)) & 1
                    bit_j = (state_idx >> (self.num_claims - 1 - j)) & 1
                    
                    if bit_i == bit_j:
                        prob_match += amp
                
                # Normalize to shift: 0.5 = neutral, 1.0 = perfect correlation
                # We map [0.5, 1.0] -> [0.0, 1.0] for bias strength
                bias_strength = max(0.0, (prob_match - 0.5) * 2.0)
                
                corr[i, j] = bias_strength
                corr[j, i] = bias_strength
                
        return corr

    def bias_for_state(self, h: torch.Tensor, step_idx: int) -> torch.Tensor:
        """
        Compute bias force based on cached quantum correlations.
        
        Args:
            h: Current claim states [N, dim]
            step_idx: Iteration step
        
        Returns:
            Bias force vector [N, dim]
        """
        # Calculate attraction based on quantum correlation
        # Each claim is pulled toward its entangled partners
        
        # bias[i] = sum(correlation[i,j] * vector[j])
        # This effectively mixes in the partner's vector
        
        bias = torch.matmul(self.correlation_matrix, h)
        
        # Scale by global strength
        bias = bias * QUANTUM_BIAS_STRENGTH
        
        return bias

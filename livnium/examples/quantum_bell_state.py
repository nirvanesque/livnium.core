"""
Quantum Bell State Demo

Demonstrates real quantum entanglement using Livnium's quantum layer.
Creates a Bell state (|00⟩ + |11⟩)/√2 and verifies perfect correlation.
"""

import sys
sys.path.insert(0, '.')

from livnium.quantum import QuantumRegister, QuantumGates

def create_bell_state():
    """Create Bell state: H on qubit 0, then CNOT(0, 1)"""
    qr = QuantumRegister([0, 1])
    
    # Apply Hadamard to qubit 0: |0⟩ → (|0⟩ + |1⟩)/√2
    qr.apply_gate(QuantumGates.hadamard(), 0)
    
    # Apply CNOT with control=0, target=1
    qr.apply_cnot(0, 1)
    
    return qr

def demonstrate_entanglement():
    """Demonstrate Bell state entanglement"""
    print("=" * 60)
    print("Quantum Bell State Demonstration")
    print("=" * 60)
    print()
    print("Creating Bell state: (|00⟩ + |11⟩)/√2")
    print("  Step 1: H|0⟩ → (|0⟩ + |1⟩)/√2")
    print("  Step 2: CNOT → (|00⟩ + |11⟩)/√2")
    print()
    
    # Collect measurement results
    results_00 = 0
    results_11 = 0
    results_01 = 0
    results_10 = 0
    
    num_trials = 100
    
    for _ in range(num_trials):
        qr = create_bell_state()
        
        # Measure both qubits
        m0 = qr.measure_qubit(0)
        m1 = qr.measure_qubit(1)
        
        if m0 == 0 and m1 == 0:
            results_00 += 1
        elif m0 == 1 and m1 == 1:
            results_11 += 1
        elif m0 == 0 and m1 == 1:
            results_01 += 1
        elif m0 == 1 and m1 == 0:
            results_10 += 1
    
    print(f"Measurement Results (from {num_trials} trials):")
    print(f"  |00⟩: {results_00:3d} ({results_00/num_trials*100:.1f}%)")
    print(f"  |11⟩: {results_11:3d} ({results_11/num_trials*100:.1f}%)")
    print(f"  |01⟩: {results_01:3d} ({results_01/num_trials*100:.1f}%) ← should be 0")
    print(f"  |10⟩: {results_10:3d} ({results_10/num_trials*100:.1f}%) ← should be 0")
    print()
    
    # Verify perfect correlation
    correlation_perfect = (results_01 == 0 and results_10 == 0)
    distribution_balanced = abs(results_00 - results_11) < 20
    
    print("Verification:")
    if correlation_perfect:
        print("  ✓ Perfect correlation: No |01⟩ or |10⟩ outcomes")
    else:
        print("  ✗ Correlation broken: Found mismatched outcomes")
    
    if distribution_balanced:
        print("  ✓ Balanced distribution: ~50% |00⟩, ~50% |11⟩")
    else:
        print("  ✗ Unbalanced distribution")
    
    print()
    if correlation_perfect and distribution_balanced:
        print("✓ Bell state entanglement verified!")
    else:
        print("⚠ Unexpected results")
    
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_entanglement()

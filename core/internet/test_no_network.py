"""
Test: Entangled Basins WITHOUT Network Communication

This demonstrates the core concept of Idea A: two machines can achieve
correlation WITHOUT any communication during the run. They only need:
- Same seed (shared beforehand)
- Same input (shared beforehand)

No network needed - just compare results after both process independently.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import time
from core.internet.entangled_basins import (
    initialize_shared_system,
    process_to_basin,
    verify_correlation,
    BasinSignatureGenerator,
    CorrelationVerifier
)


def simulate_machine_a(seed: int, input_text: str, output_file: str):
    """
    Simulate Machine A processing independently.
    Saves result to file (simulating "Machine A's result").
    """
    print("=" * 70)
    print("MACHINE A: Processing Independently")
    print("=" * 70)
    print()
    
    print(f"Seed: {seed}")
    print(f"Input: '{input_text}'")
    print()
    
    print("Processing...")
    start_time = time.time()
    
    system = initialize_shared_system(seed)
    signature = process_to_basin(system, input_text, max_steps=50)
    hash_str = BasinSignatureGenerator.compute_basin_hash(signature)
    
    elapsed = time.time() - start_time
    
    print(f"✅ Processing complete ({elapsed:.2f}s)")
    print(f"  Basin signature: {len(signature)} cells")
    print(f"  Basin hash: {hash_str}")
    print()
    
    # Save result
    result = {
        'machine': 'A',
        'seed': seed,
        'input': input_text,
        'signature_length': len(signature),
        'signature_hash': hash_str,
        'signature': signature,  # Full signature for comparison
        'processing_time': elapsed
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"Result saved to: {output_file}")
    print()
    
    return result


def simulate_machine_b(seed: int, input_text: str, output_file: str):
    """
    Simulate Machine B processing independently.
    Saves result to file (simulating "Machine B's result").
    """
    print("=" * 70)
    print("MACHINE B: Processing Independently")
    print("=" * 70)
    print()
    
    print(f"Seed: {seed}")
    print(f"Input: '{input_text}'")
    print()
    
    print("Processing...")
    start_time = time.time()
    
    system = initialize_shared_system(seed)
    signature = process_to_basin(system, input_text, max_steps=50)
    hash_str = BasinSignatureGenerator.compute_basin_hash(signature)
    
    elapsed = time.time() - start_time
    
    print(f"✅ Processing complete ({elapsed:.2f}s)")
    print(f"  Basin signature: {len(signature)} cells")
    print(f"  Basin hash: {hash_str}")
    print()
    
    # Save result
    result = {
        'machine': 'B',
        'seed': seed,
        'input': input_text,
        'signature_length': len(signature),
        'signature_hash': hash_str,
        'signature': signature,  # Full signature for comparison
        'processing_time': elapsed
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"Result saved to: {output_file}")
    print()
    
    return result


def compare_results(file_a: str, file_b: str):
    """
    Compare results from two machines (no network needed).
    This simulates comparing results after both machines process independently.
    """
    print("=" * 70)
    print("COMPARISON: No Network Communication Needed!")
    print("=" * 70)
    print()
    
    # Load results
    with open(file_a, 'r') as f:
        result_a = json.load(f)
    
    with open(file_b, 'r') as f:
        result_b = json.load(f)
    
    print("Machine A Results:")
    print(f"  Seed: {result_a['seed']}")
    print(f"  Input: '{result_a['input']}'")
    print(f"  Signature length: {result_a['signature_length']}")
    print(f"  Hash: {result_a['signature_hash']}")
    print()
    
    print("Machine B Results:")
    print(f"  Seed: {result_b['seed']}")
    print(f"  Input: '{result_b['input']}'")
    print(f"  Signature length: {result_b['signature_length']}")
    print(f"  Hash: {result_b['signature_hash']}")
    print()
    
    # Convert signatures back to tuples for comparison
    sig_a = tuple(tuple(x) if isinstance(x, list) else x for x in result_a['signature'])
    sig_b = tuple(tuple(x) if isinstance(x, list) else x for x in result_b['signature'])
    
    # Verify correlation
    result = CorrelationVerifier.verify_correlation(sig_a, sig_b)
    
    print("Correlation Check:")
    print(f"  Hash match: {result_a['signature_hash'] == result_b['signature_hash']}")
    print(f"  Correlated: {result.correlated}")
    print(f"  Match type: {result.match_details['match_type']}")
    print()
    
    if result.correlated:
        print("✅ SUCCESS: Both machines reached the SAME basin!")
        print()
        print("Key Insight:")
        print("  - NO network communication during processing")
        print("  - Only shared seed and input (beforehand)")
        print("  - Both process independently")
        print("  - Results match perfectly!")
        print()
        print("This is the 'spooky action at a distance' effect:")
        print("  Apparent non-local correlation from shared structure!")
    else:
        print("⚠️  Basins differ (may need more evolution steps)")
    
    print()


def test_no_network_correlation():
    """
    Full test: Two machines process independently, then compare.
    No network communication needed!
    """
    print("\n" + "=" * 70)
    print("TEST: Entangled Basins WITHOUT Network")
    print("=" * 70)
    print("\nThis demonstrates that Idea A works WITHOUT any network")
    print("communication. Both machines just need:")
    print("  - Same seed (shared beforehand)")
    print("  - Same input (shared beforehand)")
    print("\nThey process independently, then we compare results.")
    print()
    
    seed = 42
    input_text = "hello world"
    
    file_a = "/tmp/machine_a_result.json"
    file_b = "/tmp/machine_b_result.json"
    
    # Machine A processes
    result_a = simulate_machine_a(seed, input_text, file_a)
    
    # Small delay to simulate different machines
    time.sleep(0.5)
    
    # Machine B processes (independently, no communication)
    result_b = simulate_machine_b(seed, input_text, file_b)
    
    # Compare results (simulating checking after both finish)
    compare_results(file_a, file_b)
    
    return result_a, result_b


def test_multiple_inputs_no_network():
    """Test multiple inputs to show consistent correlation."""
    print("\n" + "=" * 70)
    print("TEST: Multiple Inputs (No Network)")
    print("=" * 70)
    print()
    
    seed = 42
    inputs = ["hello", "world", "quantum", "geometry"]
    
    results = []
    for input_text in inputs:
        # Machine A
        system_a = initialize_shared_system(seed)
        sig_a = process_to_basin(system_a, input_text, max_steps=50)
        hash_a = BasinSignatureGenerator.compute_basin_hash(sig_a)
        
        # Machine B (independent processing)
        system_b = initialize_shared_system(seed)
        sig_b = process_to_basin(system_b, input_text, max_steps=50)
        hash_b = BasinSignatureGenerator.compute_basin_hash(sig_b)
        
        # Verify (no network - just compare)
        result = CorrelationVerifier.verify_correlation(sig_a, sig_b)
        results.append((input_text, result.correlated, hash_a == hash_b))
        
        status = "✅" if result.correlated else "❌"
        print(f"{status} '{input_text}': "
              f"Correlated={result.correlated}, "
              f"Hash match={hash_a == hash_b}")
    
    print()
    correlated_count = sum(1 for _, corr, _ in results if corr)
    print(f"Summary: {correlated_count}/{len(results)} inputs showed correlation")
    print("  (All processed independently, no network communication!)")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("ENTANGLED BASINS - NO NETWORK TEST")
    print("=" * 70)
    print("\nThis proves that Idea A works WITHOUT network communication.")
    print("Both machines just need the same seed and input beforehand.")
    print()
    
    # Main test
    test_no_network_correlation()
    
    # Multiple inputs test
    test_multiple_inputs_no_network()
    
    print("=" * 70)
    print("✅ Test complete!")
    print("=" * 70)
    print("\nKey Takeaway:")
    print("  Idea A demonstrates correlation WITHOUT network communication.")
    print("  The 'entanglement' comes from shared structure (seed + input),")
    print("  not from communication during processing.")
    print()


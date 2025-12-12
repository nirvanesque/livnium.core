"""
AES-32 Pairwise Search (Level 2 Recursive Geometry)

Problem: Single-byte coordinate descent gets stuck in local minima because 
         AES MixColumns entangles adjacent bytes.

Solution: Search 2 bytes (16 bits) simultaneously to break the entanglement.

Complexity: 
- 1-Byte Search: 2^8 = 256 checks
- 2-Byte Search: 2^16 = 65,536 checks (Trivial for AES-32)
"""

import time
import sys
import os
import itertools
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    from experiments.crypto.aes32_simple import AES32
except ImportError:
    from aes32_simple import AES32


def compute_tension(cipher, plaintext, key, target_ciphertext):
    try:
        computed = cipher.encrypt(plaintext, key)
        hamming = sum(bin(c1 ^ c2).count('1') for c1, c2 in zip(computed, target_ciphertext))
        return hamming / (len(target_ciphertext) * 8)
    except:
        return 1.0


def pairwise_search(plaintext, target_ciphertext):
    cipher = AES32()
    
    # Start with a blank key
    current_key = bytearray(4)
    best_tension = compute_tension(cipher, plaintext, bytes(current_key), target_ciphertext)
    
    print("="*60)
    print("Layer 0 (Level 2): Pairwise Geometric Search")
    print("="*60)
    print(f"Target: {target_ciphertext.hex()}")
    print("-" * 60)

    start_time = time.time()
    
    # Define the pairs to search. 
    # (0,1) are mixed directly in simplified AES, so they are most critical.
    # Search all 6 pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    pairs = list(itertools.combinations(range(4), 2))
    
    # Also try starting from a better initial key (from smart_pattern result)
    # This helps if we're already close
    if best_tension > 0.1:
        # Try a few random starting points
        for _ in range(3):
            test_start = bytearray(os.urandom(4))
            t = compute_tension(cipher, plaintext, bytes(test_start), target_ciphertext)
            if t < best_tension:
                current_key = test_start
                best_tension = t
                print(f"  Better starting point found: {current_key.hex()} (tension: {t:.4f})")
    
    # We loop through pairs. Usually one pass is enough for AES-32.
    # Increase cycles to allow more refinement
    for cycle in range(5):
        print(f"\n[Cycle {cycle+1}] refining 16-bit pairwise correlations...")
        improved = False
        
        for i, j in pairs:
            # print(f"  Checking pair ({i}, {j})...")
            
            # We will search 2^16 = 65536 possibilities for this pair
            # keeping the other 2 bytes fixed.
            
            local_best_pair = (current_key[i], current_key[j])
            local_min_tension = best_tension
            
            # Optimization: Don't re-check if tension is already 0
            if best_tension == 0.0: break
            
            # Brute force the pair (i, j)
            # We construct the loop to avoid recreating the bytearray every time for speed
            test_key = bytearray(current_key)
            
            # Iterate 0..65535
            for val_i in range(256):
                test_key[i] = val_i
                for val_j in range(256):
                    test_key[j] = val_j
                    
                    # Fast encryption check
                    # (In a real attack, we would optimize the encrypt function to only 
                    # re-compute affected parts, but AES-32 is fast enough)
                    try:
                        computed = cipher.encrypt(plaintext, bytes(test_key))
                        
                        # Manual hamming calc for speed
                        diff = 0
                        for k in range(4):
                            diff += bin(computed[k] ^ target_ciphertext[k]).count('1')
                        
                        t = diff / 32.0
                        
                        if t < local_min_tension:
                            local_min_tension = t
                            local_best_pair = (val_i, val_j)
                            
                            # Live update
                            if t < best_tension:
                                best_tension = t
                                current_key[i] = val_i
                                current_key[j] = val_j
                                improved = True
                                print(f"  ★ Improved! Tension: {t:.4f} | Key: {current_key.hex()}")
                        
                        if t == 0.0:
                            current_key[i] = val_i
                            current_key[j] = val_j
                            best_tension = 0.0
                            break
                            
                    except:
                        continue
                if best_tension == 0.0: break
            if best_tension == 0.0: break
            
        if best_tension == 0.0: break
        if not improved:
            print("  (Stable - No pairwise improvements possible)")
            break

    # Final refinement: If we're very close (tension < 0.1), do exhaustive local search
    if best_tension > 0.0 and best_tension < 0.1:
        print(f"\n[Final Refinement] Exhaustive search around best key (tension: {best_tension:.4f})...")
        print(f"  Best key so far: {current_key.hex()}")
        
        # Search all keys within ±16 of each byte (33 values per byte)
        # Total: 33^4 = 1,185,921 combinations (still manageable, ~10-15 seconds)
        best_refined_key = bytearray(current_key)
        best_refined_tension = best_tension
        checked = 0
        
        for byte0_offset in range(-16, 17):
            val0 = (current_key[0] + byte0_offset) % 256
            for byte1_offset in range(-16, 17):
                val1 = (current_key[1] + byte1_offset) % 256
                for byte2_offset in range(-16, 17):
                    val2 = (current_key[2] + byte2_offset) % 256
                    for byte3_offset in range(-16, 17):
                        val3 = (current_key[3] + byte3_offset) % 256
                        test_key = bytearray([val0, val1, val2, val3])
                        
                        t = compute_tension(cipher, plaintext, bytes(test_key), target_ciphertext)
                        checked += 1
                        
                        if t < best_refined_tension:
                            best_refined_tension = t
                            best_refined_key = bytearray(test_key)
                            print(f"  ★ Refined to: {best_refined_key.hex()} (tension: {best_refined_tension:.4f})")
                            
                            if t == 0.0:
                                current_key = best_refined_key
                                best_tension = 0.0
                                break
                    if best_tension == 0.0:
                        break
                if best_tension == 0.0:
                    break
            if best_tension == 0.0:
                break
        
        if best_refined_tension < best_tension:
            current_key = best_refined_key
            best_tension = best_refined_tension
        print(f"  Checked {checked:,} keys in refinement")
        
        if best_refined_tension < best_tension:
            current_key = best_refined_key
            best_tension = best_refined_tension
            print(f"  ★ Refined to: {current_key.hex()} (tension: {best_tension:.4f})")

    elapsed = time.time() - start_time
    print("="*60)
    
    if best_tension == 0.0:
        print(f"SUCCESS! Key Found: {current_key.hex()}")
        print(f"Time: {elapsed:.4f}s")
        # Relevant to your [2025-11-11] note:
        # We can log this 'success' event here for the reward system later.
    else:
        print(f"Best Key: {current_key.hex()} (Tension: {best_tension:.4f})")
        print(f"Time: {elapsed:.4f}s")

    return bytes(current_key)


if __name__ == "__main__":
    cipher = AES32()
    true_key = b'\x12\x34\x56\x78'
    plaintext = b'\x9a\xbc\xde\xf0'
    target_ciphertext = cipher.encrypt(plaintext, true_key)
    
    print(f"True Key (Hidden): {true_key.hex()}")
    print()
    
    pairwise_search(plaintext, target_ciphertext)


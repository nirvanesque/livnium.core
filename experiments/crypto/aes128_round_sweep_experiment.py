"""
AES-128 Round Sweep Experiment: Measuring Diffusion Phase Transition

This experiment sweeps over different round counts (2, 3, 4, 5, ...) and measures:
- Success rate of key recovery
- Final tension values
- Time to convergence

The goal is to find the "phase transition" where the gradient landscape dies
and the geometric structure becomes non-navigable.
"""

import time
import sys
import os
import itertools
import random
from typing import Tuple, List


# Make repo root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


# -------------------------------------------
# 1. Cipher factory: get AES cipher by rounds
# -------------------------------------------


def get_aes_cipher_for_rounds(num_rounds: int):
    """
    Returns an AES-128 cipher object with `num_rounds` rounds.

    You have options:
      - Implement AES128_2Round, AES128_3Round, AES128_4Round, ...
        in experiments/crypto/, each with an encrypt(plaintext, key) method.
      - Or modify this to wrap a generic AES128 class that takes `num_rounds`.

    For now, we try to import per-round classes.
    """
    module_name = f"experiments.crypto.aes128_{num_rounds}round"
    class_name = f"AES128_{num_rounds}Round"

    try:
        mod = __import__(module_name, fromlist=[class_name])
        cls = getattr(mod, class_name)
        return cls()
    except Exception as e:
        print(f"[ERROR] Could not load {class_name} from {module_name}: {e}")
        raise


# -------------------------------------------
# 2. Tension + pairwise code (same as breaker)
# -------------------------------------------


def compute_triangulated_tension(cipher, key: bytes, constraints) -> float:
    """
    Tension = (# of wrong bits across all constraints) / (128 * num_constraints)

    constraints: list of (p1, p2, expected_delta)
    """
    total_errors = 0
    max_errors = 0
    try:
        for (p1, p2, expected_delta) in constraints:
            c1 = cipher.encrypt(p1, key)
            c2 = cipher.encrypt(p2, key)
            actual_delta = bytes(a ^ b for a, b in zip(c1, c2))

            # Count bit errors between expected_delta and actual_delta
            errors = sum(bin(a ^ b).count("1") for a, b in zip(actual_delta, expected_delta))
            total_errors += errors
            max_errors += 128  # 128 bits per block

        return total_errors / max_errors if max_errors > 0 else 1.0
    except Exception:
        # If anything blows up, treat as max tension
        return 1.0


def quantum_pairwise_solve(cipher,
                           current_key: bytearray,
                           pair_indices: Tuple[int, int],
                           constraints):
    """
    Simulated 'quantum' pairwise search (brute-forcing 256x256 for a given pair).

    Returns:
        (best_pair_val, min_tension)
    where best_pair_val = (v1, v2).
    """
    idx1, idx2 = pair_indices
    best_pair_val = (current_key[idx1], current_key[idx2])
    min_tension = 1.0

    test_key = bytearray(current_key)

    for v1 in range(256):
        test_key[idx1] = v1
        for v2 in range(256):
            test_key[idx2] = v2

            t = compute_triangulated_tension(cipher, bytes(test_key), constraints)

            if t < min_tension:
                min_tension = t
                best_pair_val = (v1, v2)

                # Perfect match – we can bail early
                if t == 0.0:
                    break
        if min_tension == 0.0:
            break

    return best_pair_val, min_tension


# -------------------------------------------
# 3. Single run for a given #rounds + key
# -------------------------------------------


def run_single_break_attempt(num_rounds: int,
                             true_key: bytes,
                             num_constraints: int = 3,
                             cycles: int = 3):
    """
    Run ONE attempt of the pairwise breaker for a cipher with `num_rounds` rounds.
    Returns:
        success (bool),
        final_tension (float),
        elapsed_time (float)
    """
    cipher = get_aes_cipher_for_rounds(num_rounds)

    # 1. Build triangulation constraints for THIS key
    constraints = []
    base_p = b"\x00" * 16

    # Simple pattern: walk some bytes/bits deterministically
    bit_choices = [
        (0, 0),
        (0, 1),
        (1, 0),
        (5, 0),
        (10, 0),
        (15, 0),
    ]
    bit_choices = bit_choices[:num_constraints]

    for (byte_idx, bit_idx) in bit_choices:
        p1 = base_p
        p2_arr = bytearray(base_p)
        p2_arr[byte_idx] ^= (1 << bit_idx)
        p2 = bytes(p2_arr)

        c1 = cipher.encrypt(p1, true_key)
        c2 = cipher.encrypt(p2, true_key)
        delta = bytes(a ^ b for a, b in zip(c1, c2))

        constraints.append((p1, p2, delta))

    # 2. Initialize key at zero
    current_key = bytearray(16)
    best_tension = compute_triangulated_tension(cipher, bytes(current_key), constraints)

    # Chunks: AES-like row/column structure
    chunks = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
        (12, 13, 14, 15),
        (0, 4, 8, 12),
        (1, 5, 9, 13),
        (2, 6, 10, 14),
        (3, 7, 11, 15),
    ]

    start_time = time.time()

    for cycle in range(cycles):
        improved = False

        for chunk in chunks:
            for i, j in itertools.combinations(chunk, 2):
                if best_tension == 0.0:
                    break

                (v1, v2), t = quantum_pairwise_solve(cipher, current_key, (i, j), constraints)

                if t < best_tension:
                    best_tension = t
                    current_key[i], current_key[j] = v1, v2
                    improved = True

                if best_tension == 0.0:
                    break
            if best_tension == 0.0:
                break

        # Phase kick if stuck
        if best_tension != 0.0 and not improved:
            k = random.randint(0, 15)
            current_key[k] = random.randint(0, 255)

        if best_tension == 0.0:
            break

    elapsed = time.time() - start_time

    # Check if we truly got the key (not just zero tension by accident)
    success = (best_tension == 0.0 and bytes(current_key) == true_key)

    return success, best_tension, elapsed


# -------------------------------------------
# 4. Sweep over rounds & aggregate stats
# -------------------------------------------


def sweep_rounds(round_list: List[int],
                 trials_per_round: int = 5,
                 num_constraints: int = 3):
    print("=" * 70)
    print("AES-128 Tension Landscape vs Rounds (Livnium-style experiment)")
    print("=" * 70)
    print(f"Trials per round: {trials_per_round}, constraints per trial: {num_constraints}")
    print("")

    results = []

    for num_rounds in round_list:
        print(f"\n--- Testing {num_rounds} rounds ---")
        successes = 0
        tensions = []
        times = []

        for t_idx in range(trials_per_round):
            # Random key per trial
            true_key = os.urandom(16)

            success, final_tension, elapsed = run_single_break_attempt(
                num_rounds=num_rounds,
                true_key=true_key,
                num_constraints=num_constraints,
                cycles=3,
            )

            successes += int(success)
            tensions.append(final_tension)
            times.append(elapsed)

            print(f"  Trial {t_idx+1}: "
                  f"success={success}, final_tension={final_tension:.4f}, time={elapsed:.3f}s")

        avg_tension = sum(tensions) / len(tensions)
        min_tension = min(tensions)
        max_tension = max(tensions)
        avg_time = sum(times) / len(times)

        results.append((num_rounds, successes, avg_tension, min_tension, max_tension, avg_time))

    print("\n" + "=" * 70)
    print("SUMMARY: Tension Landscape vs Rounds")
    print("=" * 70)
    print("Rounds | Successes/Trials | AvgTension | MinTension | MaxTension | AvgTime(s)")
    for (r, succ, avg_t, min_t, max_t, avg_time) in results:
        print(f"{r:6d} | {succ:9d}/{trials_per_round:<6d} | "
              f"{avg_t:10.4f} | {min_t:10.4f} | {max_t:10.4f} | {avg_time:10.3f}")


if __name__ == "__main__":
    # Choose which rounds to test.
    # You can start with [2, 3, 4, 5] then extend up.
    rounds_to_test = [2, 3, 4, 5]

    sweep_rounds(rounds_to_test, trials_per_round=5, num_constraints=3)


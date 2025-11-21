"""
AES-128 Round Sweep Experiment: Using Full Livnium Architecture

This version uses:
- Recursive Geometry Engine: Subdivide key space into chunks
- Quantum Layer: Superposition exploration of key bits
- Tension Fields: Encode constraints as geometric tension
- Moksha Engine: Convergence detection

This measures the phase transition where diffusion destroys navigable structure.
"""

import time
import sys
import os
import itertools
import random
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

# Make repo root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig
from core.recursive import RecursiveGeometryEngine, MokshaEngine, ConvergenceState
from core.quantum import QuantumLattice, QuantumCell
# Import constraint encoder (handle space in path)
import importlib
encoder_module = importlib.import_module('core.Universal Encoder.constraint_encoder')
ConstraintEncoder = encoder_module.ConstraintEncoder
TensionField = encoder_module.TensionField


# -------------------------------------------
# 1. Cipher factory (same as before)
# -------------------------------------------


def get_aes_cipher_for_rounds(num_rounds: int):
    """Returns an AES-128 cipher object with `num_rounds` rounds."""
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
# 2. Encode AES key recovery as geometry
# -------------------------------------------


def encode_key_to_geometry(key: bytes, system: LivniumCoreSystem) -> Dict[Tuple[int, int, int], int]:
    """
    Encode 128-bit key into geometry cells.
    
    Maps each key byte to a cell coordinate.
    For 16-byte key, we use a 4×4×1 slice of the lattice.
    """
    key_cells = {}
    lattice_size = system.lattice_size
    
    # Map each byte to a cell (use first 16 cells in a pattern)
    for i, byte_val in enumerate(key):
        # Use a 4×4 pattern
        x = i % 4
        y = (i // 4) % 4
        z = 0
        
        # Ensure coordinates are valid
        if x < lattice_size and y < lattice_size and z < lattice_size:
            coords = (x, y, z)
            key_cells[coords] = byte_val
    
    return key_cells


def decode_geometry_to_key(key_cells: Dict[Tuple[int, int, int], int]) -> bytes:
    """Decode geometry cells back to key bytes."""
    # Sort by coordinates to get consistent ordering
    sorted_coords = sorted(key_cells.keys())
    key_bytes = bytearray(16)
    
    for i, coords in enumerate(sorted_coords[:16]):  # Take first 16
        key_bytes[i] = key_cells[coords]
    
    return bytes(key_bytes)


def create_tension_fields_for_constraints(
    cipher,
    constraints: List[Tuple[bytes, bytes, bytes]],
    system: LivniumCoreSystem,
    encoder: ConstraintEncoder
) -> List[TensionField]:
    """
    Create tension fields from triangulated constraints.
    
    Each constraint (p1, p2, expected_delta) creates a tension field
    that measures how far the actual delta is from expected.
    """
    tension_fields = []
    
    for idx, (p1, p2, expected_delta) in enumerate(constraints):
        # Create tension field for this constraint
        def compute_tension_for_constraint(system: LivniumCoreSystem) -> float:
            """Compute tension: how wrong is the key for this constraint."""
            # Decode key from geometry
            key_cells = {}
            for coords, cell in system.lattice.items():
                # Use symbolic weight as key byte value (mod 256)
                key_cells[coords] = int(cell.symbolic_weight) % 256
            
            key = decode_geometry_to_key(key_cells)
            
            # Encrypt with current key
            try:
                c1 = cipher.encrypt(p1, key)
                c2 = cipher.encrypt(p2, key)
                actual_delta = bytes(a ^ b for a, b in zip(c1, c2))
                
                # Count bit errors
                errors = sum(bin(a ^ b).count("1") for a, b in zip(actual_delta, expected_delta))
                return errors / 128.0  # Normalize to [0, 1]
            except Exception:
                return 1.0  # Max tension on error
        
        # Get coordinates involved (all key cells)
        involved_coords = list(system.lattice.keys())[:16]  # First 16 cells
        
        field = TensionField(
            constraint_id=f"aes_constraint_{idx}",
            involved_coords=involved_coords,
            compute_tension=compute_tension_for_constraint,
            compute_curvature=lambda s: 1.0 / (1.0 + compute_tension_for_constraint(s)),
            description=f"Triangulated constraint {idx}"
        )
        
        tension_fields.append(field)
        encoder.tension_fields.append(field)
    
    return tension_fields


# -------------------------------------------
# 3. Recursive key search with quantum
# -------------------------------------------


def recursive_quantum_key_search(
    cipher,
    constraints: List[Tuple[bytes, bytes, bytes]],
    max_iterations: int = 100
) -> Tuple[bool, float, float]:
    """
    Use recursive geometry + quantum to search for key.
    
    Returns:
        (success, final_tension, elapsed_time)
    """
    # Initialize Livnium system
    config = LivniumCoreConfig(
        lattice_size=5,  # 5×5×5 = 125 cells, enough for 16-byte key (must be odd)
        enable_quantum=True,
        enable_superposition=True,  # Required for entanglement
        enable_quantum_gates=True,
        enable_entanglement=True,
        enable_measurement=True,
        enable_geometry_quantum_coupling=True
    )
    
    system = LivniumCoreSystem(config)
    
    # Initialize recursive geometry engine
    recursive_engine = RecursiveGeometryEngine(
        base_geometry=system,
        max_depth=2  # 2 levels: full key + byte chunks
    )
    
    # Initialize quantum lattice
    quantum_lattice = QuantumLattice(system)
    
    # Initialize constraint encoder
    constraint_encoder = ConstraintEncoder(system)
    
    # Create tension fields
    tension_fields = create_tension_fields_for_constraints(
        cipher, constraints, system, constraint_encoder
    )
    
    # Initialize moksha engine
    moksha = recursive_engine.moksha
    
    # Initialize key in superposition (all bytes in |+⟩ state)
    # This means each byte is in equal superposition of all 256 values
    # We need to recreate quantum cells with 256 levels (qudits for bytes)
    for coords in list(system.lattice.keys())[:16]:
        if coords in quantum_lattice.quantum_cells:
            # Remove old cell
            del quantum_lattice.quantum_cells[coords]
        
        # Create new cell with 256 levels (one for each byte value)
        cell = QuantumCell(
            coordinates=coords,
            amplitudes=np.ones(256, dtype=complex) / np.sqrt(256),
            num_levels=256
        )
        quantum_lattice.quantum_cells[coords] = cell
    
    start_time = time.time()
    best_tension = 1.0
    best_key = None
    
    # Search loop
    for iteration in range(max_iterations):
        # Check convergence
        convergence = moksha.check_convergence()
        if convergence == ConvergenceState.MOKSHA:
            # System converged - extract final state
            break
        
        # Compute total tension
        total_tension = sum(field.get_tension(system) for field in tension_fields)
        total_tension /= len(tension_fields)  # Average
        
        if total_tension < best_tension:
            best_tension = total_tension
            # Extract best key from geometry
            key_cells = {}
            for i, (coords, cell) in enumerate(list(system.lattice.items())[:16]):
                key_cells[coords] = int(cell.symbolic_weight) % 256
            best_key = decode_geometry_to_key(key_cells)
        
        # If we found perfect match, stop
        if best_tension == 0.0:
            break
        
        # Quantum exploration: measure and update
        # Measure quantum cells to get candidate key bytes
        measured_key = bytearray(16)
        for i, coords in enumerate(list(system.lattice.keys())[:16]):
            if coords in quantum_lattice.quantum_cells:
                cell = quantum_lattice.quantum_cells[coords]
                # Measure to get a byte value
                measured_val = cell.measure()
                measured_key[i] = measured_val % 256
        
        # Update geometry based on measurement
        for i, coords in enumerate(list(system.lattice.keys())[:16]):
            if coords in system.lattice:
                cell = system.lattice[coords]
                # Update symbolic weight toward measured value
                current_sw = cell.symbolic_weight
                target_sw = float(measured_key[i])
                # Gradient descent: move toward measured value
                cell.symbolic_weight = 0.9 * current_sw + 0.1 * target_sw
        
        # Update quantum states based on tension (amplify low-tension states)
        for coords in list(system.lattice.keys())[:16]:
            if coords in quantum_lattice.quantum_cells:
                cell = quantum_lattice.quantum_cells[coords]
                # Amplify amplitudes for values that reduce tension
                # This is a simplified version - in full implementation,
                # we'd use quantum amplitude amplification
                current_byte = int(system.lattice[coords].symbolic_weight) % 256
                # Boost amplitude for current best value
                if cell.num_levels > current_byte:
                    boost = 1.1  # Small boost
                    cell.amplitudes[current_byte] *= boost
                    cell.normalize()
        
        # Note: LivniumCoreSystem doesn't have an update() method
        # We update cells directly through symbolic_weight modifications above
    
    elapsed = time.time() - start_time
    
    # Check if we found the key
    success = (best_tension == 0.0 and best_key is not None)
    
    return success, best_tension, elapsed


# -------------------------------------------
# 4. Single run (using Livnium)
# -------------------------------------------


def run_single_break_attempt_livnium(
    num_rounds: int,
    true_key: bytes,
    num_constraints: int = 3,
    max_iterations: int = 100
) -> Tuple[bool, float, float]:
    """
    Run ONE attempt using full Livnium architecture.
    
    Returns:
        (success, final_tension, elapsed_time)
    """
    cipher = get_aes_cipher_for_rounds(num_rounds)
    
    # Build triangulation constraints
    constraints = []
    base_p = b"\x00" * 16
    
    bit_choices = [
        (0, 0), (0, 1), (1, 0), (5, 0), (10, 0), (15, 0),
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
    
    # Use recursive quantum search
    success, final_tension, elapsed = recursive_quantum_key_search(
        cipher, constraints, max_iterations=max_iterations
    )
    
    return success, final_tension, elapsed


# -------------------------------------------
# 5. Sweep over rounds (same interface)
# -------------------------------------------


def sweep_rounds_livnium(
    round_list: List[int],
    trials_per_round: int = 5,
    num_constraints: int = 3,
    max_iterations: int = 100
):
    """Sweep over rounds using Livnium architecture."""
    print("=" * 70)
    print("AES-128 Tension Landscape vs Rounds (Livnium Architecture)")
    print("=" * 70)
    print(f"Using: Recursive Geometry + Quantum Layer + Tension Fields")
    print(f"Trials per round: {trials_per_round}, constraints: {num_constraints}")
    print("")
    
    results = []
    
    for num_rounds in round_list:
        print(f"\n--- Testing {num_rounds} rounds ---")
        successes = 0
        tensions = []
        times = []
        
        for t_idx in range(trials_per_round):
            true_key = os.urandom(16)
            
            success, final_tension, elapsed = run_single_break_attempt_livnium(
                num_rounds=num_rounds,
                true_key=true_key,
                num_constraints=num_constraints,
                max_iterations=max_iterations
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
    print("SUMMARY: Tension Landscape vs Rounds (Livnium)")
    print("=" * 70)
    print("Rounds | Successes/Trials | AvgTension | MinTension | MaxTension | AvgTime(s)")
    for (r, succ, avg_t, min_t, max_t, avg_time) in results:
        print(f"{r:6d} | {succ:9d}/{trials_per_round:<6d} | "
              f"{avg_t:10.4f} | {min_t:10.4f} | {max_t:10.4f} | {avg_time:10.3f}")


if __name__ == "__main__":
    rounds_to_test = [2, 3, 4, 5]
    sweep_rounds_livnium(rounds_to_test, trials_per_round=3, num_constraints=3, max_iterations=50)


"""
Recursive Moksha Demo: The Search for the Fixed Point

This example demonstrates the Recursive Geometry Engine and the Moksha Engine working together.
It initializes a recursive geometric system and evolves it until it reaches a "Moksha" state—
a fixed point where the system becomes invariant under rotations and recursion.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from livnium.classical.livnium_core_system import LivniumCoreSystem, RotationAxis
from livnium.classical.config import LivniumCoreConfig
from livnium.recursive import RecursiveGeometryEngine, MokshaEngine

def run_moksha_search():
    print("============================================================")
    print("Recursive Moksha Demo: The Search for the Fixed Point")
    print("============================================================")
    
    # 1. Initialize Base Geometry (The "Root" Universe)
    print("\n[1] Initializing Root Universe (Layer 0)...")
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_symbolic_weight=True,
        enable_class_structure=True,
        enable_90_degree_rotations=True
    )
    base_geometry = LivniumCoreSystem(config)
    print(f"    - Lattice Size: {base_geometry.lattice_size}x{base_geometry.lattice_size}x{base_geometry.lattice_size}")
    print(f"    - Total Symbolic Weight: {base_geometry.get_total_symbolic_weight()}")
    
    # 2. Initialize Recursive Engine (Fractal Architecture)
    print("\n[2] initializing Recursive Geometry Engine...")
    recursive_engine = RecursiveGeometryEngine(
        base_geometry=base_geometry,
        max_depth=2  # Universe inside Universe inside Universe
    )
    print(f"    - Max Depth: {recursive_engine.max_depth}")
    print(f"    - Hierarchy Levels: {len(recursive_engine.levels)}")
    print(f"    - Total Capacity (Cells): {recursive_engine.get_total_capacity()}")
    
    # 3. Initialize Moksha Engine (Convergence Detector)
    print("\n[3] Initializing Moksha Engine...")
    moksha_engine = MokshaEngine(recursive_engine)
    print("    - Goal: Find a state invariant under all operations")
    
    # 4. Evolution Loop (The Search for Truth)
    print("\n[4] Beginning Evolution Loop...")
    max_steps = 20
    
    for step in range(max_steps):
        # A. Apply a "disturbance" (Rotation) to see if system settles
        # We rotate random layers to simulate dynamical evolution
        axis = RotationAxis.X if step % 2 == 0 else RotationAxis.Y
        level = step % (recursive_engine.max_depth + 1)
        
        print(f"\n    Step {step+1}: Rotating Level {level} about {axis.name}...")
        recursive_engine.apply_recursive_rotation(level, axis, 1)
        
        # B. Check for Moksha (Fixed Point)
        convergence = moksha_engine.check_convergence()
        score = moksha_engine.get_convergence_score()
        
        print(f"      -> Convergence State: {convergence.name}")
        print(f"      -> Stability Score: {score:.4f}")
        
        if moksha_engine.should_terminate():
            print("\n    🌟 MOKSHA REACHED! System has found the Fixed Point.")
            break
            
        # Simulate time passage
        time.sleep(0.1)
    
    # 5. Final Report
    print("\n============================================================")
    print("Final Truth Report")
    print("============================================================")
    
    truth = moksha_engine.export_final_truth()
    
    if truth['moksha']:
        print("Status: LIBERATED (Moksha Reached)")
        print(f"Fixed Points Found: {len(truth['fixed_points'])}")
        # In a real run, this state would be the "Answer" to the query
    else:
        print("Status: STILL SEARCHING")
        print("System did not converge within step limit.")
        
    print("============================================================")

if __name__ == "__main__":
    run_moksha_search()

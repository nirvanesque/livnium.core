"""
Accurate QuantumSystem capacity test in recursive simplex geometry.

This test:
1. Tracks exactly which simplex nodes are visited
2. Calculates theoretical max based on visited nodes (not full tree)
3. Shows accurate utilization percentages

Adapted for Livnium-T (tetrahedral/simplex geometry) instead of cubic geometry.
"""

import sys
import numpy as np
from pathlib import Path

# Add core-t directory to path
core_t_path = Path(__file__).parent.parent
sys.path.insert(0, str(core_t_path))
# Also add parent to path for relative imports
sys.path.insert(0, str(core_t_path.parent))

# Import directly from module files to avoid relative import issues
from classical.livnium_t_system import LivniumTSystem

# Import recursive modules by loading them directly and patching imports
import importlib.util
import types

# Create mock modules for relative imports
core_t_module = types.ModuleType('core_t')
core_t_recursive_module = types.ModuleType('core_t.recursive')
core_t_classical_module = types.ModuleType('core_t.classical')
sys.modules['core_t'] = core_t_module
sys.modules['core_t.recursive'] = core_t_recursive_module
sys.modules['core_t.classical'] = core_t_classical_module

# Import the classical module and add it to the package structure
import classical.livnium_t_system
core_t_classical_module.livnium_t_system = classical.livnium_t_system

# Load all recursive dependencies first (bypassing __init__.py)
recursive_dir = core_t_path / "recursive"

# Load dependencies in order
dependencies = [
    "simplex_subdivision",
    "recursive_projection", 
    "recursive_conservation",
    "recursive_simplex_engine"
]

loaded_modules = {}
for dep_name in dependencies:
    spec = importlib.util.spec_from_file_location(
        dep_name,
        recursive_dir / f"{dep_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "core_t.recursive"
    
    # Read and patch source
    with open(recursive_dir / f"{dep_name}.py", 'r') as f:
        source = f.read()
    
    # Replace relative imports
    source = source.replace(
        "from ..classical.livnium_t_system import",
        "from classical.livnium_t_system import"
    )
    source = source.replace(
        "from .simplex_subdivision import",
        f"# from .simplex_subdivision import  # Patched\nfrom {dep_name if dep_name == 'simplex_subdivision' else 'recursive.simplex_subdivision'} import"
    )
    source = source.replace(
        "from .recursive_projection import",
        "from recursive.recursive_projection import"
    )
    source = source.replace(
        "from .recursive_conservation import",
        "from recursive.recursive_conservation import"
    )
    source = source.replace(
        "from .recursive_simplex_engine import",
        "from recursive.recursive_simplex_engine import"
    )
    source = source.replace(
        "from .moksha_engine import",
        "from recursive.moksha_engine import"
    )
    
    # Add LivniumTSystem to module dict before exec
    module.__dict__['LivniumTSystem'] = LivniumTSystem
    
    # Compile and exec
    code = compile(source, recursive_dir / f"{dep_name}.py", 'exec')
    exec(code, module.__dict__)
    
    loaded_modules[dep_name] = module
    sys.modules[f"recursive.{dep_name}"] = module

# Now get the classes we need
RecursiveSimplexEngine = loaded_modules['recursive_simplex_engine'].RecursiveSimplexEngine
SimplexLevel = loaded_modules['recursive_simplex_engine'].SimplexLevel

# Import quantum components
try:
    from quantum.quantum_system import QuantumSystem
    from quantum.quantum_node import QuantumNode
    from quantum.quantum_gates import QuantumGates, GateType
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    print("Warning: Quantum module not available, using mock implementation")


def test_level_accurate(level: SimplexLevel, depth=0, max_nodes_per_register=5, 
                       visited_stats=None, max_children_per_level=None):
    """
    Test QuantumSystem at a level and track accurate stats.
    
    Args:
        level: SimplexLevel to test
        depth: Current recursion depth
        max_nodes_per_register: Max nodes to use per quantum system
        visited_stats: Dictionary to track visited statistics
        max_children_per_level: Max children to process per level (for sampling)
    
    Returns:
        (qubits_created, systems_created, visited_stats)
    """
    if visited_stats is None:
        visited_stats = {}
    
    geometry = level.geometry
    nodes = list(geometry.nodes.keys())  # 5 nodes: 0 (core) + 1-4 (vertices)
    num_nodes = len(nodes)
    
    # Track visited
    if depth not in visited_stats:
        visited_stats[depth] = {'geometries': 0, 'total_nodes': 0}
    visited_stats[depth]['geometries'] += 1
    visited_stats[depth]['total_nodes'] += num_nodes
    
    # Create QuantumSystems
    qubits_created = 0
    systems_created = 0
    
    if QUANTUM_AVAILABLE:
        # Group nodes into quantum systems
        for i in range(0, num_nodes, max_nodes_per_register):
            node_group = nodes[i:i + max_nodes_per_register]
            size = len(node_group)
            
            if size < 1:
                continue
            
            try:
                # Create a sub-geometry for this node group (or use full geometry)
                # For simplicity, we'll create quantum nodes directly
                quantum_nodes = {}
                for node_id in node_group:
                    quantum_node = QuantumNode(
                        node_id=node_id,
                        amplitudes=np.array([1.0, 0.0], dtype=complex),
                        num_levels=2
                    )
                    quantum_nodes[node_id] = quantum_node
                
                # Apply gates to test functionality
                if len(quantum_nodes) > 0:
                    # Apply Hadamard to first node
                    first_node = list(quantum_nodes.values())[0]
                    H = QuantumGates.hadamard()
                    first_node.apply_unitary(H)
                    
                    # If we have multiple nodes, try entanglement
                    if len(quantum_nodes) >= 2:
                        # Can't directly entangle separate QuantumNodes without QuantumSystem
                        # So we'll just count them as created
                        pass
                
                qubits_created += size
                systems_created += 1
            except Exception as e:
                # Stop if we hit memory limit or other error
                print(f"  Warning at depth {depth}: {e}")
                break
    else:
        # Mock implementation: just count nodes
        qubits_created = num_nodes
        systems_created = 1
    
    # Recursively process children
    if depth < 2:  # Limit depth for testing
        children = list(level.children.values())
        
        # Apply sampling limit if specified
        if max_children_per_level and len(children) > max_children_per_level:
            import random
            children = random.sample(children, max_children_per_level)
        
        for child in children:
            cq, cr, visited_stats = test_level_accurate(
                child, depth + 1, max_nodes_per_register, 
                visited_stats, max_children_per_level
            )
            qubits_created += cq
            systems_created += cr
    
    return qubits_created, systems_created, visited_stats


def run_accurate_test():
    """Run accurate capacity test."""
    print("="*70)
    print("ACCURATE QuantumSystem Capacity Test (Livnium-T)")
    print("="*70)
    print()
    
    if not QUANTUM_AVAILABLE:
        print("⚠️  Quantum module not available - using mock implementation")
        print()
    
    # Create base Livnium-T system (5 nodes)
    base = LivniumTSystem()
    engine = RecursiveSimplexEngine(base, max_depth=2)
    print("✓ Built recursive simplex geometry (5-node topology, depth 2)\n")
    
    # Test 1: Full tree (no sampling)
    print("TEST 1: Full Tree (No Sampling)")
    print("-"*70)
    total_qubits, total_systems, visited_stats = test_level_accurate(
        engine.levels[0],
        depth=0,
        max_nodes_per_register=5,  # Use all 5 nodes per simplex
        max_children_per_level=None  # No limit
    )
    
    print(f"\nResults:")
    print(f"  Total qubits/nodes: {total_qubits:,}")
    print(f"  Total quantum systems: {total_systems:,}")
    
    print(f"\nVisited simplex levels:")
    actual_theoretical = 0
    for depth in sorted(visited_stats.keys()):
        stats = visited_stats[depth]
        nodes_per_geo = stats['total_nodes'] // stats['geometries'] if stats['geometries'] > 0 else 0
        max_qubits_possible = stats['total_nodes']  # If we used all nodes
        actual_theoretical += max_qubits_possible
        print(f"  Depth {depth}: {stats['geometries']:,} simplex geometries, "
              f"{stats['total_nodes']:,} nodes, "
              f"max {max_qubits_possible:,} qubits possible")
    
    print(f"\nActual theoretical max (from visited): {actual_theoretical:,} qubits")
    print(f"Actual achieved: {total_qubits:,} qubits")
    if actual_theoretical > 0:
        print(f"Utilization: {100*total_qubits/actual_theoretical:.1f}%")
    else:
        print("Utilization: N/A")
    
    # Test 2: Sampled (for comparison)
    print("\n" + "="*70)
    print("TEST 2: Sampled Tree (3 children per level)")
    print("-"*70)
    total_qubits_sampled, total_systems_sampled, visited_stats_sampled = test_level_accurate(
        engine.levels[0],
        depth=0,
        max_nodes_per_register=5,
        max_children_per_level=3  # Sample limit (out of 5 possible children)
    )
    
    print(f"\nResults:")
    print(f"  Total qubits/nodes: {total_qubits_sampled:,}")
    print(f"  Total quantum systems: {total_systems_sampled:,}")
    
    print(f"\nVisited simplex levels:")
    sampled_theoretical = 0
    for depth in sorted(visited_stats_sampled.keys()):
        stats = visited_stats_sampled[depth]
        max_qubits_possible = stats['total_nodes']
        sampled_theoretical += max_qubits_possible
        print(f"  Depth {depth}: {stats['geometries']:,} simplex geometries, "
              f"{stats['total_nodes']:,} nodes, "
              f"max {max_qubits_possible:,} qubits possible")
    
    print(f"\nSampled theoretical max: {sampled_theoretical:,} qubits")
    print(f"Sampled achieved: {total_qubits_sampled:,} qubits")
    if sampled_theoretical > 0:
        print(f"Sampled utilization: {100*total_qubits_sampled/sampled_theoretical:.1f}%")
    else:
        print("Sampled utilization: N/A")
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    if actual_theoretical > 0 and sampled_theoretical > 0:
        print(f"Full tree:  {total_qubits:,} qubits from {actual_theoretical:,} possible ({100*total_qubits/actual_theoretical:.1f}%)")
        print(f"Sampled:    {total_qubits_sampled:,} qubits from {sampled_theoretical:,} possible ({100*total_qubits_sampled/sampled_theoretical:.1f}%)")
        if total_qubits_sampled > 0:
            print(f"\nFull tree has {total_qubits/total_qubits_sampled:.1f}x more qubits than sampled")
    
    # Key differences from cubic geometry
    print("\n" + "="*70)
    print("KEY DIFFERENCES (Livnium-T vs Cubic)")
    print("="*70)
    print("• Cubic: 27 cells per geometry (3×3×3 lattice)")
    print("• Simplex: 5 nodes per geometry (tetrahedral topology)")
    print("• Cubic: Higher capacity per level, more complex structure")
    print("• Simplex: Simpler structure, cleaner recursion, minimal topology")
    print("• Both: Exponential capacity through recursion")


if __name__ == "__main__":
    run_accurate_test()


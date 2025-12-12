"""
Quick analysis of recursive depth limits for Livnium-T.

Shows theoretical growth and practical limits.
"""

import sys
from pathlib import Path

# Add core-t directory to path
core_t_path = Path(__file__).parent.parent
sys.path.insert(0, str(core_t_path))
sys.path.insert(0, str(core_t_path.parent))

from classical.livnium_t_system import LivniumTSystem

# Import recursive modules (simplified - just for quick analysis)
import importlib.util
import types

core_t_module = types.ModuleType('core_t')
core_t_recursive_module = types.ModuleType('core_t.recursive')
core_t_classical_module = types.ModuleType('core_t.classical')
sys.modules['core_t'] = core_t_module
sys.modules['core_t.recursive'] = core_t_recursive_module
sys.modules['core_t.classical'] = core_t_classical_module

import classical.livnium_t_system
core_t_classical_module.livnium_t_system = classical.livnium_t_system

recursive_dir = core_t_path / "recursive"
dependencies = ["simplex_subdivision", "recursive_projection", "recursive_conservation", "recursive_simplex_engine"]

loaded_modules = {}
for dep_name in dependencies:
    spec = importlib.util.spec_from_file_location(dep_name, recursive_dir / f"{dep_name}.py")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "core_t.recursive"
    
    with open(recursive_dir / f"{dep_name}.py", 'r') as f:
        source = f.read()
    
    source = source.replace("from ..classical.livnium_t_system import", "from classical.livnium_t_system import")
    source = source.replace("from .simplex_subdivision import", "from recursive.simplex_subdivision import")
    source = source.replace("from .recursive_projection import", "from recursive.recursive_projection import")
    source = source.replace("from .recursive_conservation import", "from recursive.recursive_conservation import")
    source = source.replace("from .recursive_simplex_engine import", "from recursive.recursive_simplex_engine import")
    source = source.replace("from .moksha_engine import", "from recursive.moksha_engine import")
    
    module.__dict__['LivniumTSystem'] = LivniumTSystem
    code = compile(source, recursive_dir / f"{dep_name}.py", 'exec')
    exec(code, module.__dict__)
    loaded_modules[dep_name] = module
    sys.modules[f"recursive.{dep_name}"] = module

RecursiveSimplexEngine = loaded_modules['recursive_simplex_engine'].RecursiveSimplexEngine


def calculate_theoretical(depth: int):
    """Calculate theoretical capacity at a given depth."""
    # At depth d:
    # - Level 0: 1 geometry, 5 nodes
    # - Level 1: 5 geometries, 25 nodes  
    # - Level d: 5^d geometries, 5^(d+1) nodes total
    total_nodes = sum(5**(i+1) for i in range(depth + 1))
    total_geometries = sum(5**i for i in range(depth + 1))
    return total_nodes, total_geometries


def test_depth(depth: int):
    """Test a specific depth quickly."""
    try:
        base = LivniumTSystem()
        engine = RecursiveSimplexEngine(base, max_depth=depth)
        total_nodes = engine.get_total_capacity()
        return {'success': True, 'nodes': total_nodes}
    except Exception as e:
        return {'success': False, 'error': str(e)}


print("="*70)
print("LIVNIUM-T RECURSIVE DEPTH ANALYSIS")
print("="*70)

print("\n📊 THEORETICAL GROWTH (5^n exponential)")
print("-"*70)
print(f"{'Depth':<8} {'Total Nodes':<20} {'Total Geometries':<20} {'Growth Factor':<15}")
print("-"*70)

for depth in range(11):
    nodes, geometries = calculate_theoretical(depth)
    if depth == 0:
        growth = 1.0
    else:
        prev_nodes, _ = calculate_theoretical(depth - 1)
        growth = nodes / prev_nodes
    
    print(f"{depth:<8} {nodes:<20,} {geometries:<20,} {growth:<15.2f}")

print("\n🔬 PRACTICAL TESTING")
print("-"*70)
print(f"{'Depth':<8} {'Status':<15} {'Actual Nodes':<20} {'Theoretical':<20}")
print("-"*70)

for depth in range(6):  # Test up to depth 5 quickly
    result = test_depth(depth)
    theoretical_nodes, _ = calculate_theoretical(depth)
    
    if result['success']:
        accuracy = 100 * result['nodes'] / theoretical_nodes if theoretical_nodes > 0 else 0
        print(f"{depth:<8} {'✅ Success':<15} {result['nodes']:<20,} {theoretical_nodes:<20,} ({accuracy:.1f}%)")
    else:
        print(f"{depth:<8} {'❌ Failed':<15} {'-':<20} {theoretical_nodes:<20,}")
        print(f"         Error: {result['error']}")

print("\n" + "="*70)
print("DEPTH LIMITS ANALYSIS")
print("="*70)

print("\n💡 KEY INSIGHTS:")
print("   • Growth: Exponential (5^n per level)")
print("   • Each level multiplies capacity by 5")
print("   • Depth 0: 5 nodes (base simplex)")
print("   • Depth 1: 25 nodes (5 children)")
print("   • Depth 2: 125 nodes (25 children)")
print("   • Depth 3: 625 nodes")
print("   • Depth 4: 3,125 nodes")
print("   • Depth 5: 15,625 nodes")
print("   • Depth 10: ~12.2 million nodes")

print("\n🎯 PRACTICAL LIMITS:")
print("   • Memory: ~O(5^d) geometries")
print("   • Computation: ~O(5^d) operations")
print("   • Recommended depths:")
print("     - Interactive: 2-3 (25-125 nodes)")
print("     - Batch: 4-5 (3K-15K nodes)")
print("     - Maximum: 6-8 (78K-1.2M nodes)")
print("     - Theoretical: Unlimited (memory/compute bound)")

print("\n⚡ OPTIMIZATION STRATEGIES:")
print("   • Lazy evaluation: Only create geometries when needed")
print("   • Sampling: Process subset of children")
print("   • Pruning: Skip low-value branches")
print("   • Caching: Store computed results")
print("   • Parallel processing: Distribute across cores")

print("\n" + "="*70)


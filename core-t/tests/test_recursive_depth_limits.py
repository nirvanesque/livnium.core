"""
Test recursive depth limits for Livnium-T.

Explores:
1. Theoretical depth limits (exponential growth)
2. Practical depth limits (memory/computation)
3. Capacity at different depths
4. Memory usage patterns
"""

import sys
import numpy as np
from pathlib import Path
import tracemalloc
import time

# Add core-t directory to path
core_t_path = Path(__file__).parent.parent
sys.path.insert(0, str(core_t_path))
sys.path.insert(0, str(core_t_path.parent))

from classical.livnium_t_system import LivniumTSystem

# Import recursive modules (using same patching approach as other tests)
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
    
    with open(recursive_dir / f"{dep_name}.py", 'r') as f:
        source = f.read()
    
    source = source.replace(
        "from ..classical.livnium_t_system import",
        "from classical.livnium_t_system import"
    )
    source = source.replace(
        "from .simplex_subdivision import",
        "from recursive.simplex_subdivision import"
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
    
    module.__dict__['LivniumTSystem'] = LivniumTSystem
    code = compile(source, recursive_dir / f"{dep_name}.py", 'exec')
    exec(code, module.__dict__)
    
    loaded_modules[dep_name] = module
    sys.modules[f"recursive.{dep_name}"] = module

RecursiveSimplexEngine = loaded_modules['recursive_simplex_engine'].RecursiveSimplexEngine
SimplexLevel = loaded_modules['recursive_simplex_engine'].SimplexLevel


def analyze_depth(depth: int, track_memory: bool = True):
    """
    Analyze a specific depth.
    
    Returns:
        dict with stats: nodes, geometries, memory_mb, time_sec
    """
    print(f"\n{'='*70}")
    print(f"Testing Depth {depth}")
    print(f"{'='*70}")
    
    if track_memory:
        tracemalloc.start()
    
    start_time = time.time()
    
    try:
        base = LivniumTSystem()
        engine = RecursiveSimplexEngine(base, max_depth=depth)
        
        build_time = time.time() - start_time
        
        # Calculate statistics
        total_nodes = engine.get_total_capacity()
        
        # Count geometries per level
        geometries_per_level = {}
        for level_id, level in engine.levels.items():
            geometries_per_level[level_id] = 1  # At least the base level
        
        # Count all geometries recursively
        def count_geometries(level, depth):
            count = 1  # This geometry
            if depth not in geometries_per_level:
                geometries_per_level[depth] = 0
            geometries_per_level[depth] += 1
            
            for child in level.children.values():
                count += count_geometries(child, depth + 1)
            return count
        
        total_geometries = count_geometries(engine.levels[0], 0)
        
        # Memory usage
        memory_mb = 0
        if track_memory:
            current, peak = tracemalloc.get_traced_memory()
            memory_mb = peak / (1024 * 1024)
            tracemalloc.stop()
        
        total_time = time.time() - start_time
        
        # Theoretical calculations
        # At depth d, we have:
        # - Level 0: 1 geometry, 5 nodes
        # - Level 1: 5 geometries, 25 nodes
        # - Level d: 5^d geometries, 5^(d+1) nodes total
        theoretical_nodes = sum(5**(i+1) for i in range(depth + 1))
        theoretical_geometries = sum(5**i for i in range(depth + 1))
        
        print(f"\nResults:")
        print(f"  Total nodes: {total_nodes:,}")
        print(f"  Total geometries: {total_geometries:,}")
        print(f"  Build time: {build_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")
        if track_memory:
            print(f"  Peak memory: {memory_mb:.2f} MB")
        
        print(f"\nTheoretical (5^n growth):")
        print(f"  Expected nodes: {theoretical_nodes:,}")
        print(f"  Expected geometries: {theoretical_geometries:,}")
        print(f"  Node accuracy: {100*total_nodes/theoretical_nodes:.1f}%")
        print(f"  Geometry accuracy: {100*total_geometries/theoretical_geometries:.1f}%")
        
        print(f"\nGeometries per level:")
        for level_id in sorted(geometries_per_level.keys()):
            expected = 5**level_id if level_id > 0 else 1
            actual = geometries_per_level[level_id]
            print(f"  Level {level_id}: {actual:,} geometries (expected: {expected:,})")
        
        return {
            'depth': depth,
            'total_nodes': total_nodes,
            'total_geometries': total_geometries,
            'memory_mb': memory_mb,
            'time_sec': total_time,
            'theoretical_nodes': theoretical_nodes,
            'theoretical_geometries': theoretical_geometries,
        }
        
    except MemoryError as e:
        if track_memory:
            tracemalloc.stop()
        print(f"\n❌ MemoryError at depth {depth}: {e}")
        return {
            'depth': depth,
            'error': 'MemoryError',
            'memory_mb': tracemalloc.get_traced_memory()[1] / (1024 * 1024) if track_memory else 0,
        }
    except Exception as e:
        if track_memory:
            tracemalloc.stop()
        print(f"\n❌ Error at depth {depth}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'depth': depth,
            'error': str(e),
        }


def find_max_depth(max_test_depth: int = 10, memory_limit_mb: float = 1000):
    """
    Find the maximum practical depth.
    
    Args:
        max_test_depth: Maximum depth to test
        memory_limit_mb: Stop if memory exceeds this (MB)
    """
    print("="*70)
    print("FINDING MAXIMUM DEPTH LIMITS")
    print("="*70)
    
    results = []
    
    for depth in range(max_test_depth + 1):
        result = analyze_depth(depth, track_memory=True)
        results.append(result)
        
        # Check if we hit limits
        if 'error' in result:
            print(f"\n⚠️  Stopped at depth {depth} due to error")
            break
        
        if result.get('memory_mb', 0) > memory_limit_mb:
            print(f"\n⚠️  Stopped at depth {depth} due to memory limit ({memory_limit_mb} MB)")
            break
        
        # Check if we're getting too slow (> 10 seconds)
        if result.get('time_sec', 0) > 10:
            print(f"\n⚠️  Depth {depth} took {result['time_sec']:.1f}s - consider stopping")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nDepth Analysis:")
    print(f"{'Depth':<8} {'Nodes':<15} {'Geometries':<15} {'Memory (MB)':<15} {'Time (s)':<12}")
    print("-" * 70)
    
    for r in results:
        if 'error' in r:
            print(f"{r['depth']:<8} {'ERROR':<15} {'ERROR':<15} {r.get('memory_mb', 0):<15.2f} {'-':<12}")
        else:
            print(f"{r['depth']:<8} {r['total_nodes']:<15,} {r['total_geometries']:<15,} "
                  f"{r.get('memory_mb', 0):<15.2f} {r.get('time_sec', 0):<12.3f}")
    
    # Growth analysis
    print(f"\nGrowth Analysis:")
    print(f"{'Depth':<8} {'Nodes':<15} {'Growth Factor':<15} {'Geometries':<15} {'Growth Factor':<15}")
    print("-" * 70)
    
    for i, r in enumerate(results):
        if 'error' in r:
            continue
        
        if i == 0:
            node_growth = 1.0
            geo_growth = 1.0
        else:
            prev = results[i-1]
            if 'error' not in prev:
                node_growth = r['total_nodes'] / prev['total_nodes'] if prev['total_nodes'] > 0 else 0
                geo_growth = r['total_geometries'] / prev['total_geometries'] if prev['total_geometries'] > 0 else 0
            else:
                node_growth = 0
                geo_growth = 0
        
        print(f"{r['depth']:<8} {r['total_nodes']:<15,} {node_growth:<15.2f} "
              f"{r['total_geometries']:<15,} {geo_growth:<15.2f}")
    
    # Theoretical limits
    print(f"\n{'='*70}")
    print("THEORETICAL LIMITS")
    print("="*70)
    print(f"\nExponential Growth Formula: 5^n")
    print(f"\nAt different depths:")
    for d in [5, 10, 15, 20, 25, 30]:
        nodes = sum(5**(i+1) for i in range(d + 1))
        geometries = sum(5**i for i in range(d + 1))
        print(f"  Depth {d:2d}: {nodes:>20,} nodes, {geometries:>15,} geometries")
    
    print(f"\n{'='*70}")
    print("PRACTICAL RECOMMENDATIONS")
    print("="*70)
    
    successful_depths = [r for r in results if 'error' not in r]
    if successful_depths:
        max_successful = max(r['depth'] for r in successful_depths)
        print(f"\n✅ Maximum successful depth tested: {max_successful}")
        print(f"   - Nodes: {successful_depths[-1]['total_nodes']:,}")
        print(f"   - Geometries: {successful_depths[-1]['total_geometries']:,}")
        print(f"   - Memory: {successful_depths[-1].get('memory_mb', 0):.2f} MB")
        print(f"   - Time: {successful_depths[-1].get('time_sec', 0):.3f}s")
    
    print(f"\n💡 Recommendations:")
    print(f"   - For interactive use: depth 3-5 (155-3,905 nodes)")
    print(f"   - For batch processing: depth 6-8 (19,530-488,280 nodes)")
    print(f"   - For maximum capacity: depth 10+ (millions of nodes)")
    print(f"   - Memory scales roughly as: ~{successful_depths[-1].get('memory_mb', 1) / max(1, max_successful):.1f} MB per depth level")


if __name__ == "__main__":
    # Test progressively increasing depths
    find_max_depth(max_test_depth=8, memory_limit_mb=2000)


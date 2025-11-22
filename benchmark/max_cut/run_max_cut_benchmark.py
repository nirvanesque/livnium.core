"""
Max-Cut Benchmark: Compare Livnium vs Baseline Solvers

Runs GSET benchmark graphs and compares results against:
- Greedy baseline
- Known literature values (if available)
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmark.max_cut.max_cut_solver_livnium import solve_max_cut_livnium, MaxCutProblem


def solve_with_greedy(problem: MaxCutProblem) -> Dict[str, Any]:
    """
    Solve Max-Cut using greedy algorithm.
    
    Greedy: Start with random partition, then repeatedly move vertices
    to the side that increases the cut.
    
    Returns:
        Dictionary with results
    """
    import random
    
    start_time = time.time()
    
    # Initialize random partition
    partition = {i: random.choice([0, 1]) for i in range(problem.num_vertices)}
    
    # Greedy improvement: move vertices to improve cut
    improved = True
    iterations = 0
    max_iterations = problem.num_vertices * 2
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try moving each vertex
        for vertex_id in range(problem.num_vertices):
            current_side = partition[vertex_id]
            other_side = 1 - current_side
            
            # Compute cut change if we move this vertex
            cut_change = 0
            for neighbor in problem.adjacency[vertex_id]:
                neighbor_side = partition[neighbor]
                if neighbor_side == current_side:
                    # Currently on same side, moving would add edge to cut
                    cut_change += 1
                else:
                    # Currently on different side, moving would remove edge from cut
                    cut_change -= 1
            
            # Move if it improves cut
            if cut_change > 0:
                partition[vertex_id] = other_side
                improved = True
    
    cut_size = problem.compute_cut_size(partition)
    tension = problem.compute_tension(partition)
    elapsed = time.time() - start_time
    
    return {
        'solved': True,
        'cut_size': cut_size,
        'partition': partition,
        'time': elapsed,
        'iterations': iterations,
        'tension': tension,
        'max_possible_cut': problem.num_edges
    }


# Known optimal/best-known values for some GSET graphs
# Source: Various Max-Cut literature
KNOWN_VALUES = {
    'G1': 11624,   # Known optimal
    'G2': 11620,   # Known optimal
    'G3': 11622,   # Known optimal
    'G11': 564,    # Known optimal
    'G12': 556,    # Known optimal
    'G14': 3064,   # Best known
    'G20': 941,    # Best known
    'G21': 931,    # Best known
    'G22': 13359,  # Best known
    'G23': 13344,  # Best known
    'G43': 6660,   # Best known
    'G44': 6650,   # Best known
    'G50': 5880,   # Best known
    'G51': 3848,   # Best known
    'G52': 3848,   # Best known
    'G54': 3364,   # Best known
    'G55': 10294,  # Best known
}


def get_known_value(graph_name: str) -> Optional[int]:
    """Get known optimal/best-known cut size for a graph."""
    return KNOWN_VALUES.get(graph_name)


def run_benchmark(
    graph_files: List[Path],
    max_steps: int = 1000,
    max_time: float = 60.0,
    verbose: bool = False,
    use_recursive: bool = False,
    recursive_depth: int = 2
) -> Dict[str, Any]:
    """
    Run benchmark on a set of graph files.
    
    Returns:
        Dictionary with benchmark results
    """
    results = {
        'livnium': [],
        'greedy': [],
        'summary': {}
    }
    
    print(f"\n{'='*70}")
    print(f"Max-Cut Benchmark: {len(graph_files)} graphs")
    print(f"{'='*70}\n")
    
    for i, graph_file in enumerate(graph_files, 1):
        graph_name = graph_file.stem
        print(f"[{i}/{len(graph_files)}] {graph_name}")
        
        # Load graph
        try:
            problem = MaxCutProblem.from_gset_file(graph_file)
            print(f"  Vertices: {problem.num_vertices}, Edges: {problem.num_edges}")
        except Exception as e:
            print(f"  Error loading: {e}")
            continue
        
        # Get known value if available
        known_value = get_known_value(graph_name)
        if known_value:
            print(f"  Known best: {known_value} edges")
        
        # Solve with Livnium
        print("  Livnium...", end=" ", flush=True)
        livnium_result = solve_max_cut_livnium(
            problem,
            max_steps=max_steps,
            max_time=max_time,
            verbose=verbose,
            use_recursive=use_recursive,
            recursive_depth=recursive_depth
        )
        livnium_result['file'] = graph_name
        livnium_result['num_vertices'] = problem.num_vertices
        livnium_result['num_edges'] = problem.num_edges
        livnium_result['known_value'] = known_value
        if known_value:
            livnium_result['ratio_to_known'] = livnium_result['cut_size'] / known_value
        results['livnium'].append(livnium_result)
        
        if livnium_result['solved']:
            ratio_str = ""
            if known_value:
                ratio = livnium_result['cut_size'] / known_value
                ratio_str = f" ({ratio:.1%} of known)"
            print(f"✓ {livnium_result['cut_size']} edges{ratio_str}, "
                  f"{livnium_result['time']:.3f}s")
        else:
            print("✗ Failed")
        
        # Solve with greedy
        print("  Greedy...", end=" ", flush=True)
        greedy_result = solve_with_greedy(problem)
        greedy_result['file'] = graph_name
        greedy_result['num_vertices'] = problem.num_vertices
        greedy_result['num_edges'] = problem.num_edges
        greedy_result['known_value'] = known_value
        if known_value:
            greedy_result['ratio_to_known'] = greedy_result['cut_size'] / known_value
        results['greedy'].append(greedy_result)
        
        if greedy_result['solved']:
            ratio_str = ""
            if known_value:
                ratio = greedy_result['cut_size'] / known_value
                ratio_str = f" ({ratio:.1%} of known)"
            print(f"✓ {greedy_result['cut_size']} edges{ratio_str}, "
                  f"{greedy_result['time']:.3f}s")
        else:
            print("✗ Failed")
        
        print()
    
    # Compute summary statistics
    livnium_solved = sum(1 for r in results['livnium'] if r.get('solved', False))
    greedy_solved = sum(1 for r in results['greedy'] if r.get('solved', False))
    
    livnium_cuts = [r['cut_size'] for r in results['livnium'] if r.get('solved', False)]
    greedy_cuts = [r['cut_size'] for r in results['greedy'] if r.get('solved', False)]
    
    livnium_times = [r['time'] for r in results['livnium'] if r.get('solved', False)]
    greedy_times = [r['time'] for r in results['greedy'] if r.get('solved', False)]
    
    # Compute ratios to known values
    livnium_ratios = [r['ratio_to_known'] for r in results['livnium'] 
                      if r.get('solved', False) and r.get('ratio_to_known') is not None]
    greedy_ratios = [r['ratio_to_known'] for r in results['greedy'] 
                     if r.get('solved', False) and r.get('ratio_to_known') is not None]
    
    results['summary'] = {
        'total_graphs': len(graph_files),
        'livnium': {
            'solved': livnium_solved,
            'unsolved': len(graph_files) - livnium_solved,
            'avg_cut_size': sum(livnium_cuts) / len(livnium_cuts) if livnium_cuts else None,
            'max_cut_size': max(livnium_cuts) if livnium_cuts else None,
            'min_cut_size': min(livnium_cuts) if livnium_cuts else None,
            'avg_time': sum(livnium_times) / len(livnium_times) if livnium_times else None,
            'avg_ratio_to_known': sum(livnium_ratios) / len(livnium_ratios) if livnium_ratios else None,
        },
        'greedy': {
            'solved': greedy_solved,
            'unsolved': len(graph_files) - greedy_solved,
            'avg_cut_size': sum(greedy_cuts) / len(greedy_cuts) if greedy_cuts else None,
            'max_cut_size': max(greedy_cuts) if greedy_cuts else None,
            'min_cut_size': min(greedy_cuts) if greedy_cuts else None,
            'avg_time': sum(greedy_times) / len(greedy_times) if greedy_times else None,
            'avg_ratio_to_known': sum(greedy_ratios) / len(greedy_ratios) if greedy_ratios else None,
        }
    }
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print benchmark summary."""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    summary = results['summary']
    
    print(f"\nTotal Graphs: {summary['total_graphs']}")
    
    print(f"\n{'Solver':<20} {'Solved':<10} {'Avg Cut':<12} {'Max Cut':<12} {'Avg Time':<12} {'Avg Ratio':<12}")
    print("-" * 70)
    
    liv = summary['livnium']
    avg_cut_liv = f"{liv['avg_cut_size']:.0f}" if liv['avg_cut_size'] is not None else "N/A"
    max_cut_liv = f"{liv['max_cut_size']:.0f}" if liv['max_cut_size'] is not None else "N/A"
    avg_time_liv = f"{liv['avg_time']:.3f}s" if liv['avg_time'] is not None else "N/A"
    avg_ratio_liv = f"{liv['avg_ratio_to_known']:.1%}" if liv['avg_ratio_to_known'] is not None else "N/A"
    
    print(f"{'Livnium':<20} {liv['solved']:<10} {avg_cut_liv:<12} {max_cut_liv:<12} "
          f"{avg_time_liv:<12} {avg_ratio_liv:<12}")
    
    gr = summary['greedy']
    avg_cut_gr = f"{gr['avg_cut_size']:.0f}" if gr['avg_cut_size'] is not None else "N/A"
    max_cut_gr = f"{gr['max_cut_size']:.0f}" if gr['max_cut_size'] is not None else "N/A"
    avg_time_gr = f"{gr['avg_time']:.3f}s" if gr['avg_time'] is not None else "N/A"
    avg_ratio_gr = f"{gr['avg_ratio_to_known']:.1%}" if gr['avg_ratio_to_known'] is not None else "N/A"
    
    print(f"{'Greedy':<20} {gr['solved']:<10} {avg_cut_gr:<12} {max_cut_gr:<12} "
          f"{avg_time_gr:<12} {avg_ratio_gr:<12}")
    
    print("\n" + "="*70)


def save_results(results: Dict[str, Any], output_path: Path):
    """Save benchmark results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Max-Cut Benchmark: Livnium vs Greedy')
    parser.add_argument('--gset-dir', type=str, help='Directory containing GSET graph files')
    parser.add_argument('--max-steps', type=int, default=1000, help='Max search steps for Livnium')
    parser.add_argument('--max-time', type=float, default=60.0, help='Max time per graph (seconds)')
    parser.add_argument('--limit', type=int, help='Limit number of graphs to test')
    parser.add_argument('--output', type=str, default='benchmark/max_cut/max_cut_results.json', 
                       help='Output JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--use-recursive', action='store_true', default=False, 
                       help='Use recursive geometry (default: False)')
    parser.add_argument('--recursive-depth', type=int, default=2, 
                       help='Recursive geometry depth (default: 2)')
    
    args = parser.parse_args()
    
    # Determine graph file directory
    if not args.gset_dir:
        benchmark_dir = Path(__file__).parent
        gset_dir = benchmark_dir / "gset"
    else:
        gset_dir = Path(args.gset_dir)
    
    # Find graph files
    graph_files = sorted(gset_dir.glob("G*"))
    
    if not graph_files:
        print(f"No GSET graph files found in {gset_dir}")
        print("Download graphs first: python benchmark/max_cut/download_gset.py")
        return
    
    if args.limit:
        graph_files = graph_files[:args.limit]
    
    # Run benchmark
    results = run_benchmark(
        graph_files,
        max_steps=args.max_steps,
        max_time=args.max_time,
        verbose=args.verbose,
        use_recursive=args.use_recursive,
        recursive_depth=args.recursive_depth
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, output_path)


if __name__ == '__main__':
    main()


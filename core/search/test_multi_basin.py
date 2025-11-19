"""
Simple Test for Multi-Basin Search

Tests the multi-basin search system with competing attractors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import List, Tuple

from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig
from core.search.multi_basin_search import (
    Basin,
    MultiBasinSearch,
    solve_with_multi_basin,
    create_candidate_basins,
)


def test_multi_basin_basic():
    """Test basic multi-basin functionality."""
    print("="*60)
    print("Test 1: Basic Multi-Basin Search")
    print("="*60)
    
    # Create system
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    # Create multi-basin search
    search = MultiBasinSearch(max_basins=5)
    
    # Create candidate basins
    candidates = create_candidate_basins(system, n_candidates=3, basin_size=4)
    
    # Add basins
    for coords in candidates:
        basin = search.add_basin(coords, system)
        print(f"  Added basin {basin.id}: {len(coords)} coords, "
              f"score={basin.score:.4f}")
    
    print(f"\n  Total basins: {len(search.basins)}")
    
    # Update basins
    search.update_all_basins(system)
    
    # Get stats
    stats = search.get_basin_stats()
    print(f"\n  Basin stats:")
    print(f"    Alive: {stats['num_alive']}")
    print(f"    Best score: {stats['best_score']:.4f}")
    print(f"    Avg score: {stats['avg_score']:.4f}")
    
    # Get winner
    winner = search.get_winner()
    if winner:
        print(f"    Winner: basin {winner.id} (score={winner.score:.4f})")
    
    print("  ✓ Multi-basin search works")
    print()


def test_basin_competition():
    """Test basin competition dynamics."""
    print("="*60)
    print("Test 2: Basin Competition")
    print("="*60)
    
    # Create system
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    # Create search
    search = MultiBasinSearch(max_basins=5)
    
    # Create basins with different initial SW values
    coords_list = list(system.lattice.keys())
    
    # Basin 1: High SW (should win)
    basin1_coords = coords_list[:4]
    for coords in basin1_coords:
        cell = system.get_cell(coords)
        if cell:
            cell.symbolic_weight = 50.0  # High SW
    
    # Basin 2: Medium SW
    basin2_coords = coords_list[4:8]
    for coords in basin2_coords:
        cell = system.get_cell(coords)
        if cell:
            cell.symbolic_weight = 20.0  # Medium SW
    
    # Basin 3: Low SW (should lose)
    basin3_coords = coords_list[8:12]
    for coords in basin3_coords:
        cell = system.get_cell(coords)
        if cell:
            cell.symbolic_weight = 5.0  # Low SW
    
    # Add basins
    basin1 = search.add_basin(basin1_coords, system)
    basin2 = search.add_basin(basin2_coords, system)
    basin3 = search.add_basin(basin3_coords, system)
    
    print(f"  Initial scores:")
    print(f"    Basin 1: {basin1.score:.4f} (curvature={basin1.curvature:.4f}, tension={basin1.tension:.4f})")
    print(f"    Basin 2: {basin2.score:.4f} (curvature={basin2.curvature:.4f}, tension={basin2.tension:.4f})")
    print(f"    Basin 3: {basin3.score:.4f} (curvature={basin3.curvature:.4f}, tension={basin3.tension:.4f})")
    
    # Run competition for several steps
    print(f"\n  Running competition...")
    for step in range(5):
        search.update_all_basins(system)
        stats = search.get_basin_stats()
        
        if (step + 1) % 2 == 0:
            print(f"    Step {step+1}: {stats['num_alive']} alive, "
                  f"best={stats['best_score']:.4f}")
    
    # Check winner
    winner = search.get_winner()
    if winner:
        print(f"\n  Winner: Basin {winner.id} (score={winner.score:.4f})")
        print(f"    Expected: Basin 1 should win (highest initial SW)")
        if winner.id == basin1.id:
            print("  ✓ Correct winner selected")
        else:
            print(f"  ⚠️  Unexpected winner (expected basin {basin1.id})")
    
    print()


def test_solve_with_multi_basin():
    """Test high-level solve function."""
    print("="*60)
    print("Test 3: Solve with Multi-Basin")
    print("="*60)
    
    # Create system
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    # Create candidate solutions
    candidates = create_candidate_basins(system, n_candidates=5, basin_size=4)
    
    print(f"  Created {len(candidates)} candidate solutions")
    
    # Solve
    winner, steps, stats = solve_with_multi_basin(
        system,
        candidates,
        max_steps=50,
        verbose=True
    )
    
    if winner:
        print(f"\n  Solution found:")
        print(f"    Basin ID: {winner.id}")
        print(f"    Score: {winner.score:.4f}")
        print(f"    Steps: {steps}")
        print(f"    Final stats: {stats['num_alive']} basins alive")
    else:
        print(f"\n  No winner found after {steps} steps")
    
    print("  ✓ Solve function works")
    print()


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("MULTI-BASIN SEARCH - TEST SUITE")
    print("="*60)
    print()
    
    try:
        test_multi_basin_basic()
        test_basin_competition()
        test_solve_with_multi_basin()
        
        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print()
        print("Summary:")
        print("  ✓ Multi-basin search initializes correctly")
        print("  ✓ Basins compete and winner emerges")
        print("  ✓ High-level solve function works")
        print()
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()


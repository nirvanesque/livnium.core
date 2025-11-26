"""
Test assertions for multi-basin search.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

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


def test_basin_dataclass():
    """Test Basin dataclass."""
    coords = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    basin = Basin(
        id=1,
        active_coords=coords,
        score=0.5,
        curvature=0.8,
        tension=0.3,
        entropy=0.2
    )
    
    assert basin.id == 1, "Should have correct ID"
    assert basin.active_coords == coords, "Should have correct coords"
    assert basin.score == 0.5, "Should have correct score"
    assert basin.curvature == 0.8, "Should have correct curvature"
    assert basin.tension == 0.3, "Should have correct tension"
    assert basin.entropy == 0.2, "Should have correct entropy"
    assert basin.is_alive == True, "Should be alive by default"
    assert basin.is_winning == False, "Should not be winning by default"
    
    # Test update_score
    basin.update_score()
    expected_score = basin.curvature - basin.tension
    assert abs(basin.score - expected_score) < 1e-6, "Score should be curvature - tension"


def test_multi_basin_search_initialization():
    """Test MultiBasinSearch initialization."""
    search = MultiBasinSearch(max_basins=5)
    
    assert search.max_basins == 5, "Should set max_basins"
    assert len(search.basins) == 0, "Should start with no basins"
    assert search.basin_counter == 0, "Should start at counter 0"


def test_add_basin():
    """Test adding a basin."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    search = MultiBasinSearch(max_basins=5)
    
    coords_list = list(system.lattice.keys())[:4]
    basin = search.add_basin(coords_list, system)
    
    assert isinstance(basin, Basin), "Should return Basin"
    assert basin.id == 0, "Should have ID 0"
    assert basin.active_coords == coords_list, "Should have correct coords"
    assert len(search.basins) == 1, "Should have 1 basin"
    assert search.basin_counter == 1, "Counter should be 1"


def test_update_all_basins():
    """Test updating all basins."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    search = MultiBasinSearch(max_basins=5)
    
    # Add multiple basins
    coords_list = list(system.lattice.keys())
    basin1 = search.add_basin(coords_list[:4], system)
    basin2 = search.add_basin(coords_list[4:8], system)
    
    initial_score1 = basin1.score
    initial_score2 = basin2.score
    
    # Update basins
    search.update_all_basins(system)
    
    # Scores should be updated
    assert basin1.age == 1, "Basin 1 age should increase"
    assert basin2.age == 1, "Basin 2 age should increase"
    
    # One basin should be winning
    alive_basins = [b for b in search.basins if b.is_alive]
    winning_basins = [b for b in alive_basins if b.is_winning]
    assert len(winning_basins) == 1, "Should have exactly one winner"


def test_get_winner():
    """Test getting winner basin."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    search = MultiBasinSearch(max_basins=5)
    
    # No basins
    winner = search.get_winner()
    assert winner is None, "Should return None with no basins"
    
    # Add basins
    coords_list = list(system.lattice.keys())
    search.add_basin(coords_list[:4], system)
    search.add_basin(coords_list[4:8], system)
    
    # Update to identify winner
    search.update_all_basins(system)
    
    winner = search.get_winner()
    assert winner is not None, "Should have a winner"
    assert winner.is_winning == True, "Winner should be marked as winning"


def test_get_best_basin():
    """Test getting best basin."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    search = MultiBasinSearch(max_basins=5)
    
    # No basins
    best = search.get_best_basin()
    assert best is None, "Should return None with no basins"
    
    # Add basins
    coords_list = list(system.lattice.keys())
    basin1 = search.add_basin(coords_list[:4], system)
    basin2 = search.add_basin(coords_list[4:8], system)
    
    # Update to compute scores
    search.update_all_basins(system)
    
    best = search.get_best_basin()
    assert best is not None, "Should have a best basin"
    assert best in search.basins, "Best should be in basins list"


def test_get_basin_stats():
    """Test basin statistics."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    search = MultiBasinSearch(max_basins=5)
    
    # No basins
    stats = search.get_basin_stats()
    assert stats['num_basins'] == 0, "Should have 0 basins"
    assert stats['num_alive'] == 0, "Should have 0 alive"
    
    # Add basins
    coords_list = list(system.lattice.keys())
    search.add_basin(coords_list[:4], system)
    search.add_basin(coords_list[4:8], system)
    
    search.update_all_basins(system)
    stats = search.get_basin_stats()
    
    assert stats['num_basins'] == 2, "Should have 2 basins"
    assert stats['num_alive'] == 2, "Should have 2 alive"
    assert 'best_score' in stats, "Should have best_score"
    assert 'avg_score' in stats, "Should have avg_score"
    assert stats['best_score'] is not None, "Best score should not be None"


def test_basin_competition():
    """Test basin competition dynamics."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    search = MultiBasinSearch(max_basins=5)
    
    coords_list = list(system.lattice.keys())
    
    # Basin 1: High SW (should win)
    basin1_coords = coords_list[:4]
    for coords in basin1_coords:
        cell = system.get_cell(coords)
        if cell:
            cell.symbolic_weight = 50.0  # High SW
    
    # Basin 2: Low SW (should lose)
    basin2_coords = coords_list[4:8]
    for coords in basin2_coords:
        cell = system.get_cell(coords)
        if cell:
            cell.symbolic_weight = 5.0  # Low SW
    
    # Add basins
    basin1 = search.add_basin(basin1_coords, system)
    basin2 = search.add_basin(basin2_coords, system)
    
    # Run competition for several steps
    for step in range(3):
        search.update_all_basins(system)
    
    # Check winner
    winner = search.get_winner()
    assert winner is not None, "Should have a winner"
    
    # Verify basins are still valid
    assert basin1.is_alive or basin2.is_alive, "At least one basin should be alive"


def test_create_candidate_basins():
    """Test creating candidate basins."""
    config = LivniumCoreConfig(lattice_size=3)
    system = LivniumCoreSystem(config)
    
    candidates = create_candidate_basins(system, n_candidates=3, basin_size=4)
    
    assert isinstance(candidates, list), "Should return list"
    assert len(candidates) == 3, "Should have 3 candidates"
    
    for candidate in candidates:
        assert isinstance(candidate, list), "Each candidate should be a list"
        assert len(candidate) == 4, "Each candidate should have 4 coords"
        for coord in candidate:
            assert isinstance(coord, tuple), "Coords should be tuples"
            assert len(coord) == 3, "Coords should be 3D"


def test_solve_with_multi_basin():
    """Test high-level solve function."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    # Create candidate solutions
    candidates = create_candidate_basins(system, n_candidates=3, basin_size=4)
    
    # Solve
    winner, steps, stats = solve_with_multi_basin(
        system,
        candidates,
        max_steps=10,
        verbose=False
    )
    
    assert isinstance(steps, int), "Steps should be int"
    assert steps <= 10, "Steps should not exceed max_steps"
    assert isinstance(stats, dict), "Stats should be dict"
    assert 'num_alive' in stats, "Stats should have num_alive"


def test_basin_pruning():
    """Test basin pruning when max_basins exceeded."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    search = MultiBasinSearch(max_basins=3)
    
    coords_list = list(system.lattice.keys())
    
    # Add more basins than max
    for i in range(5):
        start_idx = i * 2
        end_idx = start_idx + 4
        if end_idx <= len(coords_list):
            search.add_basin(coords_list[start_idx:end_idx], system)
    
    # Pruning marks basins as dead, then _prune_dead_basins removes them
    # Trigger update to prune dead basins
    search.update_all_basins(system)
    
    # After pruning, should have at most max_basins alive
    alive_basins = [b for b in search.basins if b.is_alive]
    assert len(alive_basins) <= search.max_basins, \
        f"Should have at most {search.max_basins} alive basins, got {len(alive_basins)}"


if __name__ == "__main__":
    print("Running multi-basin search tests...")
    
    test_basin_dataclass()
    print("✓ Basin dataclass")
    
    test_multi_basin_search_initialization()
    print("✓ MultiBasinSearch initialization")
    
    test_add_basin()
    print("✓ Add basin")
    
    test_update_all_basins()
    print("✓ Update all basins")
    
    test_get_winner()
    print("✓ Get winner")
    
    test_get_best_basin()
    print("✓ Get best basin")
    
    test_get_basin_stats()
    print("✓ Get basin stats")
    
    test_basin_competition()
    print("✓ Basin competition")
    
    test_create_candidate_basins()
    print("✓ Create candidate basins")
    
    test_solve_with_multi_basin()
    print("✓ Solve with multi-basin")
    
    test_basin_pruning()
    print("✓ Basin pruning")
    
    print("\nAll tests passed! ✓")


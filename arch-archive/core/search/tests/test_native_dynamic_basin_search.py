"""
Test assertions for native dynamic basin search.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from typing import List, Tuple

from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig
from core.search.native_dynamic_basin_search import (
    compute_local_curvature,
    compute_symbolic_tension,
    compute_noise_entropy,
    get_geometry_signals,
    update_basin_dynamic,
    DynamicBasinSearch,
    apply_dynamic_basin,
)


class SimpleTask:
    """Simple task for testing - 3-bit parity."""
    
    def __init__(self, system: LivniumCoreSystem, rng: np.random.Generator):
        self.system = system
        self.rng = rng
        
        # Choose 3 input cells and 1 output cell
        coords_list = list(system.lattice.keys())
        selected_indices = rng.choice(len(coords_list), size=3, replace=False)
        self.input_coords = [coords_list[i] for i in selected_indices]
        
        # Choose output from remaining cells
        remaining = [c for c in coords_list if c not in self.input_coords]
        if remaining:
            self.output_coord = remaining[rng.integers(0, len(remaining))]
        else:
            self.output_coord = coords_list[0]
        
        # Random 3-bit input
        self.target_input = [rng.integers(0, 2) for _ in range(3)]
        self.target_output = sum(self.target_input) % 2  # Parity
        
        # Encode input
        self._encode_input()
    
    def _encode_input(self):
        """Encode input bits into lattice cells."""
        for coords, bit in zip(self.input_coords, self.target_input):
            cell = self.system.get_cell(coords)
            if cell:
                if bit == 1:
                    cell.symbolic_weight = 20.0  # High = 1
                else:
                    cell.symbolic_weight = 10.0  # Low = 0
    
    def decode_answer(self) -> int:
        """Decode answer from output cell."""
        output_cell = self.system.get_cell(self.output_coord)
        if output_cell:
            return 1 if output_cell.symbolic_weight > 15.0 else 0
        return 0
    
    def is_correct(self) -> bool:
        """Check if task is solved."""
        answer = self.decode_answer()
        return answer == self.target_output


def test_compute_local_curvature():
    """Test local curvature computation."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    # Get some coordinates
    coords_list = list(system.lattice.keys())[:5]
    
    curvature = compute_local_curvature(system, coords_list)
    
    assert isinstance(curvature, float), "Curvature should be float"
    assert 0.0 <= curvature <= 2.0, f"Curvature out of range: {curvature}"
    
    # Empty list should return 0
    assert compute_local_curvature(system, []) == 0.0, "Empty coords should return 0"


def test_compute_symbolic_tension():
    """Test symbolic tension computation."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    coords_list = list(system.lattice.keys())[:5]
    
    tension = compute_symbolic_tension(system, coords_list)
    
    assert isinstance(tension, float), "Tension should be float"
    assert 0.0 <= tension <= 2.0, f"Tension out of range: {tension}"
    
    # Empty list should return 0
    assert compute_symbolic_tension(system, []) == 0.0, "Empty coords should return 0"


def test_compute_noise_entropy():
    """Test noise entropy computation."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    coords_list = list(system.lattice.keys())[:5]
    
    entropy = compute_noise_entropy(system, coords_list)
    
    assert isinstance(entropy, float), "Entropy should be float"
    assert 0.0 <= entropy <= 2.0, f"Entropy out of range: {entropy}"
    
    # Empty list should return 0
    assert compute_noise_entropy(system, []) == 0.0, "Empty coords should return 0"


def test_get_geometry_signals():
    """Test geometry signals retrieval."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    rng = np.random.Generator(np.random.PCG64(42))
    task = SimpleTask(system, rng)
    
    active_coords = task.input_coords + [task.output_coord]
    
    # Get signals
    curvature = compute_local_curvature(system, active_coords)
    tension = compute_symbolic_tension(system, active_coords)
    entropy = compute_noise_entropy(system, active_coords)
    
    # Get all signals at once
    signals_tuple = get_geometry_signals(system, active_coords)
    signals_curv, signals_tens, signals_entr = signals_tuple
    
    assert isinstance(signals_tuple, tuple), "Should return tuple"
    assert len(signals_tuple) == 3, "Should have 3 values"
    
    # Verify signals match individual calls
    assert abs(signals_curv - curvature) < 1e-6, "Curvature should match"
    assert abs(signals_tens - tension) < 1e-6, "Tension should match"
    assert abs(signals_entr - entropy) < 1e-6, "Entropy should match"


def test_update_basin_dynamic_correct():
    """Test dynamic basin update for correct task."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    rng = np.random.Generator(np.random.PCG64(42))
    task = SimpleTask(system, rng)
    
    # Get initial SW values
    active_coords = task.input_coords + [task.output_coord]
    initial_sw = {}
    for coords in active_coords:
        cell = system.get_cell(coords)
        if cell:
            initial_sw[coords] = cell.symbolic_weight
    
    # Update with correct
    is_correct = task.is_correct()
    update_basin_dynamic(system, task, is_correct)
    
    # Get updated SW values
    updated_sw = {}
    for coords in active_coords:
        cell = system.get_cell(coords)
        if cell:
            updated_sw[coords] = cell.symbolic_weight
    
    # Verify SW bounds
    for coords, cell in system.lattice.items():
        assert 0.0 <= cell.symbolic_weight <= 200.0, \
            f"SW out of bounds: {coords} = {cell.symbolic_weight}"


def test_update_basin_dynamic_incorrect():
    """Test dynamic basin update for incorrect task."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    rng = np.random.Generator(np.random.PCG64(43))
    task = SimpleTask(system, rng)
    
    # Get initial SW values
    active_coords = task.input_coords + [task.output_coord]
    initial_sw = {}
    for coords in active_coords:
        cell = system.get_cell(coords)
        if cell:
            initial_sw[coords] = cell.symbolic_weight
    
    # Update with incorrect
    is_correct = task.is_correct()
    update_basin_dynamic(system, task, is_correct)
    
    # Verify SW bounds
    for coords, cell in system.lattice.items():
        assert 0.0 <= cell.symbolic_weight <= 200.0, \
            f"SW out of bounds: {coords} = {cell.symbolic_weight}"


def test_dynamic_basin_search_class():
    """Test DynamicBasinSearch class."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    rng = np.random.Generator(np.random.PCG64(42))
    task = SimpleTask(system, rng)
    
    # Create DynamicBasinSearch instance
    basin_search = DynamicBasinSearch(
        base_alpha=0.10,
        base_beta=0.15,
        base_noise=0.03
    )
    
    assert basin_search.base_alpha == 0.10, "Should set base_alpha"
    assert basin_search.base_beta == 0.15, "Should set base_beta"
    assert basin_search.base_noise == 0.03, "Should set base_noise"
    
    # Get signals
    active_coords = task.input_coords + [task.output_coord]
    signals = basin_search.get_signals(system, active_coords)
    
    assert isinstance(signals, dict), "Should return dictionary"
    assert 'curvature' in signals, "Should have curvature"
    assert 'tension' in signals, "Should have tension"
    assert 'entropy' in signals, "Should have entropy"
    
    # Update basin
    is_correct = task.is_correct()
    basin_search.update(system, task, is_correct)
    
    # Verify system still valid
    for coords, cell in system.lattice.items():
        assert 0.0 <= cell.symbolic_weight <= 200.0, \
            f"SW out of bounds: {coords} = {cell.symbolic_weight}"


def test_apply_dynamic_basin():
    """Test convenience function."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    rng = np.random.Generator(np.random.PCG64(42))
    task = SimpleTask(system, rng)
    
    # Get initial SW
    active_coords = task.input_coords + [task.output_coord]
    initial_sw = system.get_cell(active_coords[0]).symbolic_weight
    
    # Apply dynamic basin
    is_correct = task.is_correct()
    apply_dynamic_basin(system, task, is_correct)
    
    # Get updated SW
    updated_sw = system.get_cell(active_coords[0]).symbolic_weight
    
    # Verify SW bounds
    assert 0.0 <= updated_sw <= 200.0, "SW should be in bounds"


def test_multiple_updates():
    """Test multiple basin updates."""
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    rng = np.random.Generator(np.random.PCG64(42))
    
    # Run multiple tasks
    n_tasks = 5
    
    for i in range(n_tasks):
        task = SimpleTask(system, rng)
        is_correct = task.is_correct()
        
        # Get signals before update
        active_coords = task.input_coords + [task.output_coord]
        curv_before, tens_before, entr_before = get_geometry_signals(system, active_coords)
        
        # Update basin
        update_basin_dynamic(system, task, is_correct)
        
        # Get signals after update
        curv_after, tens_after, entr_after = get_geometry_signals(system, active_coords)
        
        # Verify signals are valid
        assert 0.0 <= curv_after <= 2.0, "Curvature should be in range"
        assert 0.0 <= tens_after <= 2.0, "Tension should be in range"
        assert 0.0 <= entr_after <= 2.0, "Entropy should be in range"
    
    # Verify system still valid
    for coords, cell in system.lattice.items():
        assert 0.0 <= cell.symbolic_weight <= 200.0, \
            f"SW out of bounds: {coords} = {cell.symbolic_weight}"


if __name__ == "__main__":
    print("Running native dynamic basin search tests...")
    
    test_compute_local_curvature()
    print("✓ Compute local curvature")
    
    test_compute_symbolic_tension()
    print("✓ Compute symbolic tension")
    
    test_compute_noise_entropy()
    print("✓ Compute noise entropy")
    
    test_get_geometry_signals()
    print("✓ Get geometry signals")
    
    test_update_basin_dynamic_correct()
    print("✓ Update basin dynamic (correct)")
    
    test_update_basin_dynamic_incorrect()
    print("✓ Update basin dynamic (incorrect)")
    
    test_dynamic_basin_search_class()
    print("✓ DynamicBasinSearch class")
    
    test_apply_dynamic_basin()
    print("✓ Apply dynamic basin")
    
    test_multiple_updates()
    print("✓ Multiple updates")
    
    print("\nAll tests passed! ✓")


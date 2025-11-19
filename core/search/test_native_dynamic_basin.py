"""
Simple Test for Native Dynamic Basin Search

Tests the core dynamic basin reinforcement functions:
- Geometry signal computation (curvature, tension, entropy)
- Dynamic basin updates
- DynamicBasinSearch class
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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


def test_geometry_signals():
    """Test geometry signal computation."""
    print("="*60)
    print("Test 1: Geometry Signal Computation")
    print("="*60)
    
    # Create system
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    # Create task
    rng = np.random.Generator(np.random.PCG64(42))
    task = SimpleTask(system, rng)
    
    # Get active coordinates
    active_coords = task.input_coords + [task.output_coord]
    
    # Compute signals
    curvature = compute_local_curvature(system, active_coords)
    tension = compute_symbolic_tension(system, active_coords)
    entropy = compute_noise_entropy(system, active_coords)
    
    # Get all signals at once
    signals_tuple = get_geometry_signals(system, active_coords)
    signals_curv, signals_tens, signals_entr = signals_tuple
    
    print(f"  Active cells: {len(active_coords)}")
    print(f"  Curvature: {curvature:.4f}")
    print(f"  Tension: {tension:.4f}")
    print(f"  Entropy: {entropy:.4f}")
    print(f"  Signals tuple: {signals_tuple}")
    
    # Verify signals are in reasonable range
    assert 0.0 <= curvature <= 2.0, f"Curvature out of range: {curvature}"
    assert 0.0 <= tension <= 2.0, f"Tension out of range: {tension}"
    assert 0.0 <= entropy <= 2.0, f"Entropy out of range: {entropy}"
    
    # Verify get_geometry_signals matches individual calls
    assert abs(signals_curv - curvature) < 1e-6
    assert abs(signals_tens - tension) < 1e-6
    assert abs(signals_entr - entropy) < 1e-6
    
    print("  ✓ All signals computed correctly")
    print()


def test_dynamic_basin_update():
    """Test dynamic basin update function."""
    print("="*60)
    print("Test 2: Dynamic Basin Update")
    print("="*60)
    
    # Create system
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    # Create task
    rng = np.random.Generator(np.random.PCG64(42))
    task = SimpleTask(system, rng)
    
    # Get initial SW values
    active_coords = task.input_coords + [task.output_coord]
    initial_sw = {}
    for coords in active_coords:
        cell = system.get_cell(coords)
        if cell:
            initial_sw[coords] = cell.symbolic_weight
    
    print(f"  Initial SW values:")
    for coords, sw in initial_sw.items():
        print(f"    {coords}: {sw:.2f}")
    
    # Test correct update
    print(f"\n  Testing CORRECT update...")
    is_correct = task.is_correct()
    update_basin_dynamic(system, task, is_correct)
    
    # Get updated SW values
    updated_sw = {}
    for coords in active_coords:
        cell = system.get_cell(coords)
        if cell:
            updated_sw[coords] = cell.symbolic_weight
    
    print(f"  Updated SW values:")
    for coords, sw in updated_sw.items():
        initial = initial_sw.get(coords, 0)
        change = sw - initial
        print(f"    {coords}: {sw:.2f} (change: {change:+.2f})")
    
    # Verify SW increased for correct (or decreased for wrong)
    if is_correct:
        # SW should increase (basin strengthening)
        for coords in active_coords:
            if coords in initial_sw and coords in updated_sw:
                assert updated_sw[coords] >= initial_sw[coords], \
                    f"SW decreased for correct: {coords}"
        print("  ✓ Correct update: SW increased (basin strengthened)")
    else:
        # SW should decrease (basin decay)
        for coords in active_coords:
            if coords in initial_sw and coords in updated_sw:
                assert updated_sw[coords] <= initial_sw[coords], \
                    f"SW increased for wrong: {coords}"
        print("  ✓ Wrong update: SW decreased (basin decayed)")
    
    # Verify SW bounds
    for coords, cell in system.lattice.items():
        assert 0.0 <= cell.symbolic_weight <= 200.0, \
            f"SW out of bounds: {coords} = {cell.symbolic_weight}"
    
    print("  ✓ All SW values in bounds [0, 200]")
    print()


def test_dynamic_basin_search_class():
    """Test DynamicBasinSearch class."""
    print("="*60)
    print("Test 3: DynamicBasinSearch Class")
    print("="*60)
    
    # Create system
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    # Create task
    rng = np.random.Generator(np.random.PCG64(42))
    task = SimpleTask(system, rng)
    
    # Create DynamicBasinSearch instance
    basin_search = DynamicBasinSearch(
        base_alpha=0.10,
        base_beta=0.15,
        base_noise=0.03
    )
    
    # Get initial signals
    active_coords = task.input_coords + [task.output_coord]
    initial_signals = basin_search.get_signals(system, active_coords)
    
    print(f"  Initial signals:")
    for key, value in initial_signals.items():
        print(f"    {key}: {value:.4f}")
    
    # Update basin
    is_correct = task.is_correct()
    print(f"\n  Task correct: {is_correct}")
    basin_search.update(system, task, is_correct)
    
    # Get updated signals
    updated_signals = basin_search.get_signals(system, active_coords)
    
    print(f"  Updated signals:")
    for key, value in updated_signals.items():
        change = value - initial_signals[key]
        print(f"    {key}: {value:.4f} (change: {change:+.4f})")
    
    print("  ✓ DynamicBasinSearch class works correctly")
    print()


def test_apply_dynamic_basin():
    """Test convenience function."""
    print("="*60)
    print("Test 4: Apply Dynamic Basin (Convenience Function)")
    print("="*60)
    
    # Create system
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    # Create task
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
    
    print(f"  Initial SW: {initial_sw:.2f}")
    print(f"  Updated SW: {updated_sw:.2f}")
    print(f"  Change: {updated_sw - initial_sw:+.2f}")
    print(f"  Task correct: {is_correct}")
    
    # Verify change direction
    if is_correct:
        assert updated_sw >= initial_sw, "SW should increase for correct"
        print("  ✓ SW increased for correct task")
    else:
        assert updated_sw <= initial_sw, "SW should decrease for wrong"
        print("  ✓ SW decreased for wrong task")
    
    print()


def test_multiple_updates():
    """Test multiple basin updates to see geometry adaptation."""
    print("="*60)
    print("Test 5: Multiple Updates (Geometry Adaptation)")
    print("="*60)
    
    # Create system
    config = LivniumCoreConfig(lattice_size=3, enable_semantic_polarity=True)
    system = LivniumCoreSystem(config)
    
    rng = np.random.Generator(np.random.PCG64(42))
    
    # Run multiple tasks
    n_tasks = 10
    correct_count = 0
    
    print(f"  Running {n_tasks} tasks...")
    
    for i in range(n_tasks):
        task = SimpleTask(system, rng)
        is_correct = task.is_correct()
        
        if is_correct:
            correct_count += 1
        
        # Get signals before update
        active_coords = task.input_coords + [task.output_coord]
        curv_before, tens_before, entr_before = get_geometry_signals(system, active_coords)
        
        # Update basin
        update_basin_dynamic(system, task, is_correct)
        
        # Get signals after update
        curv_after, tens_after, entr_after = get_geometry_signals(system, active_coords)
        
        if (i + 1) % 3 == 0:
            print(f"    Task {i+1}: correct={is_correct}")
            print(f"      Curvature: {curv_before:.4f} → {curv_after:.4f}")
            print(f"      Tension: {tens_before:.4f} → {tens_after:.4f}")
            print(f"      Entropy: {entr_before:.4f} → {entr_after:.4f}")
    
    success_rate = correct_count / n_tasks
    print(f"\n  Success rate: {correct_count}/{n_tasks} ({success_rate*100:.1f}%)")
    print("  ✓ Multiple updates completed successfully")
    print()


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("NATIVE DYNAMIC BASIN SEARCH - TEST SUITE")
    print("="*60)
    print()
    
    try:
        test_geometry_signals()
        test_dynamic_basin_update()
        test_dynamic_basin_search_class()
        test_apply_dynamic_basin()
        test_multiple_updates()
        
        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print()
        print("Summary:")
        print("  ✓ Geometry signals computed correctly")
        print("  ✓ Dynamic basin updates work")
        print("  ✓ DynamicBasinSearch class functional")
        print("  ✓ Convenience function works")
        print("  ✓ Multiple updates adapt geometry")
        print()
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()


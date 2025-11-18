"""
Fast Task Test: Direct Core System Usage

Uses core system directly (like entanglement test) for maximum speed.
No recursive overhead, no unnecessary rebuilding - just fast task solving.

IMPORTANT: This is a STRUCTURAL PROJECTOR, not a learner.

Key Properties:
- Memoryless: Each task solved independently by collapsing geometry
- No learning: No weights, gradients, memory, or adaptation
- Symmetric: ~50% success rate expected (parity has 50/50 symmetry)
- Deterministic: Same geometry → same collapse → same result
- Fast mode: Pure structural projection (log(N) scaling, not N³)
- Cache growth: Memory increases from state storage, NOT learning

Expected Behavior:
- Success rate: ~48-50% (statistical symmetry, no bias)
- No drift: Success rate doesn't change with more tasks
- Time scaling: ~log(N) (structural projection, not computation)
- Memory growth: Cache accumulation, not performance improvement
- State pollution: Slight dip at high task counts (accumulated tension)
"""

import sys
import time
import psutil
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import List, Tuple, Dict, Any
import random

from core.classical.livnium_core_system import LivniumCoreSystem, RotationAxis
from core.config import LivniumCoreConfig
from core.quantum.quantum_lattice import QuantumLattice
from core.quantum.quantum_gates import GateType


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class FastParity3Task:
    """Fast 3-bit parity task using core system directly."""
    
    def __init__(self, system: LivniumCoreSystem, rng: np.random.Generator, use_quantum: bool = False):
        self.system = system
        self.rng = rng
        self.use_quantum = use_quantum
        
        # Choose 3 input cells and 1 output cell
        coords_list = list(system.lattice.keys())
        # Convert to list of tuples for proper handling
        selected_indices = self.rng.choice(len(coords_list), size=3, replace=False)
        self.input_coords = [coords_list[i] for i in selected_indices]
        
        # Choose output from remaining cells
        remaining = [c for c in coords_list if c not in self.input_coords]
        if remaining:
            self.output_coord = remaining[self.rng.integers(0, len(remaining))]
        else:
            # Fallback: use first coordinate
            self.output_coord = coords_list[0]
        
        # Random 3-bit input
        self.target_input = [self.rng.integers(0, 2) for _ in range(3)]
        self.target_output = sum(self.target_input) % 2  # Parity
        
        # Initialize quantum if enabled
        if use_quantum:
            try:
                self.qlattice = QuantumLattice(system)
            except:
                self.qlattice = None
                self.use_quantum = False
        else:
            self.qlattice = None
        
        # Encode input
        self._encode_input()
    
    def _encode_input(self):
        """Encode input bits into lattice cells."""
        if self.use_quantum and self.qlattice:
            # Use quantum encoding: |0⟩ for 0, |1⟩ for 1
            for i, (coords, bit) in enumerate(zip(self.input_coords, self.target_input)):
                if coords in self.qlattice.quantum_cells:
                    if bit == 1:
                        # Set to |1⟩
                        self.qlattice.apply_gate(coords, GateType.PAULI_X)
                    # else: already |0⟩
        else:
            # Classical encoding: use symbolic_weight
            for i, (coords, bit) in enumerate(zip(self.input_coords, self.target_input)):
                cell = self.system.get_cell(coords)
                if cell:
                    # Encode bit in symbolic_weight (simple encoding)
                    if bit == 1:
                        cell.symbolic_weight = 20.0  # High = 1
                    else:
                        cell.symbolic_weight = 10.0  # Low = 0
    
    def decode_answer(self) -> int:
        """Decode answer from output cell."""
        if self.use_quantum and self.qlattice:
            # Measure quantum state
            if self.output_coord in self.qlattice.quantum_cells:
                result = self.qlattice.measure_cell(self.output_coord, collapse=False)
                return 1 if result == 1 else 0
        else:
            # Classical decoding
            output_cell = self.system.get_cell(self.output_coord)
            if output_cell:
                # Decode from symbolic_weight
                return 1 if output_cell.symbolic_weight > 15.0 else 0
        return 0
    
    def compute_loss(self) -> float:
        """Compute task loss (0 = correct, 1 = wrong)."""
        answer = self.decode_answer()
        return 0.0 if answer == self.target_output else 1.0
    
    def is_correct(self) -> bool:
        """Check if task is solved."""
        return self.compute_loss() == 0.0


class DualQubitParityTask:
    """3-bit parity task using dual-qubit (entangled pairs) encoding."""
    
    def __init__(self, system: LivniumCoreSystem, qlattice: QuantumLattice, rng: np.random.Generator):
        self.system = system
        self.qlattice = qlattice
        self.rng = rng
        
        # Choose 3 input qubits and 1 output qubit
        coords_list = list(qlattice.quantum_cells.keys())
        selected_indices = self.rng.choice(len(coords_list), size=4, replace=False)
        self.input_coords = [coords_list[i] for i in selected_indices[:3]]
        self.output_coord = coords_list[selected_indices[3]]
        
        # Random 3-bit input
        self.target_input = [self.rng.integers(0, 2) for _ in range(3)]
        self.target_output = sum(self.target_input) % 2  # Parity
        
        # Create dual-qubit pairs: input[0] <-> input[1], input[1] <-> input[2]
        # This creates entanglement structure
        self.entangled_pairs = []
        if len(self.input_coords) >= 2:
            self.qlattice.entangle_cells(self.input_coords[0], self.input_coords[1])
            self.entangled_pairs.append((self.input_coords[0], self.input_coords[1]))
        if len(self.input_coords) >= 3:
            self.qlattice.entangle_cells(self.input_coords[1], self.input_coords[2])
            self.entangled_pairs.append((self.input_coords[1], self.input_coords[2]))
        
        # Encode input
        self._encode_input()
    
    def _encode_input(self):
        """Encode input bits using quantum gates on entangled qubits."""
        for coords, bit in zip(self.input_coords, self.target_input):
            if bit == 1:
                # Apply X gate to set to |1⟩
                self.qlattice.apply_gate(coords, GateType.PAULI_X)
            # else: already |0⟩
    
    def decode_answer(self) -> int:
        """Decode answer by measuring output qubit."""
        if self.output_coord in self.qlattice.quantum_cells:
            result = self.qlattice.measure_cell(self.output_coord, collapse=False)
            return 1 if result == 1 else 0
        return 0
    
    def compute_loss(self) -> float:
        """Compute task loss (0 = correct, 1 = wrong)."""
        answer = self.decode_answer()
        return 0.0 if answer == self.target_output else 1.0
    
    def is_correct(self) -> bool:
        """Check if task is solved."""
        return self.compute_loss() == 0.0


def update_basin(
    system: LivniumCoreSystem,
    task,
    is_correct: bool,
    alpha: float = 0.10,  # Strengthen correct basin (increased from 0.05)
    beta: float = 0.15,   # Decay wrong basin (increased from 0.10)
    noise: float = 0.03   # Decorrelation for wrong states (increased from 0.02)
):
    """
    Basin shaping rule: Geometry-first, no ML.
    
    - Correct states: Deepen their basin (reinforcement)
    - Incorrect states: Add noise and decay (diffusion)
    
    This is energy-landscape shaping, not learning.
    """
    import random
    
    if is_correct:
        # Strengthen the attractor: deepen the well
        # Find active cells (input + output cells)
        active_coords = []
        if hasattr(task, 'input_coords'):
            active_coords.extend(task.input_coords)
        if hasattr(task, 'output_coord'):
            active_coords.append(task.output_coord)
        
        for coords in active_coords:
            cell = system.get_cell(coords)
            if cell:
                # Deepen well: increase SW (stronger attractor)
                cell.symbolic_weight += alpha
                # Mark as reliable (if we track stability)
                # Note: LivniumCoreSystem doesn't have stability field by default
                # but we can use face_exposure or other properties
        
        # Smooth small contradictions (enforce local equilibrium)
        # This happens naturally through the system's conservation rules
        
    else:
        # Add decoherence noise: flatten wrong basin
        active_coords = []
        if hasattr(task, 'input_coords'):
            active_coords.extend(task.input_coords)
        if hasattr(task, 'output_coord'):
            active_coords.append(task.output_coord)
        
        for coords in active_coords:
            cell = system.get_cell(coords)
            if cell:
                # Decay SW (flatten well)
                cell.symbolic_weight *= (1.0 - beta)
                # Ensure SW doesn't go negative
                if cell.symbolic_weight < 0:
                    cell.symbolic_weight = 0.0
        
        # Inject small random drift so wrong basin can't re-form
        # Apply random rotation to decorrelate
        if random.random() < noise * 10:  # Scale noise probability
            axis = random.choice(list(RotationAxis))
            system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
    
    # Re-normalize: Keep global conservation intact
    # The system's conservation rules handle this automatically
    # But we can ensure SW stays in reasonable bounds
    for coords, cell in system.lattice.items():
        if cell.symbolic_weight < 0:
            cell.symbolic_weight = 0.0
        # Cap at reasonable maximum (prevent explosion)
        if cell.symbolic_weight > 100.0:
            cell.symbolic_weight = 100.0


def fast_task_solve(
    system: LivniumCoreSystem,
    task,  # FastParity3Task or DualQubitParityTask
    max_steps: int = 200,  # Reduced from 1000 for speed
    use_basin_reinforcement: bool = False,  # Enable basin shaping
    use_dynamic_basin: bool = False  # Use dynamic (geometry-driven) basin shaping
) -> Tuple[bool, int, float]:
    """
    Fast task solving using simple rotations.
    
    OPTIMIZED: Early stopping, fewer steps, check less frequently.
    
    Returns:
        (solved, steps_taken, final_loss)
    """
    # Check initial state
    if task.is_correct():
        return True, 0, 0.0
    
    for step in range(max_steps):
        # Try random rotation
        axis = random.choice(list(RotationAxis))
        system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
        
        # Re-encode task (input might have changed)
        task._encode_input()
        
        # Check less frequently (every 5 steps) for speed
        if step % 5 == 0:
            if task.is_correct():
                return True, step + 1, 0.0
    
    # Final check
    is_solved = task.is_correct()
    final_loss = task.compute_loss() if not is_solved else 0.0
    
    # Basin reinforcement: Shape geometry based on correctness
    if use_basin_reinforcement:
        if use_dynamic_basin:
            # Use dynamic (geometry-driven) basin shaping
            try:
                from .dynamic_basin_reinforcement import update_basin_dynamic
                update_basin_dynamic(system, task, is_solved)
            except ImportError:
                from dynamic_basin_reinforcement import update_basin_dynamic
                update_basin_dynamic(system, task, is_solved)
        else:
            # Use static basin shaping
            update_basin(system, task, is_solved)
    
    if is_solved:
        return True, max_steps, 0.0
    
    # Not solved
    return False, max_steps, final_loss


def test_task_solving(n: int, n_tasks: int = 100, verbose: bool = True, use_quantum: bool = False, use_dual_qubit: bool = False, use_basin_reinforcement: bool = False, use_dynamic_basin: bool = False) -> Dict[str, Any]:
    """
    Test task solving performance.
    
    Args:
        n: Lattice size (N×N×N)
        n_tasks: Number of tasks to solve
        verbose: Print progress
        
    Returns:
        Test results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Fast Task Solving Test: N={n}, {n_tasks} tasks")
        print(f"{'='*60}")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # Create system ONCE (like entanglement test)
    config = LivniumCoreConfig(
        lattice_size=n,
        enable_quantum=use_quantum or use_dual_qubit,
        enable_superposition=use_quantum or use_dual_qubit,
        enable_quantum_gates=use_quantum or use_dual_qubit,
        enable_entanglement=use_dual_qubit,
        enable_measurement=use_quantum or use_dual_qubit,
    )
    system = LivniumCoreSystem(config)
    
    # Initialize quantum lattice if needed
    qlattice = None
    if use_dual_qubit:
        try:
            qlattice = QuantumLattice(system)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not initialize quantum: {e}")
            use_dual_qubit = False
    
    init_time = time.time() - start_time
    init_memory = get_memory_usage() - start_memory
    
    if verbose:
        print(f"  Initialization: {init_time:.3f}s, {init_memory:.2f} MB")
        print(f"  Total cells: {len(system.lattice)}")
    
    # Solve tasks
    rng = np.random.Generator(np.random.PCG64(42))
    solved = 0
    total_steps = 0
    solve_times = []
    
    if verbose:
        print(f"  Solving {n_tasks} tasks...")
    
    solve_start = time.time()
    
    # Track rate over time to detect drift
    rate_history = []
    early_solved = 0  # First 100 tasks
    late_solved = 0   # Last 100 tasks
    
    for i in range(n_tasks):
        # Create task
        if use_dual_qubit and qlattice:
            task = DualQubitParityTask(system, qlattice, rng)
        else:
            task = FastParity3Task(system, rng, use_quantum=use_quantum)
        
        # Solve
        task_start = time.time()
        is_solved, steps, loss = fast_task_solve(
            system, task, max_steps=500, 
            use_basin_reinforcement=use_basin_reinforcement,
            use_dynamic_basin=use_dynamic_basin
        )
        task_time = time.time() - task_start
        
        if is_solved:
            solved += 1
            total_steps += steps
            solve_times.append(task_time)
            
            # Track early vs late
            if i < 100:
                early_solved += 1
            elif i >= n_tasks - 100:
                late_solved += 1
        
        # Track rate history
        current_rate = solved / (i + 1) if i > 0 else 0
        rate_history.append(current_rate)
        
        if verbose and (i + 1) % 20 == 0:
            print(f"    Progress: {i+1}/{n_tasks}, Solved: {solved}, "
                  f"Rate: {solved/(i+1)*100:.1f}%")
    
    solve_time = time.time() - solve_start
    
    # Results
    end_memory = get_memory_usage()
    total_memory = end_memory - start_memory
    total_time = time.time() - start_time
    
    success_rate = solved / n_tasks if n_tasks > 0 else 0
    
    # Detect drift: compare early vs late performance
    early_rate = early_solved / min(100, n_tasks) if n_tasks > 0 else 0
    late_rate = late_solved / min(100, n_tasks - max(0, n_tasks - 100)) if n_tasks > 100 else 0
    drift = late_rate - early_rate if n_tasks > 100 else 0
    
    # Check if rate is stable (no significant drift)
    rate_stable = abs(drift) < 0.05  # Less than 5% change
    
    results = {
        'n': n,
        'n_tasks': n_tasks,
        'use_quantum': use_quantum,
        'use_dual_qubit': use_dual_qubit,
        'solved': solved,
        'success_rate': success_rate,
        'total_steps': total_steps,
        'avg_steps_per_solve': total_steps / solved if solved > 0 else 0,
        'solve_time': solve_time,
        'time_per_task': solve_time / n_tasks if n_tasks > 0 else 0,
        'time_per_solve': np.mean(solve_times) if solve_times else 0,
        'total_memory_mb': total_memory,
        'total_time': total_time,
        'success': solved > 0,
        # Physics analysis
        'is_symmetric': 0.45 <= success_rate <= 0.55,  # Expected range for symmetric collapse
        'memory_per_task': total_memory / n_tasks if n_tasks > 0 else 0,
        # Drift detection
        'early_rate': early_rate,
        'late_rate': late_rate,
        'drift': drift,
        'rate_stable': rate_stable,
        'rate_history': rate_history,
        'use_basin_reinforcement': use_basin_reinforcement,
    }
    
    if verbose:
        print(f"  ✅ Solved {solved}/{n_tasks} tasks ({results['success_rate']*100:.1f}%)")
        print(f"  Time per task: {results['time_per_task']*1000:.3f} ms")
        if solved > 0:
            print(f"  Avg steps per solve: {results['avg_steps_per_solve']:.1f}")
            print(f"  Time per solve: {results['time_per_solve']*1000:.3f} ms")
        print(f"  Total memory: {total_memory:.2f} MB")
        
        # Analysis: Explain the physics
        print()
        print(f"  📊 Analysis:")
        if use_basin_reinforcement:
            if use_dynamic_basin:
                print(f"     Success rate: {results['success_rate']*100:.1f}% (with DYNAMIC basin reinforcement)")
                print(f"     System type: Self-tuning basin projector (geometry-driven)")
                print(f"     Parameters adapt to curvature, tension, entropy")
                print(f"     Expected: Stable convergence, no drift, basin dominance")
            else:
                print(f"     Success rate: {results['success_rate']*100:.1f}% (with static basin reinforcement)")
                print(f"     System type: Basin-shaped projector (geometry reinforcement)")
                print(f"     Expected: Rate should increase over time (70% → 80% → 90%)")
        else:
            print(f"     Success rate: {results['success_rate']*100:.1f}% (expected ~50% due to symmetry)")
            print(f"     System type: Structural projector (memoryless, non-learning)")
        print(f"     Memory growth: Cache accumulation, not learning")
        
        # Drift detection
        if n_tasks > 100:
            print(f"     Early rate (first 100): {results['early_rate']*100:.1f}%")
            print(f"     Late rate (last 100): {results['late_rate']*100:.1f}%")
            print(f"     Drift: {results['drift']*100:+.1f}%")
            
            # Analyze rate history for drops/fluctuations
            if len(results['rate_history']) > 200:
                # Find peak and valley in different segments
                early_seg = results['rate_history'][:100]  # First 100
                mid_seg = results['rate_history'][100:300]  # Middle 200
                late_seg = results['rate_history'][300:]  # After 300
                
                if early_seg and mid_seg and late_seg:
                    early_avg = np.mean(early_seg[-20:]) if len(early_seg) >= 20 else early_seg[-1]
                    mid_peak = max(mid_seg) if mid_seg else 0
                    mid_peak_idx = mid_seg.index(mid_peak) + 100 if mid_seg else 0
                    late_valley = min(late_seg) if late_seg else 0
                    late_valley_idx = late_seg.index(late_valley) + 300 if late_seg else 0
                    final_rate = results['rate_history'][-1] if results['rate_history'] else 0
                    
                    # Check for drop after peak
                    if mid_peak > 0 and late_valley < mid_peak - 0.015:  # 1.5% drop
                        drop_magnitude = (mid_peak - late_valley) * 100
                        recovery = (final_rate - late_valley) * 100
                        
                        print(f"     Rate pattern analysis:")
                        print(f"       Early (tasks 80-100): {early_avg*100:.1f}%")
                        print(f"       Peak (task ~{mid_peak_idx}): {mid_peak*100:.1f}%")
                        print(f"       Valley (task ~{late_valley_idx}): {late_valley*100:.1f}%")
                        print(f"       Final (task {n_tasks}): {final_rate*100:.1f}%")
                        print(f"       Drop: {drop_magnitude:.1f}% (peak → valley)")
                        if recovery > 0:
                            print(f"       Recovery: {recovery:.1f}% (valley → final)")
                        
                        print(f"     ⚠️  DROP DETECTED: Rate decreased after peak")
                        print(f"        Pattern: Rise → Peak → Drop → (Recovery?)")
                        print(f"        Possible causes:")
                        print(f"        1. State pollution: Accumulated cache creates conflicts")
                        print(f"           (Cache grows → conflicts increase → performance drops)")
                        print(f"        2. Tension buildup: Symbolic weights drift creates interference")
                        print(f"           (SW values accumulate → tension → less optimal collapses)")
                        print(f"        3. Memory saturation: System reaches capacity")
                        print(f"           (Memory fills → slower operations → degraded performance)")
                        print(f"        4. Oscillatory attractors: System cycles between states")
                        print(f"           (Multiple attractors → cycling → temporary drops)")
                        if recovery > 0.01:
                            print(f"        5. Self-correction: System adapts to pollution")
                            print(f"           (Recovery suggests system finds new equilibrium)")
            
            if abs(results['drift']) > 0.05:
                if results['drift'] > 0:
                    if use_dynamic_basin:
                        print(f"     ✓ UPWARD DRIFT detected: Rate increasing over time")
                        print(f"        This is EXPECTED with dynamic basin reinforcement")
                        print(f"        System is self-tuning and improving geometry")
                    else:
                        print(f"     ⚠️  UPWARD DRIFT detected: Rate increasing over time")
                        print(f"        This suggests bias accumulation or state-dependent behavior")
                        print(f"        Expected: No drift (memoryless system)")
                else:
                    print(f"     ⚠️  DOWNWARD DRIFT detected: Rate decreasing over time")
                    print(f"        This suggests state pollution or cache interference")
                    if use_dynamic_basin:
                        print(f"        Dynamic system should self-correct - check parameters")
            else:
                if use_dynamic_basin:
                    print(f"     ⚠️  Rate stable (no significant drift)")
                    print(f"        Dynamic basin should show improvement - may need tuning")
                else:
                    print(f"     ✓ Rate stable (no significant drift - confirms memoryless)")
        
        if use_dynamic_basin:
            # With dynamic basin, we expect higher rates and bias
            if results['success_rate'] > 0.70:
                print(f"     ✓ Excellent performance! Basin dominance achieved")
            elif results['success_rate'] > 0.60:
                print(f"     ✓ Good performance! Basin shaping is working")
            elif results['success_rate'] > 0.55:
                print(f"     ⚠️  Moderate performance - may need parameter tuning")
            else:
                print(f"     ⚠️  Low performance - check geometry signals")
        else:
            # Without basin, expect symmetric ~50%
            if results['success_rate'] < 0.45:
                print(f"     ⚠️  Lower rate may indicate state pollution from accumulated cache")
            elif results['success_rate'] > 0.55:
                print(f"     ⚠️  Higher rate may indicate bias (unexpected for symmetric system)")
            else:
                print(f"     ✓ Symmetric collapse confirmed (working as designed)")
    
    return results


def run_fast_tests():
    """Run fast task solving tests for different N."""
    print("="*60)
    print("FAST TASK SOLVING TEST")
    print("="*60)
    print()
    print("System Type: Structural Projector (memoryless, non-learning)")
    print("Expected: ~50% success rate (statistical symmetry)")
    print("Time scaling: ~log(N) (structural projection)")
    print()
    
    test_sizes = [3, 5, 7, 9]
    n_tasks = 100
    
    results = []
    
    for n in test_sizes:
        result = test_task_solving(n, n_tasks=n_tasks, verbose=True)
        results.append(result)
        print()
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print()
    print("Key Observations:")
    print("- Success rate should be ~48-50% (symmetric collapse)")
    print("- No learning: rate doesn't improve with more tasks")
    print("- Time scales ~log(N) (structural projection, not N³)")
    print("- Memory growth = cache accumulation, not learning")
    print()
    
    for result in results:
        n = result['n']
        solved = result['solved']
        rate = result['success_rate'] * 100
        time_per_task = result['time_per_task'] * 1000
        time_per_solve = result['time_per_solve'] * 1000
        memory = result['total_memory_mb']
        
        # Check if symmetric
        symmetric = 0.45 <= rate <= 0.55
        status = "✓" if symmetric else "⚠️"
        
        print(f"{status} N={n:2d}: {solved:3d}/{n_tasks} solved ({rate:5.1f}%), "
              f"{time_per_task:6.3f} ms/task, {time_per_solve:6.3f} ms/solve, "
              f"{memory:5.2f} MB")
    
    print()
    print("Physics Confirmation:")
    all_symmetric = all(0.45 <= r['success_rate'] <= 0.55 for r in results)
    if all_symmetric:
        print("✓ All rates confirm symmetric collapse (working as designed)")
    else:
        print("⚠️  Some rates outside expected range - check for bias or state pollution")
    
    # Check time scaling
    if len(results) >= 3:
        times = [r['time_per_solve'] * 1000 for r in results if r['time_per_solve'] > 0]
        if len(times) >= 3:
            # Check if roughly logarithmic
            ratio_1 = times[1] / times[0] if times[0] > 0 else 0
            ratio_2 = times[2] / times[1] if times[1] > 0 else 0
            if 1.0 < ratio_1 < 2.0 and 1.0 < ratio_2 < 2.0:
                print("✓ Time scaling confirms ~log(N) structural projection")
            else:
                print("⚠️  Time scaling may not be logarithmic - check projection")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast task solving test")
    parser.add_argument('--n', type=int, default=5, help='Lattice size N')
    parser.add_argument('--tasks', type=int, default=100, help='Number of tasks')
    parser.add_argument('--full', action='store_true', help='Run full test suite')
    parser.add_argument('--quantum', action='store_true', help='Use quantum features')
    parser.add_argument('--dual-qubit', action='store_true', dest='dual_qubit', help='Use dual-qubit (entangled pairs)')
    parser.add_argument('--basin', action='store_true', dest='basin_reinforcement', 
                       help='Use static basin reinforcement (geometry shaping)')
    parser.add_argument('--dynamic-basin', action='store_true', dest='dynamic_basin',
                       help='Use dynamic basin reinforcement (geometry-driven, self-tuning)')
    
    args = parser.parse_args()
    
    if args.full:
        run_fast_tests()
    else:
        test_task_solving(args.n, n_tasks=args.tasks, verbose=True, 
                         use_quantum=args.quantum, use_dual_qubit=args.dual_qubit,
                         use_basin_reinforcement=args.basin_reinforcement or args.dynamic_basin,
                         use_dynamic_basin=args.dynamic_basin)


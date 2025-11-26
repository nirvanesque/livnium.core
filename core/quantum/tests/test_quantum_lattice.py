"""
Test assertions for QuantumLattice.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.quantum.quantum_lattice import QuantumLattice
from core.classical.livnium_core_system import LivniumCoreSystem
from core.quantum.quantum_gates import GateType
from core.config import LivniumCoreConfig


def test_basic_initialization():
    """Test basic quantum lattice initialization."""
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_quantum=True,
        enable_superposition=True
    )
    core_system = LivniumCoreSystem(config)
    
    lattice = QuantumLattice(core_system)
    
    assert lattice.core_system == core_system, "Should reference core system"
    assert len(lattice.quantum_cells) == 27, "Should have 27 quantum cells for 3×3×3"
    
    # Check all cells initialized
    for coords in core_system.lattice.keys():
        assert coords in lattice.quantum_cells, f"Cell {coords} should have quantum state"


def test_quantum_not_enabled():
    """Test that quantum lattice requires quantum to be enabled."""
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_quantum=False
    )
    core_system = LivniumCoreSystem(config)
    
    try:
        lattice = QuantumLattice(core_system)
        assert False, "Should raise ValueError when quantum not enabled"
    except ValueError:
        pass  # Expected


def test_apply_gate():
    """Test applying quantum gate to cell."""
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_quantum=True,
        enable_superposition=True,
        enable_quantum_gates=True
    )
    core_system = LivniumCoreSystem(config)
    lattice = QuantumLattice(core_system)
    
    # Apply Hadamard to center cell
    lattice.apply_gate((0, 0, 0), GateType.HADAMARD)
    
    # Check state changed
    cell = lattice.quantum_cells[(0, 0, 0)]
    probs = cell.get_probabilities()
    assert abs(probs[0] - 0.5) < 1e-6, "Hadamard should create superposition"


def test_apply_gate_to_all():
    """Test applying gate to all cells."""
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_quantum=True,
        enable_superposition=True,
        enable_quantum_gates=True
    )
    core_system = LivniumCoreSystem(config)
    lattice = QuantumLattice(core_system)
    
    # Apply Hadamard to all
    lattice.apply_gate_to_all(GateType.HADAMARD)
    
    # Check all cells have superposition
    for cell in lattice.quantum_cells.values():
        probs = cell.get_probabilities()
        assert abs(probs[0] - 0.5) < 1e-6, "All cells should have superposition"


def test_entangle_cells():
    """Test entangling cells."""
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_quantum=True,
        enable_superposition=True,
        enable_entanglement=True
    )
    core_system = LivniumCoreSystem(config)
    lattice = QuantumLattice(core_system)
    
    # Entangle two cells
    lattice.entangle_cells((0, 0, 0), (1, 0, 0))
    
    # Check entanglement manager has the pair
    if lattice.entanglement_manager:
        assert lattice.entanglement_manager.is_entangled((0, 0, 0), (1, 0, 0)), \
            "Cells should be entangled"


def test_entangle_by_face_exposure():
    """Test entangling by face exposure."""
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_quantum=True,
        enable_superposition=True,
        enable_entanglement=True
    )
    core_system = LivniumCoreSystem(config)
    lattice = QuantumLattice(core_system)
    
    # Entangle corner cell (high face exposure)
    lattice.entangle_by_face_exposure((1, 1, 1))
    
    # Should create entanglements
    if lattice.entanglement_manager:
        entangled = lattice.entanglement_manager.get_entangled_cells((1, 1, 1))
        assert len(entangled) > 0, "Should have entanglements"


def test_measure_cell():
    """Test measuring a cell."""
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_quantum=True,
        enable_superposition=True,
        enable_measurement=True
    )
    core_system = LivniumCoreSystem(config)
    lattice = QuantumLattice(core_system)
    
    # Create superposition first
    if config.enable_quantum_gates:
        lattice.apply_gate((0, 0, 0), GateType.HADAMARD)
    
    # Measure
    result = lattice.measure_cell((0, 0, 0), collapse=True)
    
    assert result.measured_state in [0, 1], "Measurement should be 0 or 1"
    assert result.collapsed, "State should be collapsed"
    assert 0.0 <= result.probability <= 1.0, "Probability should be valid"


def test_measure_all():
    """Test measuring all cells."""
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_quantum=True,
        enable_superposition=True,
        enable_measurement=True
    )
    core_system = LivniumCoreSystem(config)
    lattice = QuantumLattice(core_system)
    
    results = lattice.measure_all(collapse=True)
    
    assert len(results) == 27, "Should measure all 27 cells"
    for coords, result in results.items():
        assert result.measured_state in [0, 1], "Measurement should be 0 or 1"


def test_quantum_state_summary():
    """Test quantum state summary."""
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_quantum=True,
        enable_superposition=True,
        enable_entanglement=True,
        enable_measurement=True
    )
    core_system = LivniumCoreSystem(config)
    lattice = QuantumLattice(core_system)
    
    summary = lattice.get_quantum_state_summary()
    
    assert 'total_quantum_cells' in summary, "Should have cell count"
    assert 'features_enabled' in summary, "Should have features"
    assert summary['total_quantum_cells'] == 27, "Should have 27 cells"


if __name__ == "__main__":
    print("Running QuantumLattice tests...")
    
    test_basic_initialization()
    print("✓ Basic initialization")
    
    test_quantum_not_enabled()
    print("✓ Quantum not enabled")
    
    test_apply_gate()
    print("✓ Apply gate")
    
    test_apply_gate_to_all()
    print("✓ Apply gate to all")
    
    test_entangle_cells()
    print("✓ Entangle cells")
    
    test_entangle_by_face_exposure()
    print("✓ Entangle by face exposure")
    
    test_measure_cell()
    print("✓ Measure cell")
    
    test_measure_all()
    print("✓ Measure all")
    
    test_quantum_state_summary()
    print("✓ Quantum state summary")
    
    print("\nAll tests passed! ✓")


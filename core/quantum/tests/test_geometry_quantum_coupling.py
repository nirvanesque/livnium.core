"""
Test assertions for GeometryQuantumCoupling.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from core.quantum.geometry_quantum_coupling import GeometryQuantumCoupling
from core.classical.livnium_core_system import LivniumCoreSystem, CellClass
from core.quantum.quantum_cell import QuantumCell
from core.config import LivniumCoreConfig


def test_basic_initialization():
    """Test basic coupling initialization."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    
    coupling = GeometryQuantumCoupling(core_system)
    
    assert coupling.core_system == core_system, "Should reference core system"


def test_initialize_quantum_state_from_geometry():
    """Test initializing quantum state from geometry."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    coupling = GeometryQuantumCoupling(core_system)
    
    # Get corner cell (high SW)
    geometric_cell = core_system.get_cell((1, 1, 1))
    quantum_cell = QuantumCell(coordinates=(1, 1, 1), amplitudes=None, num_levels=2)
    
    initialized = coupling.initialize_quantum_state_from_geometry(quantum_cell, geometric_cell)
    
    assert initialized == quantum_cell, "Should return same cell"
    probs = quantum_cell.get_probabilities()
    assert abs(np.sum(probs) - 1.0) < 1e-6, "Should be normalized"


def test_apply_polarity_to_phase():
    """Test applying polarity to phase."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    coupling = GeometryQuantumCoupling(core_system)
    
    cell = QuantumCell(coordinates=(1, 0, 0), amplitudes=None, num_levels=2)
    
    # Apply polarity
    result = coupling.apply_polarity_to_phase(cell, polarity=1.0)
    
    assert result == cell, "Should return same cell"
    # State should still be normalized
    probs = cell.get_probabilities()
    assert abs(np.sum(probs) - 1.0) < 1e-6, "Should remain normalized"


def test_face_exposure_to_entanglement_strength():
    """Test mapping face exposure to entanglement strength."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    coupling = GeometryQuantumCoupling(core_system)
    
    strength_0 = coupling.face_exposure_to_entanglement_strength(0)
    strength_3 = coupling.face_exposure_to_entanglement_strength(3)
    
    assert 0.0 <= strength_0 <= 1.0, "Strength should be in [0, 1]"
    assert 0.0 <= strength_3 <= 1.0, "Strength should be in [0, 1]"
    assert strength_3 > strength_0, "Higher face exposure should give higher strength"


def test_symbolic_weight_to_amplitude_modulation():
    """Test mapping symbolic weight to amplitude modulation."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    coupling = GeometryQuantumCoupling(core_system)
    
    factor_0 = coupling.symbolic_weight_to_amplitude_modulation(0.0)
    factor_27 = coupling.symbolic_weight_to_amplitude_modulation(27.0)
    
    assert 0.0 <= factor_0 <= 1.0, "Factor should be in [0, 1]"
    assert 0.0 <= factor_27 <= 1.0, "Factor should be in [0, 1]"
    assert factor_27 > factor_0, "Higher SW should give higher factor"


def test_class_to_initial_state():
    """Test mapping cell class to initial state."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    coupling = GeometryQuantumCoupling(core_system)
    
    # Core → |0⟩
    state_core = coupling.class_to_initial_state(CellClass.CORE)
    assert np.allclose(state_core, [1.0, 0.0]), "Core should be |0⟩"
    
    # Corner → |1⟩
    state_corner = coupling.class_to_initial_state(CellClass.CORNER)
    assert np.allclose(state_corner, [0.0, 1.0]), "Corner should be |1⟩"
    
    # Center → superposition
    state_center = coupling.class_to_initial_state(CellClass.CENTER)
    assert len(state_center) == 2, "Should be 2-element state"


def test_phi_straight_line():
    """Test straight-line Φ calculation."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    coupling = GeometryQuantumCoupling(core_system)
    
    phi = coupling.phi_straight_line((1, 0, 0))
    
    assert -1.0 <= phi <= 1.0, "Phi should be in [-1, 1]"


def test_phi_rotated():
    """Test rotated Φ calculation."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    coupling = GeometryQuantumCoupling(core_system)
    
    rotated_phi, rotation_matrix = coupling.phi_rotated((1, 0, 0), rotation_axis="Y", quarter_turns=1)
    
    assert -1.0 <= rotated_phi <= 1.0, "Rotated phi should be in [-1, 1]"
    assert rotation_matrix.shape == (2, 2), "Rotation matrix should be 2×2"


def test_phi_dual_representation():
    """Test dual representation of Φ."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    coupling = GeometryQuantumCoupling(core_system)
    
    result = coupling.phi_dual_representation((1, 0, 0))
    
    assert 'straight_phi' in result, "Should have straight phi"
    assert 'rotated_phi' in result, "Should have rotated phi"
    assert 'invariant_preserved' in result, "Should have invariant flag"
    assert 'interpretation' in result, "Should have interpretation"


def test_update_quantum_from_geometry():
    """Test updating quantum states from geometry."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    coupling = GeometryQuantumCoupling(core_system)
    
    quantum_cells = {}
    for coords in core_system.lattice.keys():
        quantum_cells[coords] = QuantumCell(coords, amplitudes=None, num_levels=2)
    
    coupling.update_quantum_from_geometry(quantum_cells)
    
    # All cells should be normalized
    for cell in quantum_cells.values():
        probs = cell.get_probabilities()
        assert abs(np.sum(probs) - 1.0) < 1e-6, "All cells should be normalized"


if __name__ == "__main__":
    print("Running GeometryQuantumCoupling tests...")
    
    test_basic_initialization()
    print("✓ Basic initialization")
    
    test_initialize_quantum_state_from_geometry()
    print("✓ Initialize quantum state from geometry")
    
    test_apply_polarity_to_phase()
    print("✓ Apply polarity to phase")
    
    test_face_exposure_to_entanglement_strength()
    print("✓ Face exposure to entanglement strength")
    
    test_symbolic_weight_to_amplitude_modulation()
    print("✓ Symbolic weight to amplitude modulation")
    
    test_class_to_initial_state()
    print("✓ Class to initial state")
    
    test_phi_straight_line()
    print("✓ Phi straight line")
    
    test_phi_rotated()
    print("✓ Phi rotated")
    
    test_phi_dual_representation()
    print("✓ Phi dual representation")
    
    test_update_quantum_from_geometry()
    print("✓ Update quantum from geometry")
    
    print("\nAll tests passed! ✓")


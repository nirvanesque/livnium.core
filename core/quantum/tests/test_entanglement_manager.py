"""
Test assertions for EntanglementManager.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from core.quantum.entanglement_manager import EntanglementManager, EntangledPair


def test_basic_initialization():
    """Test basic entanglement manager initialization."""
    manager = EntanglementManager(lattice_size=3)
    
    assert manager.lattice_size == 3, "Lattice size should be 3"
    assert len(manager.entangled_pairs) == 0, "Should start with no pairs"
    assert len(manager.entanglement_graph) == 0, "Should start with empty graph"


def test_create_bell_pair():
    """Test creating Bell pair."""
    manager = EntanglementManager(lattice_size=3)
    
    pair = manager.create_bell_pair((0, 0, 0), (1, 0, 0), bell_type="phi_plus")
    
    assert isinstance(pair, EntangledPair), "Should return EntangledPair"
    assert pair.cell1 == (0, 0, 0), "Cell1 should match"
    assert pair.cell2 == (1, 0, 0), "Cell2 should match"
    assert pair.entanglement_strength == 1.0, "Should be maximally entangled"
    
    # Check pair is stored
    assert manager.is_entangled((0, 0, 0), (1, 0, 0)), "Cells should be entangled"


def test_entangled_pair():
    """Test EntangledPair dataclass."""
    state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    pair = EntangledPair((0, 0, 0), (1, 0, 0), state)
    
    assert pair.cell1 == (0, 0, 0), "Cell1 should match"
    assert pair.cell2 == (1, 0, 0), "Cell2 should match"
    assert len(pair.state_vector) == 4, "Should have 4-element state vector"
    
    concurrence = pair.get_concurrence()
    assert 0.0 <= concurrence <= 1.0, "Concurrence should be in [0, 1]"
    
    is_max = pair.is_maximally_entangled()
    assert isinstance(is_max, bool), "Should return boolean"


def test_entangle_by_distance():
    """Test entangling by distance."""
    manager = EntanglementManager(lattice_size=3)
    
    pairs = manager.entangle_by_distance((0, 0, 0), max_distance=1.5)
    
    assert isinstance(pairs, list), "Should return list"
    assert len(pairs) > 0, "Should create some pairs"


def test_entangle_by_face_exposure():
    """Test entangling by face exposure."""
    manager = EntanglementManager(lattice_size=3)
    
    # Corner cell (face_exposure=3) should get more entanglements
    pairs = manager.entangle_by_face_exposure((1, 1, 1), face_exposure=3)
    
    assert isinstance(pairs, list), "Should return list"
    assert len(pairs) >= 3, "High face exposure should create more pairs"


def test_get_entangled_cells():
    """Test getting entangled cells."""
    manager = EntanglementManager(lattice_size=3)
    
    manager.create_bell_pair((0, 0, 0), (1, 0, 0))
    manager.create_bell_pair((0, 0, 0), (0, 1, 0))
    
    entangled = manager.get_entangled_cells((0, 0, 0))
    
    assert len(entangled) == 2, "Should have 2 entangled cells"
    assert (1, 0, 0) in entangled, "Should include first pair"
    assert (0, 1, 0) in entangled, "Should include second pair"


def test_is_entangled():
    """Test checking if cells are entangled."""
    manager = EntanglementManager(lattice_size=3)
    
    assert not manager.is_entangled((0, 0, 0), (1, 0, 0)), "Should not be entangled initially"
    
    manager.create_bell_pair((0, 0, 0), (1, 0, 0))
    
    assert manager.is_entangled((0, 0, 0), (1, 0, 0)), "Should be entangled"
    assert manager.is_entangled((1, 0, 0), (0, 0, 0)), "Should be entangled (symmetric)"


def test_break_entanglement():
    """Test breaking entanglement."""
    manager = EntanglementManager(lattice_size=3)
    
    manager.create_bell_pair((0, 0, 0), (1, 0, 0))
    assert manager.is_entangled((0, 0, 0), (1, 0, 0)), "Should be entangled"
    
    manager.break_entanglement((0, 0, 0), (1, 0, 0))
    assert not manager.is_entangled((0, 0, 0), (1, 0, 0)), "Should not be entangled"


def test_entanglement_statistics():
    """Test entanglement statistics."""
    manager = EntanglementManager(lattice_size=3)
    
    manager.create_bell_pair((0, 0, 0), (1, 0, 0))
    manager.create_bell_pair((0, 0, 0), (0, 1, 0))
    
    stats = manager.get_entanglement_statistics()
    
    assert 'total_entangled_pairs' in stats, "Should have pair count"
    assert 'max_connections_per_cell' in stats, "Should have max connections"
    assert 'entangled_cells' in stats, "Should have cell count"
    assert stats['total_entangled_pairs'] == 2, "Should have 2 pairs"


if __name__ == "__main__":
    print("Running EntanglementManager tests...")
    
    test_basic_initialization()
    print("✓ Basic initialization")
    
    test_create_bell_pair()
    print("✓ Create Bell pair")
    
    test_entangled_pair()
    print("✓ Entangled pair")
    
    test_entangle_by_distance()
    print("✓ Entangle by distance")
    
    test_entangle_by_face_exposure()
    print("✓ Entangle by face exposure")
    
    test_get_entangled_cells()
    print("✓ Get entangled cells")
    
    test_is_entangled()
    print("✓ Is entangled")
    
    test_break_entanglement()
    print("✓ Break entanglement")
    
    test_entanglement_statistics()
    print("✓ Entanglement statistics")
    
    print("\nAll tests passed! ✓")


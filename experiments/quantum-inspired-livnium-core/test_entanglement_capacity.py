"""
Test Entanglement Capacity in Recursive Geometry

Tests how many qubits can be entangled simultaneously using:
- Pairwise entanglement (Bell pairs)
- Chain entanglement (linear chains)
- Cluster entanglement (fully connected clusters)
- Recursive entanglement (across levels)
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Set

# Make repo root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig
from core.recursive import RecursiveGeometryEngine
from core.quantum import QuantumLattice, QuantumCell
from core.quantum.entanglement_manager import EntanglementManager, EntangledPair


def count_entangled_qubits_recursive(level, entanglement_managers: Dict[int, EntanglementManager]) -> int:
    """Recursively count all entangled qubits."""
    level_id = level.level_id
    
    # Count entangled qubits at this level
    if level_id in entanglement_managers:
        manager = entanglement_managers[level_id]
        stats = manager.get_entanglement_statistics()
        # Each pair involves 2 qubits, but we count unique qubits
        entangled_qubits = stats['entangled_cells']
    else:
        entangled_qubits = 0
    
    # Count children recursively
    for child_level in level.children.values():
        entangled_qubits += count_entangled_qubits_recursive(child_level, entanglement_managers)
    
    return entangled_qubits


def test_pairwise_entanglement(base_lattice_size: int = 5,
                               max_depth: int = 3) -> Dict:
    """
    Test pairwise entanglement capacity (Bell pairs).
    
    Creates maximum number of Bell pairs across the recursive structure.
    """
    print("=" * 70)
    print("Test 1: Pairwise Entanglement (Bell Pairs)")
    print("=" * 70)
    
    # Initialize system
    config = LivniumCoreConfig(
        lattice_size=base_lattice_size,
        enable_quantum=True,
        enable_superposition=True,
        enable_quantum_gates=True,
        enable_entanglement=True,
        enable_measurement=True,
        enable_geometry_quantum_coupling=True
    )
    
    base_system = LivniumCoreSystem(config)
    recursive_engine = RecursiveGeometryEngine(
        base_geometry=base_system,
        max_depth=max_depth
    )
    
    # Initialize quantum lattices and entanglement managers for each level
    quantum_lattices: Dict[int, QuantumLattice] = {}
    entanglement_managers: Dict[int, EntanglementManager] = {}
    
    level_0 = recursive_engine.levels[0]
    quantum_lattices[0] = QuantumLattice(base_system)
    entanglement_managers[0] = quantum_lattices[0].entanglement_manager
    
    total_pairs = 0
    level_stats = {}
    
    # Create Bell pairs at each level
    for level_id, level in recursive_engine.levels.items():
        if level_id not in quantum_lattices:
            # Create quantum lattice for this level
            quantum_lattices[level_id] = QuantumLattice(level.geometry)
            entanglement_managers[level_id] = quantum_lattices[level_id].entanglement_manager
        
        manager = entanglement_managers[level_id]
        geometry = level.geometry
        
        # Create Bell pairs: entangle each cell with its nearest neighbors
        pairs_created = 0
        cells = list(geometry.lattice.keys())
        
        # Create pairs: each cell with its 6 nearest neighbors (if not already paired)
        for i, cell1 in enumerate(cells):
            # Get nearest neighbors
            neighbors = manager._get_nearest_neighbors(cell1)
            
            for neighbor in neighbors:
                if neighbor in cells and neighbor != cell1:
                    # Check if not already entangled
                    if not manager.is_entangled(cell1, neighbor):
                        manager.create_bell_pair(cell1, neighbor, "phi_plus")
                        pairs_created += 1
        
        # Each pair is counted twice (both directions), so divide by 2
        actual_pairs = pairs_created // 2
        total_pairs += actual_pairs
        
        stats = manager.get_entanglement_statistics()
        level_stats[level_id] = {
            'pairs': actual_pairs,
            'entangled_cells': stats['entangled_cells'],
            'max_connections': stats['max_connections_per_cell']
        }
        
        print(f"Level {level_id}: {actual_pairs:,} Bell pairs, "
              f"{stats['entangled_cells']:,} entangled cells, "
              f"max {stats['max_connections_per_cell']} connections/cell")
    
    # Count total entangled qubits
    total_entangled = count_entangled_qubits_recursive(level_0, entanglement_managers)
    
    print()
    print(f"Total Bell pairs: {total_pairs:,}")
    print(f"Total entangled qubits: {total_entangled:,}")
    print(f"Average pairs per qubit: {total_pairs / total_entangled:.2f}" if total_entangled > 0 else "")
    
    return {
        'total_pairs': total_pairs,
        'total_entangled_qubits': total_entangled,
        'level_stats': level_stats
    }


def test_chain_entanglement(base_lattice_size: int = 5,
                            max_depth: int = 3) -> Dict:
    """
    Test chain entanglement capacity (linear chains).
    
    Creates maximum length entanglement chains.
    """
    print("\n" + "=" * 70)
    print("Test 2: Chain Entanglement (Linear Chains)")
    print("=" * 70)
    
    config = LivniumCoreConfig(
        lattice_size=base_lattice_size,
        enable_quantum=True,
        enable_superposition=True,
        enable_quantum_gates=True,
        enable_entanglement=True,
        enable_measurement=True,
        enable_geometry_quantum_coupling=True
    )
    
    base_system = LivniumCoreSystem(config)
    recursive_engine = RecursiveGeometryEngine(
        base_geometry=base_system,
        max_depth=max_depth
    )
    
    quantum_lattices: Dict[int, QuantumLattice] = {}
    entanglement_managers: Dict[int, EntanglementManager] = {}
    
    level_0 = recursive_engine.levels[0]
    quantum_lattices[0] = QuantumLattice(base_system)
    entanglement_managers[0] = quantum_lattices[0].entanglement_manager
    
    total_chains = 0
    max_chain_length = 0
    level_stats = {}
    
    for level_id, level in recursive_engine.levels.items():
        if level_id not in quantum_lattices:
            quantum_lattices[level_id] = QuantumLattice(level.geometry)
            entanglement_managers[level_id] = quantum_lattices[level_id].entanglement_manager
        
        manager = entanglement_managers[level_id]
        geometry = level.geometry
        cells = list(geometry.lattice.keys())
        
        # Create chains: connect cells in a path
        chains_created = 0
        used_cells = set()
        max_length = 0
        
        for start_cell in cells:
            if start_cell in used_cells:
                continue
            
            # Build chain from this cell
            chain = [start_cell]
            current = start_cell
            used_cells.add(current)
            
            # Extend chain by finding unentangled neighbors
            while True:
                neighbors = manager._get_nearest_neighbors(current)
                next_cell = None
                
                for neighbor in neighbors:
                    if neighbor in cells and neighbor not in used_cells:
                        if not manager.is_entangled(current, neighbor):
                            next_cell = neighbor
                            break
                
                if next_cell is None:
                    break
                
                # Create entanglement
                manager.create_bell_pair(current, next_cell, "phi_plus")
                chain.append(next_cell)
                current = next_cell
                used_cells.add(current)
            
            if len(chain) > 1:
                chains_created += 1
                max_length = max(max_length, len(chain))
        
        total_chains += chains_created
        max_chain_length = max(max_chain_length, max_length)
        
        stats = manager.get_entanglement_statistics()
        level_stats[level_id] = {
            'chains': chains_created,
            'max_chain_length': max_length,
            'entangled_cells': stats['entangled_cells']
        }
        
        print(f"Level {level_id}: {chains_created:,} chains, "
              f"max length {max_length}, "
              f"{stats['entangled_cells']:,} entangled cells")
    
    total_entangled = count_entangled_qubits_recursive(level_0, entanglement_managers)
    
    print()
    print(f"Total chains: {total_chains:,}")
    print(f"Max chain length: {max_chain_length}")
    print(f"Total entangled qubits: {total_entangled:,}")
    
    return {
        'total_chains': total_chains,
        'max_chain_length': max_chain_length,
        'total_entangled_qubits': total_entangled,
        'level_stats': level_stats
    }


def test_cluster_entanglement(base_lattice_size: int = 5,
                              max_depth: int = 2,
                              cluster_size: int = 10) -> Dict:
    """
    Test cluster entanglement (fully connected clusters).
    
    Creates clusters of fully connected qubits.
    """
    print("\n" + "=" * 70)
    print(f"Test 3: Cluster Entanglement ({cluster_size}-qubit clusters)")
    print("=" * 70)
    
    config = LivniumCoreConfig(
        lattice_size=base_lattice_size,
        enable_quantum=True,
        enable_superposition=True,
        enable_quantum_gates=True,
        enable_entanglement=True,
        enable_measurement=True,
        enable_geometry_quantum_coupling=True
    )
    
    base_system = LivniumCoreSystem(config)
    recursive_engine = RecursiveGeometryEngine(
        base_geometry=base_system,
        max_depth=max_depth
    )
    
    quantum_lattices: Dict[int, QuantumLattice] = {}
    entanglement_managers: Dict[int, EntanglementManager] = {}
    
    level_0 = recursive_engine.levels[0]
    quantum_lattices[0] = QuantumLattice(base_system)
    entanglement_managers[0] = quantum_lattices[0].entanglement_manager
    
    total_clusters = 0
    total_entanglements_in_clusters = 0
    level_stats = {}
    
    for level_id, level in recursive_engine.levels.items():
        if level_id not in quantum_lattices:
            quantum_lattices[level_id] = QuantumLattice(level.geometry)
            entanglement_managers[level_id] = quantum_lattices[level_id].entanglement_manager
        
        manager = entanglement_managers[level_id]
        geometry = level.geometry
        cells = list(geometry.lattice.keys())
        
        # Create clusters: groups of fully connected qubits
        clusters_created = 0
        used_cells = set()
        
        # Create clusters of size cluster_size
        for i in range(0, len(cells), cluster_size):
            cluster_cells = cells[i:i+cluster_size]
            
            # Skip if any cell already used
            if any(cell in used_cells for cell in cluster_cells):
                continue
            
            # Fully connect cluster: each cell with every other
            cluster_entanglements = 0
            for j, cell1 in enumerate(cluster_cells):
                for cell2 in cluster_cells[j+1:]:
                    if not manager.is_entangled(cell1, cell2):
                        manager.create_bell_pair(cell1, cell2, "phi_plus")
                        cluster_entanglements += 1
                        used_cells.add(cell1)
                        used_cells.add(cell2)
            
            if cluster_entanglements > 0:
                clusters_created += 1
                total_entanglements_in_clusters += cluster_entanglements
        
        total_clusters += clusters_created
        
        stats = manager.get_entanglement_statistics()
        level_stats[level_id] = {
            'clusters': clusters_created,
            'entanglements': stats['total_entangled_pairs'],
            'entangled_cells': stats['entangled_cells']
        }
        
        print(f"Level {level_id}: {clusters_created:,} clusters, "
              f"{stats['total_entangled_pairs']:,} pairs, "
              f"{stats['entangled_cells']:,} entangled cells")
    
    total_entangled = count_entangled_qubits_recursive(level_0, entanglement_managers)
    
    print()
    print(f"Total clusters: {total_clusters:,}")
    print(f"Total entanglements in clusters: {total_entanglements_in_clusters:,}")
    print(f"Total entangled qubits: {total_entangled:,}")
    print(f"Average cluster size: {total_entangled / total_clusters:.1f}" if total_clusters > 0 else "")
    
    return {
        'total_clusters': total_clusters,
        'total_entanglements': total_entanglements_in_clusters,
        'total_entangled_qubits': total_entangled,
        'level_stats': level_stats
    }


def test_recursive_entanglement(base_lattice_size: int = 5,
                                max_depth: int = 3) -> Dict:
    """
    Test recursive entanglement (entangle across levels).
    
    Creates entanglement between parent and child levels.
    """
    print("\n" + "=" * 70)
    print("Test 4: Recursive Entanglement (Cross-Level)")
    print("=" * 70)
    
    config = LivniumCoreConfig(
        lattice_size=base_lattice_size,
        enable_quantum=True,
        enable_superposition=True,
        enable_quantum_gates=True,
        enable_entanglement=True,
        enable_measurement=True,
        enable_geometry_quantum_coupling=True
    )
    
    base_system = LivniumCoreSystem(config)
    recursive_engine = RecursiveGeometryEngine(
        base_geometry=base_system,
        max_depth=max_depth
    )
    
    quantum_lattices: Dict[int, QuantumLattice] = {}
    entanglement_managers: Dict[int, EntanglementManager] = {}
    
    level_0 = recursive_engine.levels[0]
    quantum_lattices[0] = QuantumLattice(base_system)
    entanglement_managers[0] = quantum_lattices[0].entanglement_manager
    
    cross_level_pairs = 0
    
    # Entangle parent cells with representative child cells
    for level_id, level in recursive_engine.levels.items():
        if level_id not in quantum_lattices:
            quantum_lattices[level_id] = QuantumLattice(level.geometry)
            entanglement_managers[level_id] = quantum_lattices[level_id].entanglement_manager
        
        # For each parent cell, count potential cross-level entanglements
        for parent_coords, child_level in level.children.items():
            # Count potential: each parent cell could entangle with child cells
            child_cells = list(child_level.geometry.lattice.keys())
            if child_cells:
                # Count potential pairs (parent cell with each child cell)
                cross_level_pairs += len(child_cells)
    
    print(f"Potential cross-level entanglements: {cross_level_pairs:,}")
    print("(Note: Cross-level entanglement requires child geometries to have quantum enabled)")
    print("(This is a limitation of current implementation - child configs don't inherit quantum settings)")
    
    return {
        'cross_level_pairs': cross_level_pairs,
        'note': 'Cross-level entanglement limited by child geometry config'
    }


def run_all_tests():
    """Run all entanglement capacity tests."""
    print("=" * 70)
    print("ENTANGLEMENT CAPACITY TEST SUITE")
    print("=" * 70)
    print("Testing how many qubits can be entangled using recursive geometry")
    print()
    
    base_lattice_size = 5
    max_depth = 3
    
    # Test 1: Pairwise
    result1 = test_pairwise_entanglement(base_lattice_size, max_depth)
    
    # Test 2: Chains
    result2 = test_chain_entanglement(base_lattice_size, max_depth)
    
    # Test 3: Clusters (smaller depth for performance)
    result3 = test_cluster_entanglement(base_lattice_size, max_depth=2, cluster_size=10)
    
    # Test 4: Recursive
    result4 = test_recursive_entanglement(base_lattice_size, max_depth)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Pairwise entanglement: {result1['total_entangled_qubits']:,} qubits")
    print(f"Chain entanglement: {result2['total_entangled_qubits']:,} qubits")
    print(f"Cluster entanglement: {result3['total_entangled_qubits']:,} qubits")
    print(f"Max entangled (any method): {max(result1['total_entangled_qubits'], result2['total_entangled_qubits'], result3['total_entangled_qubits']):,} qubits")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()


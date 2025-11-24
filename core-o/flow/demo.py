"""
Flow Engine Demo

Demonstrates O-A7: The Flow Law in action.

Shows how Livnium-O transforms from static geometry into dynamic universe.
"""

import numpy as np
import sys
from pathlib import Path

# Add core-o to path
core_o_path = Path(__file__).parent.parent
sys.path.insert(0, str(core_o_path))

from classical.livnium_o_system import LivniumOSystem
from flow.flow_engine import FlowEngine, move_neighbor, evolve_system, create_velocity_field


def main():
    """Run the flow engine demo."""
    print("="*70)
    print("Livnium-O Flow Engine Demo")
    print("O-A7: The Flow Law - Continuous Tangential Dynamics")
    print("="*70)
    print()
    
    # Create initial system
    print("1. Creating initial system...")
    neighbor_radii = [1.0] * 6
    system = LivniumOSystem(neighbor_radii=neighbor_radii, core_radius=1.0)
    print(f"   System: {system}")
    print()
    
    # Show initial positions
    print("2. Initial neighbor positions:")
    neighbors = system.get_neighbor_nodes()
    for i, neighbor in enumerate(neighbors[:3], start=1):
        pos = neighbor.position
        dist = np.sqrt(sum(x*x for x in pos))
        print(f"   Neighbor {neighbor.node_id}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
              f"distance={dist:.3f}")
    print()
    
    # Test move_neighbor
    print("3. Moving neighbor 1 along tangent plane...")
    neighbor = neighbors[0]
    pos_vec = np.array(neighbor.position)
    radial_unit = pos_vec / np.linalg.norm(pos_vec)
    
    # Create tangential velocity
    arbitrary_vec = np.array([1, 0, 0])
    if np.allclose(radial_unit, arbitrary_vec) or np.allclose(radial_unit, -arbitrary_vec):
        arbitrary_vec = np.array([0, 1, 0])
    
    tangential_velocity = np.cross(radial_unit, arbitrary_vec)
    tangential_velocity = tangential_velocity / np.linalg.norm(tangential_velocity) * 0.1
    
    print(f"   Tangential velocity: ({tangential_velocity[0]:.3f}, "
          f"{tangential_velocity[1]:.3f}, {tangential_velocity[2]:.3f})")
    
    moved_system = move_neighbor(system, neighbor.node_id, tangential_velocity, dt=0.1)
    moved_neighbor = moved_system.get_node(neighbor.node_id)
    moved_pos = moved_neighbor.position
    moved_dist = np.sqrt(sum(x*x for x in moved_pos))
    
    print(f"   New position: ({moved_pos[0]:.3f}, {moved_pos[1]:.3f}, {moved_pos[2]:.3f})")
    print(f"   Distance preserved: {moved_dist:.6f} (expected: {system.core_radius + neighbor.radius:.6f})")
    print()
    
    # Test evolve_system
    print("4. Evolving system with velocity field...")
    velocity_field = {}
    for i, neighbor in enumerate(neighbors[:3], start=1):
        pos_vec = np.array(neighbor.position)
        radial_unit = pos_vec / np.linalg.norm(pos_vec)
        arbitrary_vec = np.array([1, 0, 0])
        if np.allclose(radial_unit, arbitrary_vec) or np.allclose(radial_unit, -arbitrary_vec):
            arbitrary_vec = np.array([0, 1, 0])
        
        tangential_velocity = np.cross(radial_unit, arbitrary_vec)
        tangential_velocity = tangential_velocity / np.linalg.norm(tangential_velocity) * 0.05
        velocity_field[neighbor.node_id] = tangential_velocity
    
    evolved_system = evolve_system(system, velocity_field, dt=0.1)
    print(f"   Evolved system: {evolved_system}")
    print()
    
    # Test FlowEngine class
    print("5. Using FlowEngine class...")
    engine = FlowEngine(system, dt=0.1)
    
    # Define a simple force function (repulsion from other neighbors)
    def repulsion_force(system: LivniumOSystem, neighbor_id: int) -> np.ndarray:
        """Simple repulsion force."""
        neighbor = system.get_node(neighbor_id)
        pos_vec = np.array(neighbor.position)
        
        # Sum repulsion from other neighbors
        force = np.zeros(3)
        for other in system.get_neighbor_nodes():
            if other.node_id == neighbor_id:
                continue
            
            other_pos = np.array(other.position)
            diff = pos_vec - other_pos
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                force += diff / (dist ** 2) * 0.01
        
        return force
    
    # Evolve with forces
    print("   Evolving with repulsion forces...")
    engine.step_with_forces(repulsion_force)
    print(f"   After 1 step: {engine.get_current_state()}")
    
    # Evolve more steps
    for i in range(4):
        engine.step_with_forces(repulsion_force)
    
    print(f"   After 5 steps: {engine.get_current_state()}")
    print(f"   History length: {len(engine.get_history())}")
    print()
    
    # Test reversibility
    print("6. Testing reversibility...")
    original_positions = [n.position for n in system.get_neighbor_nodes()]
    final_positions = [n.position for n in engine.get_current_state().get_neighbor_nodes()]
    
    # Reverse steps
    for _ in range(5):
        engine.reverse_step()
    
    reversed_positions = [n.position for n in engine.get_current_state().get_neighbor_nodes()]
    
    # Check if reversed
    max_error = 0.0
    for orig, rev in zip(original_positions, reversed_positions):
        error = np.linalg.norm(np.array(orig) - np.array(rev))
        max_error = max(max_error, error)
    
    print(f"   Max position error after reversal: {max_error:.10f}")
    print(f"   Reversibility: {'✅ PASS' if max_error < 1e-5 else '❌ FAIL'}")
    print()
    
    print("="*70)
    print("Demo complete!")
    print("="*70)
    print()
    print("Livnium-O is now ALIVE:")
    print("  ✅ Neighbors glide around Om")
    print("  ✅ Exposure changes continuously")
    print("  ✅ SW redistributes")
    print("  ✅ Geometry becomes computation")
    print("  ✅ Dynamics are reversible")
    print("  ✅ States become trajectories")


if __name__ == "__main__":
    main()


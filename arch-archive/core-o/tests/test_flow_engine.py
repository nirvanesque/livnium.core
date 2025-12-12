"""
Test suite for Flow Engine (O-A7)

Tests the dynamic flow layer that makes Livnium-O alive.
"""

import unittest
import sys
from pathlib import Path
import math

# Add core-o directory to path
core_o_path = Path(__file__).parent.parent
sys.path.insert(0, str(core_o_path))

import numpy as np

from classical.livnium_o_system import LivniumOSystem
from flow.flow_engine import (
    FlowEngine,
    move_neighbor,
    evolve_system,
    compute_tangential_velocity,
    create_velocity_field,
)


class TestFlowEngine(unittest.TestCase):
    """Test suite for Flow Engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.neighbor_radii = [1.0] * 6
        self.system = LivniumOSystem(neighbor_radii=self.neighbor_radii, core_radius=1.0)
    
    def test_compute_tangential_velocity(self):
        """Test tangential velocity projection."""
        pos_vec = np.array([2.0, 0.0, 0.0])
        velocity = np.array([0.1, 0.2, 0.3])
        
        tangential = compute_tangential_velocity(pos_vec, velocity)
        
        # Should be perpendicular to radial
        dot_product = np.dot(tangential, pos_vec)
        self.assertAlmostEqual(dot_product, 0.0, places=10)
        
        # Should have no radial component
        radial_unit = pos_vec / np.linalg.norm(pos_vec)
        radial_component = np.dot(tangential, radial_unit)
        self.assertAlmostEqual(radial_component, 0.0, places=10)
    
    def test_move_neighbor_preserves_tangency(self):
        """Test move_neighbor preserves tangency."""
        neighbor = self.system.get_neighbor_nodes()[0]
        neighbor_id = neighbor.node_id
        
        # Create tangential velocity
        pos_vec = np.array(neighbor.position)
        radial_unit = pos_vec / np.linalg.norm(pos_vec)
        arbitrary_vec = np.array([1, 0, 0])
        if np.allclose(radial_unit, arbitrary_vec) or np.allclose(radial_unit, -arbitrary_vec):
            arbitrary_vec = np.array([0, 1, 0])
        
        tangential_velocity = np.cross(radial_unit, arbitrary_vec)
        tangential_velocity = tangential_velocity / np.linalg.norm(tangential_velocity) * 0.1
        
        moved_system = move_neighbor(self.system, neighbor_id, tangential_velocity, dt=0.1)
        moved_neighbor = moved_system.get_node(neighbor_id)
        
        # Verify tangency
        distance = math.sqrt(sum(x*x for x in moved_neighbor.position))
        expected_distance = self.system.core_radius + neighbor.radius
        self.assertAlmostEqual(distance, expected_distance, places=6)
    
    def test_evolve_system(self):
        """Test evolve_system moves multiple neighbors."""
        neighbors = self.system.get_neighbor_nodes()
        
        # Create velocity field
        velocity_field = {}
        for neighbor in neighbors[:3]:
            pos_vec = np.array(neighbor.position)
            radial_unit = pos_vec / np.linalg.norm(pos_vec)
            arbitrary_vec = np.array([1, 0, 0])
            if np.allclose(radial_unit, arbitrary_vec) or np.allclose(radial_unit, -arbitrary_vec):
                arbitrary_vec = np.array([0, 1, 0])
            
            tangential_velocity = np.cross(radial_unit, arbitrary_vec)
            tangential_velocity = tangential_velocity / np.linalg.norm(tangential_velocity) * 0.1
            velocity_field[neighbor.node_id] = tangential_velocity
        
        evolved_system = evolve_system(self.system, velocity_field, dt=0.1)
        
        # Verify neighbors moved
        for neighbor_id in velocity_field.keys():
            original_pos = np.array(self.system.get_node(neighbor_id).position)
            evolved_pos = np.array(evolved_system.get_node(neighbor_id).position)
            distance_change = np.linalg.norm(evolved_pos - original_pos)
            self.assertGreater(distance_change, 1e-6)
    
    def test_flow_engine_step(self):
        """Test FlowEngine.step()."""
        engine = FlowEngine(self.system, dt=0.1)
        
        # Create velocity field
        neighbor = self.system.get_neighbor_nodes()[0]
        pos_vec = np.array(neighbor.position)
        radial_unit = pos_vec / np.linalg.norm(pos_vec)
        arbitrary_vec = np.array([1, 0, 0])
        if np.allclose(radial_unit, arbitrary_vec) or np.allclose(radial_unit, -arbitrary_vec):
            arbitrary_vec = np.array([0, 1, 0])
        
        tangential_velocity = np.cross(radial_unit, arbitrary_vec)
        tangential_velocity = tangential_velocity / np.linalg.norm(tangential_velocity) * 0.1
        
        velocity_field = {neighbor.node_id: tangential_velocity}
        
        # Step forward
        engine.step(velocity_field)
        
        # Verify system changed
        self.assertNotEqual(id(engine.get_current_state()), id(self.system))
        self.assertEqual(len(engine.get_history()), 2)
    
    def test_flow_engine_with_forces(self):
        """Test FlowEngine with force function."""
        engine = FlowEngine(self.system, dt=0.1)
        
        def simple_force(system: LivniumOSystem, neighbor_id: int) -> np.ndarray:
            """Simple force pointing in +x direction."""
            return np.array([0.1, 0.0, 0.0])
        
        # Step with forces
        engine.step_with_forces(simple_force)
        
        # Verify system evolved
        self.assertEqual(len(engine.get_history()), 2)
    
    def test_flow_engine_reversibility(self):
        """Test FlowEngine reversibility."""
        engine = FlowEngine(self.system, dt=0.1)
        
        # Create velocity field
        neighbor = self.system.get_neighbor_nodes()[0]
        pos_vec = np.array(neighbor.position)
        radial_unit = pos_vec / np.linalg.norm(pos_vec)
        arbitrary_vec = np.array([1, 0, 0])
        if np.allclose(radial_unit, arbitrary_vec) or np.allclose(radial_unit, -arbitrary_vec):
            arbitrary_vec = np.array([0, 1, 0])
        
        tangential_velocity = np.cross(radial_unit, arbitrary_vec)
        tangential_velocity = tangential_velocity / np.linalg.norm(tangential_velocity) * 0.1
        
        velocity_field = {neighbor.node_id: tangential_velocity}
        
        # Step forward
        engine.step(velocity_field)
        forward_state = engine.get_current_state()
        
        # Reverse
        engine.reverse_step()
        reversed_state = engine.get_current_state()
        
        # Should return to original
        original_positions = [n.position for n in self.system.get_neighbor_nodes()]
        reversed_positions = [n.position for n in reversed_state.get_neighbor_nodes()]
        
        for orig, rev in zip(original_positions, reversed_positions):
            error = np.linalg.norm(np.array(orig) - np.array(rev))
            self.assertLess(error, 1e-5)
    
    def test_flow_engine_reset(self):
        """Test FlowEngine.reset()."""
        engine = FlowEngine(self.system, dt=0.1)
        
        # Evolve
        neighbor = self.system.get_neighbor_nodes()[0]
        pos_vec = np.array(neighbor.position)
        radial_unit = pos_vec / np.linalg.norm(pos_vec)
        arbitrary_vec = np.array([1, 0, 0])
        if np.allclose(radial_unit, arbitrary_vec) or np.allclose(radial_unit, -arbitrary_vec):
            arbitrary_vec = np.array([0, 1, 0])
        
        tangential_velocity = np.cross(radial_unit, arbitrary_vec)
        tangential_velocity = tangential_velocity / np.linalg.norm(tangential_velocity) * 0.1
        
        velocity_field = {neighbor.node_id: tangential_velocity}
        
        for _ in range(5):
            engine.step(velocity_field)
        
        self.assertEqual(len(engine.get_history()), 6)
        
        # Reset
        engine.reset()
        
        self.assertEqual(len(engine.get_history()), 1)
        self.assertEqual(id(engine.get_current_state()), id(self.system))
    
    def test_create_velocity_field_with_forces(self):
        """Test create_velocity_field with force function."""
        def force_function(system: LivniumOSystem, neighbor_id: int) -> np.ndarray:
            return np.array([0.1, 0.1, 0.1])
        
        velocity_field = create_velocity_field(self.system, force_function=force_function)
        
        # Should have velocities for all neighbors
        self.assertGreater(len(velocity_field), 0)
        
        # All velocities should be tangential
        for neighbor_id, velocity in velocity_field.items():
            neighbor = self.system.get_node(neighbor_id)
            pos_vec = np.array(neighbor.position)
            dot_product = np.dot(velocity, pos_vec)
            self.assertAlmostEqual(dot_product, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()


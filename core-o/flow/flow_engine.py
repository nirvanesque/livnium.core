"""
Flow Engine: O-A7 Implementation

The Flow Law transforms Livnium-O from static geometry into dynamic universe.

Core Functions:
- move_neighbor(): Move one neighbor along tangent plane
- evolve_system(): Evolve entire system forward in time
- compute_tangential_velocity(): Project velocity onto tangent plane
- create_velocity_field(): Generate velocity field from forces/gradients

This is where:
- search happens
- optimization happens
- meaning flows
- collapse occurs
- baseline flow emerges
- semantic transitions propagate
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Callable
import sys
from pathlib import Path

# Add core-o to path
core_o_path = Path(__file__).parent.parent
sys.path.insert(0, str(core_o_path))

from classical.livnium_o_system import LivniumOSystem, SphericalRotationGroup


def compute_tangential_velocity(
    position: np.ndarray,
    velocity: np.ndarray
) -> np.ndarray:
    """
    Project velocity onto tangent plane (O-A7 constraint).
    
    Ensures: v_tangential · (N - Om) = 0
    
    Args:
        position: 3D position vector (N - Om)
        velocity: 3D velocity vector (may have radial component)
        
    Returns:
        Tangential velocity vector (perpendicular to radial)
    """
    radial_vec = position
    radial_norm = np.linalg.norm(radial_vec)
    
    if radial_norm < 1e-10:
        # At origin, return zero velocity
        return np.zeros(3)
    
    radial_unit = radial_vec / radial_norm
    
    # Project velocity onto tangent plane
    # v_tangential = v - (v · r̂) r̂
    radial_component = np.dot(velocity, radial_unit)
    tangential_velocity = velocity - radial_component * radial_unit
    
    return tangential_velocity


def move_neighbor(
    system: LivniumOSystem,
    neighbor_id: int,
    tangential_velocity: np.ndarray,
    dt: float = 0.01
) -> LivniumOSystem:
    """
    Move a neighbor along the tangent plane (O-A7: The Flow Law).
    
    The neighbor moves only along the tangent plane, preserving tangency:
    v_i(t) · (N_i - Om) = 0
    
    Update rule:
    N_i(t + Δt) = Om + (1+r_i) · R_i(Δt) · û_i(t)
    
    Args:
        system: Current Livnium-O system
        neighbor_id: ID of neighbor to move (1-based)
        tangential_velocity: 3D velocity vector (will be projected if needed)
        dt: Time step
        
    Returns:
        New system with moved neighbor
    """
    if neighbor_id < 1 or neighbor_id > system.n_neighbors:
        raise ValueError(f"Neighbor ID must be in [1, {system.n_neighbors}], got {neighbor_id}")
    
    neighbor = system.get_node(neighbor_id)
    if neighbor.is_om:
        raise ValueError("Cannot move core (Om)")
    
    # Get current position relative to core
    pos_vec = np.array(neighbor.position)
    
    # Ensure velocity is tangential
    tangential_velocity = compute_tangential_velocity(pos_vec, tangential_velocity)
    
    # Current distance from core
    current_distance = np.linalg.norm(pos_vec)
    expected_distance = system.core_radius + neighbor.radius
    
    # Normalize position to ensure correct distance
    if abs(current_distance - expected_distance) > 1e-10:
        pos_vec = pos_vec * (expected_distance / current_distance)
    
    # Compute incremental rotation
    velocity_norm = np.linalg.norm(tangential_velocity)
    if velocity_norm < 1e-10:
        # No movement
        return system
    
    # Rotation axis is perpendicular to both radial and velocity
    rotation_axis = np.cross(pos_vec, tangential_velocity)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    
    if rotation_axis_norm < 1e-10:
        # Velocity is parallel to radial (shouldn't happen after projection)
        return system
    
    rotation_axis = rotation_axis / rotation_axis_norm
    
    # Rotation angle: move by |v| * dt along sphere surface
    rotation_angle = (velocity_norm * dt) / expected_distance
    
    # Create incremental SO(3) rotation matrix
    rotation_group = system.rotation_group
    rotation_matrix = rotation_group.rotation_matrix_axis_angle(rotation_axis, rotation_angle)
    
    # Apply rotation to position
    new_pos = rotation_matrix @ pos_vec
    
    # Ensure distance is preserved
    new_distance = np.linalg.norm(new_pos)
    if abs(new_distance - expected_distance) > 1e-10:
        new_pos = new_pos * (expected_distance / new_distance)
    
    # Create new system with updated position
    neighbor_radii = [system.nodes[i].radius for i in range(1, system.n_neighbors + 1)]
    positions = [system.nodes[i].position for i in range(1, system.n_neighbors + 1)]
    positions[neighbor_id - 1] = tuple(new_pos)
    
    new_system = LivniumOSystem(
        neighbor_radii=neighbor_radii,
        core_radius=system.core_radius,
        positions=positions
    )
    
    # Verify ledger is preserved
    new_system._verify_ledger()
    
    return new_system


def evolve_system(
    system: LivniumOSystem,
    velocity_field: Dict[int, np.ndarray],
    dt: float = 0.01
) -> LivniumOSystem:
    """
    Evolve the system according to O-A7: The Flow Law.
    
    All neighbors move along their tangent planes simultaneously.
    
    This is where:
    - search happens
    - optimization happens
    - meaning flows
    - collapse occurs
    - baseline flow emerges
    
    Args:
        system: Current Livnium-O system
        velocity_field: Dictionary mapping neighbor_id -> velocity vector
        dt: Time step
        
    Returns:
        New system after evolution
    """
    current_system = system
    
    # Move each neighbor in the velocity field
    for neighbor_id, velocity in velocity_field.items():
        if neighbor_id < 1 or neighbor_id > current_system.n_neighbors:
            continue
        current_system = move_neighbor(current_system, neighbor_id, velocity, dt)
    
    return current_system


def create_velocity_field(
    system: LivniumOSystem,
    force_function: Optional[Callable[[LivniumOSystem, int], np.ndarray]] = None,
    gradient_function: Optional[Callable[[LivniumOSystem, int], np.ndarray]] = None,
    reward_function: Optional[Callable[[LivniumOSystem, int], float]] = None
) -> Dict[int, np.ndarray]:
    """
    Create velocity field from forces, gradients, or rewards.
    
    This is the bridge between optimization/search and geometry.
    
    Args:
        system: Current Livnium-O system
        force_function: Function (system, neighbor_id) -> force vector
        gradient_function: Function (system, neighbor_id) -> gradient vector
        reward_function: Function (system, neighbor_id) -> reward scalar
        
    Returns:
        Dictionary mapping neighbor_id -> velocity vector
    """
    velocity_field = {}
    
    for neighbor in system.get_neighbor_nodes():
        neighbor_id = neighbor.node_id
        velocity = np.zeros(3)
        
        # Apply force function if provided
        if force_function is not None:
            force = force_function(system, neighbor_id)
            velocity += force
        
        # Apply gradient function if provided
        if gradient_function is not None:
            gradient = gradient_function(system, neighbor_id)
            velocity += gradient
        
        # Apply reward function if provided (convert scalar to direction)
        if reward_function is not None:
            reward = reward_function(system, neighbor_id)
            # Convert reward to velocity direction
            pos_vec = np.array(neighbor.position)
            radial_unit = pos_vec / np.linalg.norm(pos_vec)
            # Create perpendicular direction
            if abs(radial_unit[0]) < 0.9:
                perp_vec = np.array([1, 0, 0])
            else:
                perp_vec = np.array([0, 1, 0])
            reward_direction = np.cross(radial_unit, perp_vec)
            reward_direction = reward_direction / np.linalg.norm(reward_direction)
            velocity += reward * reward_direction
        
        # Project onto tangent plane
        if np.linalg.norm(velocity) > 1e-10:
            velocity = compute_tangential_velocity(np.array(neighbor.position), velocity)
            velocity_field[neighbor_id] = velocity
    
    return velocity_field


class FlowEngine:
    """
    Flow Engine: Complete dynamics system for Livnium-O.
    
    Implements O-A7: The Flow Law with full support for:
    - Tangential motion
    - SO(3) rotations
    - Force fields
    - Gradient descent
    - Reward maximization
    - Reversible evolution
    """
    
    def __init__(self, system: LivniumOSystem, dt: float = 0.01):
        """
        Initialize flow engine.
        
        Args:
            system: Initial Livnium-O system
            dt: Default time step
        """
        self.system = system
        self.dt = dt
        self.history: List[LivniumOSystem] = [system]
        self.rotation_group = system.rotation_group
    
    def step(self, velocity_field: Dict[int, np.ndarray], dt: Optional[float] = None) -> 'FlowEngine':
        """
        Evolve system one step forward.
        
        Args:
            velocity_field: Dictionary mapping neighbor_id -> velocity
            dt: Time step (uses default if None)
            
        Returns:
            Self (for chaining)
        """
        if dt is None:
            dt = self.dt
        
        self.system = evolve_system(self.system, velocity_field, dt)
        self.history.append(self.system)
        return self
    
    def step_with_forces(
        self,
        force_function: Callable[[LivniumOSystem, int], np.ndarray],
        dt: Optional[float] = None
    ) -> 'FlowEngine':
        """
        Evolve system using force function.
        
        Args:
            force_function: Function (system, neighbor_id) -> force vector
            dt: Time step (uses default if None)
            
        Returns:
            Self (for chaining)
        """
        velocity_field = create_velocity_field(self.system, force_function=force_function)
        return self.step(velocity_field, dt)
    
    def step_with_gradient(
        self,
        gradient_function: Callable[[LivniumOSystem, int], np.ndarray],
        dt: Optional[float] = None
    ) -> 'FlowEngine':
        """
        Evolve system using gradient function.
        
        Args:
            gradient_function: Function (system, neighbor_id) -> gradient vector
            dt: Time step (uses default if None)
            
        Returns:
            Self (for chaining)
        """
        velocity_field = create_velocity_field(self.system, gradient_function=gradient_function)
        return self.step(velocity_field, dt)
    
    def step_with_reward(
        self,
        reward_function: Callable[[LivniumOSystem, int], float],
        dt: Optional[float] = None
    ) -> 'FlowEngine':
        """
        Evolve system using reward function.
        
        Args:
            reward_function: Function (system, neighbor_id) -> reward scalar
            dt: Time step (uses default if None)
            
        Returns:
            Self (for chaining)
        """
        velocity_field = create_velocity_field(self.system, reward_function=reward_function)
        return self.step(velocity_field, dt)
    
    def reverse_step(self) -> 'FlowEngine':
        """
        Reverse last step (if history available).
        
        Returns:
            Self (for chaining)
        """
        if len(self.history) > 1:
            self.history.pop()
            self.system = self.history[-1]
        return self
    
    def reset(self) -> 'FlowEngine':
        """
        Reset to initial system.
        
        Returns:
            Self (for chaining)
        """
        if len(self.history) > 0:
            self.system = self.history[0]
            self.history = [self.system]
        return self
    
    def get_current_state(self) -> LivniumOSystem:
        """Get current system state."""
        return self.system
    
    def get_history(self) -> List[LivniumOSystem]:
        """Get evolution history."""
        return self.history.copy()


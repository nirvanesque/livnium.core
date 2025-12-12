"""
Livnium Geometric Laws: The Physics

Derives forces purely from geometric gradients.
Implements:
1. Soft Repulsion (Pauli Exclusion)
2. Geometric Gravity (Inward Fall via SW maximization)
3. Smooth Differentiable Kernels
"""

import numpy as np
from typing import Tuple


class GeometricPotential:
    """
    The 'Laws of Physics' for Livnium.
    
    Derives forces purely from geometric gradients.
    Implements the Unified Principle: Minimizing Geometric Stress.
    """
    
    def __init__(
        self,
        radius: float = 1.0,
        k_repulsion: float = 500.0,
        k_gravity: float = 2.0,
        sw_target: float = 12.0
    ):
        """
        Initialize geometric potential laws.
        
        Args:
            radius: Sphere radius (R)
            k_repulsion: Stiffness of spheres (repulsion strength)
            k_gravity: Strength of the "Inward Fall" (density attraction)
            sw_target: Ideal density (target SW value, e.g., 12 neighbors)
        """
        self.R = radius
        self.D = 2.0 * radius  # Diameter (hard core)
        self.R_cut = 3.0 * self.D  # Gravity range (3 diameters)
        
        # Tuning Constants
        self.k_rep = k_repulsion    # Stiffness of spheres
        self.k_grav = k_gravity     # Strength of the "Inward Fall"
        self.sw_target = sw_target  # Ideal density (e.g., 12 neighbors)
    
    def _kernel_smooth_step(self, r: float) -> Tuple[float, float]:
        """
        A smooth differentiable kernel for SW calculation.
        
        Returns: 
            (value, derivative)
            
        Logic:
            1.0 when touching (r=D)
            0.0 at cutoff (r=R_cut)
            Uses cubic spline interpolation for smoothness.
        """
        if r >= self.R_cut:
            return 0.0, 0.0
        
        # Clamp distance for kernel calculation to avoid singularities inside spheres
        # We treat r < D as having max influence (1.0)
        eff_r = max(r, self.D)
        
        # Normalized distance x in [0, 1]
        dist_range = self.R_cut - self.D
        x = (eff_r - self.D) / dist_range
        
        # Cubic spline (smooth falloff): (1 - x)^3
        # This guarantees value=1 at x=0, value=0 at x=1
        # And derivative=0 at x=1 (smooth landing)
        val = (1.0 - x)**3
        
        # Derivative d(val)/dr = d(val)/dx * dx/dr
        # d(val)/dx = -3(1-x)^2
        # dx/dr = 1 / dist_range
        deriv = -3.0 * (1.0 - x)**2 / dist_range
        
        return val, deriv
    
    def compute_forces(
        self,
        positions: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Computes Potentials and Forces for the N-body system.
        
        Returns:
            forces (N,3), total_potential_energy, sw_values (N,)
        """
        N = len(positions)
        forces = np.zeros((N, 3))
        potential_energy = 0.0
        
        # Array to store current SW for each sphere
        current_sw = np.zeros(N)
        
        # Temporary storage for pairwise data to avoid re-calculating dists
        # (i, j, r_hat, kernel_deriv)
        pairs = []
        
        # --- PASS 1: Calculate Distances, Repulsion, and SW ---
        for i in range(N):
            for j in range(i + 1, N):
                r_vec = positions[i] - positions[j]
                dist = np.linalg.norm(r_vec)
                
                if dist == 0:
                    continue
                
                r_hat = r_vec / dist  # Direction j -> i
                
                # A. Repulsion (Hard Constraint)
                # V = 0.5 * k * overlap^2
                if dist < self.D:
                    overlap = self.D - dist
                    potential_energy += 0.5 * self.k_rep * overlap**2
                    
                    # F = - grad V = k * overlap * r_hat
                    f_push = self.k_rep * overlap * r_hat
                    forces[i] += f_push
                    forces[j] -= f_push
                
                # B. SW Density Accumulation
                if dist < self.R_cut:
                    val, deriv = self._kernel_smooth_step(dist)
                    current_sw[i] += val
                    current_sw[j] += val
                    
                    # Store data for Gravity Pass
                    pairs.append((i, j, r_hat, deriv))
        
        # --- PASS 2: Geometric Gravity (The Inward Fall) ---
        # The system wants to minimize (SW - Target)^2
        # Force depends on the SW "deficit" of BOTH interacting particles.
        
        for i in range(N):
            # Energy due to density mismatch
            diff = current_sw[i] - self.sw_target
            potential_energy += 0.5 * self.k_grav * diff**2
        
        for (i, j, r_hat, deriv) in pairs:
            # Chain Rule for Gravity Gradient:
            # Interaction i-j contributes to SW_i AND SW_j.
            # We must pull them together if EITHER needs more density.
            
            diff_i = current_sw[i] - self.sw_target
            diff_j = current_sw[j] - self.sw_target
            
            # Combined Gradient Magnitude
            # deriv is negative (kernel drops with distance)
            # diff is negative (if under-dense)
            # We want attraction -> Force opposite to r_hat (separation vector)
            
            f_grav_mag = self.k_grav * (diff_i + diff_j) * deriv
            
            # Apply force
            # if deriv < 0 and diff < 0 (needs neighbors), f_grav_mag > 0.
            # We want attraction. r_hat points j->i.
            # Force on i should be toward j (-r_hat).
            # Wait, let's check signs carefully:
            # Force = - dV/dr
            # V ~ (SW - T)^2
            # dV/dr = 2 * (SW - T) * dSW/dr
            # dSW/dr = deriv (negative)
            # (SW - T) is negative.
            # Result: dV/dr is Positive.
            # Force is Negative (Attractive). 
            # So we subtract from i (toward j) and add to j.
            
            f_vec = f_grav_mag * r_hat
            
            forces[i] -= f_vec
            forces[j] += f_vec
        
        return forces, potential_energy, current_sw

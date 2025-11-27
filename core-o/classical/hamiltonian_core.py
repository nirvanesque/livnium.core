"""
Livnium Hamiltonian Core: The Engine

Manages State (q, p), Time, and Thermodynamics.
Implements symplectic integrator with Langevin thermal bath.

Doesn't know about spheres or SW - just integrates momentum and applies thermostat.
"""

import numpy as np
from typing import Dict, Optional
import os
import sys
from .forces import GeometricPotential

# Try to import psutil for better RAM monitoring, fall back to basic method
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class LivniumHamiltonian:
    """
    The Engine.
    
    Manages State (q, p), Time, and Thermodynamics.
    Implements the Unified Principle: Minimizing Geometric Stress.
    """
    
    def __init__(
        self,
        n_spheres: int,
        temp: float = 0.1,
        friction: float = 0.05,
        dt: float = 0.01,
        radius: float = 1.0,
        k_repulsion: float = 500.0,
        k_gravity: float = 2.0,
        sw_target: float = 12.0,
        mass: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
        max_spheres: int = 500,
            enable_performance_monitoring: bool = True,
            max_ram_gb: float = 8.0,
            ram_check_interval: int = 10
        ):
        """
        Initialize Hamiltonian engine.
        
        Args:
            n_spheres: Number of spheres in the system
            temp: Temperature (T) for thermal bath
            friction: Friction coefficient (Gamma) for Langevin dynamics
            dt: Time step
            radius: Sphere radius
            k_repulsion: Repulsion strength
            k_gravity: Gravity strength
            sw_target: Target SW density
            mass: Optional mass array (default: all ones)
            positions: Optional initial positions (default: random)
            max_spheres: Maximum N for naive O(N²) solver (safety limit)
            enable_performance_monitoring: Track step times and warn if slow
            max_ram_gb: Maximum RAM usage in GB before auto-kill (default: 8.0)
            ram_check_interval: Check RAM every N steps (default: 10)
        """
        # Safety: Limit N for naive O(N²) solver
        if n_spheres > max_spheres:
            raise ValueError(
                f"n_spheres={n_spheres} exceeds safety limit of {max_spheres}. "
                f"This would require ~{n_spheres * (n_spheres - 1) // 2:,} pair computations per step. "
                f"Use neighbor lists (Verlet/cell lists) for larger systems, or increase max_spheres if you know what you're doing."
            )
        
        if n_spheres > 300:
            import warnings
            warnings.warn(
                f"Large system (N={n_spheres}): O(N²) solver will be slow. "
                f"Consider using neighbor lists for better performance.",
                UserWarning
            )
        
        # Configuration
        self.N = n_spheres
        self.dt = dt
        self.max_spheres = max_spheres
        self.enable_performance_monitoring = enable_performance_monitoring
        
        # Thermodynamics
        self.temperature = temp  # T
        self.friction = friction  # Gamma
        
        # The Laws
        self.laws = GeometricPotential(
            radius=radius,
            k_repulsion=k_repulsion,
            k_gravity=k_gravity,
            sw_target=sw_target
        )
        
        # State Vector
        if positions is not None:
            self.q = positions.copy()
        else:
            # Random initial positions
            self.q = np.random.randn(self.N, 3) * 2.0
        
        self.p = np.zeros((self.N, 3))  # Momenta (start at rest)
        
        if mass is not None:
            self.mass = mass.copy()
        else:
            self.mass = np.ones(self.N)  # Inertia (can make dynamic later)
        
        # Metrics
        self.time = 0.0
        self.energy_log = []
        self.step_times = []  # Performance monitoring
        self.slow_step_warning_shown = False
        
        # RAM monitoring
        self.max_ram_gb = max_ram_gb
        self.ram_check_interval = ram_check_interval
        self.ram_warning_shown = False
    
    def step(self) -> Dict:
        """
        Velocity Verlet Integration with Langevin Thermostat.
        
        Preserves Symplectic structure (energy conservation when friction=0).
        Implements the Unified Principle: Minimizing Geometric Stress.
        
        Returns:
            Dictionary with:
            - time: Current time
            - total_energy: Total energy (kinetic + potential)
            - kinetic_energy: Kinetic energy
            - potential_energy: Potential energy
            - avg_sw: Average SW density
            - positions: Current positions
            - step_time: Time taken for this step (seconds)
        """
        import time
        step_start = time.time()
        
        # RAM monitoring (check periodically)
        if len(self.energy_log) % self.ram_check_interval == 0:
            ram_used_gb = self._get_ram_usage_gb()
            
            # Sanity check: if RAM reading is clearly wrong (e.g., > 100GB), disable monitoring
            if ram_used_gb > 100.0:
                # Likely reading system RAM instead of process RAM - disable check
                if not self.ram_warning_shown:
                    import warnings
                    warnings.warn(
                        f"RAM monitoring detected invalid reading ({ram_used_gb:.2f} GB). "
                        f"Disabling RAM checks. This is likely a measurement issue, not actual usage.",
                        UserWarning
                    )
                    self.ram_warning_shown = True
                # Skip the check for this step
            elif ram_used_gb > self.max_ram_gb:
                raise MemoryError(
                    f"RAM usage ({ram_used_gb:.2f} GB) exceeds limit ({self.max_ram_gb:.2f} GB). "
                    f"Simulation stopped to protect system. "
                    f"Reduce N or enable neighbor lists for larger systems."
                )
            elif ram_used_gb > self.max_ram_gb * 0.8 and not self.ram_warning_shown:
                import warnings
                warnings.warn(
                    f"RAM usage is high: {ram_used_gb:.2f} GB / {self.max_ram_gb:.2f} GB limit. "
                    f"Consider reducing N or stopping soon.",
                    UserWarning
                )
                self.ram_warning_shown = True
        
        dt = self.dt
        
        # 1. First Half-Kick (Hamiltonian)
        forces, pot_energy, sw = self.laws.compute_forces(self.q)
        self.p += 0.5 * forces * dt
        
        # 2. Drift (Kinematic)
        self.q += (self.p / self.mass[:, None]) * dt
        
        # 3. Second Half-Kick (Hamiltonian)
        forces_new, pot_energy_new, sw_new = self.laws.compute_forces(self.q)
        self.p += 0.5 * forces_new * dt
        
        # Use updated values
        pot_energy = pot_energy_new
        sw = sw_new
        
        # 4. Thermal Bath (Langevin Dynamics)
        # Adds Entropy (friction) and Fluctuations (noise)
        if self.friction > 0:
            sigma = np.sqrt(2.0 * self.friction * self.temperature)
            noise = np.random.randn(self.N, 3)
            
            # Update momentum
            self.p -= self.friction * self.p * dt  # Drag
            self.p += sigma * noise * np.sqrt(dt)  # Kick
        
        self.time += dt
        
        # 5. Logging (The "Observer")
        kinetic_energy = 0.5 * np.sum(self.p**2 / self.mass[:, None])
        total_energy = kinetic_energy + pot_energy
        
        # Performance monitoring
        step_time = time.time() - step_start
        self.step_times.append(step_time)
        
        # Keep only last 100 step times
        if len(self.step_times) > 100:
            self.step_times = self.step_times[-100:]
        
        # Warn if step is taking too long (> 1 second)
        if self.enable_performance_monitoring:
            if step_time > 1.0 and not self.slow_step_warning_shown:
                import warnings
                warnings.warn(
                    f"Step taking {step_time:.2f}s (very slow for N={self.N}). "
                    f"Consider reducing N or using neighbor lists. "
                    f"Press Ctrl+C to interrupt if needed.",
                    UserWarning
                )
                self.slow_step_warning_shown = True
        
        self.energy_log.append({
            'time': self.time,
            'total_energy': total_energy,
            'kinetic_energy': kinetic_energy,
            'potential_energy': pot_energy,
            'avg_sw': np.mean(sw),
            'step_time': step_time
        })
        
        return {
            'time': self.time,
            'total_energy': total_energy,
            'kinetic_energy': kinetic_energy,
            'potential_energy': pot_energy,
            'avg_sw': np.mean(sw),
            'sw': sw.copy(),
            'positions': self.q.copy(),
            'step_time': step_time
        }
    
    def get_energy_history(self) -> Dict:
        """
        Get energy history for analysis.
        
        Returns:
            Dictionary with arrays of time, energies, SW, and step times
        """
        if not self.energy_log:
            return {
                'time': np.array([]),
                'total_energy': np.array([]),
                'kinetic_energy': np.array([]),
                'potential_energy': np.array([]),
                'avg_sw': np.array([]),
                'step_time': np.array([])
            }
        
        return {
            'time': np.array([e['time'] for e in self.energy_log]),
            'total_energy': np.array([e['total_energy'] for e in self.energy_log]),
            'kinetic_energy': np.array([e['kinetic_energy'] for e in self.energy_log]),
            'potential_energy': np.array([e['potential_energy'] for e in self.energy_log]),
            'avg_sw': np.array([e['avg_sw'] for e in self.energy_log]),
            'step_time': np.array([e.get('step_time', 0.0) for e in self.energy_log])
        }
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.step_times:
            return {
                'avg_step_time': 0.0,
                'min_step_time': 0.0,
                'max_step_time': 0.0,
                'total_steps': 0,
                'estimated_pairs_per_step': self.N * (self.N - 1) // 2
            }
        
        step_times = np.array(self.step_times)
        return {
            'avg_step_time': float(np.mean(step_times)),
            'min_step_time': float(np.min(step_times)),
            'max_step_time': float(np.max(step_times)),
            'total_steps': len(self.energy_log),
            'estimated_pairs_per_step': self.N * (self.N - 1) // 2,
            'pairs_per_second': (self.N * (self.N - 1) // 2) / np.mean(step_times) if np.mean(step_times) > 0 else 0
        }
    
    def _get_ram_usage_gb(self) -> float:
        """
        Get current RAM usage in GB for this process.
        
        Returns:
            RAM usage in GB
        """
        if HAS_PSUTIL:
            # Use psutil for accurate process memory
            process = psutil.Process(os.getpid())
            ram_bytes = process.memory_info().rss  # Resident Set Size
            return ram_bytes / (1024 ** 3)  # Convert to GB
        else:
            # Fallback: platform-specific methods
            try:
                import resource
                # Unix/Linux/Mac
                # ru_maxrss is in KB on both macOS and Linux
                ram_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # Convert KB to GB: KB / 1024 / 1024 = GB
                return ram_kb / (1024 ** 2)  # KB to GB
            except (ImportError, AttributeError):
                # Last resort: return 0 (can't measure, but won't crash)
                return 0.0
    
    def get_ram_usage(self) -> Dict:
        """
        Get current RAM usage information.
        
        Returns:
            Dictionary with RAM usage stats
        """
        ram_used = self._get_ram_usage_gb()
        return {
            'ram_used_gb': ram_used,
            'ram_limit_gb': self.max_ram_gb,
            'ram_percent': (ram_used / self.max_ram_gb * 100) if self.max_ram_gb > 0 else 0,
            'has_psutil': HAS_PSUTIL
        }


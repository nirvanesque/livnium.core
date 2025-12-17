"""
Recursive Hamiltonian: Add Hamiltonian dynamics to recursive geometry

Integrates core-o Hamiltonian core into recursive geometry engine.
Each level evolves with momentum and forces, not just static structure.
"""

from typing import Dict, Optional, Any
import numpy as np
import sys
from pathlib import Path

# Lazy import - will be done in __init__
LivniumHamiltonian = None
HAS_HAMILTONIAN = False
_import_error = None

def _import_hamiltonian():
    """Lazy import of Hamiltonian core from core-o."""
    global LivniumHamiltonian, HAS_HAMILTONIAN, _import_error
    
    if HAS_HAMILTONIAN:
        return True  # Already imported
    
    # Import from core-o using importlib for more reliable path handling
    # File is at: core/recursive/recursive_hamiltonian.py
    # Need to go up 3 levels to reach project root
    project_root = Path(__file__).parent.parent.parent
    core_o_path = project_root / "core-o"
    hamiltonian_file = core_o_path / "classical" / "hamiltonian_core.py"
    
    if not hamiltonian_file.exists():
        HAS_HAMILTONIAN = False
        _import_error = f"File not found: {hamiltonian_file}"
        return False
    
    # Add core-o to path
    if str(core_o_path) not in sys.path:
        sys.path.insert(0, str(core_o_path))
    
    try:
        # Try direct import
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "classical.hamiltonian_core",
            str(hamiltonian_file)
        )
        if spec is None or spec.loader is None:
            raise ImportError("Could not load spec")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules["classical.hamiltonian_core"] = module
        spec.loader.exec_module(module)
        
        LivniumHamiltonian = module.LivniumHamiltonian
        HAS_HAMILTONIAN = True
        return True
    except Exception as e:
        # Fallback: try normal import
        try:
            from classical.hamiltonian_core import LivniumHamiltonian as _LivniumHamiltonian
            LivniumHamiltonian = _LivniumHamiltonian
            HAS_HAMILTONIAN = True
            return True
        except ImportError as e2:
            HAS_HAMILTONIAN = False
            _import_error = f"{str(e)}; {str(e2)}"
            return False


class RecursiveHamiltonian:
    """
    Adds Hamiltonian dynamics to recursive geometry.
    
    Each level in the recursive hierarchy can evolve with:
    - Momentum (p) - how fast SW is changing
    - Forces (F) - gradients of potential energy
    - Thermal bath - temperature and friction
    
    This makes the recursive system dynamic, not just structural.
    """
    
    def __init__(
        self,
        recursive_engine,
        temp: float = 0.1,
        friction: float = 0.05,
        dt: float = 0.01,
        enable_dynamics: bool = True
    ):
        """
        Initialize recursive Hamiltonian dynamics.
        
        Args:
            recursive_engine: RecursiveGeometryEngine instance
            temp: Temperature for thermal bath
            friction: Friction coefficient
            dt: Time step
            enable_dynamics: Whether to enable Hamiltonian evolution
        """
        # Try to import Hamiltonian if not already done
        if not _import_hamiltonian():
            error_msg = "core-o Hamiltonian core not available. Cannot enable recursive dynamics."
            if _import_error:
                error_msg += f" Import error: {_import_error}"
            raise ImportError(error_msg)
        
        self.recursive_engine = recursive_engine
        self.temp = temp
        self.friction = friction
        self.dt = dt
        self.enable_dynamics = enable_dynamics
        
        # Create Hamiltonian for each level
        self.level_hamiltonians: Dict[int, LivniumHamiltonian] = {}
        self._initialize_level_hamiltonians()
    
    def _initialize_level_hamiltonians(self):
        """Initialize Hamiltonian for each recursive level."""
        if not self.enable_dynamics:
            return
        
        for level_id, level in self.recursive_engine.levels.items():
            geometry = level.geometry
            num_cells = len(geometry.lattice)
            
            # Create Hamiltonian for this level
            # Use number of cells as "spheres" - each cell is like a sphere
            # Positions come from cell coordinates
            # SW values become the "mass" or "energy density"
            
            # Extract positions from cell coordinates
            positions = np.array(list(geometry.lattice.keys()), dtype=float)
            
            # Create Hamiltonian with appropriate parameters
            hamiltonian = LivniumHamiltonian(
                n_spheres=num_cells,
                temp=self.temp,
                friction=self.friction,
                dt=self.dt,
                positions=positions,
                max_spheres=500  # Safety limit
            )
            
            self.level_hamiltonians[level_id] = hamiltonian
    
    def evolve_level(self, level_id: int) -> Dict[str, Any]:
        """
        Evolve a specific level using Hamiltonian dynamics.
        
        Args:
            level_id: Level to evolve
            
        Returns:
            Evolution statistics
        """
        if not self.enable_dynamics or level_id not in self.level_hamiltonians:
            return {}
        
        hamiltonian = self.level_hamiltonians[level_id]
        stats = hamiltonian.step()
        
        # Update geometry based on evolved positions
        # Map Hamiltonian positions back to cell coordinates
        level = self.recursive_engine.levels[level_id]
        geometry = level.geometry
        
        # Update SW values based on evolved state
        # SW can evolve based on forces and momentum
        sw_values = stats.get('sw', [])
        if len(sw_values) == len(geometry.lattice):
            for (coords, cell), sw_val in zip(geometry.lattice.items(), sw_values):
                # Update cell SW based on Hamiltonian evolution
                # This creates dynamic SW that evolves with forces
                cell.symbolic_weight = float(sw_val)
        
        return stats
    
    def evolve_all_levels(self) -> Dict[int, Dict[str, Any]]:
        """
        Evolve all levels in the recursive hierarchy.
        
        Returns:
            Dictionary mapping level_id to evolution statistics
        """
        results = {}
        
        for level_id in self.recursive_engine.levels.keys():
            results[level_id] = self.evolve_level(level_id)
        
        return results
    
    def get_energy_history(self, level_id: int) -> Dict:
        """Get energy history for a specific level."""
        if level_id not in self.level_hamiltonians:
            return {}
        
        return self.level_hamiltonians[level_id].get_energy_history()
    
    def get_all_energy_histories(self) -> Dict[int, Dict]:
        """Get energy history for all levels."""
        histories = {}
        
        for level_id in self.level_hamiltonians.keys():
            histories[level_id] = self.get_energy_history(level_id)
        
        return histories


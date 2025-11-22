"""
DataCube: Even-Dimensional Resource Grids

DataCubes are even-numbered N×N×N grids (2×2×2, 4×4×4, 6×6×6, ...) that serve
as resource containers, data staging areas, and I/O buffers.

**Critical Distinction:**
- **Omcubes** (odd N ≥ 3): Livnium Core Universes - implement all axioms, collapse, recursion
- **DataCubes** (even N ≥ 2): Resource Grids - NO axioms, NO collapse, NO computation

DataCubes are NOT Livnium cores. They cannot:
- Execute Livnium collapse mechanics
- Implement symbolic weight (SW) system
- Use face exposure rules
- Perform recursive geometry operations
- Anchor observers (no center cell)
- Maintain Livnium invariants

They CAN:
- Store data (lookup tables, feature maps)
- Act as I/O buffers (input snapshots, output surfaces)
- Serve as preprocessing/postprocessing containers
- Hold embeddings or temporary state
- Connect to Omcubes as data sources/sinks

Think of them as "RAM blocks around a CPU":
- Omcubes = CPU (core geometry, computation)
- DataCubes = RAM (resource/data layers, storage)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class DataCell:
    """
    A single cell in an even-dimensional DataCube.
    
    Unlike LatticeCell (in Omcubes), this has:
    - NO face exposure calculation
    - NO symbolic weight
    - NO cell class (core/center/edge/corner)
    - NO observer anchor
    - Just coordinates and data storage
    """
    coordinates: Tuple[int, int, int]
    data: Optional[Any] = None  # Can store any data type
    
    def set_data(self, value: Any) -> None:
        """Set data in this cell."""
        self.data = value
    
    def get_data(self) -> Optional[Any]:
        """Get data from this cell."""
        return self.data


class DataCube:
    """
    Even-dimensional resource grid (2×2×2, 4×4×4, 6×6×6, ...).
    
    This is NOT a Livnium core. It cannot execute:
    - Livnium collapse mechanics
    - Symbolic weight calculations
    - Face exposure rules
    - Recursive geometry
    - Observer anchoring
    - Rotation-based transformations
    
    It is a simple data container for:
    - Input/output buffers
    - Feature maps
    - Lookup tables
    - Temporary storage
    - Embedding carriers
    - Preprocessing/postprocessing data
    """
    
    def __init__(self, size: int):
        """
        Initialize DataCube.
        
        Args:
            size: Even integer ≥ 2 (2, 4, 6, 8, ...)
        
        Raises:
            ValueError: If size is odd or < 2
        """
        if size < 2:
            raise ValueError(f"DataCube size must be >= 2, got {size}")
        if size % 2 != 0:
            raise ValueError(f"DataCube size must be even (2, 4, 6, ...), got {size}")
        
        self.size = size
        self.lattice: Dict[Tuple[int, int, int], DataCell] = {}
        self._initialize_lattice()
    
    def _initialize_lattice(self) -> None:
        """Initialize all cells in the even-dimensional grid."""
        # Coordinate range: {-(N/2-1), ..., N/2-1} for even N
        # For N=4: {-1, 0, 1} (no center at 0,0,0)
        # For N=2: {0} (just one cell per axis)
        coord_range = list(range(-(self.size // 2 - 1), self.size // 2 + 1))
        
        for x in coord_range:
            for y in coord_range:
                for z in coord_range:
                    coords = (x, y, z)
                    self.lattice[coords] = DataCell(coords)
    
    def get_cell(self, coords: Tuple[int, int, int]) -> Optional[DataCell]:
        """Get cell at coordinates."""
        return self.lattice.get(coords)
    
    def set_data(self, coords: Tuple[int, int, int], data: Any) -> None:
        """Set data at coordinates."""
        if coords not in self.lattice:
            raise ValueError(f"Coordinates {coords} out of bounds for {self.size}×{self.size}×{self.size} DataCube")
        self.lattice[coords].set_data(data)
    
    def get_data(self, coords: Tuple[int, int, int]) -> Optional[Any]:
        """Get data at coordinates."""
        cell = self.get_cell(coords)
        return cell.get_data() if cell else None
    
    def get_all_data(self) -> Dict[Tuple[int, int, int], Any]:
        """Get all data as a dictionary mapping coordinates to values."""
        return {coords: cell.get_data() for coords, cell in self.lattice.items()}
    
    def clear(self) -> None:
        """Clear all data from cells."""
        for cell in self.lattice.values():
            cell.set_data(None)
    
    def to_numpy(self, dtype=np.float32) -> np.ndarray:
        """
        Convert DataCube to numpy array.
        
        Returns:
            3D numpy array of shape (size, size, size)
        """
        arr = np.zeros((self.size, self.size, self.size), dtype=dtype)
        coord_range = list(range(-(self.size // 2 - 1), self.size // 2 + 1))
        
        for i, x in enumerate(coord_range):
            for j, y in enumerate(coord_range):
                for k, z in enumerate(coord_range):
                    data = self.get_data((x, y, z))
                    if data is not None:
                        try:
                            arr[i, j, k] = float(data)
                        except (ValueError, TypeError):
                            arr[i, j, k] = 0.0
        
        return arr
    
    def from_numpy(self, arr: np.ndarray) -> None:
        """
        Load data from numpy array into DataCube.
        
        Args:
            arr: 3D numpy array of shape (size, size, size)
        """
        if arr.shape != (self.size, self.size, self.size):
            raise ValueError(f"Array shape {arr.shape} does not match DataCube size {self.size}")
        
        coord_range = list(range(-(self.size // 2 - 1), self.size // 2 + 1))
        
        for i, x in enumerate(coord_range):
            for j, y in enumerate(coord_range):
                for k, z in enumerate(coord_range):
                    self.set_data((x, y, z), float(arr[i, j, k]))
    
    def __repr__(self) -> str:
        return f"DataCube(size={self.size}, cells={len(self.lattice)})"


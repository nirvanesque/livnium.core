"""
Geometry Embedding: Map Rule 30 center column into Livnium cube

Embeds a binary sequence into a vertical path in a Livnium omcube.
Each bit becomes a geometric state: Φ = +1 for 1, Φ = -1 for 0.
"""

import numpy as np
from typing import List, Tuple, Optional
from core.classical.livnium_core_system import LivniumCoreSystem, LatticeCell


class Rule30Path:
    """
    Represents a geometric path through a Livnium cube.
    
    Each step in the path corresponds to a bit in the Rule 30 center column.
    The path is vertical (along Z-axis) through the center of the cube.
    """
    
    def __init__(self, sequence: List[int], cube_size: int = 3):
        """
        Initialize path from binary sequence.
        
        Args:
            sequence: List of 0s and 1s from Rule 30 center column
            cube_size: Size of the omcube (3, 5, etc. - must be odd)
        """
        self.sequence = sequence
        self.cube_size = cube_size
        self.path_coords = []
        self.path_states = []  # Φ values: +1 for 1, -1 for 0
        
        # Generate path coordinates (vertical column through center)
        self._generate_path()
    
    def _generate_path(self):
        """Generate vertical path coordinates through cube center."""
        center_x = 0
        center_y = 0
        
        # For a cube of size N, coordinates range from -(N-1)/2 to (N-1)/2
        # We'll create a path along the Z-axis through the center
        z_range = (self.cube_size - 1) // 2
        z_coords = list(range(-z_range, z_range + 1))
        
        # If sequence is longer than cube depth, wrap or truncate
        # If shorter, repeat or pad
        for i, bit in enumerate(self.sequence):
            if i < len(z_coords):
                z = z_coords[i]
            else:
                # Wrap around if sequence is longer
                z = z_coords[i % len(z_coords)]
            
            coord = (center_x, center_y, z)
            self.path_coords.append(coord)
            
            # Map bit to geometric state: 1 → +1, 0 → -1
            phi = 1.0 if bit == 1 else -1.0
            self.path_states.append(phi)
    
    def get_path_length(self) -> int:
        """Get length of the path."""
        return len(self.path_coords)
    
    def get_coordinates(self) -> List[Tuple[int, int, int]]:
        """Get list of coordinates in the path."""
        return self.path_coords
    
    def get_states(self) -> List[float]:
        """Get list of Φ states (geometric values)."""
        return self.path_states


def embed_into_cube(
    center_seq: List[int],
    cube_size: int = 3
) -> Tuple[LivniumCoreSystem, Rule30Path]:
    """
    Embed Rule 30 center column sequence into a Livnium cube.
    
    Creates a vertical path through the cube center, mapping each bit
    to a geometric state (Φ = +1 for 1, Φ = -1 for 0).
    
    Args:
        center_seq: Binary sequence from Rule 30 center column
        cube_size: Size of omcube (3, 5, etc. - must be odd)
        
    Returns:
        Tuple of (LivniumCoreSystem instance, Rule30Path object)
    """
    # Create Livnium core system with config
    from core.config import LivniumCoreConfig
    
    config = LivniumCoreConfig(
        lattice_size=cube_size,
        enable_symbolic_weight=True,
        enable_face_exposure=True,
        enable_global_observer=True
    )
    
    system = LivniumCoreSystem(config)
    
    # Create path representation
    path = Rule30Path(center_seq, cube_size)
    
    # Embed states into cube cells
    # For each coordinate in the path, store the Φ value as metadata
    # The actual SW is still computed from face exposure
    # But we can use Φ to influence the geometric field
    for coord, phi in zip(path.get_coordinates(), path.get_states()):
        # Check if coordinate is in valid range
        boundary = (cube_size - 1) // 2
        x, y, z = coord
        if -boundary <= x <= boundary and -boundary <= y <= boundary and -boundary <= z <= boundary:
            cell = system.get_cell(coord)
            if cell:
                # Store Φ value as metadata
                cell.phi_value = phi
    
    return system, path


def create_sequence_vectors(sequence: List[int]) -> List[np.ndarray]:
    """
    Convert binary sequence to vector representation for Layer0/Layer1.
    
    Each bit becomes a 1D vector: [1.0] for 1, [-1.0] for 0.
    This allows us to use existing Layer0/Layer1 computation.
    
    Args:
        sequence: Binary sequence
        
    Returns:
        List of numpy arrays (vectors)
    """
    vectors = []
    for bit in sequence:
        vec = np.array([1.0 if bit == 1 else -1.0], dtype=np.float32)
        vectors.append(vec)
    return vectors


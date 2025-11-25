"""
Geometric Diagnostics for Rule 30 Path

Computes divergence, tension, and basin depth for a Rule 30 sequence
embedded in Livnium geometry.
"""

import numpy as np
import math
from typing import List, Dict, Tuple
from experiments.rule30.geometry_embed import Rule30Path, create_sequence_vectors


class MockEncodedPair:
    """
    Mock encoded pair to interface with Layer0/Layer1.
    
    Since Rule 30 is a single sequence (not premise/hypothesis pairs),
    we treat the sequence as both premise and hypothesis.
    """
    
    def __init__(self, vectors: List[np.ndarray]):
        self.vectors = vectors
        self.premise_vecs = vectors
        self.hypothesis_vecs = vectors  # Same sequence
    
    def get_resonance(self) -> float:
        """Compute resonance from sequence structure."""
        if not self.vectors:
            return 0.0
        
        # Resonance: measure of internal coherence
        # For Rule 30, we compute self-similarity
        similarities = []
        for i in range(len(self.vectors) - 1):
            v1 = self.vectors[i]
            v2 = self.vectors[i + 1]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-6 and norm2 > 1e-6:
                sim = np.dot(v1, v2) / (norm1 * norm2)
                similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def get_word_vectors(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return word vectors for Layer0/Layer1."""
        return self.premise_vecs, self.hypothesis_vecs


def _compute_field_divergence(premise_vecs: List[np.ndarray], hypothesis_vecs: List[np.ndarray]) -> float:
    """
    Compute field divergence using angle-based method.
    
    Based on Layer0Resonance implementation.
    """
    if not premise_vecs or not hypothesis_vecs:
        return 0.0
    
    # Compute mean vectors
    premise_mean = np.mean([v for v in premise_vecs if np.linalg.norm(v) > 1e-6], axis=0)
    hypothesis_mean = np.mean([v for v in hypothesis_vecs if np.linalg.norm(v) > 1e-6], axis=0)
    
    # Normalize
    p_norm = np.linalg.norm(premise_mean)
    h_norm = np.linalg.norm(hypothesis_mean)
    
    if p_norm < 1e-6 or h_norm < 1e-6:
        return 0.0
    
    premise_unit = premise_mean / p_norm
    hypothesis_unit = hypothesis_mean / h_norm
    
    # Cosine similarity
    cos_sim = np.dot(premise_unit, hypothesis_unit)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
    # Calculate angle (0 to π)
    theta = np.arccos(cos_sim)
    theta_deg = theta * (180.0 / math.pi)
    
    # Normalize to [0, 1]
    theta_norm = theta / math.pi
    
    # Equilibrium angle (normalized)
    equilibrium_angle_deg = 41.2
    theta_eq = equilibrium_angle_deg * (math.pi / 180.0)
    theta_eq_norm = theta_eq / math.pi
    
    # Divergence: negative = convergence (E), positive = divergence (C)
    divergence_scale = 2.5
    divergence = (theta_norm - theta_eq_norm) * divergence_scale
    
    # Neutral basin clamp
    neutral_window = 0.20
    neutral_clamp_factor = 0.25
    if abs(divergence) < neutral_window:
        divergence = divergence * neutral_clamp_factor
    
    return float(np.clip(divergence, -1.0, 1.0))


def compute_divergence_path(path: Rule30Path, show_progress: bool = False) -> List[float]:
    """
    Compute divergence along the Rule 30 path.
    
    Uses angle-based divergence computation at each step.
    
    Args:
        path: Rule30Path object
        show_progress: Show progress indicator for large sequences
        
    Returns:
        List of divergence values along the path
    """
    import sys
    
    vectors = create_sequence_vectors(path.sequence)
    divergence_path = []
    
    # For very long sequences, we can optimize by sampling
    # But for accuracy, we'll compute for all windows
    window_size = min(5, len(vectors))
    total_windows = len(vectors) - window_size + 1
    
    # Progress tracking for large sequences
    if show_progress and total_windows > 1000:
        progress_interval = max(1, total_windows // 50)
    
    for i in range(total_windows):
        # Show progress for large sequences
        if show_progress and total_windows > 1000 and i % progress_interval == 0:
            percent = (i / total_windows) * 100
            sys.stdout.write(f"\rComputing divergence: {i:,}/{total_windows:,} ({percent:.1f}%)")
            sys.stdout.flush()
        
        window_vecs = vectors[i:i + window_size]
        
        # Compute divergence (treating window as both premise and hypothesis)
        divergence = _compute_field_divergence(window_vecs, window_vecs)
        divergence_path.append(divergence)
    
    if show_progress and total_windows > 1000:
        sys.stdout.write(f"\rComputing divergence: {total_windows:,}/{total_windows:,} (100.0%)\n")
        sys.stdout.flush()
    
    # Pad to match path length
    if len(divergence_path) < len(path.sequence):
        # Repeat last value or interpolate
        while len(divergence_path) < len(path.sequence):
            divergence_path.append(divergence_path[-1] if divergence_path else 0.0)
    
    return divergence_path[:len(path.sequence)]


def compute_tension_curve(path: Rule30Path) -> List[float]:
    """
    Compute tension along the Rule 30 path.
    
    Tension measures internal contradictions in the geometric field.
    Computes curvature-based tension from sequence variations.
    
    Args:
        path: Rule30Path object
        
    Returns:
        List of tension values along the path
    """
    vectors = create_sequence_vectors(path.sequence)
    tension_curve = []
    resonance_history = []
    
    # Compute resonance for each position
    for i in range(len(vectors) - 1):
        v1 = vectors[i]
        v2 = vectors[i + 1]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 1e-6 and norm2 > 1e-6:
            sim = np.dot(v1, v2) / (norm1 * norm2)
            resonance_history.append(sim)
        else:
            resonance_history.append(0.0)
    
    # Compute tension from curvature (second derivative)
    for i in range(len(path.sequence)):
        if len(resonance_history) >= 3 and i >= 2:
            r_t = resonance_history[i] if i < len(resonance_history) else 0.0
            r_t1 = resonance_history[i-1] if i-1 < len(resonance_history) else 0.0
            r_t2 = resonance_history[i-2] if i-2 < len(resonance_history) else 0.0
            curvature = abs(r_t - 2.0 * r_t1 + r_t2)
        else:
            curvature = 0.0
        
        # Tension = curvature magnitude
        tension_curve.append(curvature)
    
    return tension_curve[:len(path.sequence)]


def compute_basin_depth(path: Rule30Path) -> List[float]:
    """
    Compute basin depth along the Rule 30 path.
    
    Basin depth measures how deep the system is in an attraction well.
    Computes from cold density (convergence) and divergence force.
    
    Args:
        path: Rule30Path object
        
    Returns:
        List of basin depth values along the path
    """
    vectors = create_sequence_vectors(path.sequence)
    basin_depths = []
    
    # Shared basin depths (simplified)
    cold_depth = 0.5
    far_depth = 0.5
    
    # Compute basin depth for sliding windows
    window_size = min(5, len(vectors))
    for i in range(len(vectors) - window_size + 1):
        window_vecs = vectors[i:i + window_size]
        
        # Compute cold density (convergence measure)
        cold_density = 0.0
        for v1 in window_vecs:
            for v2 in window_vecs:
                dot_prod = np.dot(v1, v2)
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_prod / (norm1 * norm2)
                    cold_density += similarity
        cold_density = cold_density / (len(window_vecs) ** 2) if window_vecs else 0.0
        
        # Compute divergence force
        divergence = _compute_field_divergence(window_vecs, window_vecs)
        divergence_force = max(0.0, divergence)  # Only positive divergence
        
        # Basin attractions
        cold_attraction = cold_density * cold_depth
        far_attraction = divergence_force * far_depth
        
        # Basin depth = max attraction
        basin_depth = max(cold_attraction, far_attraction)
        basin_depths.append(basin_depth)
    
    # Pad to match path length
    if len(basin_depths) < len(path.sequence):
        while len(basin_depths) < len(path.sequence):
            basin_depths.append(basin_depths[-1] if basin_depths else 0.0)
    
    return basin_depths[:len(path.sequence)]


def compute_all_diagnostics(path: Rule30Path) -> Dict[str, List[float]]:
    """
    Compute all geometric diagnostics for a Rule 30 path.
    
    Args:
        path: Rule30Path object
        
    Returns:
        Dictionary with 'divergence', 'tension', 'basin_depth' lists
    """
    return {
        'divergence': compute_divergence_path(path),
        'tension': compute_tension_curve(path),
        'basin_depth': compute_basin_depth(path)
    }


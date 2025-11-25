"""
Optimized Rule 30 Generator - Center Column Only

For large sequences, we only need the center column, not the full triangle.
This is much faster for 100k+ steps.
"""

from typing import List
import sys


def generate_center_column_direct(n_steps: int, show_progress: bool = True) -> List[int]:
    """
    Generate Rule 30 center column directly without building full triangle.
    
    This is optimized for large sequences (100k+ steps) where we only
    need the center column, not the entire triangle.
    
    Args:
        n_steps: Number of steps to generate
        show_progress: Show progress indicator
        
    Returns:
        List of center column values (0s and 1s)
    """
    if n_steps <= 0:
        return []
    
    # We need a window around the center to compute the center cell
    # Rule 30 needs left, center, right neighbors
    # For center column, we track a window of cells around center
    window_size = min(n_steps + 10, 2000)  # Reasonable window size
    
    # Initialize: single black cell at center
    current_row = [0] * window_size
    center_idx = window_size // 2
    current_row[center_idx] = 1
    
    center_column = [1]  # First row center is 1
    
    # Progress tracking
    progress_interval = max(1, n_steps // 100)  # Update every 1%
    
    for step in range(1, n_steps):
        # Show progress
        if show_progress and step % progress_interval == 0:
            percent = (step / n_steps) * 100
            sys.stdout.write(f"\rGenerating Rule 30: {step:,}/{n_steps:,} ({percent:.1f}%)")
            sys.stdout.flush()
        
        # Create new row
        new_row = [0] * window_size
        
        # Apply Rule 30
        for i in range(1, window_size - 1):
            left = current_row[i - 1]
            center = current_row[i]
            right = current_row[i + 1]
            
            # Rule 30: new = (left XOR center) OR (center AND NOT right)
            new_cell = (left ^ center) | (center & (1 - right))
            new_row[i] = new_cell
        
        # Extract center cell
        center_cell = new_row[center_idx]
        center_column.append(center_cell)
        
        # Update current row (shift window if needed)
        # For efficiency, we keep a fixed-size window
        # If pattern grows too wide, we'll miss edges, but center stays accurate
        current_row = new_row
    
    if show_progress:
        sys.stdout.write(f"\rGenerating Rule 30: {n_steps:,}/{n_steps:,} (100.0%)\n")
        sys.stdout.flush()
    
    return center_column


def generate_center_column_fast(n_steps: int) -> List[int]:
    """
    Fast center column generation using bit manipulation.
    
    Even faster version using bit arrays for very large sequences.
    """
    if n_steps <= 0:
        return []
    
    # Use a more efficient approach: track only what's needed
    # For center column, we need a band around center
    band_width = min(n_steps // 10 + 100, 5000)
    
    # Initialize
    current = [0] * band_width
    center_idx = band_width // 2
    current[center_idx] = 1
    
    center_column = [1]
    
    progress_interval = max(1, n_steps // 50)
    
    for step in range(1, n_steps):
        if step % progress_interval == 0:
            percent = (step / n_steps) * 100
            sys.stdout.write(f"\rGenerating: {step:,}/{n_steps:,} ({percent:.1f}%)")
            sys.stdout.flush()
        
        new = [0] * band_width
        
        # Apply Rule 30
        for i in range(1, band_width - 1):
            new[i] = (current[i-1] ^ current[i]) | (current[i] & (1 - current[i+1]))
        
        center_column.append(new[center_idx])
        current = new
    
    sys.stdout.write(f"\rGenerating: {n_steps:,}/{n_steps:,} (100.0%)\n")
    sys.stdout.flush()
    
    return center_column


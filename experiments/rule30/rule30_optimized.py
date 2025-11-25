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
    
    Matches the original generate_rule30() logic exactly, but only extracts
    the center column instead of storing all rows.
    
    Args:
        n_steps: Number of steps to generate
        show_progress: Show progress indicator
        
    Returns:
        List of center column values (0s and 1s)
    """
    if n_steps <= 0:
        return []
    
    # Match original initialization exactly
    # Start with single black cell in center
    current_row = [0] * (n_steps + 1)
    center_idx = n_steps // 2
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
        
        # Create new row (one cell wider on each side) - matches original
        new_row = [0] * (len(current_row) + 2)
        
        # Apply Rule 30 to each cell - matches original exactly
        for i in range(1, len(current_row) - 1):
            left = current_row[i - 1] if i > 0 else 0
            center = current_row[i]
            right = current_row[i + 1] if i < len(current_row) - 1 else 0
            
            # Rule 30 formula - matches original
            new_cell = (left ^ center) | (center & (1 - right))
            new_row[i + 1] = new_cell  # Offset by 1 to account for growth
        
        # Trim row like original FIRST (before extracting center)
        first_nonzero = next((i for i, x in enumerate(new_row) if x != 0), 0)
        last_nonzero = next((i for i in range(len(new_row) - 1, -1, -1) if new_row[i] != 0), len(new_row) - 1)
        
        # Keep padding for symmetry (matches original exactly)
        padding = max(1, (n_steps - step) // 2)
        start_idx = max(0, first_nonzero - padding)
        end_idx = min(len(new_row), last_nonzero + padding + 1)
        
        trimmed_row = new_row[start_idx:end_idx]
        
        # Extract center cell from TRIMMED row (matches extract_center_column logic)
        center_cell = trimmed_row[len(trimmed_row) // 2]
        center_column.append(center_cell)
        
        current_row = trimmed_row
    
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


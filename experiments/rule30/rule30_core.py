"""
Rule 30 Cellular Automaton Core

Generates Rule 30 CA patterns starting from a single black cell.
Rule 30: new cell = XOR(left neighbor, current cell) OR (current cell AND NOT right neighbor)
"""

from typing import List


def generate_rule30(n_steps: int) -> List[List[int]]:
    """
    Generate Rule 30 cellular automaton for n_steps rows.
    
    Starts with a single black cell (1) in the center.
    Each row is computed from the previous row using Rule 30.
    
    Args:
        n_steps: Number of rows to generate
        
    Returns:
        List of lists, where each inner list is a row of the triangle
        Each cell is 0 (white) or 1 (black)
    """
    if n_steps <= 0:
        return []
    
    # Start with single black cell in center
    # Use a list that can grow symmetrically
    current_row = [0] * (n_steps + 1)  # Pre-allocate enough space
    center_idx = n_steps // 2
    current_row[center_idx] = 1
    
    triangle = [current_row[:]]  # Store first row
    
    for step in range(1, n_steps):
        # Create new row (one cell wider on each side)
        new_row = [0] * (len(current_row) + 2)
        
        # Apply Rule 30 to each cell
        # Rule 30: new = XOR(left, current) OR (current AND NOT right)
        # Simplified: new = (left XOR current) OR (current AND NOT right)
        for i in range(1, len(current_row) - 1):
            left = current_row[i - 1] if i > 0 else 0
            center = current_row[i]
            right = current_row[i + 1] if i < len(current_row) - 1 else 0
            
            # Rule 30 formula
            new_cell = (left ^ center) | (center & (1 - right))
            new_row[i + 1] = new_cell  # Offset by 1 to account for growth
        
        # Trim leading/trailing zeros to keep triangle compact
        # Find first and last non-zero indices
        first_nonzero = next((i for i, x in enumerate(new_row) if x != 0), 0)
        last_nonzero = next((i for i in range(len(new_row) - 1, -1, -1) if new_row[i] != 0), len(new_row) - 1)
        
        # Keep some padding for symmetry
        padding = max(1, (n_steps - step) // 2)
        start_idx = max(0, first_nonzero - padding)
        end_idx = min(len(new_row), last_nonzero + padding + 1)
        
        trimmed_row = new_row[start_idx:end_idx]
        triangle.append(trimmed_row)
        current_row = trimmed_row
    
    return triangle


def rule30_next_cell(left: int, center: int, right: int) -> int:
    """
    Compute next cell value using Rule 30.
    
    Rule 30 truth table:
    left | center | right | new
    -----|--------|-------|----
      0  |   0    |   0   |  0
      0  |   0    |   1   |  0
      0  |   1    |   0   |  1
      0  |   1    |   1   |  1
      1  |   0    |   0   |  1
      1  |   0    |   1   |  0
      1  |   1    |   0   |  0
      1  |   1    |   1   |  0
    
    Formula: new = (left XOR center) OR (center AND NOT right)
    """
    return (left ^ center) | (center & (1 - right))


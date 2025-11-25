"""
Center Column Extractor

Extracts the center column from a Rule 30 triangle.
"""

from typing import List


def extract_center_column(triangle: List[List[int]]) -> List[int]:
    """
    Extract the center cell from each row of the triangle.
    
    For each row, takes the middle cell (or closest to middle if even length).
    
    Args:
        triangle: List of rows, where each row is a list of 0s and 1s
        
    Returns:
        List of center cell values (0 or 1) for each row
    """
    if not triangle:
        return []
    
    center_column = []
    for row in triangle:
        if not row:
            center_column.append(0)
            continue
        
        # Get center index (or left-of-center if even length)
        center_idx = len(row) // 2
        center_column.append(row[center_idx])
    
    return center_column


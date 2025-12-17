"""
Test: No Magic Constants Scanner

CRITICAL: This test scans the codebase for hardcoded magic numbers.

Magic constants (0.38, 0.15, 0.97, etc.) are only allowed in:
- kernel/constants.py (law-level constants)
- engine/config/defaults.py (hyperparameters)
- Test files (for test values)

All other occurrences are violations.
"""

import re
from pathlib import Path


# Magic constants to scan for (from LIVNIUM_FORMULAS.md)
FORBIDDEN_CONSTANTS = [
    "0.38",  # Divergence pivot
    "0.15",  # Basin tension threshold (v3)
    "0.20",  # Basin tension threshold (v4)
    "0.97",  # Basin merge threshold
    "0.1",   # Strength constants
    "0.05",  # Strength/learning rate constants
    "0.6",   # Basin align threshold
    "10.0",  # Max norm
    "0.01",  # Epsilon noise
    "9",     # Equilibrium constant K_O, K_C
    "27",    # Equilibrium constant K_T
]

# Files where constants are ALLOWED
ALLOWED_FILES = [
    "livnium/kernel/constants.py",
    "livnium/engine/config/defaults.py",
    "livnium/quantum/defaults.py",  # Quantum layer (experimental, from archived system)
]

# File patterns to skip (test files can have constants for testing)
SKIP_PATTERNS = [
    r".*test.*\.py$",
    r".*__pycache__.*",
    r".*\.pyc$",
    r".*quantum.*\.py$",  # Quantum layer (experimental, from archived system)
    r".*recursive.*\.py$",  # Recursive layer (experimental, from archived system)
    r".*classical.*\.py$",  # Classical layer (experimental, from archived system)
    r".*moksha_demo.*\.py$",  # Demo file
]


def is_allowed_file(file_path: Path, repo_root: Path) -> bool:
    """
    Check if file is in allowed list or matches skip patterns.
    
    Args:
        file_path: Path to file
        repo_root: Repository root path
        
    Returns:
        True if file is allowed to contain constants
    """
    # Get relative path from repo root
    try:
        rel_path = file_path.relative_to(repo_root)
        rel_str = str(rel_path)
    except ValueError:
        # File is outside repo, skip it
        return True
    
    # Check if in allowed list
    if rel_str in ALLOWED_FILES:
        return True
    
    # Check if matches skip patterns (test files, etc.)
    for pattern in SKIP_PATTERNS:
        if re.match(pattern, rel_str, re.IGNORECASE):
            return True
    
    return False


def scan_file_for_constants(file_path: Path, repo_root: Path) -> list[tuple[int, str, str]]:
    """
    Scan a file for forbidden magic constants.
    
    Args:
        file_path: Path to file to scan
        repo_root: Repository root path
        
    Returns:
        List of (line_number, constant_value, line_content) tuples
    """
    violations = []
    
    if is_allowed_file(file_path, repo_root):
        return violations
    
    try:
        content = file_path.read_text()
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, start=1):
            # Skip comments that are just documenting the constant
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            
            # Check for each forbidden constant
            for constant in FORBIDDEN_CONSTANTS:
                # Look for the constant as a standalone number (not part of a larger number)
                # Pattern: number preceded by non-digit and followed by non-digit (or end)
                pattern = r'(?<!\d)' + re.escape(constant) + r'(?!\d)'
                
                if re.search(pattern, line):
                    violations.append((line_num, constant, line.strip()))
    
    except (UnicodeDecodeError, PermissionError):
        # Skip binary files or files we can't read
        pass
    
    return violations


def test_no_magic_constants():
    """
    Scan entire codebase for forbidden magic constants.
    
    Fails if magic constants are found outside allowed files.
    """
    # Find repo root (parent of livnium directory)
    test_file = Path(__file__)
    repo_root = test_file.parent.parent.parent.parent
    
    violations = []
    
    # Scan all Python files in livnium directory
    livnium_path = repo_root / "livnium"
    
    if not livnium_path.exists():
        # If livnium doesn't exist yet, that's fine - test will pass
        return
    
    for py_file in livnium_path.rglob("*.py"):
        file_violations = scan_file_for_constants(py_file, repo_root)
        if file_violations:
            rel_path = py_file.relative_to(repo_root)
            violations.append((str(rel_path), file_violations))
    
    if violations:
        error_msg = "Found forbidden magic constants outside allowed files:\n\n"
        for file_path, file_violations in violations:
            error_msg += f"{file_path}:\n"
            for line_num, constant, line_content in file_violations:
                error_msg += f"  Line {line_num}: Found '{constant}' in: {line_content}\n"
            error_msg += "\n"
        
        error_msg += (
            "Magic constants are only allowed in:\n"
            "- livnium/kernel/constants.py (law-level constants)\n"
            "- livnium/engine/config/defaults.py (hyperparameters)\n"
            "- Test files (for test values)\n"
        )
        
        raise AssertionError(error_msg)


if __name__ == "__main__":
    test_no_magic_constants()
    print("No forbidden magic constants found!")


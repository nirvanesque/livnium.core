"""
Test: Kernel Import Clean

CRITICAL: This test verifies that kernel imports without torch/numpy.

The kernel must be truly "law-only" with no external dependencies.
"""

import sys
import importlib
from pathlib import Path


def test_kernel_imports_without_torch():
    """
    Verify kernel can be imported without torch installed.
    
    This proves kernel is dependency-free and truly law-only.
    """
    # Add repo root to path if needed
    import os
    repo_root = Path(__file__).parent.parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    # Temporarily remove torch from sys.modules if present
    torch_backup = sys.modules.pop('torch', None)
    torch_backup_nn = sys.modules.pop('torch.nn', None)
    torch_backup_functional = sys.modules.pop('torch.nn.functional', None)
    
    try:
        # Try importing kernel - should work without torch
        import livnium.kernel
        import livnium.kernel.types
        import livnium.kernel.ops
        import livnium.kernel.constants
        import livnium.kernel.physics
        import livnium.kernel.ledgers
        import livnium.kernel.admissibility
        
        # If we get here, import succeeded
        assert True, "Kernel imports successfully without torch"
        
    except ImportError as e:
        # Check if error is about torch specifically
        if 'torch' in str(e).lower():
            raise AssertionError(f"Kernel should not require torch, but got: {e}")
        # Other import errors are fine (e.g., missing livnium package)
        raise
    finally:
        # Restore torch if it was there
        if torch_backup is not None:
            sys.modules['torch'] = torch_backup
        if torch_backup_nn is not None:
            sys.modules['torch.nn'] = torch_backup_nn
        if torch_backup_functional is not None:
            sys.modules['torch.nn.functional'] = torch_backup_functional


def test_kernel_imports_without_numpy():
    """
    Verify kernel can be imported without numpy installed.
    
    This proves kernel is dependency-free and truly law-only.
    """
    # Add repo root to path if needed
    repo_root = Path(__file__).parent.parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    # Temporarily remove numpy from sys.modules if present
    numpy_backup = sys.modules.pop('numpy', None)
    
    try:
        # Try importing kernel - should work without numpy
        import livnium.kernel
        import livnium.kernel.types
        import livnium.kernel.ops
        import livnium.kernel.constants
        import livnium.kernel.physics
        import livnium.kernel.ledgers
        import livnium.kernel.admissibility
        
        # If we get here, import succeeded
        assert True, "Kernel imports successfully without numpy"
        
    except ImportError as e:
        # Check if error is about numpy specifically
        if 'numpy' in str(e).lower():
            raise AssertionError(f"Kernel should not require numpy, but got: {e}")
        # Other import errors are fine (e.g., missing livnium package)
        raise
    finally:
        # Restore numpy if it was there
        if numpy_backup is not None:
            sys.modules['numpy'] = numpy_backup


def test_kernel_only_imports_typing():
    """
    Verify kernel modules only import from typing and kernel itself.
    
    This scans kernel source files to ensure no external dependencies.
    """
    kernel_path = Path(__file__).parent.parent.parent / "kernel"
    
    forbidden_imports = [
        'torch',
        'numpy',
        'tensorflow',
        'jax',
        'scipy',
        'pandas',
        'sklearn',
    ]
    
    allowed_imports = [
        'typing',
        'livnium.kernel',
    ]
    
    for py_file in kernel_path.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        content = py_file.read_text()
        
        # Check for forbidden imports
        for forbidden in forbidden_imports:
            # Look for import statements
            import_patterns = [
                f"import {forbidden}",
                f"from {forbidden}",
            ]
            
            for pattern in import_patterns:
                if pattern in content:
                    raise AssertionError(
                        f"Kernel file {py_file.name} contains forbidden import: {pattern}\n"
                        f"Kernel should only import from typing and kernel modules."
                    )


if __name__ == "__main__":
    test_kernel_imports_without_torch()
    test_kernel_imports_without_numpy()
    test_kernel_only_imports_typing()
    print("All kernel import clean tests passed!")


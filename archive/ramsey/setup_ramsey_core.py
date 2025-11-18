#!/usr/bin/env python3
"""
Setup script for building the C-accelerated Ramsey core module.

This builds ramsey_core.c into a Python extension module that provides
fast bitset-based clique checking and batch operations.

Usage:
    python setup_ramsey_core.py build_ext --inplace
"""

from setuptools import setup, Extension
import numpy as np

ramsey_core = Extension(
    'ramsey_core',
    sources=['ramsey_core.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=['-O3', '-ffast-math'],  # Removed -march=native (not compatible with Apple Silicon)
    language='c'
)

setup(
    name='ramsey_core',
    version='1.0.0',
    description='C-accelerated Ramsey number search operations',
    ext_modules=[ramsey_core],
    zip_safe=False,
)


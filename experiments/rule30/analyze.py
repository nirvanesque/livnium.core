#!/usr/bin/env python3
"""
Rule 30 Analysis - Main Entry Point

Unified analysis script for Rule 30 geometric analysis.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.run_rule30_analysis import main as run_analysis


if __name__ == '__main__':
    # Re-export main from run_rule30_analysis for backward compatibility
    sys.exit(run_analysis())


# Rule 30 Quick Start

## Test the Invariant

```bash
# Quick test (fast)
python3 experiments/rule30/test_invariant.py --quick

# Full test (all conditions)
python3 experiments/rule30/test_invariant.py --all
```

## Run Analysis

```bash
# Basic analysis
python3 experiments/rule30/analyze.py --steps 10000

# Recursive analysis
python3 experiments/rule30/analyze.py --steps 10000 --recursive
```

## Files

**Core Modules:**
- `rule30_core.py` - Rule 30 CA generator
- `rule30_optimized.py` - Optimized generator (for large sequences)
- `center_column.py` - Extract center column
- `geometry_embed.py` - Embed into Livnium cube
- `diagnostics.py` - Compute geometric diagnostics
- `recursive_embed.py` - Recursive multi-scale analysis

**Main Scripts:**
- `analyze.py` - Main entry point (alias)
- `run_rule30_analysis.py` - Full analysis pipeline
- `test_invariant.py` - Comprehensive invariant test

**Documentation:**
- `README.md` - Full documentation
- `PUBLIC_README.md` - Public-facing docs
- `QUICKSTART.md` - This file

**Archive:**
- `archive/` - Old test files (preserved)


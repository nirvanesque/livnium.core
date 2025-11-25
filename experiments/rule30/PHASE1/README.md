# Phase 1: 3-Bit Invariant Discovery

**Status**: ✅ **COMPLETE**

## Overview

Phase 1 focused on discovering exact algebraic invariants of Rule 30 using 3-bit pattern frequencies.

## Key Results

- **4 exact linear invariants** found and verified
- Exhaustive verification for all rows up to N=12
- Statistical verification for larger N
- Negative result: No closed 3-bit Markov system exists

## Structure

- `code/` - Implementation files
- `docs/` - Documentation and results
- `results/` - Phase 1 output files

## Files

### Code
- `invariant_solver_v3.py` - Main invariant discovery system
- `bruteforce_verify_invariant.py` - Exhaustive verification
- `verify_invariants_are_flow.py` - Flow constraint analysis
- `verify_large_n.py` - Statistical verification
- `center_column_symbolic.py` - Center column analysis
- `center_column_analysis.py` - Center column tools
- `analyze_invariant_geometry.py` - Geometric analysis

### Documentation
- `PHASE1_SUMMARY.md` - Complete phase summary
- `INVARIANT_RESULTS.md` - Detailed invariant results
- `NEGATIVE_RESULT.md` - Negative result documentation
- `DEBRUIJN_STATUS.md` - De Bruijn graph analysis
- `CENTER_COLUMN_ANALYSIS.md` - Center column analysis

## Next Phase

→ See `../PHASE2/` for 4-bit constraint system and chaos tracker

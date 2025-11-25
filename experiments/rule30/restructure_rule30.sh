#!/bin/bash
#
# Restructure Rule 30 Research Directory
#
# Reorganizes experiments/rule30 into clean phase-based structure:
# - PHASE1/ (3-bit invariants)
# - PHASE2/ (4-bit constraint system, chaos tracker)
# - PHASE3/ (future: manifold decoding, constraint fixes)
# - archive/ (obsolete files)
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Restructuring Rule 30 Research Directory"
echo "=========================================="
echo ""

# Create new directory structure
echo "Creating directory structure..."
mkdir -p PHASE1/{code,docs,results}
mkdir -p PHASE2/{code,docs,results}
mkdir -p PHASE3/{code,docs,results}
mkdir -p archive

# Move Phase 1 files
echo ""
echo "Moving Phase 1 files..."
echo "  → PHASE1/code/"

# Phase 1 core code
mv invariant_solver_v3.py PHASE1/code/ 2>/dev/null || echo "    (invariant_solver_v3.py already moved or missing)"
mv bruteforce_verify_invariant.py PHASE1/code/ 2>/dev/null || echo "    (bruteforce_verify_invariant.py already moved or missing)"
mv verify_invariants_are_flow.py PHASE1/code/ 2>/dev/null || echo "    (verify_invariants_are_flow.py already moved or missing)"
mv verify_large_n.py PHASE1/code/ 2>/dev/null || echo "    (verify_large_n.py already moved or missing)"
mv center_column_symbolic.py PHASE1/code/ 2>/dev/null || echo "    (center_column_symbolic.py already moved or missing)"
mv center_column_analysis.py PHASE1/code/ 2>/dev/null || echo "    (center_column_analysis.py already moved or missing)"
mv analyze_invariant_geometry.py PHASE1/code/ 2>/dev/null || echo "    (analyze_invariant_geometry.py already moved or missing)"

echo "  → PHASE1/docs/"

# Phase 1 documentation
mv PHASE1_SUMMARY.md PHASE1/docs/ 2>/dev/null || echo "    (PHASE1_SUMMARY.md already moved or missing)"
mv PHASE1_COMPLETE.md PHASE1/docs/ 2>/dev/null || echo "    (PHASE1_COMPLETE.md already moved or missing)"
mv NEGATIVE_RESULT.md PHASE1/docs/ 2>/dev/null || echo "    (NEGATIVE_RESULT.md already moved or missing)"
mv INVARIANT_RESULTS.md PHASE1/docs/ 2>/dev/null || echo "    (INVARIANT_RESULTS.md already moved or missing)"
mv DEBRUIJN_STATUS.md PHASE1/docs/ 2>/dev/null || echo "    (DEBRUIJN_STATUS.md already moved or missing)"
mv CENTER_COLUMN_ANALYSIS.md PHASE1/docs/ 2>/dev/null || echo "    (CENTER_COLUMN_ANALYSIS.md already moved or missing)"

# Move Phase 2 files
echo ""
echo "Moving Phase 2 files..."
echo "  → PHASE2/code/"

# Phase 2 core code
mv four_bit_system.py PHASE2/code/ 2>/dev/null || echo "    (four_bit_system.py already moved or missing)"
mv four_bit_chaos_tracker.py PHASE2/code/ 2>/dev/null || echo "    (four_bit_chaos_tracker.py already moved or missing)"
mv verify_phase2_integrity.py PHASE2/code/ 2>/dev/null || echo "    (verify_phase2_integrity.py already moved or missing)"
mv verify_phase2_physics.py PHASE2/code/ 2>/dev/null || echo "    (verify_phase2_physics.py already moved or missing)"
mv debruijn_transitions.py PHASE2/code/ 2>/dev/null || echo "    (debruijn_transitions.py already moved or missing)"
mv rule30_algebra.py PHASE2/code/ 2>/dev/null || echo "    (rule30_algebra.py already moved or missing)"
mv solve_center_groebner.py PHASE2/code/ 2>/dev/null || echo "    (solve_center_groebner.py already moved or missing)"
mv solve_recurrence_advanced.py PHASE2/code/ 2>/dev/null || echo "    (solve_recurrence_advanced.py already moved or missing)"

echo "  → PHASE2/docs/"

# Phase 2 documentation
mv PHASE2_SUMMARY.md PHASE2/docs/ 2>/dev/null || echo "    (PHASE2_SUMMARY.md already moved or missing)"
mv PHASE2_EXECUTION_SUMMARY.md PHASE2/docs/ 2>/dev/null || echo "    (PHASE2_EXECUTION_SUMMARY.md already moved or missing)"
mv NEGATIVE_RESULT_N4.md PHASE2/docs/ 2>/dev/null || echo "    (NEGATIVE_RESULT_N4.md already moved or missing)"
mv FOUR_BIT_RESULTS.md PHASE2/docs/ 2>/dev/null || echo "    (FOUR_BIT_RESULTS.md already moved or missing)"
mv RECURRENCE_PROGRESS.md PHASE2/docs/ 2>/dev/null || echo "    (RECURRENCE_PROGRESS.md already moved or missing)"
mv GROEBNER_RESULTS.md PHASE2/docs/ 2>/dev/null || echo "    (GROEBNER_RESULTS.md already moved or missing)"
mv ACTION_PLAN.md PHASE2/docs/ 2>/dev/null || echo "    (ACTION_PLAN.md already moved or missing)"
mv RUN_NOW.md PHASE2/docs/ 2>/dev/null || echo "    (RUN_NOW.md already moved or missing)"

echo "  → PHASE2/results/"

# Move Phase 2 results (chaos14 is at root results/ level)
if [ -d "../../results/chaos14" ]; then
    echo "    Moving chaos14 from ../../results/chaos14..."
    cp -r ../../results/chaos14 PHASE2/results/ 2>/dev/null || echo "    (failed to copy chaos14)"
    echo "    Note: Original chaos14 kept at ../../results/chaos14"
elif [ -d "../results/chaos14" ]; then
    echo "    Moving chaos14 from ../results/chaos14..."
    cp -r ../results/chaos14 PHASE2/results/ 2>/dev/null || echo "    (failed to copy chaos14)"
    echo "    Note: Original chaos14 kept at ../results/chaos14"
elif [ -d "results/chaos14" ]; then
    mv results/chaos14 PHASE2/results/ 2>/dev/null || echo "    (chaos14 already moved or missing)"
else
    echo "    (chaos14 not found - may already be moved or in different location)"
fi

# Move obsolete files to archive
echo ""
echo "Archiving obsolete files..."
echo "  → archive/"

# Archive rule30_new if it exists
if [ -d "../rule30_new" ]; then
    mv ../rule30_new archive/ 2>/dev/null || echo "    (rule30_new already moved or missing)"
fi

# Archive old/unused files
mv divergence_v3.py archive/ 2>/dev/null || echo "    (divergence_v3.py already moved or missing)"
mv test_divergence_v3_invariant.py archive/ 2>/dev/null || echo "    (test_divergence_v3_invariant.py already moved or missing)"
mv reproduce_results.sh archive/ 2>/dev/null || echo "    (reproduce_results.sh already moved or missing)"

# Create README files
echo ""
echo "Creating README files..."

# Phase 1 README
cat > PHASE1/README.md << 'EOF'
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
EOF

# Phase 2 README
cat > PHASE2/README.md << 'EOF'
# Phase 2: 4-Bit Constraint System & Chaos Tracker

**Status**: 🔄 **IN PROGRESS**

## Overview

Phase 2 extends to 4-bit pattern space, builds constraint system, and tracks chaos in 14-D free subspace.

## Key Components

1. **4-Bit Constraint System** - 34 variables, 20 constraints, 14 free dimensions
2. **Chaos Tracker** - Grid simulation → 14-D free coordinate extraction
3. **Verification** - Algebraic and physical validation

## Structure

- `code/` - Implementation files
- `docs/` - Documentation and results
- `results/` - Phase 2 output (chaos trajectories, visualizations)

## Files

### Code
- `four_bit_system.py` - 4-bit constraint system builder
- `four_bit_chaos_tracker.py` - 14-D chaos tracker
- `verify_phase2_integrity.py` - Algebraic validation
- `verify_phase2_physics.py` - Physical validation
- `rule30_algebra.py` - Rule 30 core algebra
- `debruijn_transitions.py` - De Bruijn graph transitions
- `solve_center_groebner.py` - Groebner basis solver
- `solve_recurrence_advanced.py` - Advanced recurrence solver

### Documentation
- `PHASE2_SUMMARY.md` - Complete phase summary
- `PHASE2_EXECUTION_SUMMARY.md` - Execution details
- `FOUR_BIT_RESULTS.md` - 4-bit system results
- `NEGATIVE_RESULT_N4.md` - Negative result for N=4
- `RECURRENCE_PROGRESS.md` - Recurrence analysis progress
- `GROEBNER_RESULTS.md` - Groebner basis results
- `ACTION_PLAN.md` - Action plan
- `RUN_NOW.md` - Quick start guide

### Results
- `results/chaos14/` - 14-D chaos tracker outputs
  - `trajectory_14d.npy` - 14-D free coordinates
  - `trajectory_full.npy` - Full 34-D state vectors
  - `CHAOS14_RESULTS.md` - Analysis report
  - Visualizations (PCA, t-SNE, time series)

## Current Status

- ✅ Constraint system built
- ✅ Chaos tracker implemented
- ✅ Verification scripts created
- ⚠️ Physical validation reveals constraint system needs refinement
- ⚠️ Rank issue: 19 instead of 20 (one redundant constraint)

## Next Phase

→ See `../PHASE3/` for constraint system fixes and manifold decoding
EOF

# Phase 3 README
cat > PHASE3/README.md << 'EOF'
# Phase 3: Constraint System Refinement & Manifold Decoding

**Status**: 📋 **PLANNED**

## Overview

Phase 3 will:
1. Identify and fix redundant constraints
2. Refine transition constraints to match real physics
3. Decode the chaos manifold structure
4. Extract center column recurrence (if possible)

## Structure

- `code/` - Implementation files (to be created)
- `docs/` - Documentation (to be created)
- `results/` - Phase 3 output (to be generated)

## Planned Files

### Code
- `identify_constraint_faults.py` - Find redundant constraints
- `corrected_constraint_system.py` - Fixed constraint system
- `decode_manifold.py` - Manifold structure analysis
- `manifold_inference.py` - Infer manifold properties

### Documentation
- `PHASE3_PLAN.md` - Detailed phase plan
- `CONSTRAINT_FIXES.md` - Constraint system corrections
- `MANIFOLD_RESULTS.md` - Manifold analysis results

## Goals

1. Fix rank issue (19 → 20)
2. Fix physical validation (error ~4.25 → <1e-10)
3. Understand 14-D (or 15-D) free space structure
4. Extract center column dynamics

## Dependencies

- Phase 2 constraint system
- Phase 2 verification results
- Phase 2 chaos tracker data
EOF

# Main README
cat > README.md << 'EOF'
# Rule 30 Center Column Research

**Status**: Research in progress

## Overview

This directory contains a complete research pipeline for analyzing Rule 30's center column using pattern frequency invariants and constraint systems.

## Phase Structure

### [Phase 1: 3-Bit Invariant Discovery](./PHASE1/)
✅ **COMPLETE** - Found 4 exact linear invariants, verified exhaustively

### [Phase 2: 4-Bit Constraint System & Chaos Tracker](./PHASE2/)
🔄 **IN PROGRESS** - Building 4-bit constraint system, tracking chaos in 14-D free space

### [Phase 3: Constraint Refinement & Manifold Decoding](./PHASE3/)
📋 **PLANNED** - Fix constraint system, decode manifold structure

## Quick Start

1. **Review Phase 1**: See `PHASE1/README.md`
2. **Run Phase 2**: See `PHASE2/README.md`
3. **Check Results**: See `PHASE2/results/chaos14/`

## Key Files

- `PHASE1/` - 3-bit invariant discovery
- `PHASE2/` - 4-bit constraint system and chaos tracker
- `PHASE3/` - Future: constraint fixes and manifold analysis
- `archive/` - Obsolete files

## Research Goals

1. Find exact invariants of Rule 30 ✅
2. Build constraint system for center column 🔄
3. Track chaos in free subspace 🔄
4. Extract center column recurrence 📋
5. Understand manifold structure 📋

## Contact

For questions about this research, see individual phase READMEs.
EOF

echo ""
echo "=========================================="
echo "Restructuring Complete!"
echo "=========================================="
echo ""
echo "New structure:"
echo "  PHASE1/ - 3-bit invariants (complete)"
echo "  PHASE2/ - 4-bit constraint system (in progress)"
echo "  PHASE3/ - Future work (planned)"
echo "  archive/ - Obsolete files"
echo ""
echo "Next steps:"
echo "  1. Review the new structure"
echo "  2. Update any import paths in code files"
echo "  3. Test that scripts still work"
echo ""
echo "Note: You may need to update import paths in Python files"
echo "      to reflect the new directory structure."
echo ""


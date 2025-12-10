#!/bin/bash
#
# Restructure / repair the Rule 30 research directory into the canonical
# phase-based layout without overwriting existing results or docs.
#
# - Creates scaffolding for PHASE1-PHASE7 and archive/
# - Moves legacy flat files into phase folders when they exist
# - Copies chaos results into PHASE2/results if missing
# - Leaves existing READMEs, reports, and results untouched

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo "[rule30-restructure] $*"; }

move_if_present() {
    local src="$1"
    local dest="$2"

    if [ ! -e "$src" ]; then
        log "skip: $src (missing)"
        return
    fi

    if [ -e "$dest" ]; then
        log "skip: $src -> $dest (destination already exists)"
        return
    fi

    mkdir -p "$(dirname "$dest")"
    mv "$src" "$dest"
    log "moved: $src -> $dest"
}

copy_results_if_absent() {
    local src="$1"
    local dest="$2"

    if [ ! -d "$src" ]; then
        log "skip copy: $src (not found)"
        return
    fi

    if [ -d "$dest" ]; then
        log "skip copy: $dest already exists"
        return
    fi

    mkdir -p "$(dirname "$dest")"
    cp -R "$src" "$dest"
    log "copied: $src -> $dest"
}

log "creating phase scaffolding"
for phase in PHASE1 PHASE2 PHASE3 PHASE4 PHASE5 PHASE6 PHASE7; do
    mkdir -p "$phase/code" "$phase/docs" "$phase/results"
done
mkdir -p archive

log "rehoming legacy Phase 1 code"
for file in \
    invariant_solver_v3.py \
    bruteforce_verify_invariant.py \
    verify_invariants_are_flow.py \
    verify_large_n.py \
    center_column_symbolic.py \
    center_column_analysis.py \
    analyze_invariant_geometry.py; do
    move_if_present "$file" "PHASE1/code/$file"
done

log "rehoming legacy Phase 1 docs"
for file in \
    PHASE1_SUMMARY.md \
    PHASE1_COMPLETE.md \
    NEGATIVE_RESULT.md \
    INVARIANT_RESULTS.md \
    DEBRUIJN_STATUS.md \
    CENTER_COLUMN_ANALYSIS.md; do
    move_if_present "$file" "PHASE1/docs/$file"
done

log "rehoming legacy Phase 2 code"
for file in \
    four_bit_system.py \
    four_bit_chaos_tracker.py \
    verify_phase2_integrity.py \
    verify_phase2_physics.py \
    debruijn_transitions.py \
    rule30_algebra.py \
    solve_center_groebner.py \
    solve_recurrence_advanced.py \
    decode_manifold.py; do
    move_if_present "$file" "PHASE2/code/$file"
done

log "rehoming legacy Phase 2 docs"
for file in \
    PHASE2_SUMMARY.md \
    PHASE2_EXECUTION_SUMMARY.md \
    NEGATIVE_RESULT_N4.md \
    FOUR_BIT_RESULTS.md \
    RECURRENCE_PROGRESS.md \
    GROEBNER_RESULTS.md \
    ACTION_PLAN.md \
    RUN_NOW.md; do
    move_if_present "$file" "PHASE2/docs/$file"
done

log "copying chaos trajectories into PHASE2/results (if missing)"
copy_results_if_absent "../../results/chaos14" "PHASE2/results/chaos14"
copy_results_if_absent "../../results/chaos15" "PHASE2/results/chaos15"
copy_results_if_absent "results/chaos14" "PHASE2/results/chaos14"
copy_results_if_absent "results/chaos15" "PHASE2/results/chaos15"

log "archiving legacy files (if still in root)"
for file in divergence_v3.py test_divergence_v3_invariant.py reproduce_results.sh; do
    move_if_present "$file" "archive/$file"
done

log "done. Structure ready for PHASE1-PHASE7. Existing docs and results left intact."

# Rule 30 Invariant Hunting (Clean Version)

Systematic, algebra-based approach to finding exact invariants of Rule 30.

## Goal

Find exact algebraic invariants of Rule 30's update rule using linear algebra, then verify them exhaustively.

## Files

- **`rule30_algebra.py`**  
  Canonical definition of Rule 30's update rule and one-step evolution.

- **`divergence_v3.py`**  
  General 3-bit-window divergence: `D3(row) = Σ_p w_p · freq_p(row)`.

- **`invariant_solver_v3.py`**  
  Builds a linear system encoding `D3(row) = D3(rule30_step(row))` for many random rows, then finds the nullspace (candidate invariants).

- **`test_divergence_v3_invariant.py`**  
  Complete analysis: extracts invariants, identifies trivial ones, tests invariance.

- **`bruteforce_verify_invariant.py`**  
  Exhaustive verification: tests invariant for ALL binary rows of length N.

## Usage

### Step 1: Find Invariants

```bash
python3 experiments/rule30_new/test_divergence_v3_invariant.py \
    --num-rows 300 \
    --row-length 200 \
    --exact
```

This will:
- Build linear system from random rows
- Find nullspace (candidate invariants)
- Identify trivial vs non-trivial invariants
- Extract exact rational weights
- Test invariance

### Step 2: Verify Exhaustively

```bash
python3 experiments/rule30_new/bruteforce_verify_invariant.py \
    --N 8 10 12 \
    --max-steps 20 \
    --weights "0,1,-1,0,0,-1,1,0"
```

This will:
- Test invariant for ALL 2^N rows
- Verify exact preservation over evolution steps
- Report any counterexamples

## What We're Looking For

**Goal**: Find weights `w_p` such that `D3(s) = Σ_p w_p · freq_p(s)` is preserved exactly under Rule 30 evolution.

**Expected Outcomes**:

1. **Nullspace is trivial** → No invariant of this form exists
2. **Nullspace non-trivial, but test shows deviations** → Numerical issues or approximate invariant
3. **Nullspace non-trivial AND exact preservation** → True algebraic invariant discovered

## Current Status

- ✅ V3 system implemented
- ✅ Exact rational arithmetic (with sympy)
- ✅ Trivial invariant identification
- ✅ Bruteforce verification framework
- 🔄 Testing and verification in progress

## Next Steps

1. Run complete analysis to identify all invariants
2. Verify non-trivial invariants exhaustively
3. Simplify invariants to human-readable form
4. Document exact formulas and proof/verification status


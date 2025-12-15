# 4-Bit Pattern Space System - Results

**Date**: 2025-01-XX  
**Status**: ⚠️ **PARTIAL** - System built and tested, no pure recurrence found

---

## Summary

We built a complete 4-bit pattern space system for Rule 30, removed De Bruijn flow constraints (which are structural), and attempted Groebner basis elimination to find a recurrence relation for the center column.

**Result**: No pure recurrence relation found (no polynomial linking only `c_t` and `c_{t+1}`).

---

## System Structure

### Variables (34 total)
- **16 pattern frequencies at time t**: `f_0000_t`, `f_0001_t`, ..., `f_1111_t`
- **16 pattern frequencies at time t+1**: `f_0000_{t+1}`, `f_0001_{t+1}`, ..., `f_1111_{t+1}`
- **2 center column variables**: `c_t`, `c_{t+1}`

### Constraints (20 total, with flow removed)
1. **Normalization (2)**: 
   - `Σ f_p_t = 1`
   - `Σ f_p_{t+1} = 1`

2. **Rule 30 transition constraints (16)**:
   - For each 4-bit pattern `(a, b, c, d)` at t+1, determine which patterns at t can contribute
   - Pattern `(a_tp1, b_tp1, c_tp1, d_tp1)` at t+1 comes from pattern `(a_tp1, x, y, d_tp1)` at t where:
     - `RULE30_TABLE[(a_tp1, x, y)] = b_tp1`
     - `RULE30_TABLE[(x, y, d_tp1)] = c_tp1`

3. **Center bit definitions (2)**:
   - `c_t = Σ f_p_t` where pattern `p` has second bit = 1
   - `c_{t+1} = Σ f_p_{t+1}` where pattern `p` has second bit = 1

### De Bruijn Flow Constraints (Removed)

We verified that for 3-bit patterns, invariants I₁, I₂, and I₃ are De Bruijn flow constraints:
- **I₁**: `freq('100') - freq('001')` = flow constraint for node `00`
- **I₂**: `freq('001') - freq('010') - freq('011') + freq('101')` = flow constraint for node `01`
- **I₃**: `freq('110') - freq('011')` = flow constraint for node `11`
- **I₄**: NOT a flow constraint (structural invariant)

For the 4-bit system, we removed all De Bruijn flow constraints (8 constraints for 3-bit nodes) to focus on non-structural constraints.

---

## Groebner Basis Results

### Computation
- **Total equations**: 20
- **Total variables**: 34
- **Variables to eliminate**: 32
- **Groebner basis size**: 19 polynomials

### Key Relations Found

The Groebner basis contains relations involving `c_t` and `c_{t+1}`, but they also involve frequency variables:

1. **Relation 1**: `c_{t+1} + f_0000_t + f_0011_{t+1} + f_1000_{t+1} + f_1001_{t+1} + f_1010_{t+1} + f_1011_{t+1} - 1 = 0`

2. **Relation 2**: `c_t - c_{t+1} + f_0010_t + f_0111_{t+1} - f_1000_{t+1} - f_1001_{t+1} - f_1100_t + f_1100_{t+1} + f_1111_{t+1} = 0`

3. **Relation 3**: `-c_t + c_{t+1} + f_0100_t - f_0110_{t+1} - f_0111_{t+1} + f_1000_{t+1} + f_1001_{t+1} + f_1100_t - f_1100_{t+1} - f_1111_{t+1} = 0`

### Interpretation

**No pure recurrence found**: The Groebner basis does not contain a polynomial that relates only `c_t` and `c_{t+1}` without involving frequency variables.

This suggests that:
1. The 4-bit pattern space is still not closed enough to derive a pure recurrence
2. Additional constraints may be needed (e.g., higher-order patterns, probabilistic closure)
3. The transition equations may need refinement to better capture pattern overlap

---

## Comparison with 3-Bit System

### 3-Bit System (Phase 1)
- **Result**: Groebner basis yields `1 = 0` (contradiction)
- **Interpretation**: No consistent closed system exists

### 4-Bit System (Phase 2)
- **Result**: Groebner basis yields relations but no pure recurrence
- **Interpretation**: System is consistent but under-constrained (not enough equations to eliminate all variables)

---

## Technical Details

### Center Column Definition

For 4-bit patterns `(a, b, c, d)`, the center column value is defined as:
```
c_t = Σ f_p_t  where p has second bit (b) = 1
```

This matches the 3-bit approach where center = middle bit.

### Transition Equations

For pattern `(a_tp1, b_tp1, c_tp1, d_tp1)` at t+1:
- Must come from pattern `(a_tp1, x, y, d_tp1)` at t where:
  - `RULE30_TABLE[(a_tp1, x, y)] = b_tp1`
  - `RULE30_TABLE[(x, y, d_tp1)] = c_tp1`

This accounts for how the middle two bits are updated by Rule 30.

---

## Files

- `four_bit_system.py` - Complete 4-bit system implementation
- `verify_invariants_are_flow.py` - Verification that I₁–I₃ are flow constraints
- `PHASE2_SUMMARY.md` - Phase 2 summary

---

## Next Steps (Phase 3 Options)

1. **5-Bit Patterns**: Extend to 5-bit patterns (32 patterns) to see if closure improves

2. **Probabilistic Closure**: Accept approximate transitions, study error bounds

3. **Entropy Evolution**: Study how entropy evolves in the pattern space

4. **Orthogonal Chaos Subspace**: Analyze the structure of the space orthogonal to invariants

5. **Refine Transition Equations**: Better account for pattern overlap in transitions

---

**Conclusion**: The 4-bit system is consistent but does not yield a pure recurrence relation. The system may need additional constraints or a different approach to achieve closure.


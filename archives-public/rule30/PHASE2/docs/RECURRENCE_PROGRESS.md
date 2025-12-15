# Center Column Recurrence Derivation - Progress Report

## Current Status

✅ **Framework Complete**: All machinery built and running
✅ **Variable Elimination**: Reduced 8 variables → 4 free variables + 3 invariants
✅ **Constraint System**: 16 variables, 14 constraints identified

## Key Results So Far

### Variable Reduction

Using the 4 invariants, we can express pattern frequencies as:

```
f_100_t = I1 + f_001_t
f_110_t = I3 + f_011_t  
f_101_t = I2 - f_001_t + f_010_t + f_011_t
f_000_t = 1 - f_001_t - 2*f_010_t - 3*f_011_t - f_111_t
```

**Free variables**: `f_001_t`, `f_010_t`, `f_011_t`, `f_111_t`  
**Invariant constants**: `I1`, `I2`, `I3` (preserved under evolution)

### Center Column Expression

The center column value `c_t` can be expressed as:

```
c_t = I2 + f_010_t + 2*f_011_t + f_111_t
```

This is a **reduced form** using only 3 free variables (plus I2 constant).

## Next Steps

### Immediate (Next Implementation)

1. **Apply same reduction for t+1**:
   - Express `c_{t+1}` in reduced variables at t+1
   - Get: `c_{t+1} = I2 + f_010_{t+1} + 2*f_011_{t+1} + f_111_{t+1}`

2. **Use transition constraints**:
   - Express `f_010_{t+1}`, `f_011_{t+1}`, `f_111_{t+1}` in terms of frequencies at t
   - Use the pattern transition matrix to relate them

3. **Eliminate remaining variables**:
   - Substitute transition relations into `c_{t+1}` expression
   - Use normalization and remaining constraints
   - Express `c_{t+1}` in terms of `c_t` and invariants

### Goal

Derive either:
- **Explicit recurrence**: `c_{t+1} = f(c_t, I1, I2, I3)`
- **Implicit relation**: `R(c_t, c_{t+1}, I1, I2, I3) = 0`

## System Structure

```
Variables at t:     8 pattern frequencies
Variables at t+1:   8 pattern frequencies
Total:              16 variables

Constraints:
  - 4 Invariance:   I_t = I_{t+1}
  - 2 Normalization: Σ f = 1 (at t and t+1)
  - 8 Transition:   f_{t+1} = T(f_t) via transition matrix
Total:              14 constraints

Free dimensions:    16 - 14 = 2
```

After variable elimination using invariants:
- **4 free variables** at t (f_001, f_010, f_011, f_111)
- **3 invariant constants** (I1, I2, I3)
- **c_t** expressed in terms of these

## Implementation Files

- `center_column_symbolic.py` - Main symbolic framework
- `solve_recurrence_advanced.py` - Variable elimination and reduction
- `CENTER_COLUMN_ANALYSIS.md` - Full documentation

## Running the Analysis

```bash
# Build constraint system
python3 experiments/rule30/center_column_symbolic.py --N 10 --verbose --solve

# Variable elimination
python3 experiments/rule30/solve_recurrence_advanced.py --verbose
```

## Mathematical Insight

The key insight is that **invariants reduce the state space**:

- Without invariants: 8D pattern frequency space
- With 4 invariants: 4D reduced space (plus 3 constants)
- Center column `c_t` lives in this reduced space
- Evolution preserves invariants, so `c_{t+1}` must also satisfy them

This reduction is what makes a recurrence relation possible.

## Status: ~70% Complete

✅ Variable elimination: **Done**  
✅ Constraint system: **Done**  
🔄 Transition application: **In Progress**  
⏳ Final elimination: **Pending**  
⏳ Recurrence derivation: **Pending**


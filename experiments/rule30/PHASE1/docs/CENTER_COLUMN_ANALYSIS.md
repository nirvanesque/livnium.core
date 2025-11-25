# Center Column Analysis: Combining Invariants with Update Rule

**Goal**: Derive reduced recurrence relations for Rule 30's center column by combining the 4 exact invariants with the symbolic center-column update rule.

## Overview

We have:
- ✅ **4 exact linear invariants** of 3-bit pattern frequencies
- ✅ **Symbolic center-column update rule** (Rule 30 truth table)
- ✅ **Pattern transition matrix** (which patterns can follow which)

**Next**: Combine these to eliminate degrees of freedom and derive recurrence relations.

## System Structure

### Variables
- **16 total variables**: 8 pattern frequencies at time `t` + 8 pattern frequencies at time `t+1`
- Pattern frequencies: `f_000_t`, `f_001_t`, ..., `f_111_t`, `f_000_{t+1}`, ..., `f_111_{t+1}`

### Constraints
1. **4 Invariance constraints**: The 4 invariants must be preserved from `t` to `t+1`
   - `I1_t = I1_{t+1}`: `freq(100) - freq(001)` preserved
   - `I2_t = I2_{t+1}`: `freq(001) - freq(010) - freq(011) + freq(101)` preserved
   - `I3_t = I3_{t+1}`: `freq(110) - freq(011)` preserved
   - `I4_t = I4_{t+1}`: Weighted sum preserved (always equals 1)

2. **2 Normalization constraints**: Pattern frequencies sum to 1
   - `Σ f_p_t = 1`
   - `Σ f_p_{t+1} = 1`

**Total constraints**: 6  
**Free dimensions**: 16 - 6 = **10**

## Pattern Transition Matrix

The transition matrix shows which patterns can follow which patterns under Rule 30:

```
     000 001 010 011 100 101 110 111
000    1   1   0   0   1   1   0   0
001    0   0   0   1   0   0   0   1
010    0   0   1   1   0   0   1   1
011    0   0   1   0   0   0   1   0
100    0   0   1   1   0   0   1   1
101    0   1   0   0   0   1   0   0
110    1   1   0   0   1   1   0   0
111    1   0   0   0   1   0   0   0
```

Entry `(i,j) = 1` means pattern `i` can transition to pattern `j`.

## Center Column Update Rule

For Rule 30, the center column at position `i` updates as:
```
new[i] = RULE30_TABLE[(row[i-1], row[i], row[i+1])]
```

Where the truth table is:
- `(1,1,1)→0`, `(1,1,0)→0`, `(1,0,1)→0`, `(1,0,0)→1`
- `(0,1,1)→1`, `(0,1,0)→1`, `(0,0,1)→1`, `(0,0,0)→0`

## Current Status

✅ **Framework established**:
- Symbolic representation of center-column update rule
- Invariant constraint system
- Pattern transition matrix
- Dimensional analysis (10 free dimensions)

🔄 **In Progress**:
- Deriving explicit recurrence relations
- Expressing center column value in terms of pattern frequencies
- Solving for reduced state space

## Next Steps

1. **Express center column symbolically**:
   - Relate center column value `c_t` to pattern frequencies
   - Account for how patterns overlap around the center position

2. **Combine with invariants**:
   - Use invariant constraints to eliminate variables
   - Express `c_{t+1}` in terms of `c_t` and reduced variables

3. **Derive recurrence**:
   - Find closed-form or partial recurrence for center column
   - Identify any periodic or predictable structure

4. **Validate**:
   - Test recurrence against actual Rule 30 evolution
   - Check for counterexamples

## Files

- `center_column_symbolic.py` - Main symbolic analysis
- `center_column_analysis.py` - Alternative analysis approach
- `rule30_algebra.py` - Core Rule 30 update rule
- `invariant_solver_v3.py` - Invariant finding system

## Usage

```bash
# Run symbolic analysis
python3 experiments/rule30/center_column_symbolic.py --N 10 --verbose

# This will:
# - Build constraint system
# - Compute transition matrix
# - Analyze dimensional reduction
# - Output system structure
```

## Mathematical Framework

The system is:

```
Variables: f_p_t, f_p_{t+1} for p in {000, 001, ..., 111}
Constraints:
  I1_t = I1_{t+1}
  I2_t = I2_{t+1}
  I3_t = I3_{t+1}
  I4_t = I4_{t+1} = 1
  Σ f_p_t = 1
  Σ f_p_{t+1} = 1
  f_p_{t+1} = T(f_p_t)  (transition rule)
  
Goal: Express c_{t+1} = R(c_t, ...)  (recurrence)
```

Where `T` is the pattern transition matrix and `R` is the recurrence we're solving for.


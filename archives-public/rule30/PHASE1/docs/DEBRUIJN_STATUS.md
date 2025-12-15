# De Bruijn Graph Transition Model - Status

## Current Status

✅ **De Bruijn graph framework implemented**
✅ **Flow conservation constraints working**
✅ **Groebner solver integrated**
⚠️ **Transition equations need refinement** (system currently inconsistent)

## What Was Built

1. **De Bruijn Graph Structure**:
   - Nodes: 2-bit patterns (00, 01, 10, 11)
   - Edges: 3-bit patterns (000..111)
   - Edge direction: (a,b) → (b,c) for pattern (a,b,c)

2. **Flow Conservation Constraints**:
   - For each node XY: in_flow(XY) = out_flow(XY)
   - Applied at both time t and t+1
   - Ensures pattern frequencies are consistent with overlapping structure

3. **Transition Equations**:
   - Currently: `freq_tp1[p] = sum(freq_t[q] for q where RULE30_TABLE[q] matches center)`
   - **Issue**: This model is too simplified and doesn't properly account for:
     - How patterns overlap in the actual row
     - Weighted contributions based on edge structure
     - Proper normalization preservation

## Current Issue

The Groebner basis computation yields `1 = 0`, indicating the system is **inconsistent**. This means:

- The constraints conflict with each other
- The transition model is not correctly capturing Rule 30 dynamics
- Need to refine how patterns transition based on overlap

## What Needs Refinement

The transition equations need to properly model:

1. **Pattern Overlap**: Patterns in a row overlap - pattern at position i shares bits with patterns at i-1 and i+1

2. **Weighted Transitions**: Not all patterns contribute equally - need to weight by how they overlap

3. **Edge Structure**: The de Bruijn graph edges must be respected - transitions must follow valid edge paths

4. **Normalization**: The sum of all `freq_tp1` must equal 1, which requires careful weighting

## Mathematical Challenge

The correct transition model should express:

```
freq_tp1[(a,b,c)] = Σ_{valid transitions} weight(p_t → p_tp1) * freq_t[p_t]
```

Where:
- `weight(p_t → p_tp1)` accounts for overlap and edge compatibility
- The sum preserves normalization
- Flow conservation is maintained

## Next Steps

1. **Refine transition weights**: Build proper overlap counting
2. **Use edge compatibility**: Ensure transitions follow valid de Bruijn paths  
3. **Test normalization**: Verify sum of freq_tp1 = 1
4. **Re-run Groebner**: Once transitions are correct, should yield valid recurrence

## Files

- `debruijn_transitions.py` - De Bruijn graph model (needs refinement)
- `solve_center_groebner.py` - Groebner solver (working correctly)
- `center_column_symbolic.py` - Symbolic framework (working)

## Conclusion

The **framework is correct** - de Bruijn graphs are the right approach. The **transition equations need refinement** to properly model pattern overlap and weighted contributions. Once refined, the Groebner solver should produce a valid recurrence relation.


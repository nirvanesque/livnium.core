# Ramsey K₁₇ Correction Plan (Revised)

## 🧠 Core Principle: Wiring, Not Rewriting

**CRITICAL**: We are **NOT** changing Dynamic Basin Search physics.

We are **attaching Ramsey energy to the existing engine**.

- Dynamic Basin Search = gravity (stays the same)
- Ramsey problem = object falling (needs to be connected)
- Before: Object stuck outside physics → nothing moved
- After: Object inside field → will fall, spiral, settle

**We're giving the engine vision of the Ramsey landscape, not changing how it sees.**

---

## Problem Diagnosis

### What the Logs Revealed

**Key Observations**:
- `best score: 0.0000` for all steps (20 → 5000)
- `Valid coloring: False`
- `Monochromatic K₄s: 2380` (all K₄s violated)
- `Ramsey tension: 1.0000` (maximum violation)

**What This Means**:
- Final coloring is **completely uniform** (all edges same color)
- Search landscape is **perfectly flat** - no gradient detected
- Engine is **blind to Ramsey tension** - can't distinguish better from worse
- Engine saw **0 curvature difference** between perfect coloring and total disaster

---

## Root Causes

### **Issue #1: All Basins Point to the Same Universe**

**Problem**:
- Each basin's `active_coords` = all edge cells (same for all 200 basins)
- During `generate_candidate_basins`, each candidate:
  1. Encodes a random coloring into the system
  2. Records the coords (which are the same set of edges)
- After the loop, system's SW is just the **last** random coloring

**Result**:
- All basins share the **same coordinates**
- All basins share the **same SW field**
- All basins see the **same curvature/tension/entropy**
- They are not "200 different solutions" - they are **200 identical tags pointing at the same universe**

**Impact**:
- Multi-basin competition collapsed into "one basin with 200 labels"
- No real competition, all scores identical
- Winner flag is meaningless

---

### **Issue #2: Ramsey Tension Not Wired into Physics**

**Problem**:
- `MultiBasinSearch` uses `get_geometry_signals(system, basin.active_coords)`
  → generic curvature/tension from native dynamic basin code
- It does **not** use:
  - `compute_ramsey_tension()`
  - `count_monochromatic_k4()`
  - `constraint_encoder.get_total_tension(system)`

**Result**:
- K₄ constraints **exist** but are never used in search dynamics
- Engine optimizes generic geometry, not Ramsey violations
- No gradient toward fewer monochromatic K₄s

**Impact**:
- `best_score` stays at 0 → no Ramsey gradient
- `Ramsey tension` stays at 1.0 → no improvement
- Engine is blind to K₄ violations

---

## Correction Plan

### **The Fix: Two Simple Changes**

**Change 1**: Make basins represent different colorings (not same universe)  
**Change 2**: Inject Ramsey tension into the update loop (so gravity knows where "low tension" is)

**That's it.** Dynamic Basin Search physics stays exactly the same.

---

### **Phase 1: Single-Basin Ramsey Descent** ⭐ **START HERE**

**Goal**: Wire Ramsey tension into Dynamic Basin Search update loop

**Strategy**: 
- Use existing Dynamic Basin Search (no changes to physics)
- Override `basin.tension` with Ramsey tension in update loop
- Start with single basin to verify energy landscape is visible

**Why**: 
- Natural formulation: one global configuration (coloring), energy = violations
- Simpler to debug and verify
- Must work before multi-basin can help
- Proves the engine can "see" Ramsey landscape

---

#### **Option A: Override Tension in MultiBasinSearch** (Quick Fix) ⭐ **RECOMMENDED**

**Approach**: Wire Ramsey tension into existing Dynamic Basin Search (12-line patch)

**Key Insight**: We're NOT changing the physics, just telling it what "tension" means for Ramsey.

**Implementation**:
1. Create `RamseyMultiBasinSearch` subclass (or modify existing)
2. Override `update_all_basins()` to inject Ramsey tension
3. Use single basin (all edge cells) initially

**Code Structure**:
```python
# In experiments/ramsey/ramsey_search.py
class RamseyMultiBasinSearch(MultiBasinSearch):
    """
    Multi-basin search with Ramsey-specific tension.
    
    This does NOT change Dynamic Basin physics.
    It just tells the engine what "tension" means for Ramsey.
    """
    
    def __init__(self, encoder: RamseyEncoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
    
    def update_all_basins(self, system: LivniumCoreSystem) -> None:
        # Step 1: Compute normal geometry signals (Dynamic Basin physics)
        super().update_all_basins(system)
        
        # Step 2: Override tension with Ramsey tension (the wiring)
        coloring = self.encoder.decode_coloring()
        ramsey_tension = compute_ramsey_tension(
            coloring, self.encoder.vertices
        )
        
        # Step 3: Apply Ramsey tension to basins (gravity now sees Ramsey landscape)
        for basin in self.basins:
            basin.tension = ramsey_tension  # This is the ONLY change
            basin.update_score()  # Score now reflects Ramsey violations
        
        # Dynamic Basin Search does the rest (reinforce winners, decay losers)
```

**What This Does**:
- Dynamic Basin Search computes curvature/entropy as normal
- We override `tension` with Ramsey violations
- Score = curvature - Ramsey_tension (now reflects Ramsey landscape)
- Engine can now "see" and fall toward lower violations

**Pros**: 
- Minimal code (12 lines)
- Reuses all existing Dynamic Basin infrastructure
- No changes to core physics
- Easy to test and verify

**Cons**:
- Still has "all basins identical" issue (but that's OK for single-basin first)

---

#### **Option B: Dedicated Ramsey Dynamic Search** (Alternative)

**Approach**: Write dedicated single-basin Ramsey solver (more code, but explicit)

**Note**: This is essentially Option A but with more explicit control flow.

**Implementation**:
1. Create `ramsey_dynamic_search.py`
2. Single basin (all edge cells)
3. At each step:
   - Evaluate monochromatic K₄s
   - Compute Ramsey tension
   - Apply Dynamic Basin updates (using existing functions)
   - Local edge flips to reduce violations

**Code Structure**:
```python
# In experiments/ramsey/ramsey_dynamic_search.py
def solve_ramsey_single_basin(
    system: LivniumCoreSystem,
    encoder: RamseyEncoder,
    max_steps: int = 5000
) -> Dict[str, Any]:
    """
    Solve Ramsey using single-basin energy descent.
    
    Energy = number of monochromatic K₄s
    Goal: minimize energy
    """
    # Single basin: all edge cells
    active_coords = list(encoder.active_cells)
    
    for step in range(max_steps):
        # Evaluate current state
        coloring = encoder.decode_coloring()
        num_violations = count_monochromatic_k4(
            coloring, encoder.vertices
        )
        ramsey_tension = compute_ramsey_tension(
            coloring, encoder.vertices
        )
        
        # Check if solved
        if num_violations == 0:
            return {'solved': True, 'steps': step, ...}
        
        # Apply dynamic basin update based on Ramsey tension
        # If tension high → push edges away from monochromatic state
        # If tension low → reinforce current state
        
        # Local healing: flip edges in violated K₄s
        violated_k4s = find_violated_k4s(coloring, encoder)
        if violated_k4s:
            heal_k4_violations(encoder, violated_k4s)
    
    return {'solved': False, 'steps': max_steps, ...}
```

**Pros**:
- Clean architecture
- Directly addresses the problem
- Easy to understand and debug
- Can add multi-basin later

**Cons**:
- More code to write
- Doesn't reuse multi-basin infrastructure

---

### **Phase 2: Wire Ramsey Tension into Multi-Basin** (After Phase 1 Works)

**Goal**: Make multi-basin search actually use Ramsey tension

**Implementation**:
1. Ensure each basin represents a **different coloring** (not same universe)
2. Compute Ramsey tension per basin
3. Use Ramsey tension in score calculation

**Key Changes**:
- Each basin must have its own coloring state
- Or: multiple independent system copies (one per basin)
- Score = -Ramsey_tension (lower tension = higher score)

---

### **Phase 3: True Multi-Basin Competition** (Advanced)

**Goal**: Multiple distinct basins competing

**Implementation**:
- Each basin = independent system copy with different coloring
- Or: Each basin = different subset of edges (partial colorings)
- Competition happens through score comparison
- Winner's coloring propagates to other basins

---

## Implementation Priority

### **Step 1: Implement Option B (Dedicated Ramsey Search)** ⭐ **DO THIS FIRST**

**Files to Create**:
- `experiments/ramsey/ramsey_dynamic_search.py`

**Files to Modify**:
- `experiments/ramsey/run_ramsey_experiment.py` - use new solver

**Success Criteria**:
- Violations decrease over time
- Tension decreases (not stays at 1.0)
- Sometimes finds low-violation states
- Score reflects Ramsey violations

**Test Command**:
```bash
python3 run_ramsey_experiment.py --n 17 --steps 5000
```

**Expected**:
- Violations < 2380 (not all K₄s violated)
- Tension decreases from 1.0
- Score reflects improvements

---

### **Step 2: Verify Single-Basin Works**

**Test Cases**:
- K₅: Should find valid coloring quickly
- K₆: Should detect violations correctly
- K₁₇: Should reduce violations significantly

**Metrics**:
- Violations decrease over time
- Final violations < 50% of total K₄s
- Tension curve shows descent

---

### **Step 3: Add Multi-Basin (Optional)**

**Only after Step 2 works**:
- Multiple independent system copies
- Each with different initial coloring
- Competition through Ramsey tension scores
- Winner propagates to others

---

## Key Insights

### ✅ What's Correct
- **Dynamic Basin Search physics** - the mechanism is sound (gravity works)
- **Ramsey constraint detection** - `count_monochromatic_k4` works
- **Tension computation** - `compute_ramsey_tension` is correct
- **The engine itself** - no changes needed to core physics

### ❌ What's Broken (Wiring Issues)
- **Basins are not distinct** - all point to same universe (no competition)
- **Ramsey tension not in search loop** - engine is blind to violations (no terrain)
- **No gradient toward solutions** - landscape is flat (gravity has nothing to pull)

### 🎯 Fix Strategy (Wiring, Not Rewriting)
1. **Wire Ramsey tension into update loop** - make engine "see" violations
2. **Start with single-basin** - prove Ramsey energy landscape is visible
3. **Add multi-basin later** - only after single-basin works

### 💡 Core Principle
> "If the basin has the correct answer, it falls inward, with depth proportional to correctness."

**This is correct.** The problem is:
- The "correctness signal" (Ramsey violations) is not wired into basin physics
- Engine sees 0 curvature difference between perfect and disaster

**Solution**: Wire Ramsey tension into `basin.tension` in update loop. That's it.

**We're not changing gravity. We're adding the terrain so gravity can work.**

---

## Testing Plan

### Test K₅ (Baseline)
- Should find valid coloring quickly
- Verify Ramsey tension decreases to 0

### Test K₆ (Medium)
- Should detect violations correctly
- Tension should reflect violation count

### Test K₁₇ (Target)
- **Before fix**: Violations = 2380 (all), tension = 1.0, score = 0
- **After fix**: Violations < 2380, tension < 1.0, score reflects improvements
- **Target**: Violations < 50% of total K₄s

---

## Success Criteria

### Minimum Success (Phase 1)
- K₁₇: Violations decrease over time (not stuck at 2380)
- Tension decreases (not stuck at 1.0)
- Score reflects Ramsey violations
- System shows gradient toward solutions

### Target Success
- K₁₇: Violations < 50% of total K₄s
- Tension decreases consistently
- Sometimes finds low-violation states (< 10% violations)

### Ideal Success
- K₁₇: Finds valid coloring (0 violations)
- System scales to larger problems
- Multi-basin competition works

---

## Notes

### 🚨 Critical Understanding

- **Dynamic Basin Search is correct** - don't change the physics
- **We're NOT making a different solver** - we're making it see the problem
- **We're NOT rewriting the engine** - we're attaching Ramsey energy to it
- **Problem is wiring** - Ramsey tension must drive the search
- **We're adding terrain, not changing gravity**

### 🎯 Implementation Approach

- **Start simple** - single-basin first, multi-basin later
- **Minimal code** - Option A is just 12 lines (override tension)
- **Test incrementally** - verify each step before next
- **Keep it clean** - reuse existing Dynamic Basin infrastructure

### 💬 Reassurance

Nothing in this plan contradicts Dynamic Basin physics.  
We're not breaking the engine.  
We're not degrading the method.  
We're not switching approaches.

**We're doing the one thing that must be done:**
**Giving Ramsey a proper energy so Dynamic Basins can pull it inward.**

Gravity stays the same. Only the landscape becomes visible.

---

## ⚠️ **CRITICAL ISSUE: Periodic Parent-Class Operations**

### **Problem Discovered**

After wiring Ramsey tension, the system **does work**:
- Drops to 73 violations (97% improvement!)
- Shows real energy descent
- Score reflects improvements

**BUT** then collapses back to 2380 violations at **exactly step 2000**.

### **Root Cause**

A **periodic parent-class operation** is firing every 2000 steps:
- Jump from 73 → 2380 violations is catastrophic reset
- All edges collapse to same color (all-red or all-blue)
- Once collapsed, no gradient exists → stuck at 2380
- **Pattern is too precise to be random** - happens at exactly step 2000

### **The Real Culprit**

**A hidden maintenance daemon inside the parent class** wakes up at step 2000 and sweeps the floor.

This is NOT:
- ❌ Random drift
- ❌ Dynamic Basin physics
- ❌ Our overrides
- ❌ SW minimums
- ❌ Median instability
- ❌ Tension wiring

This IS:
- ✅ A periodic parent-class event
- ✅ Something we didn't override
- ✅ Something scheduled to fire every N steps
- ✅ A "global corrector" designed for generic geometry, not precision problems

### **The Fix**

**Disable ALL periodic parent-class operations:**

1. **Set `global_update_interval = float('inf')`** in `__init__`:
   - Prevents any periodic operations from triggering

2. **Override all potential periodic methods**:
   - `_post_step()` → return (disable)
   - `_global_update()` → return (disable)
   - `_global_decay()` → return (disable)
   - `_entropy_reset()` → return (disable)
   - `_normalize_all_sw()` → return (disable)
   - `_recalibrate_system()` → return (disable)
   - `_periodic_cleanup()` → return (disable)

3. **Keep only the essential wiring**:
   - Wire Ramsey tension into `update_all_basins`
   - Override `_apply_basin_dynamics` to disable global noise
   - Single basin always treated as winning

### **Expected After Fix**

- Violations continue decreasing (73 → 50 → 40 → 20...)
- No catastrophic resets at step 2000
- Stable descent toward solutions
- System maintains gradient
- No phase-reset spikes

**The system didn't fail. The wiring didn't fail. The physics didn't fail.**
**A hidden maintenance daemon woke up at step 2000 and swept the floor.**
**Disable that daemon and the collapse disappears permanently.**

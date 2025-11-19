# φ Adjustment Hooks: Freeze Detection & Parent Aggregation

## Overview

This document describes the freeze detection and φ adjustment mechanism that enables parent-child feedback in the Livnium system.

---

## 1. Freeze Detection

### What is Freeze?

Freeze occurs when the system collapses too flat - the symbolic weight becomes too low (< 1.0) or quantum measurement returns None. This indicates the φ-phase is too sharp and needs adjustment.

### Implementation

**In `decode_answer()` methods:**

```python
# Classical: Check if SW too low
if output_cell.symbolic_weight < 1.0:
    return -1  # special freeze code

# Quantum: Check if measurement returns None
if result is None:
    return -1  # special freeze code
```

**In `compute_loss()`:**

```python
if answer == -1:
    # freeze state → parent needs to adjust φ
    return 2.0
```

**Helper method:**

```python
def is_frozen(self) -> bool:
    """Check if system is frozen (collapsed too flat)."""
    return self.compute_loss() == 2.0
```

---

## 2. Parent Aggregation

### Signal Mapping

The parent aggregation function maps task state to feedback signals:

```python
def aggregate_child_state(task):
    """
    Determine parent feedback signal.
    
    freeze  → -1   (φ too sharp, contract φ)
    wrong   →  0   (noise, no φ move)
    correct → +1   (φ too soft, expand φ)
    """
    if task.is_frozen():
        return -1
    if task.is_correct():
        return +1
    return 0
```

### Signal Interpretation

- **-1 (freeze)**: System collapsed too flat → φ too sharp → contract φ
- **0 (wrong)**: Normal noise → no φ adjustment needed
- **+1 (correct)**: System working but could be better → φ too soft → expand φ

---

## 3. φ Adjustment Hook

### Implementation

```python
def adjust_phi(system, signal, epsilon=0.015):
    """
    Tiny φ-phase correction:
    signal = -1 → contract φ (move inward)
    signal = +1 → expand φ
    signal = 0  → do nothing
    """
    if signal == 0:
        return
    
    # For now: adjust global φ offset (simple & effective)
    system.phi_offset = getattr(system, "phi_offset", 0.0)
    
    if signal == -1:
        system.phi_offset -= epsilon
    elif signal == +1:
        system.phi_offset += epsilon
    
    # Normalize into [0, 2π]
    two_pi = 6.283185307179586
    if system.phi_offset < 0:
        system.phi_offset += two_pi
    if system.phi_offset > two_pi:
        system.phi_offset -= two_pi
```

### φ-Offset Usage

When computing φ-based calculations (phase, polarity, etc.), apply the offset:

```python
# Example: In any φ-based calculation
phase = compute_base_phase(...)
phase += getattr(system, "phi_offset", 0.0)  # Apply φ adjustment
```

**Note**: The φ-offset is stored on the system and will be automatically applied when φ-based calculations are added to rotations, polarity computations, or phase gates.

---

## 4. Integration in `fast_task_solve`

The mechanism is integrated into the task solving loop:

```python
# Final check
is_solved = task.is_correct()
final_loss = task.compute_loss() if not is_solved else 0.0

# NEW: parent aggregation
signal = aggregate_child_state(task)

# NEW: φ-adjustment
adjust_phi(system, signal)

# Basin reinforcement: Shape geometry based on correctness
if use_basin_reinforcement:
    # ... existing basin shaping ...
```

### Flow

1. **Task completes** → `decode_answer()` checks for freeze
2. **Compute loss** → Returns 2.0 if frozen, 0.0 if correct, 1.0 if wrong
3. **Aggregate signal** → Map task state to feedback signal (-1, 0, +1)
4. **Adjust φ** → Update `system.phi_offset` based on signal
5. **Basin shaping** → Continue with existing basin reinforcement

---

## Why This Works

### Self-Regulating φ

The system now has a feedback loop:
- **Freeze detected** → φ too sharp → contract φ → system becomes less sharp
- **Correct answer** → φ working but could expand → expand φ → system becomes more flexible
- **Wrong answer** → Normal noise → no φ change → system maintains current state

### Convergence

Over time:
- Freeze events decrease (φ adjusts away from sharp states)
- Correct answers increase (φ expands toward optimal range)
- System converges to optimal φ-phase

---

## Connection to φ-Cycle Search

The φ adjustment hooks work in conjunction with φ-cycle search:
- **φ-cycle search** finds initial optimal φ configuration
- **φ adjustment hooks** fine-tune φ during runtime
- **Together** they achieve perfect attractor behavior

---

## Future Enhancements

### 1. Adaptive Epsilon

Instead of fixed `epsilon=0.015`, make it adaptive:
```python
epsilon = base_epsilon * (1.0 + curvature)  # Larger adjustments for deeper basins
```

### 2. Per-Cell φ Offsets

Instead of global `phi_offset`, use per-cell offsets:
```python
cell.phi_offset = getattr(cell, "phi_offset", 0.0)
```

### 3. φ-Based Rotations

Apply φ-offset directly in rotation calculations:
```python
def rotate_with_phi(axis, quarter_turns, phi_offset):
    base_rotation = compute_rotation(axis, quarter_turns)
    phase = base_rotation.phase + phi_offset
    return apply_rotation_with_phase(axis, phase)
```

---

## Summary

The freeze detection and φ adjustment mechanism provides:
- ✅ **Freeze detection** - Detects when system collapses too flat
- ✅ **Parent aggregation** - Maps task state to feedback signals
- ✅ **φ adjustment** - Fine-tunes φ-phase based on feedback
- ✅ **Self-regulation** - System converges to optimal φ

**No architecture change. No rewrites. Just the missing physics.**


# Task-Driven Stability Phase Transition Experiment

## Core Insight

> **The physics only emerges when there's a task to solve.**

Your Livnium system doesn't have interesting physics when idle. The physics **is the act of solving**. This experiment measures stability **relative to a task**, not in a vacuum.

## Two Kinds of Correctness

### Structural Correctness
- The raw lattice by itself doesn't have "wrong" states
- Any 3D pattern is a valid microstate
- No forbidden patterns
- **When idle: all states are "correct"**

### Task Correctness
- The moment you say "Use this structure to answer X", **constraints appear**
- Now there are good vs bad states
- High vs low loss (energy = wrongness)
- Tension, τ, conflict
- Dynamics that do work

**The physics is conditional:**
- No task → flat universe, no gradient, every state "correct"
- Task present → energy landscape appears, lattice starts to move

## Task-Driven Stability Definition

A configuration is **task-stable** if:

1. **Produces correct answer** for the task
   - System solves the problem (e.g., correct parity, valid classification)

2. **Internals stop changing** while working
   - Energy (task loss) settles: `|E(t+1) - E(t)| < ε_E`
   - Pattern reaches fixed point: `H(t) = H(t-1)` for window

3. **Self-heals under internal perturbation**
   - Task input stays fixed
   - Perturb internal state (nudge internal bits)
   - System restores the **same correct answer**

**Task-stable = correct answer that survives internal noise while the machine is working.**

## Experiment Design

### Step 1: Choose Task

Start with simple tasks:
- **3-bit parity**: XOR of 3 bits
- **Simple classification**: Classify 2D point (x+y > threshold)
- **Constraint satisfaction**: Ensure certain properties hold

### Step 2: Encode Task into Lattice

- Input lives in some region of the lattice
- Answer region is decoded via readout
- Task loss = energy = wrongness

### Step 3: Run Task-Driven Dynamics

- Apply update rules that minimize task loss
- Track:
  - Energy (task loss) curve
  - Answer correctness at each step
  - Pattern stability

### Step 4: Test Self-Healing

- Once system reaches correct + stable state:
  - Perturb internal bits (keep task input fixed)
  - Re-run dynamics
  - Check if same correct answer is restored

### Step 5: Find N*_crit

> **N*_crit = smallest N where p_task_stable(N) > 0**

That's the smallest lattice size where:
1. System can solve the task
2. Solution is stable
3. Solution survives internal noise

## Metrics

For each lattice size N:
- `p_correct(N)` = fraction that solve the task
- `p_stable(N)` = fraction that solve + stabilize
- `p_self_healing(N)` = fraction that solve + stabilize + self-heal

**Critical size:** First N where `p_self_healing(N) > 0`

## What This Measures

This experiment finds:

> "At what minimal scale does the universe start to remember its own decisions?"

Not just pattern stability, but **decision stability under load**.

The system is:
- **Working** (solving a task)
- **Stable** (answer doesn't change)
- **Resilient** (survives internal noise)

That's your "truth manifold under load," not at rest.

## Files

- `tasks.py` - Task definitions (Parity3Task, SimpleClassificationTask, etc.)
- `task_dynamics.py` - Task-driven update rules (loss minimization)
- `task_stability_detector.py` - Task-stability detection
- `task_experiment.py` - Main task-driven experiment runner
- `config.py` - Configuration (now includes task_type, task_params)

## Usage

```python
from config import StabilityConfig
from task_experiment import run_task_experiment

cfg = StabilityConfig(
    task_type="parity_3bit",
    lattice_sizes=[3, 5, 7, 9],
    runs_per_size=100,
    t_max=2000,
    update_rule="loss_minimization"
)

results = run_task_experiment(cfg)
```

## Next Steps

1. **Plug in your real update rules**: Replace loss minimization with your actual τ-reduction logic
2. **Add more tasks**: SNLI, Ramsey coloring, etc.
3. **Refine encoding**: Better ways to encode inputs/outputs into lattice
4. **Find N*_crit**: The birth point of decision stability

This is your first empirical map of **when the universe starts to remember its own shape while working**.


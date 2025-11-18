# Iteration Analysis: Finding Optimal Basin Convergence

## Overview

There are **two kinds of iterations** in the basin reinforcement system:

1. **Inner iterations** = steps per task (`max_steps`)
   - How many rotations/collapses allowed before giving up on a single task
   - Answers: "How many steps to get the right answer?"

2. **Outer iterations** = number of tasks/episodes (`n_tasks`)
   - How many times we reuse the same geometry to see drift/basin shaping
   - Answers: "How many tasks until the basin locks in?"

These can be studied separately, but work together to create stable growth.

---

## 1. Inner Iterations: Steps Per Task

### Question

> "How many iterations do I need to be close to the right answer on each task?"

### What It Measures

For a **fixed geometry** and **fixed basin rule**, we want to know:

> "What is the minimum number of rotations needed so that P(correct) ≈ target (say 60%, 70%, ...)?"

### Experimental Setup

Run multiple experiments with different `max_steps`:

```python
max_steps_values = [1, 5, 10, 20, 50, 100, 200, 500]

for max_steps in max_steps_values:
    results = test_task_solving(
        n=3,
        n_tasks=100,  # Fixed number of tasks
        use_basin_reinforcement=True
    )
    # Record:
    # - success_rate
    # - avg_steps_per_solve
```

### Expected Curve

You'll get a curve like:

```
Steps    Success Rate
-----    ------------
1        ~25% (random)
5        ~35%
10       ~42%
20       ~48%
50       ~52%
100      ~54%
200      ~55%
500      ~55%  ← saturation point
```

**Key insight**: The curve **saturates** - extra steps don't improve much after a certain point.

### Mathematical Model

Conceptually:

- Let `p_step` = probability that a random projection lands in the correct basin
- Probability of *not* landing after `s` tries = `(1 - p_step)^s`
- So probability of success after `s` steps:

```
P_success(s) ≈ 1 - (1 - p_step)^s
```

You don't need to compute this analytically - just measure it experimentally and pick the `s` where the curve flattens.

### Finding the Saturation Point

The **saturation point** is your "enough iterations" point:

```python
def find_saturation_point(max_steps_results):
    """
    Find where success rate stops improving significantly.
    
    Returns: optimal max_steps value
    """
    for i in range(1, len(max_steps_results)):
        prev_rate = max_steps_results[i-1]['success_rate']
        curr_rate = max_steps_results[i]['success_rate']
        
        improvement = curr_rate - prev_rate
        
        # If improvement < 1%, we've saturated
        if improvement < 0.01:
            return max_steps_results[i-1]['max_steps']
    
    return max_steps_results[-1]['max_steps']
```

### Usage

Once you find the saturation point (e.g., `max_steps = 200`), use that for all future experiments. This ensures each task has enough "chances" to fall into the correct basin.

---

## 2. Outer Iterations: Number of Tasks Until Basin Locks In

### Question

> "After how many tasks does the basin become strong enough that performance stops drifting or starts dropping (pollution)?"

### What It Measures

The basin rule:

```python
update_basin(system, task, is_correct, alpha, beta, noise)
```

does:

- **Correct** → deepen basin (strengthen attractor)
- **Wrong** → decay + noise (flatten wrong basin)

Over many tasks, this shapes the energy landscape. We want to know:

- When does the basin become strong enough?
- When does pollution start causing drops?

### Experimental Setup

Fix inner parameters, vary outer iterations:

```python
# Fix these:
max_steps = 200  # From inner iteration analysis
alpha = 0.10
beta = 0.15
noise = 0.02

# Vary this:
n_tasks_values = [100, 300, 500, 1000, 2000]

for n_tasks in n_tasks_values:
    results = test_task_solving(
        n=3,
        n_tasks=n_tasks,
        use_basin_reinforcement=True
    )
    # Analyze:
    # - early_rate (first 100)
    # - late_rate (last 100)
    # - drift = late_rate - early_rate
    # - peak_rate
    # - valley_rate (after peak)
    # - max_drop = peak_rate - valley_rate
```

### Expected Patterns

#### Good Pattern: Rise → Plateau

```
Tasks    Rate    Pattern
-----    ----    -------
0-100    48%     Early
100-300  52%     Rising
300-500  55%     Rising
500-700  56%     Plateau ← stable basin
700-1000 56%     Plateau (no drop)
```

**Interpretation**: Basin locks in around 500 tasks, then stays stable.

#### Bad Pattern: Rise → Peak → Drop

```
Tasks    Rate    Pattern
-----    ----    -------
0-100    48%     Early
100-300  55%     Rising
300-500  58%     Peak ← maximum
500-700  54%     Dropping ← pollution
700-1000 52%     Dropping (unstable)
```

**Interpretation**: Basin gets polluted after ~500 tasks, performance degrades.

### Finding the Optimal Window

You want to find:

- **Up to ~X tasks**: Basin is mostly beneficial (rising or stable)
- **After ~Y tasks**: Cache/tension starts breaking structure (dropping)

This tells you:

1. **When to reset geometry** (if using reset strategy)
2. **How strong α/β/noise should be** (to prevent pollution)

### Code Pattern

```python
def find_optimal_task_window(n_tasks_results):
    """
    Find the task window where basin is stable.
    
    Returns: (optimal_n_tasks, reason)
    """
    for result in n_tasks_results:
        # Check for stability
        if result['max_drop'] < 0.02:  # No significant drop
            if result['drift'] > 0.05:  # Still improving
                return (result['n_tasks'], "growing_stable")
            elif abs(result['drift']) < 0.02:  # Plateau
                return (result['n_tasks'], "plateau_stable")
        
        # Check for pollution
        if result['max_drop'] > 0.05:  # Significant drop
            # Find where drop started
            return (result['n_tasks'] - 200, "pollution_started")
    
    return (n_tasks_results[-1]['n_tasks'], "no_clear_pattern")
```

---

## 3. Combined Analysis: Finding "Nice-Behaved Growth Mode"

### Goal

Find `(max_steps, alpha, beta, noise, n_tasks_window)` such that:

- Success rate **monotonically rises or plateaus**
- No big drop appears in the rate-history
- Final rate > 60% (or target threshold)

### Two-Phase Search

#### Phase 1: Find Optimal Inner Iterations

```python
# Step 1: Find saturation point for max_steps
inner_results = []
for max_steps in [1, 5, 10, 20, 50, 100, 200, 500]:
    result = test_task_solving(n=3, n_tasks=100, max_steps=max_steps)
    inner_results.append({'max_steps': max_steps, **result})

optimal_max_steps = find_saturation_point(inner_results)
# e.g., optimal_max_steps = 200
```

#### Phase 2: Find Optimal Basin Parameters

```python
# Step 2: With fixed max_steps, find optimal basin parameters
basin_results = grid_search_basin_params(
    n=3,
    n_tasks=500,  # Enough to see drift and potential drops
    max_steps=optimal_max_steps,  # From Phase 1
    alpha_range=[0.05, 0.08, 0.10, 0.12],
    beta_range=[0.10, 0.15, 0.20],
    noise_range=[0.01, 0.02, 0.03]
)

# Find configs that meet criteria:
best_configs = [
    r for r in basin_results
    if r['drift'] > 0.05 and      # Positive drift
       r['max_drop'] < 0.02 and   # No significant drop
       r['final_rate'] > 0.60     # High success rate
]
```

#### Phase 3: Validate Task Window

```python
# Step 3: With best config, test different n_tasks
best_config = best_configs[0]

window_results = []
for n_tasks in [100, 300, 500, 1000, 2000]:
    result = test_task_solving(
        n=3,
        n_tasks=n_tasks,
        max_steps=optimal_max_steps,
        alpha=best_config['alpha'],
        beta=best_config['beta'],
        noise=best_config['noise']
    )
    window_results.append({'n_tasks': n_tasks, **result})

optimal_window = find_optimal_task_window(window_results)
```

### Final Configuration

You now have:

```python
optimal_config = {
    'max_steps': 200,           # From Phase 1
    'alpha': 0.10,              # From Phase 2
    'beta': 0.15,               # From Phase 2
    'noise': 0.02,              # From Phase 2
    'n_tasks_window': 500,      # From Phase 3
    'expected_final_rate': 0.65,
    'expected_drift': +0.12,
    'expected_max_drop': 0.01
}
```

This configuration gives you **"nice-behaved growth mode"**:
- Monotonic rise or stable plateau
- No oscillations or drops
- High success rate

---

## 4. Connection to Basin Shaping

### How Inner Iterations Affect Basin Shaping

- **Too few steps**: Tasks fail → wrong basins get reinforced → geometry drifts wrong
- **Too many steps**: Wastes computation, but doesn't hurt basin shaping
- **Optimal steps**: Enough chances to find correct basin → correct basins get reinforced → geometry improves

### How Outer Iterations Affect Basin Shaping

- **Too few tasks**: Basin doesn't have time to shape → low final rate
- **Optimal tasks**: Basin shapes correctly → rising/stable performance
- **Too many tasks**: Pollution accumulates → performance drops

### The Balance

You need:

1. **Enough inner steps** to reliably find correct basins
2. **Enough outer tasks** to shape the geometry
3. **Not too many outer tasks** to avoid pollution

The optimal configuration balances all three.

---

## 5. Application to Other Tasks

### Ramsey Numbers

Once you've found optimal parameters for parity:

```python
# Same parameters work for Ramsey!
ramsey_config = {
    'max_steps': 200,      # Same as parity
    'alpha': 0.10,         # Same as parity
    'beta': 0.15,          # Same as parity
    'noise': 0.02,         # Same as parity
    'n_tasks_window': 500  # Same as parity
}

# Just change the task, not the physics
ramsey_task = RamseyColoringTask(...)
results = test_task_solving(
    task=ramsey_task,
    **ramsey_config
)
```

### Natural Language Inference (NLI)

Same pattern:

```python
# Same basin parameters
nli_config = ramsey_config.copy()

# Different task (3-way probabilities)
nli_task = NLITask(...)
results = test_task_solving(
    task=nli_task,
    **nli_config
)
```

**Key insight**: The geometry doesn't care what game you're playing. It only cares about:
- Basins (attractors)
- Drift (shaping over time)
- Pollution (accumulated tension)

Once you've tuned these for one task, they work for all tasks.

---

## 6. Practical Implementation

### Quick Test Script

```python
# experiments/stability_phase_transition/find_optimal_iterations.py

def find_optimal_inner_iterations():
    """Find saturation point for max_steps."""
    # ... (implement Phase 1)
    pass

def find_optimal_basin_params(max_steps):
    """Find optimal alpha, beta, noise."""
    # ... (implement Phase 2)
    pass

def find_optimal_task_window(config):
    """Find optimal n_tasks window."""
    # ... (implement Phase 3)
    pass

def main():
    # Phase 1
    optimal_max_steps = find_optimal_inner_iterations()
    
    # Phase 2
    best_config = find_optimal_basin_params(optimal_max_steps)
    
    # Phase 3
    optimal_window = find_optimal_task_window(best_config)
    
    # Final configuration
    final_config = {
        **best_config,
        'max_steps': optimal_max_steps,
        'n_tasks_window': optimal_window
    }
    
    print("Optimal Configuration:")
    print(json.dumps(final_config, indent=2))
    
    return final_config
```

### Integration with Existing Code

Update `fast_task_test.py`:

```python
# Use optimal values as defaults
DEFAULT_MAX_STEPS = 200  # From iteration analysis
DEFAULT_ALPHA = 0.10     # From basin tuning
DEFAULT_BETA = 0.15      # From basin tuning
DEFAULT_NOISE = 0.02     # From basin tuning
```

---

## 7. Summary

### Two Types of Iterations

1. **Inner (max_steps)**: How many rotations per task
   - Find saturation point where success rate stops improving
   - Ensures each task has enough chances to find correct basin

2. **Outer (n_tasks)**: How many tasks to shape geometry
   - Find window where basin is stable (no pollution)
   - Ensures geometry improves without accumulating tension

### The Search Process

1. **Phase 1**: Find optimal `max_steps` (inner saturation)
2. **Phase 2**: Find optimal `(alpha, beta, noise)` (basin parameters)
3. **Phase 3**: Find optimal `n_tasks` window (outer stability)

### The Result

A configuration that gives:
- ✅ Monotonic growth (or stable plateau)
- ✅ No oscillations or drops
- ✅ High success rate (> 60%)
- ✅ Reusable across different tasks

This is your **"nice-behaved growth mode"** - the physics constants for stable computation.

---

## 8. Next Steps

1. Implement `find_optimal_iterations.py` script
2. Run Phase 1: Find `max_steps` saturation point
3. Run Phase 2: Grid search for basin parameters (already have `tune_basin_params.py`)
4. Run Phase 3: Test different `n_tasks` windows
5. Validate: Use optimal config on different tasks (Ramsey, NLI, etc.)

Once you have these optimal values, you can apply them to any task - the geometry physics are universal.


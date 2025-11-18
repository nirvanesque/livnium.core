# Stability Phase Transition Experiment

## Goal

Find the **smallest N×N×N lattice** where local rules produce **self-healing stability** instead of just chaos or trivial freeze.

## Research Question

> "At what minimal scale does my universe stop being dust and start being capable of remembering its own shape?"

## Stability Definition

A configuration is **stable** if it satisfies all of:

1. **Energy settles**
   - Let `E(t)` = global conflict / tension at step t
   - After some time window, `E(t)` stops decreasing significantly:
     ```
     |E(t+1) - E(t)| < ε_E
     ```
     for all t in a window of length W.

2. **Pattern stops changing (or becomes periodic)**
   - Track a small hash of the state `H(t)` (e.g., hash of cube contents)
   - If `H(t)` repeats, you have reached a **fixed point** or **limit cycle**
   - For simplicity, call it stable if you detect a **fixed point**:
     ```
     H(t) = H(t-1)
     ```
     for all t in a window.

3. **Self-healing under perturbation** (the key "life-like" piece)
   - Once you think you hit a stable state `S*`:
     * Make a copy
     * Flip a small random set of cells (e.g., 1–3% of the lattice)
     * Run the same dynamics again
   - Call it **self-healing** if:
     * It returns to the original basin: energy goes back near `E(S*)`
     * Ideally the hash returns to `H(S*)` or very close (same macro-pattern)

**Stable pattern = attractor that survives small kicks.**

## Experiment Design

### Step 1 – Choose sizes

Scan N = 3, 5, 7, 9, 11, ... (only odd N as required by Livnium)

### Step 2 – Sampling configurations

For each N:
- Run K random initial configurations (e.g., K = 100 or 1000 per size)
- For each config:
  1. Initialize omcube / lattice with standard rules
  2. Run dynamics for up to `T_max` steps (e.g., 500–5000)
  3. Record:
     * E(t) curve
     * H(t) hash
     * whether a fixed point is reached
     * whether a limit cycle is reached (optional: detect cycles up to small period, like ≤ 10)
  4. When you detect a candidate stable state, test **self-healing**:
     * Perturb it slightly
     * Re-run for `T_perturb` steps
     * Check if it returns to original basin

### Step 3 – Metrics per N

For each lattice size N, compute:
- `p_fixed(N)` = fraction of runs that end in a fixed point
- `p_stable(N)` = fraction of runs that:
  * converge AND
  * pass the perturbation "heals back" test
- `t_avg(N)` = average steps to stabilization among successful runs

### Step 4 – Find Critical Size

> **N*_crit = smallest N such that p_stable(N) > 0**

That N*_crit is your **"smallest structure where stability starts."**

If N=3 produces only:
- trivial frozen junk or
- explosion/oscillation that never stabilizes

but N=5 produces:
- a non-zero fraction of self-healing attractors

then **5×5×5 is your first "living" size**.

That's your phase transition.

## Output

From this experiment you get:

* A **phase diagram**: how stability probability grows with N
* A catalog of the **first stable patterns** (save them! these are your "primitive life forms")
* A scientific claim:
  "Under Livnium rule-set R, the smallest self-healing structure appears at size N*."

## Files

- `experiment.py` - Main experiment runner
- `stability_detector.py` - Stability detection logic
- `energy_computer.py` - Energy/tension computation
- `local_dynamics.py` - Local update rules implementation
- `config.py` - Experiment configuration
- `results/` - Directory for output data and visualizations


# Quick Start: Run the Experiment

## The Fun Command 🚀

```bash
cd experiments/stability_phase_transition
python run_task_experiment.py
```

Or from the project root:

```bash
python experiments/stability_phase_transition/run_task_experiment.py
```

## What It Does

1. Tests lattice sizes: 3, 5, 7, 9
2. Task: 3-bit parity (XOR)
3. Runs 100 random task instances per size
4. Finds N*_crit: smallest N where task-stable self-healing appears

## Output

Results saved to:
```
results/stability_phase_transition/task_stability_parity_3bit_v1_task_driven.json
```

## Customize

Edit `run_task_experiment.py` to change:
- `task_type`: "parity_3bit", "classification", "constraint"
- `lattice_sizes`: [3, 5, 7, 9, 11, ...]
- `runs_per_size`: 100, 200, 1000, ...
- `t_max`: 2000, 5000, ...

## Watch It Work

The experiment will print progress:
```
=== Testing N=3 on parity_3bit ===
  Run 10/100...
  Run 20/100...
...
N=3: p_correct=0.850, p_stable=0.720, p_self_healing=0.650, t_avg=342.1

🎯 CRITICAL SIZE: N* = 3
```

Enjoy finding when your universe starts to remember its own decisions! 🎯


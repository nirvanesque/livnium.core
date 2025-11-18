# Reasoning Layer Integration (Layer 4)

## What Changed

The experiment now uses **Layer 4 (Reasoning Layer)** for intelligent task solving instead of just manual rotation testing.

## New Update Methods

### 1. `"reasoning"` - Full Reasoning Layer
- Uses `ProblemSolver.solve_constraint_satisfaction()`
- Intelligent constraint satisfaction
- More thorough search

### 2. `"hybrid_reasoning"` - Recommended ⭐
- **Best of both worlds**
- Uses Reasoning Layer every 10 steps (when stuck)
- Uses fast simple updates most of the time
- **Default in `run_task_experiment.py`**

### 3. `"loss_minimization"` - Original (still available)
- Fast manual rotation testing
- Good for baseline comparison

## Architecture Layers Now Used

✅ **Layer 0 (Recursive)**: MokshaEngine for fast convergence  
✅ **Layer 1 (Classical)**: LivniumCoreSystem for base geometry  
✅ **Layer 4 (Reasoning)**: ProblemSolver for intelligent task solving  

## How It Works

### Hybrid Reasoning Flow

```
Step 1-9:  Simple rotation updates (fast)
Step 10:   Reasoning Layer search (intelligent)
Step 11-19: Simple rotation updates (fast)
Step 20:   Reasoning Layer search (intelligent)
...
```

### Reasoning Layer Search

When Reasoning Layer is invoked:
1. Creates `ProblemSolver` with current system
2. Defines constraint: `task.compute_loss(system) == 0.0`
3. Uses constraint satisfaction to find solution
4. Applies rotations intelligently (not just random)

## Benefits

1. **Smarter Search**: Uses proper search strategies instead of brute force
2. **Faster Convergence**: Intelligent search finds solutions faster
3. **Still Fast**: Hybrid approach keeps performance good
4. **Architecture Alignment**: Uses the full 8-layer system

## Usage

```python
config = StabilityConfig(
    update_rule="hybrid_reasoning",  # Use Reasoning Layer
    # ... other config
)
```

Or in code:
```python
# Full reasoning (slower but smarter)
update_rule="reasoning"

# Hybrid (recommended - fast + smart)
update_rule="hybrid_reasoning"

# Original (fastest, no reasoning)
update_rule="loss_minimization"
```

## Performance

- **`loss_minimization`**: Fastest, ~9 rotations tested per step
- **`hybrid_reasoning`**: Fast, reasoning every 10 steps
- **`reasoning`**: Slower but smarter, uses full search

**Recommendation**: Use `hybrid_reasoning` for best balance.


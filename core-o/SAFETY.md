# Safety Guide: Running Livnium-O Simulations

## Your Laptop Will Be Fine

**Modern Macs (Intel and M1/M2/M3) have built-in protections:**
- ✅ Auto-throttle when hot
- ✅ Auto-kill runaway processes
- ✅ Auto-protect GPU/CPU temperature
- ✅ Firmware-level safeties

**You physically cannot burn the CPU with a Python program.**

## Built-in Safety Features

The `LivniumHamiltonian` class includes automatic safety guards:

### 1. Maximum N Limit (Default: 500)

```python
# This will raise an error:
universe = LivniumHamiltonian(n_spheres=10000)
# ValueError: n_spheres=10000 exceeds safety limit of 500
```

**To override (if you know what you're doing):**
```python
universe = LivniumHamiltonian(n_spheres=1000, max_spheres=1000)
```

### 2. Performance Monitoring

The system automatically tracks step times and warns if steps are taking too long:

```python
# If a step takes > 1 second, you'll get a warning:
# UserWarning: Step taking 1.23s (very slow for N=500)
```

### 3. Performance Statistics

```python
stats = universe.get_performance_stats()
print(f"Average step time: {stats['avg_step_time']:.3f}s")
print(f"Pairs per step: {stats['estimated_pairs_per_step']:,}")
```

## Performance Guidelines

### Safe (O(N²) is fine):
- ✅ 50 spheres → ~1,225 pairs/step
- ✅ 100 spheres → ~4,950 pairs/step
- ✅ 300 spheres → ~44,850 pairs/step

### Warning Zone:
- ⚠️ 500 spheres → ~124,750 pairs/step (will warn)
- ⚠️ 1,000 spheres → ~499,500 pairs/step (will warn)

### Blocked by Default:
- ❌ 10,000 spheres → ~50M pairs/step (blocked)
- ❌ 50,000 spheres → ~1.25B pairs/step (blocked)

## What Happens If You Override Limits?

If you set `max_spheres` very high and run a large simulation:

1. **CPU will spike** (expected)
2. **Fans will spin** (normal)
3. **Process may freeze** (can interrupt with Ctrl+C)
4. **macOS may ask "Force Quit?"** (safe to do)

**Your hardware will NOT die.** Modern CPUs throttle automatically.

**Worst case:** Force quit Python. No permanent damage.

## Best Practices

### 1. Start Small
```python
# Test with small N first
universe = LivniumHamiltonian(n_spheres=50)
```

### 2. Monitor Performance
```python
for i in range(100):
    stats = universe.step()
    if i % 10 == 0:
        print(f"Step {i}: {stats['step_time']:.3f}s")
```

### 3. Use Interrupts
- **Ctrl+C** to stop safely
- Python will clean up properly

### 4. For Large Systems (Future)
When neighbor lists are implemented, you'll be able to run:
- 10,000+ spheres efficiently
- 50,000+ spheres with good performance

## What NOT To Do

```python
# DON'T do this:
universe = LivniumHamiltonian(n_spheres=20000, max_spheres=20000)
while True:
    universe.step()  # Will freeze your machine
```

## Summary

- ✅ Your laptop is safe
- ✅ Built-in limits protect you
- ✅ Performance monitoring warns you
- ✅ Ctrl+C stops safely
- ✅ No permanent damage possible

**You can safely explore, experiment, and simulate. The system protects you automatically.**


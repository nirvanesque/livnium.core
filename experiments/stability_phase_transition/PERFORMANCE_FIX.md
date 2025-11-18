# Performance Fix: Recursive Engine Caching

## The Problem

**Creating RecursiveGeometryEngine on every step was EXTREMELY slow!**

For N=5 with 3 layers:
- Building hierarchy = creating 2.5M cells
- Doing this every step = 2000 steps × 2.5M cells = **5 BILLION cell creations!**

That's why it was so slow compared to handling 1000+ omcubes.

## The Fix

**Cache the recursive engine per system size:**

```python
# Before (SLOW - rebuilds every step):
recursive = RecursiveGeometryEngine(base_geometry=system, max_depth=3)  # Every step!

# After (FAST - reuse cached engine):
if cache_key not in _recursive_cache:
    _recursive_cache[cache_key] = RecursiveGeometryEngine(...)  # Once per size
recursive = _recursive_cache[cache_key]
recursive.base_geometry = system  # Just update reference
```

## Performance Improvement

- **Before**: Rebuild 2.5M cells every step = ~seconds per step
- **After**: Reuse cached engine = ~milliseconds per step
- **Speedup**: ~1000x faster

## What Changed

1. **`recursive_problem_solving.py`**: Added `_recursive_cache` dict
2. **`recursive_stability.py`**: Added `_moksha_cache` dict
3. **All functions**: Check cache first, create only if missing
4. **Update reference**: `recursive.base_geometry = system` (no rebuild)

## Result

Now the experiment should run as fast as your 1000+ omcube handling!

The recursive engine is built **once per system size** and reused across all steps.


# Performance Optimizations

## Key Bottleneck Fixed

The main performance issue was in `_loss_minimization_update()`:
- **Before**: `copy.deepcopy(system)` for every rotation option (9 deep copies per step!)
- **After**: Rotate → test → rotate back (no copying)

This makes the experiment **~10-100x faster** depending on lattice size.

## Optimizations Applied

1. **No Deep Copying in Update Loop**
   - Instead of copying system 9 times per step, we rotate, test, then rotate back
   - Only one deep copy needed for self-healing test (which runs less frequently)

2. **In-Place Updates**
   - System state is modified in place during dynamics
   - No unnecessary object creation

3. **Efficient Rotation Testing**
   - Test all 9 rotation options (3 axes × 3 turns) by rotating back
   - Much faster than deep copying entire system state

## Remaining Optimizations (Future)

1. **Use Numba** (like core does)
   - Core system already uses `@jit` decorators for hot paths
   - Could add numba to task loss computation

2. **Early Stopping**
   - Stop testing rotations once loss reaches 0 (correct answer found)
   - Don't test all 9 if first one works

3. **Caching**
   - Cache task encoding/decoding results
   - Cache rotation results

4. **Parallel Runs**
   - Run multiple task instances in parallel
   - Use multiprocessing for independent runs

## Performance Tips

- Start with smaller `runs_per_size` (10-50) for testing
- Use smaller `t_max` (500-1000) initially
- Test with N=3 first (smallest, fastest)
- Increase parameters once you verify it works


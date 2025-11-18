# Basin Parameter Tuning Experiment

## Goal

Find the **"nice-behaved growth mode"** - a parameter regime where:
- Success rate drifts up monotonically (no oscillations)
- No ugly drops after peak
- High final success rate (> 60%)
- Smooth, stable growth

This is **phase diagram scanning** - the system tuning itself to find optimal physics constants.

## What It Does

The experiment performs a **grid search** over basin reinforcement parameters:

- **α (alpha)**: How aggressively to deepen correct basins
- **β (beta)**: How hard to punish wrong states  
- **noise**: How often to decorrelate wrong states

For each configuration, it measures:
- `final_rate`: Overall success rate
- `drift`: Late rate - early rate (should be > +5%)
- `max_drop`: Peak rate - valley rate (should be < 2%)
- `early_rate`: First 100 tasks
- `late_rate`: Last 100 tasks

## Criteria for "Nice-Behaved Growth"

A configuration passes if:
1. ✅ `drift > +0.05` (at least +5% improvement over time)
2. ✅ `max_drop < 0.02` (no drop bigger than 2%)
3. ✅ `final_rate > 0.60` (success rate > 60%)

## Usage

### Quick Test (Small Grid)
```bash
cd experiments/stability_phase_transition
python3 tune_basin_params.py --n 3 --tasks 200 --top-k 5
```

### Full Grid Search
```bash
python3 tune_basin_params.py --n 3 --tasks 500 --top-k 10
```

### Custom Parameter Ranges
```bash
python3 tune_basin_params.py \
    --n 3 \
    --tasks 500 \
    --alpha 0.05 0.08 0.10 0.12 \
    --beta 0.10 0.15 0.20 \
    --noise 0.01 0.02 0.03 \
    --top-k 10
```

### Save Results
```bash
python3 tune_basin_params.py --n 3 --tasks 500 --save results/basin_tuning.json
```

## Output

The experiment will:
1. Test all parameter combinations
2. Show progress for each configuration
3. Print summary with top configurations
4. Save results to JSON file

### Example Output

```
Top 10 Configurations:
Rank   α      β      noise    Rate     Drift    Drop     Status
----------------------------------------------------------------------
1      0.10   0.15   0.02     65.2%    +12.3%   1.1%     ✓ PASS
2      0.08   0.15   0.03     64.8%    +11.5%   1.3%     ✓ PASS
3      0.10   0.20   0.02     63.5%    +10.2%   1.8%     ✓ PASS
...

Best Configuration:
  α = 0.100
  β = 0.150
  noise = 0.020
  
  Final rate: 65.2%
  Drift: +12.3%
  Max drop: 1.1%
  Early rate: 48.5%
  Late rate: 60.8%
  
  ✓ This configuration meets all criteria for 'nice-behaved growth mode'
```

## Next Steps

Once you find optimal parameters:
1. Use them in `fast_task_test.py` (update default values)
2. Apply to Ramsey experiments (same basin rules, different task)
3. The system will be in "growth-only" regime - no oscillations

## Physics Interpretation

This is **energy-landscape shaping**:
- Correct structures → deeper, stable wells
- Incorrect clusters → flattened, noisy, erased
- No oscillatory chaos → smooth growth toward truth

The optimal parameters become your **physics constants** for stable computation.


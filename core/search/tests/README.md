# Search Module Tests

Assertion-based tests for the search module.

## Test Files

- **`test_native_dynamic_basin_search.py`**: Tests for dynamic basin reinforcement
  - Geometry signal computation (curvature, tension, entropy)
  - Dynamic basin updates (correct/incorrect)
  - DynamicBasinSearch class
  - Convenience functions
  - Multiple updates

- **`test_multi_basin_search.py`**: Tests for multi-basin search
  - Basin dataclass
  - MultiBasinSearch initialization
  - Adding and updating basins
  - Basin competition
  - Winner selection
  - Statistics
  - Candidate basin creation
  - High-level solve function
  - Basin pruning

- **`test_corner_rotation_policy.py`**: Tests for corner rotation policy
  - Corner rotation policy (with/without coords)
  - Convergence detection
  - Rotation affects corners check
  - Safe rotation selection
  - Auto-detection
  - Threshold testing

## Running Tests

Run individual test files:
```bash
python3 core/search/tests/test_native_dynamic_basin_search.py
python3 core/search/tests/test_multi_basin_search.py
python3 core/search/tests/test_corner_rotation_policy.py
```

Run all tests:
```bash
python3 core/search/tests/test_native_dynamic_basin_search.py && python3 core/search/tests/test_multi_basin_search.py && python3 core/search/tests/test_corner_rotation_policy.py
```

## Test Style

All tests use Python `assert` statements. Each test function:
- Has a descriptive name starting with `test_`
- Uses clear assertions with helpful error messages
- Tests one specific aspect of functionality
- Can be run independently

## Test Coverage

The tests cover:
- ✅ Geometry signal computation (curvature, tension, entropy)
- ✅ Dynamic basin reinforcement (correct/incorrect updates)
- ✅ Multi-basin competition and winner selection
- ✅ Basin statistics and pruning
- ✅ Corner rotation policy and convergence detection
- ✅ High-level solve functions
- ✅ Edge cases and error handling


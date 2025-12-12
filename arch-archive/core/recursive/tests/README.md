# Recursive Module Tests

Assertion-based tests for the recursive geometry engine.

## Test Files

- **`test_recursive_geometry_engine.py`**: Tests for `RecursiveGeometryEngine`
  - Basic initialization
  - Hierarchy building
  - GeometryLevel class
  - Cell subdivision
  - Recursive rotation
  - Recursive observer
  - Total capacity calculation
  - Level statistics
  - State projection (downward/upward)
  - Entanglement compression
  - Moksha checks
  - Subdivision rules

- **`test_geometry_subdivision.py`**: Tests for `GeometrySubdivision`
  - Basic initialization
  - Subdivision by face exposure
  - Subdivision by symbolic weight
  - Subdivide all cells
  - Subdivision statistics
  - Invalid level handling

- **`test_recursive_projection.py`**: Tests for `RecursiveProjection`
  - Basic initialization
  - Downward projection
  - Upward projection
  - Missing levels handling
  - Constraint projection
  - Value projection
  - Value aggregation

- **`test_recursive_conservation.py`**: Tests for `RecursiveConservation`
  - Basic initialization
  - Level conservation verification
  - Recursive conservation verification
  - Downward propagation
  - Upward aggregation
  - Conservation statistics
  - Invalid level handling

- **`test_moksha_engine.py`**: Tests for `MokshaEngine`
  - Basic initialization
  - ConvergenceState enum
  - FixedPointState dataclass
  - Convergence checking
  - Convergence score calculation
  - Final truth export
  - Termination check
  - Reset functionality
  - Statistics

## Running Tests

Run individual test files:
```bash
python3 core/recursive/tests/test_recursive_geometry_engine.py
python3 core/recursive/tests/test_geometry_subdivision.py
python3 core/recursive/tests/test_recursive_projection.py
python3 core/recursive/tests/test_recursive_conservation.py
python3 core/recursive/tests/test_moksha_engine.py
```

## Test Style

All tests use Python `assert` statements. Each test function:
- Has a descriptive name starting with `test_`
- Uses clear assertions with helpful error messages
- Tests one specific aspect of functionality
- Can be run independently

## Test Coverage

The tests cover:
- ✅ Recursive geometry engine initialization and hierarchy
- ✅ Cell subdivision (by face exposure, SW, or all)
- ✅ State projection (downward and upward)
- ✅ Conservation laws (SW and class counts)
- ✅ Moksha convergence detection
- ✅ Fixed point state tracking
- ✅ Statistics and reporting
- ✅ Error handling and edge cases


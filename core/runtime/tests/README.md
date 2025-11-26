# Runtime Module Tests

Assertion-based tests for the runtime orchestrator.

## Test Files

- **`test_temporal_engine.py`**: Tests for `TemporalEngine`
  - Basic initialization
  - Custom periods
  - Timestep progression
  - Timestep type determination (MACRO, MICRO, QUANTUM, MEMORY, SEMANTIC)
  - Scheduled operations
  - Timestep history
  - Statistics
  - Timestep enum and state

- **`test_orchestrator.py`**: Tests for `Orchestrator`
  - Basic initialization
  - Layer initialization (quantum, memory, etc.)
  - Step execution
  - Update types (macro, micro, quantum, memory, semantic)
  - System status
  - Multiple steps

- **`test_episode_manager.py`**: Tests for `EpisodeManager`
  - Basic initialization
  - Starting episodes
  - Running episodes
  - Termination conditions
  - Ending episodes
  - Episode counter
  - Reward calculation
  - Episode statistics
  - Episode history

## Running Tests

Run individual test files:
```bash
python3 core/runtime/tests/test_temporal_engine.py
python3 core/runtime/tests/test_orchestrator.py
python3 core/runtime/tests/test_episode_manager.py
```

## Test Style

All tests use Python `assert` statements. Each test function:
- Has a descriptive name starting with `test_`
- Uses clear assertions with helpful error messages
- Tests one specific aspect of functionality
- Can be run independently

## Test Coverage

The tests cover:
- ✅ Temporal engine timestep management
- ✅ Timestep type determination and scheduling
- ✅ Orchestrator layer coordination
- ✅ Episode management (start, run, end)
- ✅ Termination conditions
- ✅ Reward calculation
- ✅ Statistics and history tracking
- ✅ Error handling and edge cases


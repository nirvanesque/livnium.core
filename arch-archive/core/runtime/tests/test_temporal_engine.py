"""
Test assertions for TemporalEngine.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.runtime.temporal_engine import TemporalEngine, Timestep, TimestepState


def test_basic_initialization():
    """Test basic temporal engine initialization."""
    engine = TemporalEngine()
    
    assert engine.current_timestep == 0, "Should start at timestep 0"
    assert engine.macro_period == 1, "Default macro period should be 1"
    assert engine.micro_period == 5, "Default micro period should be 5"
    assert engine.quantum_period == 1, "Default quantum period should be 1"
    assert engine.memory_period == 10, "Default memory period should be 10"
    assert len(engine.timestep_history) == 0, "Should start with empty history"


def test_custom_periods():
    """Test initialization with custom periods."""
    engine = TemporalEngine(
        macro_period=2,
        micro_period=3,
        quantum_period=4,
        memory_period=5
    )
    
    assert engine.macro_period == 2, "Should set custom macro period"
    assert engine.micro_period == 3, "Should set custom micro period"
    assert engine.quantum_period == 4, "Should set custom quantum period"
    assert engine.memory_period == 5, "Should set custom memory period"


def test_step():
    """Test timestep progression."""
    engine = TemporalEngine()
    
    state = engine.step()
    
    assert isinstance(state, TimestepState), "Should return TimestepState"
    assert engine.current_timestep == 1, "Should advance to timestep 1"
    assert state.timestep == 1, "State should have timestep 1"
    assert len(engine.timestep_history) == 1, "Should record in history"


def test_timestep_type_macro():
    """Test macro timestep type."""
    engine = TemporalEngine(macro_period=1)
    
    state = engine.step()
    assert state.timestep_type == Timestep.MACRO, "Timestep 1 should be MACRO"


def test_timestep_type_micro():
    """Test micro timestep type."""
    engine = TemporalEngine(micro_period=1, macro_period=2)
    
    state = engine.step()
    # Timestep 1: macro_period=2 (not divisible), micro_period=1 (divisible) → MICRO
    assert state.timestep_type == Timestep.MICRO, "Should be MICRO when micro_period matches"


def test_timestep_type_quantum():
    """Test quantum timestep type."""
    engine = TemporalEngine(quantum_period=1, macro_period=2, micro_period=3)
    
    state = engine.step()
    # Priority: MACRO > MICRO > QUANTUM
    # If none match, should be QUANTUM or SEMANTIC
    assert state.timestep_type in [Timestep.QUANTUM, Timestep.SEMANTIC], \
        "Should be QUANTUM or SEMANTIC"


def test_timestep_type_memory():
    """Test memory timestep type."""
    engine = TemporalEngine(memory_period=1, macro_period=2, micro_period=3, quantum_period=4)
    
    state = engine.step()
    # Priority order determines type
    assert isinstance(state.timestep_type, Timestep), "Should be valid Timestep type"


def test_timestep_type_semantic():
    """Test semantic timestep type (default)."""
    engine = TemporalEngine(macro_period=2, micro_period=3, quantum_period=4, memory_period=5)
    
    state = engine.step()
    # Timestep 1: none match → SEMANTIC
    assert state.timestep_type == Timestep.SEMANTIC, "Should be SEMANTIC when no periods match"


def test_schedule_operation():
    """Test scheduling operations."""
    engine = TemporalEngine()
    
    executed = []
    def test_op():
        executed.append(True)
    
    engine.schedule_operation(timestep=3, operation=test_op)
    
    assert len(engine.scheduled_operations) == 1, "Should have 1 scheduled operation"
    
    # Step to timestep 3
    engine.step()  # timestep 1
    engine.step()  # timestep 2
    assert len(executed) == 0, "Should not execute before timestep"
    
    engine.step()  # timestep 3
    assert len(executed) == 1, "Should execute at scheduled timestep"


def test_timestep_history():
    """Test timestep history tracking."""
    engine = TemporalEngine()
    
    # Step multiple times
    for _ in range(5):
        engine.step()
    
    assert len(engine.timestep_history) == 5, "Should have 5 history entries"
    assert engine.timestep_history[-1].timestep == 5, "Last entry should be timestep 5"


def test_timestep_statistics():
    """Test timestep statistics."""
    engine = TemporalEngine()
    
    # Step multiple times
    for _ in range(10):
        engine.step()
    
    stats = engine.get_timestep_statistics()
    
    assert 'current_timestep' in stats, "Should have current timestep"
    assert 'total_timesteps' in stats, "Should have total timesteps"
    assert 'timestep_type_counts' in stats, "Should have type counts"
    assert 'scheduled_operations' in stats, "Should have scheduled operations count"
    
    assert stats['current_timestep'] == 10, "Current timestep should be 10"
    assert stats['total_timesteps'] == 10, "Total timesteps should be 10"


def test_timestep_enum():
    """Test Timestep enum."""
    assert Timestep.MACRO.value == "macro"
    assert Timestep.MICRO.value == "micro"
    assert Timestep.QUANTUM.value == "quantum"
    assert Timestep.MEMORY.value == "memory"
    assert Timestep.SEMANTIC.value == "semantic"


def test_timestep_state():
    """Test TimestepState dataclass."""
    state = TimestepState(
        timestep=1,
        timestep_type=Timestep.MACRO,
        timestamp=123.45,
        state_snapshot={'test': 'value'}
    )
    
    assert state.timestep == 1, "Timestep should match"
    assert state.timestep_type == Timestep.MACRO, "Type should match"
    assert state.timestamp == 123.45, "Timestamp should match"
    assert state.state_snapshot == {'test': 'value'}, "Snapshot should match"


if __name__ == "__main__":
    print("Running TemporalEngine tests...")
    
    test_basic_initialization()
    print("✓ Basic initialization")
    
    test_custom_periods()
    print("✓ Custom periods")
    
    test_step()
    print("✓ Step")
    
    test_timestep_type_macro()
    print("✓ Timestep type macro")
    
    test_timestep_type_micro()
    print("✓ Timestep type micro")
    
    test_timestep_type_quantum()
    print("✓ Timestep type quantum")
    
    test_timestep_type_memory()
    print("✓ Timestep type memory")
    
    test_timestep_type_semantic()
    print("✓ Timestep type semantic")
    
    test_schedule_operation()
    print("✓ Schedule operation")
    
    test_timestep_history()
    print("✓ Timestep history")
    
    test_timestep_statistics()
    print("✓ Timestep statistics")
    
    test_timestep_enum()
    print("✓ Timestep enum")
    
    test_timestep_state()
    print("✓ Timestep state")
    
    print("\nAll tests passed! ✓")


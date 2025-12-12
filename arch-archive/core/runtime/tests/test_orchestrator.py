"""
Test assertions for Orchestrator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.runtime.orchestrator import Orchestrator
from core.runtime.temporal_engine import Timestep
from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig


def test_basic_initialization():
    """Test basic orchestrator initialization."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    
    orchestrator = Orchestrator(core_system)
    
    assert orchestrator.core_system == core_system, "Should reference core system"
    assert orchestrator.config == config, "Should reference config"
    assert orchestrator.temporal_engine is not None, "Should have temporal engine"
    assert orchestrator.temporal_engine.current_timestep == 0, "Should start at timestep 0"


def test_layer_initialization():
    """Test layer initialization."""
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_quantum=True,
        enable_memory=True
    )
    core_system = LivniumCoreSystem(config)
    
    orchestrator = Orchestrator(core_system)
    
    # Quantum layer should be initialized if enabled
    assert orchestrator.quantum_lattice is not None, "Quantum lattice should be initialized"
    
    # Memory layer should be initialized if enabled
    assert orchestrator.memory_lattice is not None, "Memory lattice should be initialized"


def test_step():
    """Test orchestrator step."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    
    orchestrator = Orchestrator(core_system)
    
    result = orchestrator.step()
    
    assert isinstance(result, dict), "Should return dictionary"
    assert 'timestep' in result, "Should have timestep"
    assert 'type' in result, "Should have type"
    assert result['timestep'] == 1, "Should be timestep 1"


def test_macro_update():
    """Test macro-level update."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    
    orchestrator = Orchestrator(core_system)
    
    # Force macro timestep
    orchestrator.temporal_engine.macro_period = 1
    orchestrator.temporal_engine.micro_period = 100
    orchestrator.temporal_engine.quantum_period = 100
    orchestrator.temporal_engine.memory_period = 100
    
    result = orchestrator.step()
    
    assert result['type'] == 'macro', "Should be macro update"


def test_micro_update():
    """Test micro-level update."""
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_memory=True
    )
    core_system = LivniumCoreSystem(config)
    
    orchestrator = Orchestrator(core_system)
    
    # Force micro timestep
    orchestrator.temporal_engine.macro_period = 100
    orchestrator.temporal_engine.micro_period = 1
    orchestrator.temporal_engine.quantum_period = 100
    orchestrator.temporal_engine.memory_period = 100
    
    result = orchestrator.step()
    
    assert result['type'] == 'micro', "Should be micro update"


def test_quantum_update():
    """Test quantum-level update."""
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_quantum=True
    )
    core_system = LivniumCoreSystem(config)
    
    orchestrator = Orchestrator(core_system)
    
    # Force quantum timestep
    orchestrator.temporal_engine.macro_period = 100
    orchestrator.temporal_engine.micro_period = 100
    orchestrator.temporal_engine.quantum_period = 1
    orchestrator.temporal_engine.memory_period = 100
    
    result = orchestrator.step()
    
    assert result['type'] == 'quantum', "Should be quantum update"


def test_memory_update():
    """Test memory-level update."""
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_memory=True
    )
    core_system = LivniumCoreSystem(config)
    
    orchestrator = Orchestrator(core_system)
    
    # Force memory timestep
    orchestrator.temporal_engine.macro_period = 100
    orchestrator.temporal_engine.micro_period = 100
    orchestrator.temporal_engine.quantum_period = 100
    orchestrator.temporal_engine.memory_period = 1
    
    result = orchestrator.step()
    
    assert result['type'] == 'memory', "Should be memory update"
    assert 'memory_stats' in result, "Should have memory stats"


def test_semantic_update():
    """Test semantic-level update."""
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_semantic=True
    )
    core_system = LivniumCoreSystem(config)
    
    orchestrator = Orchestrator(core_system)
    
    # Force semantic timestep (none of the periods match)
    orchestrator.temporal_engine.macro_period = 100
    orchestrator.temporal_engine.micro_period = 100
    orchestrator.temporal_engine.quantum_period = 100
    orchestrator.temporal_engine.memory_period = 100
    
    result = orchestrator.step()
    
    assert result['type'] == 'semantic', "Should be semantic update"


def test_get_system_status():
    """Test system status retrieval."""
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_quantum=True,
        enable_memory=True
    )
    core_system = LivniumCoreSystem(config)
    
    orchestrator = Orchestrator(core_system)
    
    status = orchestrator.get_system_status()
    
    assert isinstance(status, dict), "Should return dictionary"
    assert 'timestep' in status, "Should have timestep"
    assert 'layers_active' in status, "Should have layers_active"
    
    layers = status['layers_active']
    assert layers['classical'] == True, "Classical should always be active"
    assert layers['quantum'] == True, "Quantum should be active if enabled"
    assert layers['memory'] == True, "Memory should be active if enabled"


def test_multiple_steps():
    """Test multiple orchestrator steps."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    
    orchestrator = Orchestrator(core_system)
    
    # Step multiple times
    for i in range(5):
        result = orchestrator.step()
        assert result['timestep'] == i + 1, f"Timestep should be {i+1}"
    
    assert orchestrator.temporal_engine.current_timestep == 5, "Should be at timestep 5"


if __name__ == "__main__":
    print("Running Orchestrator tests...")
    
    test_basic_initialization()
    print("✓ Basic initialization")
    
    test_layer_initialization()
    print("✓ Layer initialization")
    
    test_step()
    print("✓ Step")
    
    test_macro_update()
    print("✓ Macro update")
    
    test_micro_update()
    print("✓ Micro update")
    
    test_quantum_update()
    print("✓ Quantum update")
    
    test_memory_update()
    print("✓ Memory update")
    
    test_semantic_update()
    print("✓ Semantic update")
    
    test_get_system_status()
    print("✓ Get system status")
    
    test_multiple_steps()
    print("✓ Multiple steps")
    
    print("\nAll tests passed! ✓")


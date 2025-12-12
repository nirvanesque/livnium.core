"""
Test assertions for EpisodeManager.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.runtime.episode_manager import EpisodeManager, Episode
from core.runtime.orchestrator import Orchestrator
from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig


def test_basic_initialization():
    """Test basic episode manager initialization."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    orchestrator = Orchestrator(core_system)
    
    manager = EpisodeManager(orchestrator)
    
    assert manager.orchestrator == orchestrator, "Should reference orchestrator"
    assert manager.current_episode is None, "Should start with no episode"
    assert len(manager.episode_history) == 0, "Should start with empty history"
    assert manager.episode_counter == 0, "Should start at counter 0"


def test_start_episode():
    """Test starting an episode."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    orchestrator = Orchestrator(core_system)
    
    manager = EpisodeManager(orchestrator)
    
    episode = manager.start_episode()
    
    assert isinstance(episode, Episode), "Should return Episode"
    assert episode.episode_id == 1, "Should have episode ID 1"
    assert episode.initial_state is not None, "Should have initial state"
    assert episode.final_state is None, "Should not have final state yet"
    assert episode.terminated == False, "Should not be terminated"
    assert manager.current_episode == episode, "Should be current episode"


def test_start_episode_with_initial_state():
    """Test starting episode with custom initial state."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    orchestrator = Orchestrator(core_system)
    
    manager = EpisodeManager(orchestrator)
    
    initial_state = {'custom': 'state', 'value': 42}
    episode = manager.start_episode(initial_state=initial_state)
    
    assert episode.initial_state == initial_state, "Should use provided initial state"


def test_run_episode():
    """Test running an episode."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    orchestrator = Orchestrator(core_system)
    
    manager = EpisodeManager(orchestrator)
    
    episode = manager.run_episode(max_timesteps=5)
    
    assert isinstance(episode, Episode), "Should return Episode"
    assert episode.timesteps == 5, "Should have 5 timesteps"
    assert episode.final_state is not None, "Should have final state"
    assert episode.reward >= 0.0, "Should have non-negative reward"
    assert len(manager.episode_history) == 1, "Should be in history"


def test_run_episode_with_termination():
    """Test running episode with termination condition."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    orchestrator = Orchestrator(core_system)
    
    manager = EpisodeManager(orchestrator)
    
    termination_count = [0]
    def termination_condition():
        termination_count[0] += 1
        return termination_count[0] >= 3
    
    episode = manager.run_episode(max_timesteps=10, termination_condition=termination_condition)
    
    assert episode.terminated == True, "Should be terminated"
    assert episode.timesteps == 3, "Should stop at 3 timesteps"


def test_end_episode():
    """Test ending an episode."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    orchestrator = Orchestrator(core_system)
    
    manager = EpisodeManager(orchestrator)
    
    manager.start_episode()
    assert manager.current_episode is not None, "Should have current episode"
    
    manager.end_episode()
    
    assert manager.current_episode is None, "Should not have current episode"
    assert len(manager.episode_history) == 1, "Should be in history"
    assert manager.episode_history[0].terminated == True, "Should be terminated"


def test_episode_counter():
    """Test episode counter increment."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    orchestrator = Orchestrator(core_system)
    
    manager = EpisodeManager(orchestrator)
    
    episode1 = manager.start_episode()
    assert episode1.episode_id == 1, "First episode should be ID 1"
    
    manager.end_episode()
    
    episode2 = manager.start_episode()
    assert episode2.episode_id == 2, "Second episode should be ID 2"


def test_calculate_reward():
    """Test reward calculation."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    orchestrator = Orchestrator(core_system)
    
    manager = EpisodeManager(orchestrator)
    
    episode = manager.run_episode(max_timesteps=3)
    
    assert 0.0 <= episode.reward <= 1.0, "Reward should be in [0, 1]"
    # Reward should be high if invariants are preserved
    assert episode.reward > 0.0, "Should have positive reward for stable system"


def test_episode_statistics():
    """Test episode statistics."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    orchestrator = Orchestrator(core_system)
    
    manager = EpisodeManager(orchestrator)
    
    # Run multiple episodes
    ep1 = manager.run_episode(max_timesteps=5)
    ep2 = manager.run_episode(max_timesteps=10)
    
    stats = manager.get_episode_statistics()
    
    assert 'total_episodes' in stats, "Should have total episodes"
    assert 'average_timesteps' in stats, "Should have average timesteps"
    assert 'average_reward' in stats, "Should have average reward"
    
    assert stats['total_episodes'] == 2, "Should have 2 episodes"
    # Average should be calculated from actual episode timesteps
    expected_avg = (ep1.timesteps + ep2.timesteps) / 2.0
    assert abs(stats['average_timesteps'] - expected_avg) < 1e-6, \
        f"Average should be {expected_avg}, got {stats['average_timesteps']}"
    assert 0.0 <= stats['average_reward'] <= 1.0, "Average reward should be in [0, 1]"


def test_episode_dataclass():
    """Test Episode dataclass."""
    episode = Episode(
        episode_id=1,
        initial_state={'test': 'value'},
        final_state={'test': 'final'},
        timesteps=10,
        terminated=True,
        reward=0.8
    )
    
    assert episode.episode_id == 1, "Episode ID should match"
    assert episode.initial_state == {'test': 'value'}, "Initial state should match"
    assert episode.final_state == {'test': 'final'}, "Final state should match"
    assert episode.timesteps == 10, "Timesteps should match"
    assert episode.terminated == True, "Terminated should match"
    assert episode.reward == 0.8, "Reward should match"


def test_episode_history():
    """Test episode history tracking."""
    config = LivniumCoreConfig(lattice_size=3)
    core_system = LivniumCoreSystem(config)
    orchestrator = Orchestrator(core_system)
    
    manager = EpisodeManager(orchestrator)
    
    # Run multiple episodes
    ep1 = manager.run_episode(max_timesteps=3)
    ep2 = manager.run_episode(max_timesteps=5)
    ep3 = manager.run_episode(max_timesteps=7)
    
    assert len(manager.episode_history) == 3, "Should have 3 episodes in history"
    assert manager.episode_history[0].episode_id == ep1.episode_id, "First episode should match"
    assert manager.episode_history[1].episode_id == ep2.episode_id, "Second episode should match"
    assert manager.episode_history[2].episode_id == ep3.episode_id, "Third episode should match"


if __name__ == "__main__":
    print("Running EpisodeManager tests...")
    
    test_basic_initialization()
    print("✓ Basic initialization")
    
    test_start_episode()
    print("✓ Start episode")
    
    test_start_episode_with_initial_state()
    print("✓ Start episode with initial state")
    
    test_run_episode()
    print("✓ Run episode")
    
    test_run_episode_with_termination()
    print("✓ Run episode with termination")
    
    test_end_episode()
    print("✓ End episode")
    
    test_episode_counter()
    print("✓ Episode counter")
    
    test_calculate_reward()
    print("✓ Calculate reward")
    
    test_episode_statistics()
    print("✓ Episode statistics")
    
    test_episode_dataclass()
    print("✓ Episode dataclass")
    
    test_episode_history()
    print("✓ Episode history")
    
    print("\nAll tests passed! ✓")


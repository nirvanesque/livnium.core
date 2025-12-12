"""
Ramsey Checkpoint: Save and Resume Progress

Allows long-running Ramsey experiments to save progress and resume.
"""

import json
import os
import pickle
from typing import Dict, Any, Optional
from pathlib import Path


def get_checkpoint_path(n_vertices: int, constraint_type: str = "k4") -> Path:
    """Get checkpoint file path for given problem."""
    checkpoint_dir = Path("experiments/ramsey/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"ramsey_k{n_vertices}_{constraint_type}.ckpt"


def save_checkpoint(
    checkpoint_path: Path,
    best_coloring: Dict,
    best_violations: int,
    step: int,
    system_state: Any,  # LivniumCoreSystem state (SW values)
    encoder_state: Any,  # Edge-to-coords mapping
    metadata: Dict[str, Any]
) -> None:
    """
    Save checkpoint: best coloring, violations, step, and system state.
    
    Args:
        checkpoint_path: Path to checkpoint file
        best_coloring: Best coloring found so far
        best_violations: Best violation count
        step: Current step number
        system_state: System SW values (dict of coords -> SW)
        encoder_state: Encoder edge mapping
        metadata: Additional metadata (tension, score, etc.)
    """
    checkpoint_data = {
        'best_coloring': best_coloring,
        'best_violations': best_violations,
        'step': step,
        'system_state': system_state,
        'encoder_state': encoder_state,
        'metadata': metadata
    }
    
    # Save as pickle (handles complex objects)
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    # Also save human-readable summary
    summary_path = checkpoint_path.with_suffix('.json')
    summary = {
        'best_violations': best_violations,
        'step': step,
        'metadata': metadata
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint if it exists.
    
    Returns:
        Checkpoint data dict or None if not found
    """
    if not checkpoint_path.exists():
        return None
    
    try:
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  ⚠️  Failed to load checkpoint: {e}")
        return None


def restore_system_state(system, encoder, checkpoint_data: Dict[str, Any]) -> None:
    """
    Restore system state from checkpoint.
    
    Args:
        system: LivniumCoreSystem to restore
        encoder: RamseyEncoder to restore
        checkpoint_data: Loaded checkpoint data
    """
    # Restore SW values
    system_state = checkpoint_data.get('system_state', {})
    for coords, sw_value in system_state.items():
        cell = system.get_cell(coords)
        if cell:
            cell.symbolic_weight = sw_value
    
    # Restore encoder state (edge mapping should be same, but verify)
    encoder_state = checkpoint_data.get('encoder_state', {})
    # Edge mapping is deterministic, so we just verify it matches


def should_resume(checkpoint_path: Path, max_steps: int) -> bool:
    """
    Check if we should resume from checkpoint.
    
    Returns True if:
    - Checkpoint exists
    - Checkpoint step < max_steps (can continue)
    """
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        return False
    
    checkpoint_step = checkpoint.get('step', 0)
    return checkpoint_step < max_steps


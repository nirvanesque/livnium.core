"""
Task Framework for Task-Driven Stability Experiment

Defines tasks that can be encoded into the lattice and solved.
The physics only emerges when there's a task to solve.
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from core.classical.livnium_core_system import LivniumCoreSystem


@dataclass
class Task:
    """Base class for tasks."""
    name: str
    input_data: Any
    correct_answer: Any
    
    @abstractmethod
    def encode_into_lattice(self, system: LivniumCoreSystem) -> None:
        """Encode task input into the lattice."""
        pass
    
    @abstractmethod
    def decode_answer(self, system: LivniumCoreSystem) -> Any:
        """Decode answer from the lattice."""
        pass
    
    @abstractmethod
    def is_correct(self, answer: Any) -> bool:
        """Check if answer is correct."""
        pass
    
    @abstractmethod
    def compute_loss(self, system: LivniumCoreSystem) -> float:
        """Compute task loss (energy = wrongness)."""
        pass


class Parity3Task(Task):
    """
    3-bit parity task.
    
    Input: 3 bits (b0, b1, b2)
    Output: XOR(b0, b1, b2) (0 or 1)
    """
    
    def __init__(self, bits: Tuple[int, int, int]):
        """
        Initialize 3-bit parity task.
        
        Args:
            bits: Tuple of 3 bits (0 or 1)
        """
        if len(bits) != 3 or not all(b in [0, 1] for b in bits):
            raise ValueError("bits must be tuple of 3 values in {0, 1}")
        
        self.bits = bits
        correct = (bits[0] ^ bits[1] ^ bits[2]) % 2
        
        super().__init__(
            name="parity_3bit",
            input_data=bits,
            correct_answer=correct
        )
    
    def encode_into_lattice(self, system: LivniumCoreSystem) -> None:
        """
        Encode 3 bits into first 3 cells of lattice.
        
        Uses symbolic weight to encode: SW = 0 → bit=0, SW > 0 → bit=1
        Or uses symbol encoding.
        """
        coords_list = sorted(system.lattice.keys())[:3]
        
        for i, (coords, bit) in enumerate(zip(coords_list, self.bits)):
            cell = system.lattice[coords]
            # Encode bit in symbolic weight (normalize to 0 or 9)
            if bit == 0:
                # Set to core-like (SW = 0)
                # Actually, we'll use a different encoding: store in symbol
                if system.config.enable_symbol_alphabet:
                    # Use first symbol for 0, second for 1
                    alphabet = system.generate_alphabet(system.lattice_size)
                    symbol_idx = 0 if bit == 0 else 1
                    if symbol_idx < len(alphabet):
                        system.set_symbol(coords, alphabet[symbol_idx])
    
    def decode_answer(self, system: LivniumCoreSystem) -> int:
        """
        Decode answer from a designated output cell.
        
        Uses the last cell (or a specific cell) to read the answer.
        """
        coords_list = sorted(system.lattice.keys())
        if len(coords_list) < 4:
            return 0  # Not enough cells
        
        # Use the 4th cell as output
        output_coords = coords_list[3]
        cell = system.lattice[output_coords]
        
        # Decode from symbolic weight or symbol
        if system.config.enable_symbol_alphabet:
            symbol = cell.symbol
            if symbol:
                alphabet = system.generate_alphabet(system.lattice_size)
                # Answer is 1 if symbol is alphabet[1], else 0
                return 1 if symbol == alphabet[1] else 0
        
        # Fallback: use symbolic weight threshold
        return 1 if (cell.symbolic_weight or 0) > 4.5 else 0
    
    def is_correct(self, answer: Any) -> bool:
        """Check if answer matches correct parity."""
        return int(answer) == self.correct_answer
    
    def compute_loss(self, system: LivniumCoreSystem) -> float:
        """
        Compute task loss = 0 if correct, 1 if wrong.
        
        This is the "energy" - wrongness of the current state.
        """
        answer = self.decode_answer(system)
        return 0.0 if self.is_correct(answer) else 1.0


class SimpleClassificationTask(Task):
    """
    Simple 2D point classification task.
    
    Input: (x, y) coordinates
    Output: Class 0 or 1 (based on simple rule like x + y > threshold)
    """
    
    def __init__(self, point: Tuple[float, float], threshold: float = 0.0):
        """
        Initialize classification task.
        
        Args:
            point: (x, y) coordinates
            threshold: Classification threshold
        """
        self.point = point
        self.threshold = threshold
        correct = 1 if (point[0] + point[1]) > threshold else 0
        
        super().__init__(
            name="simple_classification",
            input_data=point,
            correct_answer=correct
        )
    
    def encode_into_lattice(self, system: LivniumCoreSystem) -> None:
        """Encode point coordinates into first 2 cells."""
        coords_list = sorted(system.lattice.keys())[:2]
        x, y = self.point
        
        # Encode x in first cell's symbolic weight (normalized)
        # Encode y in second cell's symbolic weight
        # For simplicity, map to [0, 27] range (SW range)
        if len(coords_list) >= 2:
            # Normalize coordinates to [0, 27] range
            # Assuming input is in [-1, 1] range
            x_norm = int((x + 1) * 13.5)  # Map to [0, 27]
            y_norm = int((y + 1) * 13.5)
            
            # Store in symbols or use a different encoding
            # For now, we'll use a simpler approach: encode in cell indices
            pass  # Placeholder - would need actual encoding mechanism
    
    def decode_answer(self, system: LivniumCoreSystem) -> int:
        """Decode classification from output cell."""
        coords_list = sorted(system.lattice.keys())
        if len(coords_list) < 3:
            return 0
        
        # Use 3rd cell as output
        output_coords = coords_list[2]
        cell = system.lattice[output_coords]
        
        # Decode from symbolic weight
        sw = cell.symbolic_weight or 0
        return 1 if sw > 13.5 else 0
    
    def is_correct(self, answer: Any) -> bool:
        """Check if classification is correct."""
        return int(answer) == self.correct_answer
    
    def compute_loss(self, system: LivniumCoreSystem) -> float:
        """Compute classification loss."""
        answer = self.decode_answer(system)
        return 0.0 if self.is_correct(answer) else 1.0


class ConstraintSatisfactionTask(Task):
    """
    Simple constraint satisfaction task.
    
    Example: Ensure certain cells have specific properties.
    """
    
    def __init__(self, constraints: List[Callable[[LivniumCoreSystem], bool]]):
        """
        Initialize constraint satisfaction task.
        
        Args:
            constraints: List of constraint functions
        """
        self.constraints = constraints
        
        super().__init__(
            name="constraint_satisfaction",
            input_data=constraints,
            correct_answer=True  # True when all constraints satisfied
        )
    
    def encode_into_lattice(self, system: LivniumCoreSystem) -> None:
        """Constraints are already defined, no encoding needed."""
        pass
    
    def decode_answer(self, system: LivniumCoreSystem) -> bool:
        """Answer is whether all constraints are satisfied."""
        return all(constraint(system) for constraint in self.constraints)
    
    def is_correct(self, answer: Any) -> bool:
        """Check if all constraints satisfied."""
        return bool(answer)
    
    def compute_loss(self, system: LivniumCoreSystem) -> float:
        """Loss = number of violated constraints."""
        violated = sum(1 for constraint in self.constraints if not constraint(system))
        return float(violated) / len(self.constraints) if self.constraints else 0.0


def create_task(task_type: str, **kwargs) -> Task:
    """
    Factory function to create tasks.
    
    Args:
        task_type: Type of task ("parity_3bit", "classification", "constraint")
        **kwargs: Task-specific parameters
        
    Returns:
        Task instance
    """
    if task_type == "parity_3bit":
        return Parity3Task(kwargs.get("bits", (0, 0, 0)))
    elif task_type == "classification":
        return SimpleClassificationTask(
            kwargs.get("point", (0.0, 0.0)),
            kwargs.get("threshold", 0.0)
        )
    elif task_type == "constraint":
        return ConstraintSatisfactionTask(
            kwargs.get("constraints", [])
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")


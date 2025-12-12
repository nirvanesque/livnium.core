"""
Law Extractor: Auto-Discovery of Physical Laws from Livnium Core

Extracts physical laws by observing system behavior:
- Invariants (conserved quantities)
- Functional relationships (f(x) = y)
- Convergent patterns

This enables Livnium to discover its own laws instead of having them hardcoded.
"""

import numpy as np
from typing import Dict, List, Callable, Tuple, Optional


class LivniumLawExtractor:
    """
    Extracts physical laws from Livnium Core by observing:
    - invariants (values that remain constant)
    - conserved quantities (sum/mean/energy)
    - functional relationships (f(x) = y)
    """

    def __init__(self, invariant_threshold: float = 1e-6, relationship_error_threshold: float = 1e-3):
        """
        Initialize law extractor.
        
        Args:
            invariant_threshold: Maximum std dev for a quantity to be considered invariant
            relationship_error_threshold: Maximum error for a relationship to be considered valid
        """
        self.history: List[Dict[str, float]] = []
        self.invariant_threshold = invariant_threshold
        self.relationship_error_threshold = relationship_error_threshold

    def record_state(self, state: Dict[str, float]):
        """
        Called each timestep. 
        
        'state' is a dictionary containing measurable quantities:
        {
            'SW_sum': ...,
            'alignment': ...,
            'divergence': ...,
            'energy': ...,
            'curvature': ...,
            'tension': ...
        }
        """
        self.history.append(state.copy())

    def detect_invariants(self) -> Dict[str, bool]:
        """
        Checks if some quantities remain approximately constant.
        
        Returns:
            Dictionary mapping quantity names to whether they are invariant
        """
        if len(self.history) < 2:
            return {}
        
        invariants = {}
        for key in self.history[0].keys():
            values = np.array([h[key] for h in self.history if key in h])
            
            if len(values) < 2:
                invariants[key] = False
                continue
            
            # Check if standard deviation is below threshold
            std_dev = np.std(values)
            invariants[key] = std_dev < self.invariant_threshold
        
        return invariants

    def detect_functional_relationships(self) -> Dict[str, Tuple[float, float]]:
        """
        Fit simple linear relationships of the form f(x) = a*x + b
        
        Example: divergence = 0.38 - alignment
        
        Returns:
            Dictionary mapping relationship names to (a, b) coefficients
        """
        if len(self.history) < 3:
            return {}
        
        relationships = {}
        keys = list(self.history[0].keys())

        for i, xkey in enumerate(keys):
            for j, ykey in enumerate(keys):
                if i == j:
                    continue

                # Extract x and y values
                x_values = []
                y_values = []
                
                for state in self.history:
                    if xkey in state and ykey in state:
                        x_values.append(state[xkey])
                        y_values.append(state[ykey])
                
                if len(x_values) < 3:
                    continue
                
                x = np.array(x_values)
                y = np.array(y_values)

                # Fit y = a*x + b
                try:
                    a, b = np.polyfit(x, y, 1)
                except (np.linalg.LinAlgError, ValueError):
                    continue

                # Only keep strong relationships
                y_pred = a * x + b
                error = np.mean(np.abs(y_pred - y))

                if error < self.relationship_error_threshold:
                    relationships[f"{ykey}_vs_{xkey}"] = (a, b)

        return relationships

    def extract(self) -> Dict[str, any]:
        """
        Full law extraction.
        
        Returns:
            Dictionary with:
            - 'invariants': Dict mapping quantity names to whether they're invariant
            - 'relationships': Dict mapping relationship names to (a, b) coefficients
        """
        return {
            "invariants": self.detect_invariants(),
            "relationships": self.detect_functional_relationships()
        }
    
    def get_law_summary(self) -> str:
        """
        Get human-readable summary of discovered laws.
        
        Returns:
            String describing discovered laws
        """
        laws = self.extract()
        
        summary = "=== Discovered Laws ===\n\n"
        
        # Invariants
        invariants = laws["invariants"]
        invariant_names = [name for name, is_invariant in invariants.items() if is_invariant]
        
        if invariant_names:
            summary += "Invariants (Conserved Quantities):\n"
            for name in invariant_names:
                values = [h[name] for h in self.history if name in h]
                if values:
                    summary += f"  - {name}: {values[0]:.6f} (constant)\n"
            summary += "\n"
        else:
            summary += "No invariants detected.\n\n"
        
        # Relationships
        relationships = laws["relationships"]
        
        if relationships:
            summary += "Functional Relationships:\n"
            for rel_name, (a, b) in relationships.items():
                y_name, x_name = rel_name.split("_vs_")
                if abs(a) < 1e-6:
                    summary += f"  - {y_name} = {b:.6f}\n"
                elif abs(b) < 1e-6:
                    summary += f"  - {y_name} = {a:.6f} * {x_name}\n"
                else:
                    sign = "+" if b >= 0 else ""
                    summary += f"  - {y_name} = {a:.6f} * {x_name} {sign}{b:.6f}\n"
        else:
            summary += "No functional relationships detected.\n"
        
        return summary

    def clear_history(self):
        """Clear recorded history."""
        self.history = []


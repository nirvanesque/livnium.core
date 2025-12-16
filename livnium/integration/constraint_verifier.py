"""
Constraint Verifier: High-Level API for Constraint Checking

Provides a simple API for external systems to verify constraints
and get explanations for why actions are inadmissible.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from livnium.kernel.constraints import ConstraintChecker, ConstraintCheck
from livnium.kernel.ledgers import Ledger
from livnium.kernel.types import State, Operation


@dataclass
class VerificationResult:
    """
    Result of constraint verification.
    
    Provides transparent refusal paths with explanations.
    """
    is_valid: bool
    explanation: str
    violations: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "explanation": self.explanation,
            "violations": self.violations
        }


class ConstraintVerifier:
    """
    High-level API for constraint verification.
    
    This is the main interface for external systems to:
    1. Check if actions are admissible
    2. Get explanations for why actions are inadmissible
    3. Understand what constraints are active
    """
    
    def __init__(self, ledger: Optional[Ledger] = None):
        """
        Initialize constraint verifier.
        
        Args:
            ledger: Optional ledger instance (creates new one if not provided)
        """
        if ledger is None:
            ledger = Ledger()
        self.checker = ConstraintChecker(ledger)
    
    def verify_transition(
        self,
        state_before: State,
        state_after: State,
        operation: Operation
    ) -> VerificationResult:
        """
        Verify if a transition is admissible.
        
        Args:
            state_before: State before transition
            state_after: State after transition
            operation: Operation being performed
            
        Returns:
            VerificationResult with explanation
        """
        check = self.checker.check_transition(state_before, state_after, operation)
        
        violations = [v.to_dict() for v in check.violations]
        
        return VerificationResult(
            is_valid=check.is_admissible,
            explanation=check.explain(),
            violations=violations
        )
    
    def verify_promotion(
        self,
        state: State,
        depth: int,
        energy_cost: float,
        available_energy: float
    ) -> VerificationResult:
        """
        Verify if promotion is admissible.
        
        Args:
            state: Current state
            depth: Proposed promotion depth
            energy_cost: Energy cost of promotion
            available_energy: Available energy budget
            
        Returns:
            VerificationResult with explanation
        """
        check = self.checker.check_promotion(state, depth, energy_cost, available_energy)
        
        violations = [v.to_dict() for v in check.violations]
        
        return VerificationResult(
            is_valid=check.is_admissible,
            explanation=check.explain(),
            violations=violations
        )
    
    def explain_violation(self, violation: Dict[str, Any]) -> str:
        """
        Get detailed explanation of a violation.
        
        Args:
            violation: Violation dictionary from VerificationResult
            
        Returns:
            Detailed explanation string
        """
        from livnium.kernel.constraints import ConstraintViolation, ConstraintType
        
        constraint_type = ConstraintType(violation["constraint_type"])
        violation_obj = ConstraintViolation(
            constraint_type=constraint_type,
            is_violated=violation["is_violated"],
            explanation=violation["explanation"],
            details=violation.get("details", {})
        )
        
        return self.checker.explain_violation(violation_obj)


"""
Kernel Constraints: First-Class Constraint Query and Explanation System

This module exposes kernel invariants as queryable constraints that agents can:
1. Query to check if an action is admissible
2. Get explanations for why an action is inadmissible
3. Understand what constraints are active

CRITICAL: This is observation and explanation only. Enforcement remains in admissibility.py.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .types import State, Operation
from .ledgers import Ledger
from .admissibility import is_transition_admissible, check_promotion_admissible


class ConstraintType(Enum):
    """Types of constraints that can be checked."""
    STATE_VALIDITY = "state_validity"
    TRANSITION_VALIDITY = "transition_validity"
    PROMOTION_ENERGY = "promotion_energy"
    PROMOTION_DEPTH = "promotion_depth"
    NORM_BOUND = "norm_bound"
    INVARIANT_PRESERVATION = "invariant_preservation"


@dataclass
class ConstraintViolation:
    """
    Represents a constraint violation with explanation.
    
    This provides transparent refusal paths - agents can understand
    exactly why an action is inadmissible.
    """
    constraint_type: ConstraintType
    is_violated: bool
    explanation: str
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "constraint_type": self.constraint_type.value,
            "is_violated": self.is_violated,
            "explanation": self.explanation,
            "details": self.details
        }


@dataclass
class ConstraintCheck:
    """
    Result of checking constraints for an action.
    
    Contains all constraint violations and a summary.
    """
    is_admissible: bool
    violations: List[ConstraintViolation]
    summary: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_admissible": self.is_admissible,
            "violations": [v.to_dict() for v in self.violations],
            "summary": self.summary
        }
    
    def explain(self) -> str:
        """
        Generate human-readable explanation of why action is admissible or not.
        
        Returns:
            Explanation string suitable for agent consumption
        """
        if self.is_admissible:
            return f"Action is admissible. All constraints satisfied."
        
        if not self.violations:
            return "Action is inadmissible, but no specific violations were detected."
        
        explanations = []
        for violation in self.violations:
            explanations.append(f"- {violation.explanation}")
        
        return f"Action is inadmissible because:\n" + "\n".join(explanations)


class ConstraintChecker:
    """
    First-class constraint checker that agents can query.
    
    This provides transparent refusal paths by explaining exactly
    why actions are inadmissible.
    """
    
    def __init__(self, ledger: Ledger):
        """
        Initialize constraint checker.
        
        Args:
            ledger: Ledger instance for invariant checks
        """
        self.ledger = ledger
    
    def check_transition(
        self,
        state_before: State,
        state_after: State,
        operation: Operation
    ) -> ConstraintCheck:
        """
        Check if a transition is admissible and explain why.
        
        This is the main entry point for agents to query constraints.
        
        Args:
            state_before: State before transition
            state_after: State after transition
            operation: Operation being performed
            
        Returns:
            ConstraintCheck with violations and explanations
        """
        violations = []
        
        # Check state validity
        before_valid = self.ledger.check(state_before)
        after_valid = self.ledger.check(state_after)
        
        if not before_valid:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.STATE_VALIDITY,
                is_violated=True,
                explanation=f"State before {operation.value} operation is invalid (missing vector or norm capability)",
                details={"state": "before", "operation": operation.value}
            ))
        
        if not after_valid:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.STATE_VALIDITY,
                is_violated=True,
                explanation=f"State after {operation.value} operation is invalid (missing vector or norm capability)",
                details={"state": "after", "operation": operation.value}
            ))
        
        # Check transition validity (ledger validation)
        if before_valid and after_valid:
            transition_valid = self.ledger.validate(state_before, state_after)
            if not transition_valid:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.TRANSITION_VALIDITY,
                    is_violated=True,
                    explanation=f"Transition from before to after state violates invariants (ledger validation failed)",
                    details={"operation": operation.value}
                ))
        
        # Check admissibility (enforcement layer)
        is_admissible = is_transition_admissible(
            state_before, state_after, operation, self.ledger
        ) if before_valid and after_valid else False
        
        # Generate summary
        if is_admissible:
            summary = f"{operation.value} operation is admissible"
        else:
            if violations:
                summary = f"{operation.value} operation is inadmissible: {len(violations)} constraint violation(s)"
            else:
                summary = f"{operation.value} operation is inadmissible (unknown reason)"
        
        return ConstraintCheck(
            is_admissible=is_admissible,
            violations=violations,
            summary=summary
        )
    
    def check_promotion(
        self,
        state: State,
        depth: int,
        energy_cost: float,
        available_energy: float
    ) -> ConstraintCheck:
        """
        Check if promotion is admissible and explain why.
        
        Args:
            state: Current state
            depth: Proposed promotion depth
            energy_cost: Energy cost of promotion
            available_energy: Available energy budget
            
        Returns:
            ConstraintCheck with violations and explanations
        """
        violations = []
        
        # Check state validity
        state_valid = self.ledger.check(state)
        if not state_valid:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.STATE_VALIDITY,
                is_violated=True,
                explanation="State is invalid (missing vector or norm capability)",
                details={"operation": "promote"}
            ))
        
        # Check energy sufficiency
        if energy_cost > available_energy:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.PROMOTION_ENERGY,
                is_violated=True,
                explanation=f"Insufficient energy: required {energy_cost:.4f}, available {available_energy:.4f}",
                details={
                    "energy_cost": energy_cost,
                    "available_energy": available_energy,
                    "deficit": energy_cost - available_energy
                }
            ))
        
        # Check depth constraints
        if depth < 0:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.PROMOTION_DEPTH,
                is_violated=True,
                explanation=f"Invalid promotion depth: {depth} (must be >= 0)",
                details={"depth": depth}
            ))
        
        # Check admissibility
        is_admissible = check_promotion_admissible(
            state, depth, energy_cost, available_energy, self.ledger
        ) if state_valid else False
        
        # Generate summary
        if is_admissible:
            summary = f"Promotion to depth {depth} is admissible"
        else:
            if violations:
                summary = f"Promotion to depth {depth} is inadmissible: {len(violations)} constraint violation(s)"
            else:
                summary = f"Promotion to depth {depth} is inadmissible (unknown reason)"
        
        return ConstraintCheck(
            is_admissible=is_admissible,
            violations=violations,
            summary=summary
        )
    
    def explain_violation(self, violation: ConstraintViolation) -> str:
        """
        Generate detailed explanation of a constraint violation.
        
        Args:
            violation: Constraint violation to explain
            
        Returns:
            Detailed explanation string
        """
        base_explanation = violation.explanation
        
        # Add details if available
        if violation.details:
            detail_parts = []
            for key, value in violation.details.items():
                if isinstance(value, float):
                    detail_parts.append(f"{key}={value:.4f}")
                else:
                    detail_parts.append(f"{key}={value}")
            
            if detail_parts:
                base_explanation += f" ({', '.join(detail_parts)})"
        
        return base_explanation


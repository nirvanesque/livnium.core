"""
Kernel Admissibility: Enforcement Logic

CRITICAL SEPARATION:
- Ledger.validate() → observes and reports violations
- admissibility.is_transition_admissible() → DECIDES if transition is legal (enforcement)

Ledgers observe. Admissibility enforces. This keeps "law" distinct from "policing".
"""

from .types import State, Operation
from .ledgers import Ledger


def is_transition_admissible(
    state_before: State,
    state_after: State,
    operation: Operation,
    ledger: Ledger
) -> bool:
    """
    Check if a transition is admissible (legal).
    
    This is ENFORCEMENT - it decides whether a transition is allowed.
    It uses ledger observations to make this decision.
    
    Args:
        state_before: State before transition
        state_after: State after transition
        operation: Operation being performed
        ledger: Ledger instance for invariant checks
        
    Returns:
        True if transition is admissible, False otherwise
    """
    # First, check that ledger validates the transition
    if not ledger.validate(state_before, state_after):
        return False
    
    # Additional admissibility rules can be added here
    # For example:
    # - Check operation-specific constraints
    # - Check energy conservation
    # - Check promotion depth limits
    
    # For now, if ledger validates, transition is admissible
    return True


def check_promotion_admissible(
    state: State,
    depth: int,
    energy_cost: float,
    available_energy: float,
    ledger: Ledger
) -> bool:
    """
    Check if promotion is admissible.
    
    Promotion requires:
    1. State must be valid (ledger check)
    2. Sufficient energy available
    3. Depth constraints satisfied
    
    Args:
        state: Current state
        depth: Proposed promotion depth
        energy_cost: Energy cost of promotion
        available_energy: Available energy budget
        ledger: Ledger instance for invariant checks
        
    Returns:
        True if promotion is admissible, False otherwise
    """
    # Check state validity
    if not ledger.check(state):
        return False
    
    # Check energy sufficiency
    if energy_cost > available_energy:
        return False
    
    # Check depth constraints (can be extended)
    if depth < 0:
        return False
    
    return True


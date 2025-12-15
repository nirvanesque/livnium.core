"""
Kernel Ledgers: Invariant Checks

CRITICAL RULE: Ledgers may observe and validate. They may NOT decide or enforce.

✅ Allowed:
- check(state) -> bool - observe state
- record(state) -> Dict - record observations
- validate(before, after) -> bool - validate transitions

❌ NEVER:
- enforce() - enforcement belongs to admissibility.py

This keeps "law" distinct from "policing".
"""

from typing import Dict, Any
from .types import State, LedgerRecord, Operation


class Ledger:
    """
    Ledger for tracking invariants.
    
    Ledgers observe and validate. They do not enforce.
    Enforcement is handled by admissibility.py.
    """
    
    def __init__(self):
        """Initialize empty ledger."""
        self.records: list[LedgerRecord] = []
    
    def check(self, state: State) -> bool:
        """
        Check if state satisfies invariants.
        
        This observes the state and reports whether invariants hold.
        It does NOT enforce or modify anything.
        
        Args:
            state: State to check
            
        Returns:
            True if invariants are satisfied, False otherwise
        """
        # Basic invariant: state must have valid vector and norm
        try:
            vec = state.vector()
            norm = state.norm()
            # State is valid if it has these capabilities
            return vec is not None and norm is not None
        except (AttributeError, TypeError):
            return False
    
    def record(self, state: State, operation: Operation) -> LedgerRecord:
        """
        Record state observation in ledger.
        
        This records what was observed, not what should happen.
        
        Args:
            state: State to record
            operation: Operation that produced this state
            
        Returns:
            LedgerRecord with observed values
        """
        vec = state.vector()
        norm = state.norm()
        
        record: LedgerRecord = {
            "total_sw": 0.0,  # Symbolic weight (to be computed by specific ledger types)
            "norm_bound": float(norm) if norm is not None else 0.0,
            "basin_count": 0,  # Basin count (to be computed by specific ledger types)
            "timestamp": len(self.records)
        }
        
        self.records.append(record)
        return record
    
    def validate(self, before: State, after: State) -> bool:
        """
        Validate that transition from before to after preserves invariants.
        
        This observes both states and reports whether the transition is valid.
        It does NOT prevent invalid transitions - that's admissibility's job.
        
        Args:
            before: State before transition
            after: State after transition
            
        Returns:
            True if transition preserves invariants, False otherwise
        """
        # Basic validation: both states must be valid
        before_valid = self.check(before)
        after_valid = self.check(after)
        
        if not (before_valid and after_valid):
            return False
        
        # Additional invariant checks can be added here
        # For now, we just ensure both states are valid
        
        return True


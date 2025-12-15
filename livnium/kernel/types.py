"""
Kernel Types: Minimal Protocols for State, Anchor, LedgerRecord, Operation

CRITICAL: State defines capabilities (what laws can touch), not shape.
We're not defining what a state IS - we're defining what the laws are allowed to TOUCH.
"""

from typing import Protocol, TypedDict
from enum import Enum


class State(Protocol):
    """
    State protocol: defines what laws are allowed to touch.
    
    CRITICAL: This defines capabilities, not shape.
    We're not saying "state must be a tensor" - we're saying
    "state must provide these capabilities that laws need."
    """
    
    def vector(self):
        """Return the vector representation that laws can operate on."""
        ...
    
    def norm(self):
        """Return the norm of the state vector."""
        ...


class Anchor(Protocol):
    """
    Anchor protocol: minimal interface for anchor vectors.
    
    Anchors are reference points in the geometric space.
    """
    
    def vector(self):
        """Return the anchor vector."""
        ...
    
    def norm(self):
        """Return the norm of the anchor vector."""
        ...


class LedgerRecord(TypedDict):
    """
    Ledger record structure for invariant tracking.
    
    Ledgers observe and record state invariants.
    """
    total_sw: float  # Total symbolic weight
    norm_bound: float  # Vector norm bound
    basin_count: int  # Number of basins
    timestamp: int  # Operation timestamp


class Operation(Enum):
    """
    Types of operations that can be performed.
    
    Used for admissibility checks and ledger tracking.
    """
    COLLAPSE = "collapse"
    PROMOTE = "promote"
    SPAWN = "spawn"
    MERGE = "merge"
    PRUNE = "prune"


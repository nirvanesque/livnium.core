#!/usr/bin/env python3
"""
Verify if the 4 invariants (I₁–I₄) are De Bruijn flow constraints.

For 3-bit patterns, the De Bruijn graph has:
- Nodes: 2-bit patterns (00, 01, 10, 11)
- Edges: 3-bit patterns (000..111)
- Flow conservation: For each node XY, in_flow(XY) = out_flow(XY)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import sympy
    from sympy import symbols, Eq, simplify
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.divergence_v3 import enumerate_patterns

Pattern = Tuple[int, int, int]
Node = Tuple[int, int]


def enumerate_2bit_nodes() -> List[Node]:
    """Return all 2-bit patterns (nodes in de Bruijn graph)."""
    return [(0, 0), (0, 1), (1, 0), (1, 1)]


def pattern_to_edge(p: Pattern) -> Tuple[Node, Node]:
    """
    Convert 3-bit pattern to edge in de Bruijn graph.
    
    Pattern (a, b, c) represents edge: node (a, b) → node (b, c)
    """
    return ((p[0], p[1]), (p[1], p[2]))


def build_debruijn_flow_constraints() -> List[Dict[Pattern, int]]:
    """
    Build flow conservation constraints for de Bruijn graph.
    
    For each 2-bit node XY:
        in_flow(XY) = sum of patterns where bc = XY (patterns ending at XY)
        out_flow(XY) = sum of patterns where ab = XY (patterns starting at XY)
    
    Flow conservation: in_flow(XY) = out_flow(XY)
    
    Returns:
        List of constraint vectors (one per node)
    """
    nodes = enumerate_2bit_nodes()
    patterns = enumerate_patterns()
    
    flow_constraints = []
    
    for node in nodes:
        # Build constraint: in_flow(node) - out_flow(node) = 0
        constraint = {p: 0 for p in patterns}
        
        # In-flow: patterns where this node is the destination (bc = node)
        for p in patterns:
            edge_start, edge_end = pattern_to_edge(p)
            if edge_end == node:
                constraint[p] += 1  # Contributes to in-flow
        
        # Out-flow: patterns where this node is the source (ab = node)
        for p in patterns:
            edge_start, edge_end = pattern_to_edge(p)
            if edge_start == node:
                constraint[p] -= 1  # Contributes to out-flow (subtract)
        
        flow_constraints.append(constraint)
    
    return flow_constraints


def get_invariant_vectors() -> List[Dict[Pattern, int]]:
    """Get the 4 invariant vectors as dictionaries."""
    patterns = enumerate_patterns()
    pattern_str = {p: ''.join(str(b) for b in p) for p in patterns}
    
    invariants = []
    
    # I₁: freq('100') - freq('001')
    I1 = {p: 0 for p in patterns}
    for p in patterns:
        if pattern_str[p] == '100':
            I1[p] = 1
        elif pattern_str[p] == '001':
            I1[p] = -1
    invariants.append(I1)
    
    # I₂: freq('001') - freq('010') - freq('011') + freq('101')
    I2 = {p: 0 for p in patterns}
    for p in patterns:
        if pattern_str[p] == '001':
            I2[p] = 1
        elif pattern_str[p] == '010':
            I2[p] = -1
        elif pattern_str[p] == '011':
            I2[p] = -1
        elif pattern_str[p] == '101':
            I2[p] = 1
    invariants.append(I2)
    
    # I₃: freq('110') - freq('011')
    I3 = {p: 0 for p in patterns}
    for p in patterns:
        if pattern_str[p] == '110':
            I3[p] = 1
        elif pattern_str[p] == '011':
            I3[p] = -1
    invariants.append(I3)
    
    # I₄: freq('000') + freq('001') + 2·freq('010') + 3·freq('011') + freq('111')
    I4 = {p: 0 for p in patterns}
    for p in patterns:
        if pattern_str[p] == '000':
            I4[p] = 1
        elif pattern_str[p] == '001':
            I4[p] = 1
        elif pattern_str[p] == '010':
            I4[p] = 2
        elif pattern_str[p] == '011':
            I4[p] = 3
        elif pattern_str[p] == '111':
            I4[p] = 1
    invariants.append(I4)
    
    return invariants


def check_if_flow_constraint(invariant: Dict[Pattern, int], 
                             flow_constraints: List[Dict[Pattern, int]]) -> Tuple[bool, int]:
    """
    Check if an invariant is a linear combination of flow constraints.
    
    Returns:
        (is_flow, flow_index) where flow_index is which flow constraint if is_flow=True
    """
    # Check if invariant is exactly equal to one of the flow constraints
    for i, flow in enumerate(flow_constraints):
        if invariant == flow:
            return True, i
    
    # Check if invariant is a scalar multiple of a flow constraint
    for i, flow in enumerate(flow_constraints):
        # Check if invariant = k * flow for some scalar k
        non_zero_flow = [v for v in flow.values() if v != 0]
        if not non_zero_flow:
            continue
        
        # Find scaling factor
        k = None
        matches = True
        for p in invariant:
            if flow[p] == 0:
                if invariant[p] != 0:
                    matches = False
                    break
            else:
                if k is None:
                    if invariant[p] % flow[p] != 0:
                        matches = False
                        break
                    k = invariant[p] // flow[p]
                else:
                    if invariant[p] != k * flow[p]:
                        matches = False
                        break
        
        if matches and k is not None:
            return True, i
    
    return False, -1


def main():
    """Verify if invariants are flow constraints."""
    if not SYMPY_AVAILABLE:
        print("Error: sympy required")
        return
    
    print("="*70)
    print("VERIFYING IF INVARIANTS ARE DE BRUIJN FLOW CONSTRAINTS")
    print("="*70)
    print()
    
    # Get flow constraints
    flow_constraints = build_debruijn_flow_constraints()
    nodes = enumerate_2bit_nodes()
    patterns = enumerate_patterns()
    pattern_str = {p: ''.join(str(b) for b in p) for p in patterns}
    
    print("De Bruijn Flow Constraints (for 2-bit nodes):")
    for i, (node, flow) in enumerate(zip(nodes, flow_constraints)):
        node_str = ''.join(str(b) for b in node)
        terms = []
        for p in patterns:
            if flow[p] != 0:
                p_str = pattern_str[p]
                if flow[p] == 1:
                    terms.append(f"freq('{p_str}')")
                elif flow[p] == -1:
                    terms.append(f"-freq('{p_str}')")
                else:
                    terms.append(f"{flow[p]}*freq('{p_str}')")
        print(f"  Node {node_str}: {' + '.join(terms)} = 0")
    print()
    
    # Get invariants
    invariants = get_invariant_vectors()
    invariant_names = ['I₁', 'I₂', 'I₃', 'I₄']
    invariant_formulas = [
        "freq('100') - freq('001')",
        "freq('001') - freq('010') - freq('011') + freq('101')",
        "freq('110') - freq('011')",
        "freq('000') + freq('001') + 2·freq('010') + 3·freq('011') + freq('111')"
    ]
    
    print("Checking invariants:")
    print()
    
    results = {}
    for i, (name, formula, inv) in enumerate(zip(invariant_names, invariant_formulas, invariants)):
        is_flow, flow_idx = check_if_flow_constraint(inv, flow_constraints)
        results[name] = (is_flow, flow_idx)
        
        print(f"{name}: {formula}")
        if is_flow:
            node_str = ''.join(str(b) for b in nodes[flow_idx])
            print(f"  ✅ IS a De Bruijn flow constraint (node {node_str})")
        else:
            print(f"  ❌ NOT a De Bruijn flow constraint (non-flow invariant)")
        print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    
    flow_invariants = [name for name, (is_flow, _) in results.items() if is_flow]
    non_flow_invariants = [name for name, (is_flow, _) in results.items() if not is_flow]
    
    print(f"Flow-based invariants (removable): {', '.join(flow_invariants) if flow_invariants else 'None'}")
    print(f"Non-flow invariants (structural): {', '.join(non_flow_invariants) if non_flow_invariants else 'None'}")
    print()
    
    return results


if __name__ == "__main__":
    main()


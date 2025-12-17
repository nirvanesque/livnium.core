"""
Document Recursive Projection Demo: Deep Semantic Reconciliation

This demo extends the contradiction collapse example by adding recursive
refinement. When global tension plateaus, the system "zooms in" to
high-tension basins and re-runs reconciliation at a finer semantic scale.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from livnium.domains.document.encoder import DocumentEncoder, Claim
from livnium.integration.pipeline import DocumentPipeline
from livnium.engine.collapse.engine import CollapseEngine
from livnium.domains.document.head import DocumentHead

def run_recursive_refine_demo():
    print("=" * 60)
    print("Document Recursive Projection Demo")
    print("=" * 60)
    print()
    print("Scenario: Contract dispute with multiple nested contradictions")
    print()
    
    # Setup
    dim = 64
    encoder = DocumentEncoder(dim=dim, vocab_size=1000)
    collapse_engine = CollapseEngine(dim=dim, num_layers=3)
    head = DocumentHead(dim=dim)
    pipeline = DocumentPipeline(encoder, collapse_engine, head)
    
    # Create claims with hierarchical contradictions
    claims = [
        Claim(
            claim_id="CLAIM_A",
            text="The contract expires on December 31st, 2025.",
            position=0,
            citations=[]
        ),
        Claim(
            claim_id="CLAIM_B",
            text="Termination occurs at the end of the year 2025.",
            position=1,
            citations=[]
        ),
        Claim(
            claim_id="CLAIM_C",
            text="The agreement is valid until January 2024 only.",
            position=2,
            citations=[]
        ),
        Claim(
            claim_id="CLAIM_D",
            text="The contract has no fixed expiration date.",
            position=3,
            citations=[]
        ),
        Claim(
            claim_id="CLAIM_E",
            text="The termination clause allows indefinite renewal.",
            position=4,
            citations=[]
        ),
    ]
    
    print(f"[1] Processing {len(claims)} claims with potential nested conflicts...")
    for claim in claims:
        print(f"    - {claim.claim_id}: \"{claim.text}\"")
    print()
    
    # Draft
    draft_result = pipeline.draft(claims)
    
    # Phase 1: Flat Reconciliation
    print("[2] Phase 1: Flat Reconciliation (Global Collapse)...")
    flat_result = pipeline.reconcile_contradictions(draft_result)
    
    tension_flat_initial = flat_result.global_tension_history[0]
    tension_flat_final = flat_result.global_tension_history[-1]
    
    print(f"    - Initial Tension: {tension_flat_initial:.4f}")
    print(f"    - Final Tension:   {tension_flat_final:.4f}")
    print(f"    - Reduction:       {(tension_flat_initial - tension_flat_final):.4f}")
    print(f"    - Clusters Found:  {len(flat_result.clusters)}")
    print()
    
    # Phase 2: Recursive Refinement (Opt-in)
    print("[3] Phase 2: Recursive Refinement (Zooming In)...")
    print(f"    Checking if tension plateau ({tension_flat_final:.4f}) warrants recursion...")
    print()
    
    recursive_result = pipeline.recursive_refine(flat_result, depth=1)
    
    print(f"    - Depth Reached:        {recursive_result.depth_reached}")
    print(f"    - Moksha Achieved:      {recursive_result.is_moksha}")
    print(f"    - Additional Reduction: {recursive_result.recursive_tension_reduction:.4f}")
    print(f"    - Refined Clusters:     {len(recursive_result.refined_clusters)}")
    print()
    
    # Phase 3: Results Comparison
    print("=" * 60)
    print("Final Reconciliation Report")
    print("=" * 60)
    print()
    
    print("Flat Reconciliation (Depth 0):")
    for i, cluster in enumerate(flat_result.clusters, 1):
        print(f"  Cluster {i}: {cluster}")
    print()
    
    print("Recursive Refinement (Depth 1):")
    for i, cluster in enumerate(recursive_result.refined_clusters, 1):
        print(f"  Cluster {i}: {cluster}")
    print()
    
    # Key Observables
    print("Key Observables:")
    print(f"  - Tension (Initial):           {tension_flat_initial:.4f}")
    print(f"  - Tension (After Flat):        {tension_flat_final:.4f}")
    print(f"  - Tension (After Recursive):   {tension_flat_final - recursive_result.recursive_tension_reduction:.4f}")
    print(f"  - Moksha State:                {'REACHED' if recursive_result.is_moksha else 'SEARCHING'}")
    print()
    
    print("=" * 60)
    print("Interpretation:")
    print("=" * 60)
    if recursive_result.is_moksha:
        print("✓ System reached a fixed point (Moksha).")
        print("  No further semantic refinement is possible.")
    else:
        print("⚠ System has residual tension.")
        print("  Consider deeper recursion or manual review.")
    print("=" * 60)

if __name__ == "__main__":
    run_recursive_refine_demo()

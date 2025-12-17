"""
Document Audit Demo: Project Vision

Demonstrates the instrumentation layer by auditing a reconciliation 
process step-by-step.
"""

import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from livnium.domains.document.encoder import DocumentEncoder, Claim
from livnium.domains.document.reconciler import ContradictionReconciler
from livnium.instrumentation.ledger import TensionLedger
from livnium.instrumentation.visualizer import PhysicsVisualizer

def run_audit_demo():
    print("=" * 60)
    print("Document Audit Demo: Project Vision")
    print("=" * 60)
    
    # 1. Setup Data: Multiple claims with partial consistency
    claims = [
        Claim("A", "The project is on track for Q3.", 0, []),
        Claim("B", "Deployment happens in late September.", 1, []), # Consistent with A
        Claim("C", "The timeline has slipped to 2025.", 2, []),      # Contradicts A & B
        Claim("D", "We are hiring 5 more engineers.", 3, []),       # Neutral
        Claim("E", "The budget is approved for Q3.", 4, []),        # Consistent with A
    ]
    
    encoder = DocumentEncoder(dim=64)
    reconciler = ContradictionReconciler(encoder, iterations=15)
    
    # Initialize Ledger
    ledger = TensionLedger([c.claim_id for c in claims])
    
    # 2. Run Reconciliation with Audit
    print("\n[Action] Running reconciliation with active instrumentation...")
    result = reconciler.reconcile(claims, auditor=ledger)
    
    # 3. Use Visualizer for Insights
    print("\n" + "=" * 30)
    print("VISUAL AUDIT REPORT")
    print("=" * 30)
    
    # Show Tension Curve
    PhysicsVisualizer.print_tension_curve(result.global_tension_history)
    
    # Show Force Map (Pairwise Alignments)
    # We reconstruct the final alignment matrix for the map
    final_h = torch.stack([result.final_claims_map[cid] for cid in ledger.claim_ids])
    final_align = torch.mm(final_h, final_h.t())
    PhysicsVisualizer.print_force_map(ledger.claim_ids, final_align)
    
    # 4. Final Summary
    print("\n" + "=" * 30)
    print("HYBRID PHYSICS SUMMARY")
    print("=" * 30)
    summary = result.audit_summary
    if summary:
        print(f"Reduction: {summary['reduction_pct']:.2f}%")
        print(f"Convergence: {'STABLE' if summary['is_stable'] else 'SEARCHING'}")
        print(f"Final Basins: {len(result.clusters)}")
        for i, cluster in enumerate(result.clusters):
            print(f"  Basin {i+1}: {cluster}")
            
        # Export Markdown
        markdown = PhysicsVisualizer.export_markdown_audit(ledger.claim_ids, final_align, summary)
        with open("reasoning_audit.md", "w") as f:
            f.write(markdown)
        print(f"\n[Artifact] Reasoning audit exported to: reasoning_audit.md")

if __name__ == "__main__":
    run_audit_demo()

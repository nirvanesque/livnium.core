"""
Document Contradiction Demo: Truth Reconciliation Loop

This demo showcases "Contradiction Collapse" in the document domain.
It feeds a set of conflicting claims into the Livnium pipeline and uses
mutual attraction/repulsion physics to reconcile the narrative into 
consistent clusters (Basins of Truth).

Scenario: A disputed contract clause with three different interpretations.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from livnium.domains.document.encoder import DocumentEncoder, Claim, Document, Citation
from livnium.domains.document.head import DocumentHead
from livnium.engine.collapse.engine import CollapseEngine
from livnium.integration.pipeline import DocumentPipeline

def run_contradiction_demo():
    print("============================================================")
    print("Document Contradiction Demo: The Truth Reconciliation Loop")
    print("============================================================")

    # 1. Setup Components
    dim = 256
    encoder = DocumentEncoder(dim=dim)
    engine = CollapseEngine(dim=dim)
    head = DocumentHead(dim=dim)
    pipeline = DocumentPipeline(encoder, engine, head)

    # 2. Define conflicting claims
    # We have two claims that agree and one that contradicts them.
    claims = [
        Claim(
            claim_id="CLAIM_A",
            text="The contract expires on December 31st, 2025.",
            position=0
        ),
        Claim(
            claim_id="CLAIM_B",
            text="Termination occurs at the end of the year 2025.",
            position=1
        ),
        Claim(
            claim_id="CLAIM_C",
            text="The agreement is valid until January 2024 only.",
            position=2
        )
    ]

    print(f"\n[1] Processing {len(claims)} claims:")
    for c in claims:
        print(f"    - {c.claim_id}: \"{c.text}\"")

    # 3. Draft & Reconcile
    print("\n[2] Running Contradiction Reconciliation (Mutual Physics)...")
    draft = pipeline.draft(claims)
    reconciliation = pipeline.reconcile_contradictions(draft)

    # 4. Analyze Results
    print("\n[3] Reconciliation Results:")
    print(f"    - Global Tension (Inital): {reconciliation.global_tension_history[0]:.4f}")
    print(f"    - Global Tension (Final):  {reconciliation.global_tension_history[-1]:.4f}")
    
    print("\n    Consensus Clusters (Consistent Narratives):")
    for i, cluster in enumerate(reconciliation.clusters):
        print(f"      Group {i+1}: {cluster}")

    print("\n    Detected Contradictions (Push-Apart Forces):")
    if reconciliation.contradictions:
        for a, b in reconciliation.contradictions:
            print(f"      ❌ {a} contradicts {b}")
    else:
        print("      No strong contradictions detected.")

    # 5. Final Report
    print("\n============================================================")
    print("Final Reconciliation Report")
    print("============================================================")
    if len(reconciliation.clusters) > 1:
        print("Status: MULTIPLE NARRATIVES DETECTED.")
        print("Action: System suggests manual review of contradictions.")
    else:
        print("Status: CONSENSUS REACHED.")
        print("Action: Proceed with automated validation.")
    print("============================================================")

if __name__ == "__main__":
    run_contradiction_demo()

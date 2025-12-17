"""
SNLI Audit Demo: Domain Maturity Phase 2

Demonstrates the SNLI Workflow with full step-by-step instrumentation.
"""

import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from livnium.domains.snli.workflow import SNLIWorkflow
from livnium.instrumentation.visualizer import PhysicsVisualizer

def run_snli_audit():
    print("=" * 60)
    print("SNLI Audit Demo: Domain Maturity Phase 2")
    print("=" * 60)
    
    workflow = SNLIWorkflow(dim=64, num_layers=10)
    
    # Test Cases
    pairs = [
        {
            "name": "Classical Entailment",
            "p": "A man is playing a guitar on a stage.",
            "h": "A man is performing music."
        },
        {
            "name": "Strong Contradiction",
            "p": "A person is sleeping in a bed.",
            "h": "A person is running a marathon."
        },
        {
            "name": "Neutral / Ambiguous",
            "p": "A woman is eating an apple.",
            "h": "The apple is delicious and red."
        }
    ]
    
    for pair in pairs:
        print(f"\n[Case] {pair['name']}")
        print(f"  P: {pair['p']}")
        print(f"  H: {pair['h']}")
        
        # We manually create a ledger to visualize the curve
        from livnium.instrumentation.ledger import TensionLedger
        ledger = TensionLedger(["premise", "hypothesis"])
        
        # Analyze
        result = workflow.analyze(pair['p'], pair['h'], use_instrumentation=True)
        
        # Visual Audit
        PhysicsVisualizer.print_tension_curve([r.global_tension for r in ledger.history])
        
        # Report
        print(f"\n  Prediction: {result.label.upper()} ({result.confidence:.2%})")
        print(f"  Initial Alignment: {result.alignment:.4f}")
        print(f"  Final Tension: {result.tension:.4f}")
        
        if result.audit_summary:
            print(f"  Tension Reduction: {result.audit_summary['reduction_pct']:.2f}%")

if __name__ == "__main__":
    run_snli_audit()

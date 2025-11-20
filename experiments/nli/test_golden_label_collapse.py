"""
Golden Label Collapse Test

Tests whether the 3-way collapse mechanism works correctly when given explicit labels.
This diagnoses if the issue is in the collapse engine or the energy/detector model.

Usage:
    python3 experiments/nli/test_golden_label_collapse.py
"""

import os
import sys
import argparse
import shutil
import numpy as np
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.config import LivniumCoreConfig
from experiments.nli.native_chain_encoder import NativeChainNLIEncoder, NativeEncodedPair
from experiments.nli.omcube import OmcubeNLIClassifier, CrossOmcubeCoupling
from experiments.nli.native_chain import GlobalLexicon


def test_golden_label_collapse(
    premise: str,
    hypothesis: str,
    golden_label: str,
    encoder: NativeChainNLIEncoder,
    coupling: CrossOmcubeCoupling
) -> Dict:
    """
    Test collapse behavior when given a golden label.
    
    Returns:
        Dict with collapse results, basin depths, and success flag
    """
    # Encode the pair
    encoded_pair = encoder.encode_pair(premise, hypothesis)
    
    # Create classifier
    classifier = OmcubeNLIClassifier(encoded_pair)
    classifier.coupling = coupling
    
    # Get initial basin depths
    initial_depths = coupling.get_basin_depths()
    
    # Get initial superposition state
    initial_state = classifier.get_superposition_state()
    
    # Perform collapse
    result = classifier.classify(collapse=True)
    
    # Get final basin depths
    final_depths = coupling.get_basin_depths()
    
    # Map label to index
    label_to_idx = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    golden_idx = label_to_idx.get(golden_label, -1)
    
    # Check if collapse matched golden label
    collapsed_idx = result.collapsed_omcube
    success = (collapsed_idx == golden_idx)
    
    # Calculate basin change
    basin_changes = {
        i: final_depths.get(i, 0.0) - initial_depths.get(i, 0.0)
        for i in [0, 1, 2]
    }
    
    return {
        'premise': premise,
        'hypothesis': hypothesis,
        'golden_label': golden_label,
        'golden_idx': golden_idx,
        'collapsed_label': result.label,
        'collapsed_idx': collapsed_idx,
        'success': success,
        'confidence': result.confidence,
        'probabilities': result.probabilities,
        'initial_depths': initial_depths,
        'final_depths': final_depths,
        'basin_changes': basin_changes,
        'initial_superposition': initial_state['probabilities'],
        'resonance': encoded_pair.get_resonance()
    }


def apply_golden_feedback(
    premise: str,
    hypothesis: str,
    golden_label: str,
    encoder: NativeChainNLIEncoder,
    coupling: CrossOmcubeCoupling
) -> Dict:
    """
    Apply learning feedback with golden label and observe basin response.
    """
    # Encode
    encoded_pair = encoder.encode_pair(premise, hypothesis)
    classifier = OmcubeNLIClassifier(encoded_pair)
    classifier.coupling = coupling
    
    # Get initial state
    initial_depths = coupling.get_basin_depths()
    
    # Classify
    result = classifier.classify(collapse=True)
    
    # Apply golden feedback
    label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    golden_idx = label_map[golden_label]
    
    classifier.apply_learning_feedback(
        result,
        golden_idx,
        learning_strength=1.0
    )
    
    # Get final state
    final_depths = coupling.get_basin_depths()
    
    # Calculate changes
    basin_changes = {
        i: final_depths.get(i, 0.0) - initial_depths.get(i, 0.0)
        for i in [0, 1, 2]
    }
    
    return {
        'golden_label': golden_label,
        'golden_idx': golden_idx,
        'collapsed_to': result.label,
        'initial_depths': initial_depths,
        'final_depths': final_depths,
        'basin_changes': basin_changes,
        'expected_deepening': golden_idx,
        'actual_deepening': max(basin_changes.items(), key=lambda x: x[1])[0] if basin_changes else -1
    }


def print_test_results(results: List[Dict]):
    """Print formatted test results."""
    print("="*80)
    print("GOLDEN LABEL COLLAPSE TEST RESULTS")
    print("="*80)
    print()
    
    for i, result in enumerate(results, 1):
        print(f"Test {i}: {result['premise'][:30]}... | {result['hypothesis'][:30]}...")
        print(f"  Golden Label: {result['golden_label']:12s} (idx={result['golden_idx']})")
        print(f"  Collapsed To: {result['collapsed_label']:12s} (idx={result['collapsed_idx']})")
        print(f"  Success: {'✓ YES' if result['success'] else '✗ NO'}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Resonance: {result['resonance']:.3f}")
        print()
        print("  Probabilities:")
        print(f"    Entailment:   {result['probabilities']['entailment']:.3f}")
        print(f"    Contradiction: {result['probabilities']['contradiction']:.3f}")
        print(f"    Neutral:      {result['probabilities']['neutral']:.3f}")
        print()
        print("  Initial Superposition:")
        print(f"    Entailment:   {result['initial_superposition']['entailment']:.3f}")
        print(f"    Contradiction: {result['initial_superposition']['contradiction']:.3f}")
        print(f"    Neutral:      {result['initial_superposition']['neutral']:.3f}")
        print()
        print("  Basin Depths (Before → After):")
        labels = ['Entailment', 'Contradiction', 'Neutral']
        for idx, label in enumerate(labels):
            initial = result['initial_depths'].get(idx, 0.0)
            final = result['final_depths'].get(idx, 0.0)
            change = result['basin_changes'][idx]
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            print(f"    {label:12s}: {initial:6.2f} → {final:6.2f} ({change:+6.2f}) {arrow}")
        print()
        print("-"*80)
        print()


def print_feedback_results(results: List[Dict]):
    """Print formatted feedback test results."""
    print("="*80)
    print("GOLDEN LABEL FEEDBACK TEST RESULTS")
    print("="*80)
    print()
    
    for i, result in enumerate(results, 1):
        print(f"Test {i}: Golden Label = {result['golden_label']}")
        print(f"  Collapsed To: {result['collapsed_to']}")
        print(f"  Expected Basin to Deepen: {result['expected_deepening']} ({['E', 'C', 'N'][result['expected_deepening']]})")
        print(f"  Actual Basin Deepened: {result['actual_deepening']} ({['E', 'C', 'N'][result['actual_deepening']] if result['actual_deepening'] >= 0 else 'N/A'})")
        print()
        print("  Basin Changes:")
        labels = ['Entailment', 'Contradiction', 'Neutral']
        for idx, label in enumerate(labels):
            change = result['basin_changes'][idx]
            is_expected = (idx == result['expected_deepening'])
            marker = "✓" if is_expected and change > 0 else "✗" if is_expected else " "
            print(f"    {marker} {label:12s}: {change:+6.2f}")
        print()
        print("-"*80)
        print()


def clean_all_caches(nli_dir: str = None):
    """
    Complete clean: Remove all caches, compiled files, and persistent state.
    """
    if nli_dir is None:
        nli_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    
    print("="*70)
    print("COMPLETE CLEAN: Removing all caches and persistent state")
    print("="*70)
    print()
    
    # 1. Clear in-memory state
    print("Clearing in-memory state...")
    GlobalLexicon().clear()
    print("  ✓ GlobalLexicon cleared")
    
    # 2. Remove __pycache__ directories
    print("Removing Python cache directories...")
    cache_count = 0
    cache_paths = []
    
    for root, dirs, files in os.walk(nli_dir):
        if '__pycache__' in dirs:
            cache_path = os.path.join(root, '__pycache__')
            cache_paths.append(cache_path)
            dirs.remove('__pycache__')
    
    for cache_path in cache_paths:
        try:
            shutil.rmtree(cache_path)
            cache_count += 1
            rel_path = os.path.relpath(cache_path, nli_dir)
            print(f"  ✓ Removed: {rel_path}")
        except Exception as e:
            print(f"  ⚠️  Failed to remove {cache_path}: {e}")
    
    if cache_count == 0:
        print("  ✓ No cache directories found")
    else:
        print(f"  ✓ Removed {cache_count} cache directory/ies")
    
    # 3. Remove .pyc, .pyo files
    print("Removing compiled Python files...")
    pyc_count = 0
    for root, dirs, files in os.walk(nli_dir):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for file in files:
            if file.endswith(('.pyc', '.pyo')):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    pyc_count += 1
                except Exception as e:
                    print(f"  ⚠️  Failed to remove {file_path}: {e}")
    
    if pyc_count == 0:
        print("  ✓ No compiled files found")
    else:
        print(f"  ✓ Removed {pyc_count} compiled file(s)")
    
    print()
    print("="*70)
    print("CLEAN COMPLETE: All caches and persistent state removed")
    print("="*70)
    print()


def main():
    """Run golden label collapse tests."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test golden label collapse behavior')
    parser.add_argument('--clean', action='store_true', help='Start with clean state (clear all caches)')
    args = parser.parse_args()
    
    # Clean if requested
    if args.clean:
        nli_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        clean_all_caches(nli_dir)
    
    print("="*80)
    print("GOLDEN LABEL COLLAPSE DIAGNOSTIC")
    print("="*80)
    print()
    print("This test verifies:")
    print("  1. Can the system collapse to all 3 classes?")
    print("  2. Does collapse respect golden labels?")
    print("  3. Do basins update correctly for each class?")
    print()
    
    # Clear lexicon for clean test
    lexicon = GlobalLexicon()
    lexicon.clear()
    print("✓ Cleared Global Lexicon (clean start)")
    print()
    
    # Initialize system
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_quantum=True,
        enable_symbolic_weight=True,
        enable_face_exposure=True,
    )
    
    encoder = NativeChainNLIEncoder(lattice_size=3, config=config)
    coupling = CrossOmcubeCoupling(initial_depth=1.0)
    
    print("✓ System initialized")
    print()
    
    # Test examples (one for each class)
    test_cases = [
        ("A dog runs", "A dog is running", "entailment"),
        ("A dog runs", "A dog does not run", "contradiction"),
        ("A dog runs", "A cat sleeps", "neutral"),
        ("The man is happy", "The man is not sad", "entailment"),
        ("The man is happy", "The man is sad", "contradiction"),
        ("The man is happy", "The sky is blue", "neutral"),
    ]
    
    print("="*80)
    print("TEST 1: COLLAPSE BEHAVIOR (No Feedback)")
    print("="*80)
    print()
    
    collapse_results = []
    for premise, hypothesis, golden_label in test_cases:
        result = test_golden_label_collapse(
            premise, hypothesis, golden_label,
            encoder, coupling
        )
        collapse_results.append(result)
    
    print_test_results(collapse_results)
    
    # Calculate success rates
    success_by_label = {'entailment': [], 'contradiction': [], 'neutral': []}
    for result in collapse_results:
        success_by_label[result['golden_label']].append(result['success'])
    
    print("="*80)
    print("COLLAPSE SUCCESS RATES BY LABEL")
    print("="*80)
    print()
    for label, successes in success_by_label.items():
        if successes:
            rate = sum(successes) / len(successes)
            print(f"  {label:12s}: {rate*100:5.1f}% ({sum(successes)}/{len(successes)})")
    print()
    
    # Reset coupling for feedback test
    coupling = CrossOmcubeCoupling(initial_depth=1.0)
    
    print("="*80)
    print("TEST 2: BASIN FEEDBACK BEHAVIOR (With Learning)")
    print("="*80)
    print()
    
    feedback_results = []
    for premise, hypothesis, golden_label in test_cases:
        result = apply_golden_feedback(
            premise, hypothesis, golden_label,
            encoder, coupling
        )
        feedback_results.append(result)
    
    print_feedback_results(feedback_results)
    
    # Summary
    print("="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print()
    
    # Check if all 3 classes can collapse
    collapsed_labels = set(r['collapsed_label'] for r in collapse_results)
    print(f"Classes that collapsed: {sorted(collapsed_labels)}")
    if len(collapsed_labels) == 3:
        print("✓ All 3 classes can collapse")
    else:
        print(f"✗ Only {len(collapsed_labels)} classes collapsed")
        missing = {'entailment', 'contradiction', 'neutral'} - collapsed_labels
        print(f"  Missing: {missing}")
    print()
    
    # Check if feedback works
    correct_feedback = sum(1 for r in feedback_results 
                          if r['expected_deepening'] == r['actual_deepening'])
    print(f"Feedback correctness: {correct_feedback}/{len(feedback_results)}")
    if correct_feedback == len(feedback_results):
        print("✓ All basins update correctly")
    else:
        print("✗ Some basins not updating correctly")
    print()
    
    # Final verdict
    print("="*80)
    print("VERDICT")
    print("="*80)
    print()
    
    if len(collapsed_labels) == 3 and correct_feedback == len(feedback_results):
        print("✅ COLLAPSE ENGINE IS WORKING CORRECTLY")
        print("   The issue is in the detector/energy model, not the collapse mechanism.")
        print("   The system can collapse to all 3 classes when given proper signals.")
    elif len(collapsed_labels) < 3:
        print("❌ COLLAPSE ENGINE HAS ISSUES")
        print("   The system cannot collapse to all 3 classes.")
        print("   Check: amplitude initialization, probability calculation, softmax clamping.")
    else:
        print("⚠️  COLLAPSE WORKS BUT FEEDBACK IS BROKEN")
        print("   The system can collapse but basins don't update correctly.")
        print("   Check: reinforce_geometry, basin update logic.")
    print()


if __name__ == "__main__":
    main()


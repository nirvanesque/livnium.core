"""
SNLI Test Script

Test trained model on SNLI data.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add nova_v3 and repo root (/nova) to path so quantum_embed is discoverable
nova_v3_root = Path(__file__).parent.parent
repo_root = nova_v3_root.parent
sys.path.insert(0, str(nova_v3_root))
sys.path.insert(0, str(repo_root))

from core import VectorCollapseEngine, BasinField
from tasks.snli import QuantumSNLIEncoder, SNLIHead
from quantum_embed.text_encoder_quantum import QuantumTextEncoder
from training.train_snli_vector import SNLIDataset, load_snli_data


def main():
    parser = argparse.ArgumentParser(description='Test SNLI model')
    parser.add_argument('--model-dir', type=str, required=True,
                          help='Path to model directory')
    parser.add_argument('--snli-test', type=str, required=True,
                       help='Path to SNLI test JSONL file')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of test samples')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--max-len', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--errors-file', type=str, default=None,
                       help='If set, write misclassified samples to this JSONL file')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_dir = Path(args.model_dir)
    # weights_only defaults to True on newer PyTorch; disable so we can load Vocabulary object from our own codebase
    checkpoint = torch.load(
        model_dir / 'best_model.pt',
        map_location=device,
        weights_only=False
    )
    
    model_args = checkpoint['args']
    use_dynamic_basins = checkpoint.get("use_dynamic_basins", False)
    basin_state = checkpoint.get("basin_field", None)
    basin_field = None
    if use_dynamic_basins and basin_state is not None:
        basin_field = BasinField(max_basins_per_label=basin_state.get("max_basins_per_label", 64))
        basin_field.load_state_dict(basin_state)
        basin_field.to(device)
    else:
        use_dynamic_basins = False

    quantum_ckpt = getattr(model_args, "quantum_ckpt", None)
    if not quantum_ckpt:
        raise ValueError("Checkpoint missing quantum_ckpt for quantum encoder")

    quantum_tokenizer = QuantumTextEncoder(quantum_ckpt)

    def quantum_encode(text: str, max_len: int = args.max_len):
        tokens = quantum_tokenizer.tokenize(text)
        ids = [quantum_tokenizer.word2idx.get(t, quantum_tokenizer.unk_idx) for t in tokens]
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids.extend([quantum_tokenizer.pad_idx] * (max_len - len(ids)))
        return ids

    # Create models
    collapse_engine = VectorCollapseEngine(
        dim=model_args.dim, 
        num_layers=model_args.num_layers,
        strength_entail=getattr(model_args, "strength_entail", 0.1),
        strength_contra=getattr(model_args, "strength_contra", 0.1),
        strength_neutral=getattr(model_args, "strength_neutral", 0.05),
        basin_tension_threshold=getattr(model_args, "basin_tension_threshold", 0.15),
        basin_align_threshold=getattr(model_args, "basin_align_threshold", 0.6),
        basin_anchor_lr=getattr(model_args, "basin_anchor_lr", 0.05),
        basin_prune_min_count=getattr(model_args, "basin_prune_min_count", 10),
        basin_prune_merge_cos=getattr(model_args, "basin_merge_cos_threshold", 0.97),
    ).to(device)

    encoder = QuantumSNLIEncoder(
        ckpt_path=quantum_ckpt,
    ).to(device)
    head = SNLIHead(dim=model_args.dim).to(device)
    
    # Load weights
    collapse_engine.load_state_dict(checkpoint['collapse_engine'])
    encoder.load_state_dict(checkpoint['encoder'])
    head.load_state_dict(checkpoint['head'])
    
    collapse_engine.eval()
    encoder.eval()
    head.eval()
    
    # Load test data
    print("Loading test data...")
    test_samples = load_snli_data(Path(args.snli_test), max_samples=args.max_samples)
    print(f"Loaded {len(test_samples)} test samples")
    
    # Create dataset
    test_dataset = SNLIDataset(test_samples, encode_fn=quantum_encode, max_len=args.max_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate
    print("\nEvaluating...")
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_gold_labels = []
    
    errors_fh = open(args.errors_file, "w", encoding="utf-8") if args.errors_file else None
    
    label_names = ['entailment', 'contradiction', 'neutral']
    
    with torch.no_grad():
        for batch in test_loader:
            prem_ids = batch['prem_ids'].to(device)
            hyp_ids = batch['hyp_ids'].to(device)
            labels = batch['label'].to(device)
            gold_labels = batch['gold_label']
            
            h0, v_p, v_h = encoder.build_initial_state(prem_ids, hyp_ids)
            
            if use_dynamic_basins and basin_field is not None:
                h_final, trace = collapse_engine.collapse_dynamic(
                    h0,
                    labels,
                    basin_field,
                    global_step=0,
                    spawn_new=False,
                    prune_every=0,
                    update_anchors=False,
                )
            else:
                h_final, trace = collapse_engine.collapse(h0)
            
            # Classify with directional signals
            logits = head(h_final, v_p, v_h)
            pred = logits.argmax(dim=-1)
            
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_gold_labels.extend(gold_labels)
            
            # Write misclassifications to JSONL if requested
            if errors_fh is not None:
                probs = torch.softmax(logits, dim=-1)
                for i in range(pred.size(0)):
                    if pred[i].item() != labels[i].item():
                        record = {
                            "premise": batch['premise'][i],
                            "hypothesis": batch['hypothesis'][i],
                            "gold_label": gold_labels[i],
                            "gold_index": int(labels[i].item()),
                            "pred_label": label_names[pred[i].item()],
                            "pred_index": int(pred[i].item()),
                            "probabilities": probs[i].cpu().tolist()
                        }
                        errors_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    if errors_fh is not None:
        errors_fh.close()
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Confusion matrix
    confusion = np.zeros((3, 3), dtype=int)
    for p, l in zip(all_predictions, all_labels):
        confusion[l, p] += 1
    
    print("\n" + "=" * 70)
    print("Test Results")
    print("=" * 70)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Correct: {correct}/{total}")
    print("\nConfusion Matrix:")
    print("      E    N    C")
    for i, label in enumerate(['E', 'N', 'C']):
        print(f"{label}  {confusion[i]}")
    print("=" * 70)
    
    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i, label in enumerate(label_names):
        class_total = confusion[i].sum()
        class_correct = confusion[i, i]
        class_acc = class_correct / class_total if class_total > 0 else 0.0
        print(f"  {label}: {class_acc:.4f} ({class_correct}/{class_total})")


if __name__ == '__main__':
    main()

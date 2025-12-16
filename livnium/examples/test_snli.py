"""
Test SNLI Model: Evaluate trained model on test set

Loads a trained model and evaluates on the SNLI test set.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from livnium.domains.snli.encoder import SNLIEncoder
from livnium.domains.snli.head import SNLIHead
from livnium.engine.collapse.engine import CollapseEngine
from livnium.datasets.snli import SNLIDataset
from livnium.training.evaluator import Evaluator
from livnium.training.losses import LivniumLoss
from livnium.instrumentation import LivniumLogger
from livnium.engine.config import defaults

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


class SNLIModel(nn.Module):
    """Complete model for SNLI domain."""
    
    def __init__(self, dim: int = 256, num_layers: int = 5, vocab_size: int = 2000):
        super().__init__()
        self.encoder = SNLIEncoder(dim=dim, vocab_size=vocab_size)
        self.collapse_engine = CollapseEngine(dim=dim, num_layers=num_layers)
        self.head = SNLIHead(dim=dim)
    
    def forward(self, batch):
        """Forward pass."""
        prem_ids = batch["premise_ids"]
        hyp_ids = batch["hypothesis_ids"]
        
        # Encode
        h0, v_p, v_h = self.encoder.build_initial_state(prem_ids, hyp_ids)
        
        # Collapse
        h_final, trace = self.collapse_engine.collapse(h0)
        
        # Head
        logits = self.head(h_final, v_p, v_h)
        
        return logits


def parse_args():
    parser = argparse.ArgumentParser(description="Test SNLI model on test set")
    
    # Model arguments
    parser.add_argument("--test", type=str, required=True, help="Path to test JSONL")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (optional)")
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--layers", type=int, default=5, help="Collapse engine layers")
    parser.add_argument("--vocab-size", type=int, default=2000, help="Vocabulary size")
    
    # Evaluation arguments
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda, auto if None)")
    
    return parser.parse_args()


def main():
    """Main test function."""
    args = parse_args()
    
    # Setup
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger = LivniumLogger()
    
    logger.info(f"Testing SNLI model on {device}")
    
    # Create test dataset (needs vocab from training, but for simplicity we'll rebuild)
    # In production, you'd save/load the vocab
    test_dataset = SNLIDataset(
        data_path=args.test,
        vocab_size=args.vocab_size,
        max_length=50,
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"Test set size: {len(test_dataset)} samples")
    
    # Create model
    model = SNLIModel(dim=args.dim, num_layers=args.layers, vocab_size=args.vocab_size).to(device)
    
    # Load checkpoint if provided
    if args.checkpoint and Path(args.checkpoint).exists():
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Checkpoint loaded successfully")
    else:
        logger.warning("No checkpoint provided - using randomly initialized model!")
        logger.warning("This will give random performance. Provide --checkpoint to test trained model.")
    
    # Loss function
    loss_fn = LivniumLoss(
        d_margin=defaults.D_MARGIN,
        neg_weight=defaults.NEG_WEIGHT,
        norm_reg_weight=defaults.NORM_REG_WEIGHT
    )
    
    # Evaluator
    evaluator = Evaluator(
        num_classes=3,
        class_names=["Entailment", "Neutral", "Contradiction"]
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    model.eval()
    test_metrics_list = []
    
    # Progress bar
    if HAS_TQDM:
        test_iterator = tqdm(test_loader, desc="Testing", unit="batch")
    else:
        test_iterator = test_loader
    
    with torch.no_grad():
        for batch in test_iterator:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            logits = model(batch)
            labels = batch.get("label")
            batch_metrics = evaluator.evaluate(logits, labels, loss_fn=loss_fn)
            test_metrics_list.append(batch_metrics)
    
    test_metrics = evaluator.aggregate_metrics(test_metrics_list)
    
    # Print results
    logger.info("=" * 60)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {test_metrics.get('accuracy', 0.0):.4f}")
    logger.info(f"Loss: {test_metrics.get('loss', 0.0):.4f}")
    logger.info(f"Total samples: {test_metrics.get('total', 0)}")
    logger.info(f"Correct predictions: {test_metrics.get('correct', 0)}")
    
    # Per-class accuracy
    logger.info("\nPer-class accuracy:")
    for i, class_name in enumerate(evaluator.class_names):
        acc = test_metrics.get(f"class_{i}_accuracy", 0.0)
        logger.info(f"  {class_name}: {acc:.4f}")
    
    # Confusion matrix
    if "confusion_matrix" in test_metrics:
        evaluator.print_confusion_matrix(
            test_metrics["confusion_matrix"], 
            title="Test Set Confusion Matrix"
        )
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()


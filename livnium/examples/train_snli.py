"""
Example: Training on SNLI Domain

Example showing how to train a model on the SNLI domain.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from livnium.domains.snli.encoder import SNLIEncoder
from livnium.domains.snli.head import SNLIHead
from livnium.engine.collapse.engine import CollapseEngine
from livnium.datasets.snli import SNLIDataset
from livnium.training.trainer import Trainer
from livnium.training.losses import LivniumLoss
from livnium.training.evaluator import Evaluator
from livnium.instrumentation import LivniumLogger, MetricsTracker


class SNLIModel(nn.Module):
    """Complete model for SNLI domain."""
    
    def __init__(self, dim: int = 256, num_layers: int = 5, vocab_size: int = 2000, 
                 enable_basins: bool = False, basin_threshold: str = "v4"):
        super().__init__()
        self.encoder = SNLIEncoder(dim=dim, vocab_size=vocab_size)
        self.collapse_engine = CollapseEngine(
            dim=dim, 
            num_layers=num_layers,
            enable_basins=enable_basins,
            basin_threshold=basin_threshold
        )
        self.head = SNLIHead(dim=dim)
    
    def forward(self, batch):
        """Forward pass."""
        prem_ids = batch["premise_ids"]
        hyp_ids = batch["hypothesis_ids"]
        labels = batch.get("label")  # For basin updates only (not routing - prevents leakage)
        
        # Encode
        h0, v_p, v_h = self.encoder.build_initial_state(prem_ids, hyp_ids)
        
        # Collapse (labels only used for basin maintenance, not routing)
        # Routing is physics-based to prevent label leakage
        h_final, trace = self.collapse_engine.collapse(h0, labels=labels)
        
        # Head
        logits = self.head(h_final, v_p, v_h)
        
        return logits


import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Train SNLI on Livnium")
    
    # Data arguments
    parser.add_argument("--train", type=str, required=True, help="Path to training JSONL")
    parser.add_argument("--dev", type=str, required=True, help="Path to dev JSONL")
    parser.add_argument("--test", type=str, default=None, help="Path to test JSONL (optional)")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit training samples (dry run)")
    parser.add_argument("--max-dev-samples", type=int, default=None, help="Limit dev samples (dry run)")
    
    # Model arguments
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--layers", type=int, default=5, help="Collapse engine layers")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--log-every", type=int, default=100, help="Log interval")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--save-checkpoint", action="store_true", help="Save model checkpoint after training")
    
    # Basins
    parser.add_argument("--enable-basins", action="store_true", help="Enable dynamic basins")
    parser.add_argument("--basin-threshold", type=str, default="v3", choices=["v3", "v4"], help="Basin threshold version")
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    # Setup
    # Initialize logger first
    logger = LivniumLogger(log_dir=args.log_dir)
    metrics_tracker = MetricsTracker()
    
    # Prefer MPS (Metal) on macOS, then CUDA, then CPU
    # MPS needs larger batches to be efficient (overhead for small batches)
    if torch.backends.mps.is_available():
        # MPS is slow with small batches, use CPU for batch_size < 64
        if args.batch_size < 64:
            device = "cpu"
            device_name = "CPU (MPS available but batch_size too small for efficiency)"
            logger.warning(f"MPS available but batch_size={args.batch_size} is too small for MPS efficiency.")
            logger.warning("MPS needs batch_size >= 64. Using CPU instead. Increase --batch-size for MPS acceleration.")
        else:
            device = "mps"
            device_name = "MPS (Metal Performance Shaders)"
    elif torch.cuda.is_available():
        device = "cuda"
        device_name = "CUDA"
    else:
        device = "cpu"
        device_name = "CPU"
    
    logger.info(f"Using {device_name} for GPU acceleration")
    logger.info(f"Starting SNLI domain training on {device}")
    
    # Create datasets
    vocab_size = 2000
    train_dataset = SNLIDataset(
        data_path=args.train,
        vocab_size=vocab_size,
        max_length=50,
    )
    if args.max_train_samples:
        train_dataset.samples = train_dataset.samples[:args.max_train_samples]
        logger.info(f"Limited training data to {len(train_dataset)} samples")
        
    dev_dataset = SNLIDataset(
        data_path=args.dev,
        vocab_size=vocab_size, 
        max_length=50,
        tokenizer=train_dataset._tokenize # Share vocabulary logic usually, but here simplicity checks
    )
    # Important: Share vocab if built dynamically. 
    # Current SNLIDataset rebuilds vocab in __init__ if no tokenizer passed.
    # To strictly share vocab, we should pass the tokenizer method or vocab dict.
    # For this simple script, we'll let dev build its own vocab (suboptimal) or 
    # better, update SNLIDataset to accept vocab. 
    # For now, let's just limit dev data.
    
    if args.max_dev_samples:
        dev_dataset.samples = dev_dataset.samples[:args.max_dev_samples]
        logger.info(f"Limited dev data to {len(dev_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model (encoder needs vocab_size to match dataset)
    model = SNLIModel(
        dim=args.dim, 
        num_layers=args.layers, 
        vocab_size=vocab_size,
        enable_basins=args.enable_basins,
        basin_threshold=args.basin_threshold
    ).to(device)
    
    # Move basin field to device if enabled
    if args.enable_basins and model.collapse_engine.basin_field is not None:
        model.collapse_engine.basin_field.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Configure Basins
    if args.enable_basins:
        # engine/config/defaults.py configuration injection would happen here
        # or we just rely on the default engine behavior if parameters passed
        logger.info(f"Basins ENABLED (threshold {args.basin_threshold})")
        # In a real setup we might inject this config into the engine
    else:
        logger.info("Basins DISABLED (Static Collapse)")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        max_grad_norm=1.0,
    )
    
    # Loss function (Energy + Tension minimization)
    from livnium.engine.config import defaults
    loss_fn = LivniumLoss(
        d_margin=defaults.D_MARGIN,
        neg_weight=defaults.NEG_WEIGHT,
        norm_reg_weight=defaults.NORM_REG_WEIGHT
    )
    
    # Evaluator for dev set (with SNLI class names)
    evaluator = Evaluator(
        num_classes=3,
        class_names=["Entailment", "Neutral", "Contradiction"]
    )
    
    # Train
    logger.info("Starting training loop...")
    train_metrics = trainer.fit(
        dataloader=train_loader,
        num_epochs=args.epochs,
        loss_fn=loss_fn,
        verbose=True,
    )
    
    # Evaluate on dev set
    logger.info("Evaluating on dev set...")
    model.eval()
    dev_metrics_list = []
    
    # Progress bar for evaluation
    try:
        from tqdm import tqdm
        eval_iterator = tqdm(dev_loader, desc="Evaluating", unit="batch")
    except ImportError:
        eval_iterator = dev_loader
    
    with torch.no_grad():
        for batch in eval_iterator:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            logits = model(batch)
            labels = batch.get("label")
            batch_metrics = evaluator.evaluate(logits, labels, loss_fn=loss_fn)
            dev_metrics_list.append(batch_metrics)
    
    dev_metrics = evaluator.aggregate_metrics(dev_metrics_list)
    
    # Log final metrics
    logger.info("Training complete!")
    logger.info(f"Train - Loss: {train_metrics['loss'][-1]:.4f}, Accuracy: {train_metrics['accuracy'][-1]:.4f}")
    logger.info(f"Dev - Loss: {dev_metrics.get('loss', 0.0):.4f}, Accuracy: {dev_metrics.get('accuracy', 0.0):.4f}")
    
    # Print confusion matrix
    if "confusion_matrix" in dev_metrics:
        evaluator.print_confusion_matrix(dev_metrics["confusion_matrix"], title="Dev Set Confusion Matrix")
    
    # Evaluate on test set if provided
    test_metrics = None
    if args.test:
        logger.info("Evaluating on test set...")
        test_dataset = SNLIDataset(
            data_path=args.test,
            vocab_size=vocab_size,
            max_length=50,
            tokenizer=train_dataset._tokenize  # Share vocab
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        model.eval()
        test_metrics_list = []
        
        try:
            from tqdm import tqdm
            test_iterator = tqdm(test_loader, desc="Testing", unit="batch")
        except ImportError:
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
        logger.info(f"Test - Loss: {test_metrics.get('loss', 0.0):.4f}, Accuracy: {test_metrics.get('accuracy', 0.0):.4f}")
        
        if "confusion_matrix" in test_metrics:
            evaluator.print_confusion_matrix(test_metrics["confusion_matrix"], title="Test Set Confusion Matrix")
    
    # Save metrics (convert confusion matrix to list for JSON)
    all_metrics = {
        "train": train_metrics,
        "dev": dev_metrics,
    }
    
    if test_metrics:
        all_metrics["test"] = test_metrics
    
    # Convert confusion matrices to lists if present
    if "confusion_matrix" in dev_metrics:
        all_metrics["dev"]["confusion_matrix"] = dev_metrics["confusion_matrix"].tolist()
    if test_metrics and "confusion_matrix" in test_metrics:
        all_metrics["test"]["confusion_matrix"] = test_metrics["confusion_matrix"].tolist()
    
    metrics_tracker.metrics = all_metrics
    metrics_tracker.save(f"{args.log_dir}/snli_metrics.json")
    
    # Save checkpoint if requested
    if args.save_checkpoint:
        checkpoint_path = f"{args.log_dir}/snli_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dim': args.dim,
            'layers': args.layers,
            'vocab_size': vocab_size,
            'train_accuracy': train_metrics['accuracy'][-1],
            'dev_accuracy': dev_metrics.get('accuracy', 0.0),
            'test_accuracy': test_metrics.get('accuracy', 0.0) if test_metrics else None,
        }, checkpoint_path)
        logger.info(f"Model checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()


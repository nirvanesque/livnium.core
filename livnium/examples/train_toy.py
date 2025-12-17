"""
Example: Training on Toy Domain

Simple example showing how to train a model on the toy domain.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from livnium.domains.toy.encoder import ToyEncoder
from livnium.domains.toy.head import ToyHead
from livnium.engine.collapse.engine import CollapseEngine
from livnium.datasets.toy import ToyDataset
from livnium.training.trainer import Trainer
from livnium.training.losses import LivniumLoss
from livnium.instrumentation import LivniumLogger, MetricsTracker


class ToyModel(nn.Module):
    """Complete model for toy domain."""
    
    def __init__(self, dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.encoder = ToyEncoder(dim=dim)
        self.collapse_engine = CollapseEngine(dim=dim, num_layers=3)
        self.head = ToyHead(dim=dim, num_classes=num_classes)
    
    def forward(self, batch):
        """Forward pass."""
        input_a = batch["input_a"]
        input_b = batch["input_b"]
        
        # Encode
        h0, v_a, v_b = self.encoder.build_initial_state(input_a, input_b)
        
        # Collapse
        h_final, trace = self.collapse_engine.collapse(h0)
        
        # Head
        logits = self.head(h_final, v_a, v_b)
        
        return logits


def main():
    """Main training function."""
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = LivniumLogger(log_dir="logs")
    metrics_tracker = MetricsTracker()
    
    logger.info("Starting toy domain training")
    
    # Create dataset
    train_dataset = ToyDataset(size=1000, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = ToyModel(dim=64, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        max_grad_norm=1.0,
    )
    
    # Loss function
    # Loss function
    from livnium.engine.config import defaults
    loss_fn = LivniumLoss(
        lambda_energy=defaults.LAMBDA_ENERGY,
        lambda_tension=defaults.LAMBDA_TENSION
    )
    
    # Train
    logger.info("Starting training...")
    metrics = trainer.fit(
        dataloader=train_loader,
        num_epochs=10,
        loss_fn=loss_fn,
        verbose=True,
    )
    
    # Log final metrics
    logger.info("Training complete!")
    logger.log_metrics(metrics, step=len(metrics["loss"]), prefix="Final")
    
    # Save metrics
    metrics_tracker.metrics = metrics
    metrics_tracker.save("logs/toy_training_metrics.json")


if __name__ == "__main__":
    main()


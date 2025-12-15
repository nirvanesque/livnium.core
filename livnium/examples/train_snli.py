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
from livnium.instrumentation import LivniumLogger, MetricsTracker


class SNLIModel(nn.Module):
    """Complete model for SNLI domain."""
    
    def __init__(self, dim: int = 256):
        super().__init__()
        self.encoder = SNLIEncoder(dim=dim)
        self.collapse_engine = CollapseEngine(dim=dim, num_layers=5)
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


def main():
    """Main training function."""
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = LivniumLogger(log_dir="logs")
    metrics_tracker = MetricsTracker()
    
    logger.info("Starting SNLI domain training")
    
    # Create dataset (using dummy data if file doesn't exist)
    train_dataset = SNLIDataset(
        data_path="data/snli/snli_1.0_train.jsonl",
        vocab_size=1000,
        max_length=50,
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Create model
    model = SNLIModel(dim=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        max_grad_norm=1.0,
    )
    
    # Loss function
    loss_fn = LivniumLoss(lambda_energy=0.1, lambda_tension=0.1)
    
    # Train
    logger.info("Starting training...")
    metrics = trainer.fit(
        dataloader=train_loader,
        num_epochs=5,
        loss_fn=loss_fn,
        verbose=True,
    )
    
    # Log final metrics
    logger.info("Training complete!")
    logger.log_metrics(metrics, step=len(metrics["loss"]), prefix="Final")
    
    # Save metrics
    metrics_tracker.metrics = metrics
    metrics_tracker.save("logs/snli_training_metrics.json")


if __name__ == "__main__":
    main()


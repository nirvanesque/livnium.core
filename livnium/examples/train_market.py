"""
Example: Training on Market Domain

Example showing how to train a model on the market domain.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from livnium.domains.market.encoder import MarketEncoder
from livnium.domains.market.head import MarketHead
from livnium.engine.collapse.engine import CollapseEngine
from livnium.datasets.market import MarketDataset
from livnium.training.trainer import Trainer
from livnium.training.losses import LivniumLoss
from livnium.instrumentation import LivniumLogger, MetricsTracker


class MarketModel(nn.Module):
    """Complete model for market domain."""
    
    def __init__(self, dim: int = 256):
        super().__init__()
        self.encoder = MarketEncoder(dim=dim, window=14)
        self.collapse_engine = CollapseEngine(dim=dim, num_layers=5)
        self.head = MarketHead(dim=dim)
    
    def forward(self, batch):
        """Forward pass."""
        close = batch["close"]
        high = batch["high"]
        low = batch["low"]
        open_price = batch["open"]
        volume = batch["volume"]
        
        # Encode
        h0, v_current, v_basin = self.encoder.build_initial_state(
            close, high, low, open_price, volume
        )
        
        # Collapse
        h_final, trace = self.collapse_engine.collapse(h0)
        
        # Head
        logits = self.head(h_final, v_current, v_basin)
        
        return logits


def main():
    """Main training function."""
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = LivniumLogger(log_dir="logs")
    metrics_tracker = MetricsTracker()
    
    logger.info("Starting market domain training")
    
    # Create dataset (using dummy data if file doesn't exist)
    train_dataset = MarketDataset(
        data_path="data/market/ohlcv.csv",
        window_size=20,
        create_dummy=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Create model
    model = MarketModel(dim=256)
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
        num_epochs=10,
        loss_fn=loss_fn,
        verbose=True,
    )
    
    # Log final metrics
    logger.info("Training complete!")
    logger.log_metrics(metrics, step=len(metrics["loss"]), prefix="Final")
    
    # Save metrics
    metrics_tracker.metrics = metrics
    metrics_tracker.save("logs/market_training_metrics.json")


if __name__ == "__main__":
    main()


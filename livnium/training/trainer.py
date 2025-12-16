"""
Base Trainer: Training Loop Infrastructure

Trainer.fit() API that works with any domain, engine, and model.
Loss/reward live here, not in kernel.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback: create a dummy tqdm that just returns the iterator
    def tqdm(iterable, **kwargs):
        return iterable


class Trainer:
    """
    Base trainer for LIVNIUM.
    
    Works with any domain, engine, and model.
    All loss/reward calculations happen here, not in kernel.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        max_grad_norm: Optional[float] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train (includes encoder, collapse engine, head)
            optimizer: Optimizer
            device: Device for training
            max_grad_norm: Optional gradient clipping
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.max_grad_norm = max_grad_norm
        
        self.model.to(device)
    
    def fit(
        self,
        dataloader: DataLoader,
        num_epochs: int = 1,
        loss_fn: Optional[callable] = None,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Train model.
        
        Args:
            dataloader: DataLoader for training data
            num_epochs: Number of training epochs
            loss_fn: Loss function (defaults to cross-entropy)
            verbose: Whether to print progress
            
        Returns:
            Dictionary of training metrics
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        metrics = {
            "loss": [],
            "accuracy": [],
        }
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            # Progress bar
            if verbose and HAS_TQDM:
                pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
            else:
                pbar = dataloader
            
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Domain-specific forward (encoder -> collapse -> head)
                # This is a placeholder - actual implementation depends on domain
                logits = self.model(batch)
                labels = batch.get("labels", batch.get("label"))
                
                # Loss calculation (happens here, not in kernel)
                loss = loss_fn(logits, labels)
                
                # Backward
                loss.backward()
                
                # Gradient clipping
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                
                self.optimizer.step()
                
                # Metrics
                epoch_loss += loss.item()
                preds = logits.argmax(dim=-1)
                epoch_correct += (preds == labels).sum().item()
                epoch_total += labels.size(0)
                
                # Update progress bar
                if verbose and HAS_TQDM:
                    current_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
                    avg_loss_so_far = epoch_loss / (batch_idx + 1)
                    # Use shorter format to ensure all fields fit
                    pbar.set_postfix_str(
                        f'L:{loss.item():.3f} avgL:{avg_loss_so_far:.3f} acc:{current_acc:.3f}',
                        refresh=False
                    )
                elif verbose and batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, "
                          f"Loss: {loss.item():.4f}")
            
            # Epoch metrics
            avg_loss = epoch_loss / len(dataloader)
            accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
            
            metrics["loss"].append(avg_loss)
            metrics["accuracy"].append(accuracy)
            
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} complete: "
                      f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return metrics


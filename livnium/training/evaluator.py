"""
Training Evaluator: Evaluation Metrics

Evaluation metrics and validation logic.
These observe model outputs but do not modify kernel or engine.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn


class Evaluator:
    """
    Evaluator for model performance metrics.
    
    Computes accuracy, loss, and other metrics.
    """
    
    def __init__(self, num_classes: int = 3):
        """
        Initialize evaluator.
        
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
    
    def evaluate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model predictions.
        
        Args:
            logits: Model logits [B, num_classes]
            labels: Ground truth labels [B]
            loss_fn: Optional loss function
            
        Returns:
            Dictionary of metrics
        """
        # Accuracy
        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total if total > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        # Loss
        if loss_fn is not None:
            loss = loss_fn(logits, labels)
            metrics["loss"] = loss.item()
        
        # Per-class accuracy
        per_class_correct = torch.zeros(self.num_classes)
        per_class_total = torch.zeros(self.num_classes)
        
        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                per_class_correct[c] = (preds[mask] == c).sum().item()
                per_class_total[c] = mask.sum().item()
        
        per_class_acc = per_class_correct / (per_class_total + 1e-8)
        for c in range(self.num_classes):
            metrics[f"class_{c}_accuracy"] = per_class_acc[c].item()
        
        return metrics
    
    def evaluate_batch(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """
        Evaluate single batch (alias for evaluate).
        
        Args:
            logits: Model logits [B, num_classes]
            labels: Ground truth labels [B]
            loss_fn: Optional loss function
            
        Returns:
            Dictionary of metrics
        """
        return self.evaluate(logits, labels, loss_fn)
    
    def aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate metrics across multiple batches.
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Aggregated metrics
        """
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        # Sum metrics
        sum_keys = ["correct", "total"]
        for key in sum_keys:
            aggregated[key] = sum(m.get(key, 0) for m in metrics_list)
        
        # Average metrics
        avg_keys = ["accuracy", "loss"] + [f"class_{c}_accuracy" for c in range(self.num_classes)]
        for key in avg_keys:
            values = [m.get(key, 0.0) for m in metrics_list if key in m]
            if values:
                aggregated[key] = sum(values) / len(values)
        
        # Recompute accuracy from aggregated correct/total
        if "correct" in aggregated and "total" in aggregated:
            aggregated["accuracy"] = aggregated["correct"] / aggregated["total"] if aggregated["total"] > 0 else 0.0
        
        return aggregated


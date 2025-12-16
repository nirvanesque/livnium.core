"""
Training Evaluator: Evaluation Metrics

Evaluation metrics and validation logic.
These observe model outputs but do not modify kernel or engine.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import numpy as np


class Evaluator:
    """
    Evaluator for model performance metrics.
    
    Computes accuracy, loss, and other metrics.
    """
    
    def __init__(self, num_classes: int = 3, class_names: Optional[List[str]] = None):
        """
        Initialize evaluator.
        
        Args:
            num_classes: Number of classes
            class_names: Optional list of class names for confusion matrix
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
    
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
        
        # Confusion matrix (per batch)
        confusion = self._compute_confusion_matrix(preds, labels)
        metrics["confusion_matrix"] = confusion
        
        return metrics
    
    def _compute_confusion_matrix(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor
    ) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            preds: Predicted labels [B]
            labels: Ground truth labels [B]
            
        Returns:
            Confusion matrix [num_classes, num_classes]
        """
        confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for true_label, pred_label in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            confusion[int(true_label), int(pred_label)] += 1
        return confusion
    
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
            Aggregated metrics (includes aggregated confusion matrix)
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
        
        # Aggregate confusion matrices
        confusion_matrices = [m.get("confusion_matrix") for m in metrics_list if "confusion_matrix" in m]
        if confusion_matrices:
            aggregated["confusion_matrix"] = np.sum(confusion_matrices, axis=0)
        
        return aggregated
    
    def print_confusion_matrix(self, confusion_matrix: np.ndarray, title: str = "Confusion Matrix"):
        """
        Print confusion matrix in a readable format.
        
        Args:
            confusion_matrix: Confusion matrix [num_classes, num_classes]
            title: Title for the confusion matrix
        """
        print(f"\n{title}:")
        print("=" * (len(title) + 2))
        
        # Header
        header = "True \\ Pred"
        for name in self.class_names:
            header += f"  {name[:8]:>8}"
        print(header)
        print("-" * len(header))
        
        # Rows
        for i, true_name in enumerate(self.class_names):
            row = f"{true_name[:10]:>10}"
            for j in range(self.num_classes):
                row += f"  {confusion_matrix[i, j]:>8}"
            print(row)
        
        # Summary stats
        total = confusion_matrix.sum()
        correct = np.trace(confusion_matrix)
        accuracy = correct / total if total > 0 else 0.0
        print(f"\nTotal: {total}, Correct: {correct}, Accuracy: {accuracy:.4f}")
        
        # Per-class precision and recall
        print("\nPer-class metrics:")
        for i, class_name in enumerate(self.class_names):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            print(f"  {class_name}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        print()


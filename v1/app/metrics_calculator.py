import torch
import numpy as np
from typing import Tuple, Dict


# ==================== Metrics Calculator ====================

class MetricsCalculator:
    """Calculate various metrics for model evaluation"""

    @staticmethod
    def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate accuracy"""
        correct = (predictions == targets).sum().item()
        total = targets.size(0)
        return 100.0 * correct / total

    @staticmethod
    def confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int = 2) -> np.ndarray:
        """Calculate confusion matrix"""
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(targets, predictions):
            cm[t.item(), p.item()] += 1
        return cm

    @staticmethod
    def precision_recall_f1(cm: np.ndarray) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score from confusion matrix"""
        # For binary classification
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tn = cm[0, 0]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    @staticmethod
    def compute_all_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute all metrics"""
        acc = MetricsCalculator.accuracy(predictions, targets)
        cm = MetricsCalculator.confusion_matrix(predictions, targets)
        precision, recall, f1 = MetricsCalculator.precision_recall_f1(cm)

        return {
            'accuracy': acc,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }

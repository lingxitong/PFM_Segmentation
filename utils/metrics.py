"""
Evaluation Metrics for Semantic Segmentation

This module contains comprehensive evaluation metrics including
IoU, Pixel Accuracy, Dice Score, and class-wise statistics.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F


class SegmentationMetrics:
    """
    Comprehensive metrics for semantic segmentation evaluation.
    
    Computes:
    - Mean IoU (mIoU)
    - Pixel Accuracy
    - Mean Accuracy
    - Frequency Weighted IoU
    - Per-class IoU and Accuracy
    - Dice Score
    - Precision and Recall
    
    Args:
        num_classes (int): Number of classes
        ignore_index (int): Index to ignore in calculations
        device (str): Device for computations
    """
    
    def __init__(self, num_classes: int, ignore_index: int = 255, device: str = 'cpu'):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device
        
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), 
            dtype=torch.int64, 
            device=self.device
        )
        self.total_samples = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new predictions and targets.
        
        Args:
            predictions (torch.Tensor): Model predictions of shape (B, C, H, W)
            targets (torch.Tensor): Ground truth labels of shape (B, H, W)
        """
        # Convert predictions to class indices
        if predictions.dim() == 4:  # (B, C, H, W)
            predictions = torch.argmax(predictions, dim=1)
        
        # Flatten tensors
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Create mask for valid pixels
        mask = (targets != self.ignore_index)
        predictions = predictions[mask]
        targets = targets[mask]
        
        # Update confusion matrix
        indices = self.num_classes * targets + predictions
        cm_update = torch.bincount(indices, minlength=self.num_classes**2)
        cm_update = cm_update.reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += cm_update.to(self.device)
        self.total_samples += mask.sum().item()
    
    def compute_iou(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute IoU metrics.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (per_class_iou, mean_iou)
        """
        # IoU = TP / (TP + FP + FN)
        # TP: diagonal elements
        # FP: column sum - diagonal
        # FN: row sum - diagonal
        
        tp = torch.diag(self.confusion_matrix).float()
        fp = self.confusion_matrix.sum(dim=0) - tp
        fn = self.confusion_matrix.sum(dim=1) - tp
        
        # Avoid division by zero
        denominator = tp + fp + fn
        iou = tp / (denominator + 1e-8)
        
        # Set IoU to 0 for classes that don't appear in ground truth
        valid_classes = (denominator > 0)
        iou = iou * valid_classes.float()
        
        mean_iou = iou[valid_classes].mean() if valid_classes.any() else torch.tensor(0.0)
        
        return iou, mean_iou
    
    def compute_pixel_accuracy(self) -> torch.Tensor:
        """
        Compute pixel accuracy.
        
        Returns:
            torch.Tensor: Pixel accuracy
        """
        correct_pixels = torch.diag(self.confusion_matrix).sum()
        total_pixels = self.confusion_matrix.sum()
        
        return correct_pixels / (total_pixels + 1e-8)
    
    def compute_mean_accuracy(self) -> torch.Tensor:
        """
        Compute mean class accuracy.
        
        Returns:
            torch.Tensor: Mean accuracy
        """
        # Class accuracy = TP / (TP + FN)
        tp = torch.diag(self.confusion_matrix).float()
        total_per_class = self.confusion_matrix.sum(dim=1).float()
        
        class_accuracy = tp / (total_per_class + 1e-8)
        
        # Only consider classes that appear in ground truth
        valid_classes = (total_per_class > 0)
        mean_accuracy = class_accuracy[valid_classes].mean() if valid_classes.any() else torch.tensor(0.0)
        
        return mean_accuracy
    
    def compute_frequency_weighted_iou(self) -> torch.Tensor:
        """
        Compute frequency weighted IoU.
        
        Returns:
            torch.Tensor: Frequency weighted IoU
        """
        iou, _ = self.compute_iou()
        
        # Class frequencies
        class_frequencies = self.confusion_matrix.sum(dim=1).float()
        total_pixels = class_frequencies.sum()
        weights = class_frequencies / (total_pixels + 1e-8)
        
        # Weighted IoU
        fwiou = (weights * iou).sum()
        
        return fwiou
    
    def compute_dice_score(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Dice score metrics.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (per_class_dice, mean_dice)
        """
        # Dice = 2 * TP / (2 * TP + FP + FN)
        tp = torch.diag(self.confusion_matrix).float()
        fp = self.confusion_matrix.sum(dim=0) - tp
        fn = self.confusion_matrix.sum(dim=1) - tp
        
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        
        # Set Dice to 0 for classes that don't appear
        valid_classes = ((tp + fp + fn) > 0)
        dice = dice * valid_classes.float()
        
        mean_dice = dice[valid_classes].mean() if valid_classes.any() else torch.tensor(0.0)
        
        return dice, mean_dice
    
    def compute_precision_recall(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute precision and recall metrics.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                (per_class_precision, mean_precision, per_class_recall, mean_recall)
        """
        tp = torch.diag(self.confusion_matrix).float()
        fp = self.confusion_matrix.sum(dim=0) - tp
        fn = self.confusion_matrix.sum(dim=1) - tp
        
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp + 1e-8)
        valid_precision = ((tp + fp) > 0)
        precision = precision * valid_precision.float()
        mean_precision = precision[valid_precision].mean() if valid_precision.any() else torch.tensor(0.0)
        
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn + 1e-8)
        valid_recall = ((tp + fn) > 0)
        recall = recall * valid_recall.float()
        mean_recall = recall[valid_recall].mean() if valid_recall.any() else torch.tensor(0.0)
        
        return precision, mean_precision, recall, mean_recall
    
    def compute_f1_score(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute F1 score metrics.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (per_class_f1, mean_f1)
        """
        precision, _, recall, _ = self.compute_precision_recall()
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Only consider valid classes
        valid_classes = ((precision + recall) > 0)
        f1 = f1 * valid_classes.float()
        mean_f1 = f1[valid_classes].mean() if valid_classes.any() else torch.tensor(0.0)
        
        return f1, mean_f1
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics and return as dictionary.
        
        Returns:
            Dict[str, float]: Dictionary containing all metrics
        """
        # IoU metrics
        per_class_iou, mean_iou = self.compute_iou()
        
        # Accuracy metrics
        pixel_accuracy = self.compute_pixel_accuracy()
        mean_accuracy = self.compute_mean_accuracy()
        fwiou = self.compute_frequency_weighted_iou()
        
        # Dice score
        per_class_dice, mean_dice = self.compute_dice_score()
        
        # Precision and Recall
        per_class_precision, mean_precision, per_class_recall, mean_recall = self.compute_precision_recall()
        
        # F1 Score
        per_class_f1, mean_f1 = self.compute_f1_score()
        
        # Convert to float for logging
        metrics = {
            'mIoU': mean_iou.item(),
            'Pixel_Accuracy': pixel_accuracy.item(),
            'Mean_Accuracy': mean_accuracy.item(),
            'Frequency_Weighted_IoU': fwiou.item(),
            'Mean_Dice': mean_dice.item(),
            'Mean_Precision': mean_precision.item(),
            'Mean_Recall': mean_recall.item(),
            'Mean_F1': mean_f1.item()
        }
        
        # Add per-class metrics
        for i in range(self.num_classes):
            metrics[f'IoU_Class_{i}'] = per_class_iou[i].item()
            metrics[f'Dice_Class_{i}'] = per_class_dice[i].item()
            metrics[f'Precision_Class_{i}'] = per_class_precision[i].item()
            metrics[f'Recall_Class_{i}'] = per_class_recall[i].item()
            metrics[f'F1_Class_{i}'] = per_class_f1[i].item()
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix as numpy array.
        
        Returns:
            np.ndarray: Confusion matrix
        """
        return self.confusion_matrix.cpu().numpy()
    
    def print_class_metrics(self, class_names: Optional[List[str]] = None):
        """
        Print detailed per-class metrics.
        
        Args:
            class_names (Optional[List[str]]): Names of classes
        """
        if class_names is None:
            class_names = [f"Class {i}" for i in range(self.num_classes)]
        
        # Compute metrics
        per_class_iou, mean_iou = self.compute_iou()
        per_class_dice, mean_dice = self.compute_dice_score()
        per_class_precision, mean_precision, per_class_recall, mean_recall = self.compute_precision_recall()
        per_class_f1, mean_f1 = self.compute_f1_score()
        
        print("\nPer-Class Metrics:")
        print("-" * 80)
        print(f"{'Class':<20} {'IoU':<8} {'Dice':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}")
        print("-" * 80)
        
        for i in range(self.num_classes):
            class_name = class_names[i] if i < len(class_names) else f"Class {i}"
            print(
                f"{class_name:<20} "
                f"{per_class_iou[i]:.4f}   "
                f"{per_class_dice[i]:.4f}   "
                f"{per_class_precision[i]:.4f}     "
                f"{per_class_recall[i]:.4f}   "
                f"{per_class_f1[i]:.4f}"
            )
        
        print("-" * 80)
        print(
            f"{'Mean':<20} "
            f"{mean_iou:.4f}   "
            f"{mean_dice:.4f}   "
            f"{mean_precision:.4f}     "
            f"{mean_recall:.4f}   "
            f"{mean_f1:.4f}"
        )
        print("-" * 80)


class StreamingMetrics:
    """
    Streaming version of metrics for large datasets that don't fit in memory.
    """
    
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset metrics."""
        self.tp = np.zeros(self.num_classes, dtype=np.int64)
        self.fp = np.zeros(self.num_classes, dtype=np.int64)
        self.fn = np.zeros(self.num_classes, dtype=np.int64)
        self.total_pixels = 0
        self.correct_pixels = 0
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Update metrics with new predictions and targets.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): Ground truth labels
        """
        # Flatten arrays
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Create mask for valid pixels
        mask = (targets != self.ignore_index)
        predictions = predictions[mask]
        targets = targets[mask]
        
        # Update pixel counts
        self.total_pixels += len(targets)
        self.correct_pixels += np.sum(predictions == targets)
        
        # Update per-class counts
        for c in range(self.num_classes):
            pred_mask = (predictions == c)
            target_mask = (targets == c)
            
            self.tp[c] += np.sum(pred_mask & target_mask)
            self.fp[c] += np.sum(pred_mask & ~target_mask)
            self.fn[c] += np.sum(~pred_mask & target_mask)
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute metrics from accumulated counts.
        
        Returns:
            Dict[str, float]: Computed metrics
        """
        # IoU
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-8)
        valid_classes = (self.tp + self.fp + self.fn) > 0
        mean_iou = np.mean(iou[valid_classes]) if np.any(valid_classes) else 0.0
        
        # Pixel accuracy
        pixel_accuracy = self.correct_pixels / (self.total_pixels + 1e-8)
        
        # Mean accuracy
        class_accuracy = self.tp / (self.tp + self.fn + 1e-8)
        mean_accuracy = np.mean(class_accuracy[valid_classes]) if np.any(valid_classes) else 0.0
        
        # Precision and Recall
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        
        valid_precision = (self.tp + self.fp) > 0
        valid_recall = (self.tp + self.fn) > 0
        
        mean_precision = np.mean(precision[valid_precision]) if np.any(valid_precision) else 0.0
        mean_recall = np.mean(recall[valid_recall]) if np.any(valid_recall) else 0.0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        valid_f1 = (precision + recall) > 0
        mean_f1 = np.mean(f1[valid_f1]) if np.any(valid_f1) else 0.0
        
        return {
            'mIoU': float(mean_iou),
            'Pixel_Accuracy': float(pixel_accuracy),
            'Mean_Accuracy': float(mean_accuracy),
            'Mean_Precision': float(mean_precision),
            'Mean_Recall': float(mean_recall),
            'Mean_F1': float(mean_f1)
        }


if __name__ == "__main__":
    # Test metrics
    num_classes = 19
    batch_size = 2
    height, width = 64, 64
    
    # Create dummy data
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test SegmentationMetrics
    metrics = SegmentationMetrics(num_classes)
    metrics.update(predictions, targets)
    
    computed_metrics = metrics.compute()
    print("Computed metrics:")
    for key, value in computed_metrics.items():
        if not key.startswith(('IoU_Class', 'Dice_Class', 'Precision_Class', 'Recall_Class', 'F1_Class')):
            print(f"{key}: {value:.4f}")
    
    # Test class-wise metrics
    metrics.print_class_metrics()
    
    print("\nMetrics module test completed successfully!")

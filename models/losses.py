"""
Loss Functions for Semantic Segmentation

This module contains various loss functions commonly used in semantic segmentation,
including Cross Entropy, Focal Loss, Dice Loss, IoU Loss, and OHEM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np


class CrossEntropyLoss(nn.Module):
    """
    Standard Cross Entropy Loss for semantic segmentation.
    
    Args:
        ignore_index (int): Index to ignore in loss calculation
        weight (Optional[torch.Tensor]): Class weights for handling imbalanced datasets
        reduction (str): Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, ignore_index: int = 255, weight: Optional[torch.Tensor] = None, 
                 reduction: str = 'mean'):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Cross Entropy Loss.
        
        Args:
            pred (torch.Tensor): Predictions of shape (B, C, H, W)
            target (torch.Tensor): Ground truth of shape (B, H, W)
            
        Returns:
            torch.Tensor: Loss value
        """
        return F.cross_entropy(
            pred, target, 
            weight=self.weight, 
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation, particularly effective for small objects.
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero
        ignore_index (int): Index to ignore in loss calculation
        reduction (str): Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, smooth: float = 1e-5, ignore_index: int = 255, reduction: str = 'mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Dice Loss.
        
        Args:
            pred (torch.Tensor): Predictions of shape (B, C, H, W)
            target (torch.Tensor): Ground truth of shape (B, H, W)
            
        Returns:
            torch.Tensor: Loss value
        """
        # Convert predictions to probabilities
        pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        # Create mask for valid pixels
        mask = (target != self.ignore_index).unsqueeze(1).float()
        pred = pred * mask
        target_one_hot = target_one_hot * mask
        
        # Compute Dice coefficient
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_coeff
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class IoULoss(nn.Module):
    """
    IoU (Intersection over Union) Loss for semantic segmentation.
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero
        ignore_index (int): Index to ignore in loss calculation
        reduction (str): Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, smooth: float = 1e-5, ignore_index: int = 255, reduction: str = 'mean'):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of IoU Loss.
        
        Args:
            pred (torch.Tensor): Predictions of shape (B, C, H, W)
            target (torch.Tensor): Ground truth of shape (B, H, W)
            
        Returns:
            torch.Tensor: Loss value
        """
        # Convert predictions to probabilities
        pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        # Create mask for valid pixels
        mask = (target != self.ignore_index).unsqueeze(1).float()
        pred = pred * mask
        target_one_hot = target_one_hot * mask
        
        # Compute IoU
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3)) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        iou_loss = 1.0 - iou
        
        if self.reduction == 'mean':
            return iou_loss.mean()
        elif self.reduction == 'sum':
            return iou_loss.sum()
        else:
            return iou_loss


class OHEMLoss(nn.Module):
    """
    Online Hard Example Mining (OHEM) Loss for focusing on hard examples.
    
    Args:
        thresh (float): Threshold for hard example selection
        min_kept (int): Minimum number of pixels to keep
        ignore_index (int): Index to ignore in loss calculation
        base_loss (str): Base loss function ('ce', 'focal')
    """
    
    def __init__(self, thresh: float = 0.7, min_kept: int = 100000, 
                 ignore_index: int = 255, base_loss: str = 'ce'):
        super(OHEMLoss, self).__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        
        if base_loss == 'ce':
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        else:
            raise ValueError(f"Unsupported base loss: {base_loss}")
            
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of OHEM Loss.
        
        Args:
            pred (torch.Tensor): Predictions of shape (B, C, H, W)
            target (torch.Tensor): Ground truth of shape (B, H, W)
            
        Returns:
            torch.Tensor: Loss value
        """
        # Compute pixel-wise loss
        pixel_losses = self.criterion(pred, target)
        
        # Create mask for valid pixels
        mask = (target != self.ignore_index).float()
        pixel_losses = pixel_losses * mask
        
        # Sort losses in descending order
        sorted_losses, _ = torch.sort(pixel_losses.view(-1), descending=True)
        
        # Determine number of pixels to keep
        valid_pixels = mask.sum().int().item()
        keep_num = max(self.min_kept, int(valid_pixels * self.thresh))
        keep_num = min(keep_num, valid_pixels)
        
        # Keep only the hardest examples
        if keep_num < valid_pixels:
            threshold = sorted_losses[keep_num]
            hard_mask = (pixel_losses >= threshold).float()
            return (pixel_losses * hard_mask).sum() / hard_mask.sum()
        else:
            return pixel_losses.sum() / mask.sum()


class BCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with Logits Loss for semantic segmentation.
    For multi-class segmentation, this applies BCE to each class independently.
    
    Args:
        ignore_index (int): Index to ignore in loss calculation
        weight (Optional[torch.Tensor]): Class weights for handling imbalanced datasets
        reduction (str): Reduction method ('mean', 'sum', 'none')
        pos_weight (Optional[torch.Tensor]): Weight of positive examples
    """
    
    def __init__(self, ignore_index: int = 255, weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean', pos_weight: Optional[torch.Tensor] = None):
        super(BCEWithLogitsLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.bce_loss = nn.BCEWithLogitsLoss(weight=weight, reduction='none', pos_weight=pos_weight)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of BCE with Logits Loss.
        
        Args:
            pred (torch.Tensor): Predictions of shape (B, C, H, W)
            target (torch.Tensor): Ground truth of shape (B, H, W)
            
        Returns:
            torch.Tensor: Loss value
        """
        # Convert target to one-hot encoding
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        # Create mask for valid pixels
        mask = (target != self.ignore_index).unsqueeze(1).float()
        
        # Apply BCE loss to each class
        loss = self.bce_loss(pred, target_one_hot)
        
        # Apply mask to ignore invalid pixels
        loss = loss * mask
        
        if self.reduction == 'mean':
            # Average over valid pixels only
            return loss.sum() / mask.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(loss_config: dict) -> nn.Module:
    """
    Factory function to create loss function based on configuration.
    
    Args:
        loss_config (dict): Loss configuration dictionary
        
    Returns:
        nn.Module: Loss function
    """
    loss_type = loss_config.get('type', 'cross_entropy').lower()
    ignore_index = loss_config.get('ignore_index', 255)
    
    if loss_type == 'cross_entropy' or loss_type == 'ce':
        weight = loss_config.get('class_weights')
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        return CrossEntropyLoss(ignore_index=ignore_index, weight=weight)
    
    elif loss_type == 'dice':
        smooth = loss_config.get('dice_smooth', 1e-5)
        return DiceLoss(smooth=smooth, ignore_index=ignore_index)
    
    elif loss_type == 'iou':
        smooth = loss_config.get('iou_smooth', 1e-5)
        return IoULoss(smooth=smooth, ignore_index=ignore_index)
    
    elif loss_type == 'ohem':
        thresh = loss_config.get('ohem_thresh', 0.7)
        min_kept = loss_config.get('ohem_min_kept', 100000)
        base_loss = loss_config.get('ohem_base_loss', 'ce')
        return OHEMLoss(thresh=thresh, min_kept=min_kept, 
                       ignore_index=ignore_index, base_loss=base_loss)
    
    elif loss_type == 'bce_with_logits' or loss_type == 'bce':
        weight = loss_config.get('class_weights')
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        pos_weight = loss_config.get('pos_weight')
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        return BCEWithLogitsLoss(ignore_index=ignore_index, weight=weight, pos_weight=pos_weight)
    
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    batch_size, num_classes, height, width = 2, 19, 64, 64
    
    # Create dummy data
    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test different loss functions
    losses = {
        'CrossEntropy': CrossEntropyLoss(),
        'Dice': DiceLoss(),
        'IoU': IoULoss(),
        'OHEM': OHEMLoss(),
    }
    
    for loss_name, loss_fn in losses.items():
        loss_value = loss_fn(pred, target)
        print(f"{loss_name} Loss: {loss_value.item():.4f}")

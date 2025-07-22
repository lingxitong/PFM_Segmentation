"""
Training Engine for Semantic Segmentation

This module contains the main training engine with support for
mixed precision training, gradient accumulation, and comprehensive logging.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import albumentations as A
import time
import os
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import logging
from .metrics import SegmentationMetrics
from .visualization import save_predictions
from .scheduler import get_scheduler


class SegmentationTrainer:
    """
    Comprehensive training engine for semantic segmentation models.
    
    Args:
        model (nn.Module): Segmentation model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        config (Dict): Training configuration
        device (str): Device to run training on
    """
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 criterion: nn.Module, optimizer: torch.optim.Optimizer, 
                 config: Dict[str, Any], device: str = 'cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_transform_mean_std = self._extract_validation_transform_mean_std()
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # Training settings
        self.epochs = config['training']['epochs']
        self.use_amp = config['training'].get('use_amp', False)
        self.accumulate_grad_batches = config['training'].get('accumulate_grad_batches', 1)
        self.clip_grad_norm = config['training'].get('clip_grad_norm', None)
        
        # Eval settings
        self.eval_interval = config['validation'].get('eval_interval', 1)
        log_dir = config['logging'].get('log_dir')
        experiment_name = config['logging'].get('experiment_name')
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        
        # Visualization settings
        self.save_predictions_flag = config['visualization'].get('save_predictions', True)
        self.vis_save_interval = config['visualization'].get('save_interval', 10)
        self.num_vis_samples = config['visualization'].get('num_vis_samples', 8)
        
        # Initialize components
        self.scaler = GradScaler() if self.use_amp else None
        self.scheduler = get_scheduler(optimizer, config['training'].get('scheduler', None))
        
        # Initialize metrics based on task type
        num_classes = config['dataset']['num_classes']
        ignore_index = config['dataset'].get('ignore_index', 255)
        class_names = config['dataset'].get('class_names', None)
        self.metrics = SegmentationMetrics(
            num_classes=num_classes,
            ignore_index=ignore_index
        )
        
        # Tracking variables
        self.current_epoch = 0
        self.global_step = 0
        self.best_miou = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_mious = []

        # Setup logging
        self._setup_logging()
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _extract_validation_transform_mean_std(self):
        """
        Extract mean and standard deviation from the validation transforms if A.Normalize is present.

        Returns:
            Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
                A tuple (mean_tuple, std_tuple), e.g. ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                or None if no Normalize transform is found.
        """
        validation_transform = self.val_loader.dataset.transform
        for t in validation_transform.transforms:
            if isinstance(t, A.Normalize):
                # directly return the tuple-of-floats structure
                return t.mean, t.std
        return None
    
    def _denormalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a tensor using the mean and std from the validation transforms.

        Args:
            tensor (torch.Tensor): Input tensor [C, H, W] or [B, C, H, W]

        Returns:
            torch.Tensor: Denormalized tensor
        """
        if self.val_transform_mean_std is None:
            return tensor

        mean, std = list(self.val_transform_mean_std[0]), list(self.val_transform_mean_std[1])  # assume each is a list like [0.485, 0.456, 0.406]

        mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
        std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)

        return (tensor * std + mean) * 255.0 


    def _setup_logging(self):
        """Setup logging configuration."""
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.epochs}')
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            
            # Forward pass with optional mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    main_loss = self.criterion(outputs['out'], labels.long())
                    loss = main_loss
                    
                    # Normalize loss for gradient accumulation
                    loss = loss / self.accumulate_grad_batches
            else:
                outputs = self.model(images)
                main_loss = self.criterion(outputs['out'], labels.long())
                loss = main_loss
                    
                loss = loss / self.accumulate_grad_batches
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation and optimization step
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                if self.use_amp:
                    # Gradient clipping
                    if self.clip_grad_norm:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    if self.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update learning rate
                if self.scheduler and hasattr(self.scheduler, 'step_batch'):
                    self.scheduler.step_batch(self.global_step)
                
                self.global_step += 1
            
            # Update metrics
            total_loss += loss.item() * self.accumulate_grad_batches
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item() * self.accumulate_grad_batches:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
        
        # Update learning rate scheduler (epoch-based)
        if self.scheduler and hasattr(self.scheduler, 'step'):
            self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.
        
        Returns:
            Tuple[float, Dict[str, float]]: Average validation loss and metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        # Reset metrics
        self.metrics.reset()
        
        # Collect predictions for visualization
        vis_data = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        if isinstance(outputs, dict):
                            predictions = outputs['out']
                        else:
                            predictions = outputs
                        loss = self.criterion(predictions, labels.long())
                else:
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        predictions = outputs['out']
                    else:
                        predictions = outputs
                    loss = self.criterion(predictions, labels.long())
                
                # Update metrics
                total_loss += loss.item()
                self.metrics.update(predictions, labels)
                
                # Collect data for visualization
                if (batch_idx < self.num_vis_samples // len(images) + 1 and 
                    len(vis_data) < self.num_vis_samples):
                    batch_size = min(self.num_vis_samples - len(vis_data), len(images))
                    for i in range(batch_size):
                        vis_data.append({
                            'image': self._denormalize_tensor(images[i].cpu()),
                            'label': labels[i].cpu(),
                            'prediction': torch.argmax(predictions[i], dim=0).cpu()
                        })
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Compute final metrics
        avg_loss = total_loss / num_batches
        metrics_dict = self.metrics.compute()
    
        self.val_losses.append(avg_loss)
        # Use mDice for medical tasks, mIoU for standard tasks
        primary_metric = metrics_dict.get('mDice', metrics_dict.get('mIoU', 0.0))
        self.val_mious.append(primary_metric)
        
        # Save predictions visualization
        if (self.save_predictions_flag and 
            (self.current_epoch + 1) % self.vis_save_interval == 0):
            vis_dir = os.path.join(self.log_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            save_predictions(
                vis_data[:self.num_vis_samples],
                save_dir=vis_dir,
                epoch=self.current_epoch + 1,
                class_colors=getattr(self.config['dataset'], 'class_colors', None)
            )
        
        return avg_loss, metrics_dict
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False , only_best: bool = True):
        """
        Save model checkpoint.
        
        Args:
            metrics (Dict[str, float]): Current metrics
            is_best (bool): Whether this is the best checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_mious': self.val_mious
        }
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f'New best model saved with mIoU: {metrics["mIoU"]:.4f}')
        else:
            if only_best:
                self.logger.info("Skipping regular checkpoint save as only_best is True.")
                return
            # Save regular checkpoint
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f'checkpoint_epoch_{self.current_epoch + 1:03d}.pth'
            )
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f'Checkpoint saved: {checkpoint_path}')
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f'Checkpoint not found: {checkpoint_path}')
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_mious = checkpoint.get('val_mious', [])
        
        if self.val_mious:
            self.best_miou = max(self.val_mious)
        
        self.logger.info(f'Checkpoint loaded: {checkpoint_path}')
        self.logger.info(f'Resuming from epoch {self.current_epoch}')
    
    def train(self):
        """Main training loop."""
        self.logger.info('Starting training...')
        self.logger.info(f'Total epochs: {self.epochs}')
        self.logger.info(f'Device: {self.device}')
        self.logger.info(f'Mixed precision: {self.use_amp}')
        self.logger.info(f'Gradient accumulation steps: {self.accumulate_grad_batches}')
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            if (epoch + 1) % self.eval_interval == 0:
                val_loss, val_metrics = self.validate()
                
                # Check if this is the best model
                primary_metric_name = 'mIoU'
                current_metric = val_metrics.get(primary_metric_name, val_metrics.get('mIoU', 0.0))
                is_best = current_metric > self.best_miou
                if is_best:
                    self.best_miou = current_metric
                
                # Logging
                self.logger.info(
                    f'Epoch {epoch + 1}/{self.epochs} - '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Val Loss: {val_loss:.4f}, '
                    f'Val {primary_metric_name}: {current_metric:.4f}, '
                    f'Best {primary_metric_name}: {self.best_miou:.4f}'
                )
                
                # Print detailed metrics
                for metric_name, metric_value in val_metrics.items():
                    if metric_name != 'mIoU':
                        self.logger.info(f'{metric_name}: {metric_value:.4f}')
                
            else:
                self.logger.info(
                    f'Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f}'
                )
            self.save_checkpoint(val_metrics, is_best=is_best, only_best = True)
        
        total_time = time.time() - start_time
        self.logger.info(f'Training completed in {total_time / 3600:.2f} hours')
        self.logger.info(f'Best mIoU: {self.best_miou:.4f}')
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """
        Get training statistics.
        
        Returns:
            Dict[str, List[float]]: Training statistics
        """
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_mious': self.val_mious}


if __name__ == "__main__":
    # Test trainer functionality
    print("Testing trainer module...")
    
    # This would require actual model, data loaders, etc.
    # For now, just test imports
    print("Trainer module loaded successfully!")

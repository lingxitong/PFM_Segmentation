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
        
        # LoRA/DoRA specific settings
        self.finetune_mode = config.get('model', {}).get('finetune_mode', {}).get('type', None)
        self.best_lora_state_dict = None  # Store best LoRA/DoRA weights in memory
        self.best_epoch = 0  # Record the epoch of best weights
        self.best_checkpoint = None

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
        
    def _clone_state_dict(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.detach().clone().cpu()
        elif isinstance(obj, dict):
            return {k: self._clone_state_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clone_state_dict(item) for item in obj]
        else:
            return obj
    
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
    
    def _extract_lora_dora_params(self) -> Dict[str, torch.Tensor]:
        """
        Extract LoRA/DoRA parameters from model.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing only LoRA/DoRA parameters
        """
        lora_dora_params = {}
        for name, param in self.model.named_parameters():
            # Typically LoRA/DoRA parameters have 'lora' or 'dora' in their names
            if 'lora' in name.lower() or 'dora' in name.lower():
                lora_dora_params[name] = param.detach().cpu().clone()
        return lora_dora_params
    
    def _extract_decoder_head_params(self) -> Dict[str, torch.Tensor]:
        """
        Extract decoder and segmentation_head parameters from model.
        Excludes PFM (pathology foundation model) parameters.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing only decoder and segmentation_head parameters
        """
        decoder_head_params = {}
        for name, param in self.model.named_parameters():
            # Extract decoder and segmentation_head parameters, exclude pfm parameters
            if name.startswith('decoder.') or name.startswith('segmentation_head.'):
                decoder_head_params[name] = param.detach().cpu().clone()
        return decoder_head_params
    
    def save_checkpoint_in_memory(self, metrics: Dict[str, float], is_best: bool = False , only_best: bool = True):
        """
        Save model checkpoint.
        For LoRA/DoRA mode, best weights are stored in memory first and saved to disk at the end.
        For non-full modes (frozen, lora, dora), only decoder and segmentation_head are saved.
        For full mode, entire model is saved.
        
        Args:
            metrics (Dict[str, float]): Current metrics
            is_best (bool): Whether this is the best checkpoint
            only_best (bool): Whether to save only the best checkpoint
        """
        
        if is_best:
            # Extract model state dict based on finetune mode
            if self.finetune_mode == 'full':
                # Full mode: save entire model
                model_sd = self._clone_state_dict(self.model.state_dict())
            else:
                # Non-full mode: save only decoder and segmentation_head
                model_sd = self._extract_decoder_head_params()
            
            opt_sd = self._clone_state_dict(self.optimizer.state_dict())
            sch_sd = self._clone_state_dict(self.scheduler.state_dict()) if self.scheduler else None
            sc_sd = self._clone_state_dict(self.scaler.state_dict()) if self.scaler else None
            
            checkpoint = {
                'epoch': self.current_epoch + 1,
                'model_state_dict': model_sd,
                'optimizer_state_dict': opt_sd,
                'scheduler_state_dict': sch_sd,
                'scaler_state_dict': sc_sd,
                'metrics': metrics,
                'config': self.config,
                'train_losses': self.train_losses[:],
                'val_losses': self.val_losses[:],
                'val_mious': self.val_mious[:],
                'finetune_mode': self.finetune_mode
            }
            
            self.best_checkpoint = checkpoint
            self.best_epoch = self.current_epoch + 1
            
            primary_metric_name = 'mIoU' if 'mIoU' in metrics else 'mDice'
            primary_metric = metrics.get(primary_metric_name, 0.0)
            
            if self.finetune_mode == 'full':
                self.logger.info(
                    f'New best full model state stored in memory - '
                    f'Epoch: {self.best_epoch}, {primary_metric_name}: {primary_metric:.4f}'
                )
            else:
                self.logger.info(
                    f'New best decoder+head state stored in memory - '
                    f'Epoch: {self.best_epoch}, {primary_metric_name}: {primary_metric:.4f}'
                )
            
            if self.finetune_mode in ['lora', 'dora']:
                self.best_lora_state_dict = self._extract_lora_dora_params()
                self.logger.info(
                    f'New best LoRA/DoRA weights stored in memory - '
                    f'Epoch: {self.best_epoch}, {primary_metric_name}: {primary_metric:.4f}'
                )
        else:
            #如果不是只保存最佳权重，则保存当前轮次的权重，并保存LoRA/DoRA权重
            if only_best:
                self.logger.info("Skipping regular checkpoint save as only_best is True.")
                return
            else:
                # Extract model state dict based on finetune mode
                if self.finetune_mode == 'full':
                    model_sd = self.model.state_dict()
                else:
                    model_sd = self._extract_decoder_head_params()
                
                checkpoint = {
                    'epoch': self.current_epoch + 1,
                    'model_state_dict': model_sd,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                    'metrics': metrics,
                    'config': self.config,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_mious': self.val_mious,
                    'finetune_mode': self.finetune_mode
                }
                
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, 
                    f'checkpoint_epoch_{self.current_epoch + 1:03d}.pth'
                )
                torch.save(checkpoint, checkpoint_path)
                self.logger.info(f'Checkpoint saved: {checkpoint_path}')
                #Save the LoRA/DoRA weights accroding to the current_epoch
                if self.finetune_mode in ['lora', 'dora']:
                    lora_dora_path = os.path.join(self.checkpoint_dir, f'lora_dora_weights_epoch_{self.current_epoch + 1:03d}.pth')
                    best_lora_state_dict = self._extract_lora_dora_params()
                    lora_dora_checkpoint = {
                        'epoch': self.current_epoch + 1,
                        'lora_dora_state_dict': best_lora_state_dict,
                        'best_miou': metrics.get('mIoU', 0.0),
                        'config': self.config,
                        'finetune_mode': self.finetune_mode,
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses,
                        'val_mious': self.val_mious
                    }
                    torch.save(lora_dora_checkpoint, lora_dora_path)
                    self.logger.info(f'LoRA/DoRA weights saved: {lora_dora_path}')
    
    def save_best_lora_dora_weights(self):
        """
        Save the best LoRA/DoRA weights from memory to disk at the end of training.
        This method should be called after training completes.
        """
        # Create checkpoint with LoRA/DoRA weights
        lora_dora_checkpoint = {
            'epoch': self.best_epoch,
            'lora_dora_state_dict': self.best_lora_state_dict,
            'best_miou': self.best_miou,
            'config': self.config,
            'finetune_mode': self.finetune_mode,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_mious': self.val_mious
        }
        
        # Save to disk
        best_path = os.path.join(self.checkpoint_dir, 'best_lora_dora_weights.pth')
        torch.save(lora_dora_checkpoint, best_path)
        self.logger.info(
            f'Best LoRA/DoRA weights saved to disk - '
            f'Epoch: {self.best_epoch}, mIoU: {self.best_miou:.4f}'
        )
        self.logger.info(f'lora_dora_checkpoint path: {best_path}')

    def save_best_full_model(self):
        """
        Save the best model checkpoint from memory to disk at the end of training.
        For full mode: saves entire model as 'best_full_model.pth'
        For non-full modes: saves only decoder+head as 'best_decoder_head.pth'
        """
        if self.finetune_mode == 'full':
            best_path = os.path.join(self.checkpoint_dir, 'best_full_model.pth')
            model_type_str = 'full model'
        else:
            best_path = os.path.join(self.checkpoint_dir, 'best_decoder_head.pth')
            model_type_str = 'decoder+head'
        
        torch.save(self.best_checkpoint, best_path)
        primary_metric_name = 'mIoU' if 'mIoU' in self.best_checkpoint['metrics'] else 'mDice'
        primary_metric = self.best_checkpoint['metrics'].get(primary_metric_name, 0.0)
        self.logger.info(
            f'Best {model_type_str} saved to disk - '
            f'Epoch: {self.best_checkpoint["epoch"]}, {primary_metric_name}: {primary_metric:.4f}'
        )
        self.logger.info(f'Checkpoint path: {best_path}')
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        Supports both full model checkpoints and LoRA/DoRA-only checkpoints.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        #有bug
        print(f'error, this function is not implemented yet')
        # if not os.path.exists(checkpoint_path):
        #     self.logger.warning(f'Checkpoint not found: {checkpoint_path}')
        #     return
        
        # checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # # Check if this is a LoRA/DoRA checkpoint
        # if 'lora_dora_state_dict' in checkpoint:
        #     # Load only LoRA/DoRA parameters
        #     lora_dora_params = checkpoint['lora_dora_state_dict']
        #     model_state = self.model.state_dict()
            
        #     # Update only LoRA/DoRA parameters
        #     for name, param in lora_dora_params.items():
        #         if name in model_state:
        #             model_state[name] = param.to(self.device)
        #         else:
        #             self.logger.warning(f'Parameter {name} not found in model')
            
        #     self.model.load_state_dict(model_state)
        #     self.logger.info('LoRA/DoRA weights loaded successfully')
            
        #     # Load training statistics
        #     self.current_epoch = checkpoint.get('epoch', 0)
        #     self.train_losses = checkpoint.get('train_losses', [])
        #     self.val_losses = checkpoint.get('val_losses', [])
        #     self.val_mious = checkpoint.get('val_mious', [])
        #     self.best_miou = checkpoint.get('best_miou', 0.0)
            
        # else:
        #     # Load full model checkpoint
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
            
        #     # Load optimizer state
        #     if 'optimizer_state_dict' in checkpoint:
        #         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        #     # Load scheduler state
        #     if self.scheduler and checkpoint.get('scheduler_state_dict'):
        #         self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        #     # Load scaler state
        #     if self.scaler and checkpoint.get('scaler_state_dict'):
        #         self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        #     # Load training state
        #     self.current_epoch = checkpoint.get('epoch', 0)
        #     self.train_losses = checkpoint.get('train_losses', [])
        #     self.val_losses = checkpoint.get('val_losses', [])
        #     self.val_mious = checkpoint.get('val_mious', [])
            
        #     if self.val_mious:
        #         self.best_miou = max(self.val_mious)
        
        # self.logger.info(f'Checkpoint loaded: {checkpoint_path}')
        # self.logger.info(f'Resuming from epoch {self.current_epoch}')
    
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
            self.save_checkpoint_in_memory(val_metrics, is_best=is_best, only_best = True)
        
        # Save best LoRA/DoRA weights from memory to disk
        if self.finetune_mode in ['lora', 'dora']:
            self.save_best_lora_dora_weights()
        
        self.save_best_full_model()
        
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

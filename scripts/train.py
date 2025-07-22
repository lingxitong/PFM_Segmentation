#!/usr/bin/env python3
"""
Training Script for Semantic Segmentation

This script provides a complete training pipeline for semantic segmentation models
with support for various datasets, augmentations, loss functions, and optimization techniques.

Author: @Toby
Function: Train a semantic segmentation model using a configuration file.
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import albumentations as A
import numpy as np
import logging
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_pfm_segmentation_model,count_parameters
from models.losses import get_loss_function
# from data.datasets import get_dataset
# from data.transforms import get_transforms
from data.utils import create_dataloader
from data.transforms import get_transforms,SegmentationTransforms
from data.seg_dataset import get_dataset
from utils.trainer import SegmentationTrainer
from utils.visualization import plot_training_history
from utils.logs import setup_logging



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Training script for semantic segmentation')
    
    parser.add_argument('--config', type=str, default='/mnt/sdb/lxt/PFM_Seg/base.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda:3',
                       help='Device to use (cuda/cpu/auto)')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config




def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_device(device_arg: str) -> str:
    """
    Get device for training.
    
    Args:
        device_arg (str): Device argument from command line
        
    Returns:
        str: Device string
    """
    return torch.device(device_arg)

def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to a YAML file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        save_path (str): Path to save the configuration file
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to {save_path}")




def worker_init_fn(worker_id):
    """Initialize worker with a random seed based on worker ID."""
    seed = 42  # Base seed
    random.seed(seed + worker_id)
    np.random.seed(seed+ worker_id)
    torch.manual_seed(seed+ worker_id)
    torch.cuda.manual_seed(seed+ worker_id)
    torch.cuda.manual_seed_all(seed+ worker_id)


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model (nn.Module): Model to optimize
        config (Dict[str, Any]): Training configuration
        
    Returns:
        optim.Optimizer: Optimizer
    """
    optimizer_config = config['training'].get('optimizer')
    optimizer_type = optimizer_config.get('type', 'SGD').lower()
    
    lr = config['training']['learning_rate']
    weight_decay = config['training']['optimizer'].get('weight_decay', 1e-4)
    
    if optimizer_type == 'sgd':
        momentum = config['training'].get('momentum', 0.9)
        nesterov = optimizer_config.get('nesterov', True)
        
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
    
    elif optimizer_type == 'adam':
        betas = optimizer_config.get('betas', (0.9, 0.999))
        eps = optimizer_config.get('eps', 1e-8)
        
        return optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'adamw':
        betas = optimizer_config.get('betas', (0.9, 0.999))
        eps = optimizer_config.get('eps', 1e-8)
        
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    seed = config['system'].get('seed', 42)
    set_random_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed) 
    
    # Get device
    device = get_device(args.device)
    
    # Setup logging
    log_dir = config['logging'].get('log_dir')
    experiment_name = config['logging'].get('experiment_name')
    log_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    save_config(config, os.path.join(log_dir, 'config.yaml'))
    setup_logging(log_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting training...")
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {seed}")
    
    # Create model
    logger.info(f"Creating model: {config['model']['pfm_name']}...")
    logging.info(f"Model fintune-model: {config['model']['finetune_mode']}")
    model = create_pfm_segmentation_model(config['model'])
    model = model.to(device)
    
    # Log model information
    model_params_info_dict = count_parameters(model)
    logger.info(f"Model parameters info: {model_params_info_dict}")
    # Create datasets and data loaders
    logger.info("Creating datasets...")
    dataset_config = config['dataset']
    
    # Training dataset
    # train_transforms = get_transforms(config['training']['augmentation'])
    train_transforms = SegmentationTransforms.get_training_transforms(img_size=config['training']['augmentation']['RandomResizedCropSize'],seed=seed,mean=config['model']['mean'],std=config['model']['std'])
    train_dataset = get_dataset(dataset_config, train_transforms, split='train')
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        generator=generator,
        num_workers=config['system'].get('num_workers', 4),
        pin_memory=config['system'].get('pin_memory', True),
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )
    
    # Validation dataset

    val_transforms = SegmentationTransforms.get_validation_transforms(img_size=config['validation']['augmentation']['ResizedSize'], mean=config['model']['mean'], std=config['model']['std'])
    val_dataset = get_dataset(dataset_config, val_transforms, split='val')
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['validation']['batch_size'],
        shuffle=False,
        num_workers=config['system'].get('num_workers', 4),
        pin_memory=config['system'].get('pin_memory', True),
        drop_last=False
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Create loss function
    logger.info("Creating loss function...")
    criterion = get_loss_function(config['training']['loss'])
    criterion = criterion.to(device)
    
    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = create_optimizer(model, config)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    
    trainer.train()
    
    # Plot training history
    logger.info("Generating training history plots...")
    training_stats = trainer.get_training_stats()
    
    history_plot_path = os.path.join(log_dir, 'training_history.png')
    
    plot_training_history(
        train_losses=training_stats['train_losses'],
        val_losses=training_stats['val_losses'],
        val_metrics=training_stats['val_mious'],
        metric_name='mIoU',
        save_path=history_plot_path
    )
    
    logger.info("Training completed successfully!")
    
        
if __name__ == "__main__":
    main()

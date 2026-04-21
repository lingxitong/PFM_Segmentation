#!/usr/bin/env python3
"""
Bootstrap Evaluation Script for Semantic Segmentation

This script performs bootstrap evaluation (1000 iterations) to compute
mean and 95% confidence intervals for all segmentation metrics.

Features:
- Reuses inference code from infer.py
- Computes per-sample confusion matrices for efficient bootstrap
- Calculates mean and 95% CI for all metrics

Author: @chenwm
Function: Bootstrap evaluation for semantic segmentation models
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
import json
import logging
from typing import Dict, Any, List, Tuple
import tqdm
import warnings
from torch.cuda.amp import autocast
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from infer.py
from scripts.infer import (
    load_config,
    get_device,
    load_model,
    slideWindow_preprocess,
    slideWindow_merge,
    maskPath2tensor
)

from data.seg_dataset import JSONSegmentationDataset
from data.utils import create_dataloader
from data.transforms import SegmentationTransforms, get_model_normalization
from utils.logs import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for bootstrap evaluation."""
    parser = argparse.ArgumentParser(description='Bootstrap Evaluation Script for Semantic Segmentation')
    parser.add_argument('--config', type=str, 
                       default='/Path/to/yours/PFM_Segmentation/configs/config.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, 
                       default='/Path/to/yours/PFM_Segmentation_Output/logs_musk_frozen/cpm15/checkpoints/',
                       help='Path to model checkpoint file or checkpoint directory')
    parser.add_argument('--input_json', type=str, 
                       default='/Path/to/yours/configs/data/cpm15.json',
                       help='Path to JSON file containing input data')
    parser.add_argument('--output_dir', type=str, 
                       default='/Path/to/yours/PFM_Segmentation/inference_bootstrap_MUSK',
                       help='Directory to save bootstrap evaluation results')
    parser.add_argument('--device', type=str, default='cuda:6',
                       help='Device for inference (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--input_size', type=int, default=384,
                       help='Input size for resize or window size for sliding window')
    parser.add_argument('--resize_or_windowslide', type=str, 
                       choices=['resize', 'windowslide'], default='resize',
                       help='Inference mode: resize or sliding window')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                       help='Number of bootstrap iterations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--save_vis', action='store_true',
                       help='Save visualization (masks and overlays)')
    parser.add_argument('--max_save_per_task', type=int, default=20,
                       help='Maximum number of samples to save per task (default: 20)')
    return parser.parse_args()


def visualize_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Visualize a segmentation mask with discrete colors.
    
    Args:
        mask: Segmentation mask [H, W] with integer class labels
        num_classes: Number of classes
        
    Returns:
        RGB visualization array [H, W, 3]
    """
    max_label = num_classes - 1
    
    if mask.max() == 0:
        vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    else:
        # Get discrete colors
        if max_label < 10:
            base_colors = plt.get_cmap('tab10').colors
            base_colors = np.array(base_colors)[:, :3]
        elif max_label < 20:
            base_colors = plt.get_cmap('tab20').colors
            base_colors = np.array(base_colors)[:, :3]
        else:
            base_colors = plt.get_cmap('hsv')(np.linspace(0, 1, max_label + 1))[:, :3]

        # Build palette
        palette = np.zeros((256, 3))
        num_avail = len(base_colors)
        for i in range(min(256, num_avail)):
            palette[i] = base_colors[i]
        # Background is black
        palette[0] = [0, 0, 0]

        vis = (palette[mask] * 255).astype(np.uint8)
    
    return vis


def create_overlay(image: np.ndarray, mask_vis: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create an overlay of mask visualization on original image.
    
    Args:
        image: Original RGB image [H, W, 3]
        mask_vis: Visualized mask [H, W, 3]
        alpha: Blending factor (0 = only image, 1 = only mask)
        
    Returns:
        Blended overlay [H, W, 3]
    """
    # Ensure same size
    if image.shape[:2] != mask_vis.shape[:2]:
        from PIL import Image as PILImage
        mask_vis = np.array(PILImage.fromarray(mask_vis).resize(
            (image.shape[1], image.shape[0]), PILImage.Resampling.NEAREST
        ))
    
    # Blend
    overlay = (image.astype(np.float32) * (1 - alpha) + mask_vis.astype(np.float32) * alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay


def save_visualization(
    pred_mask: torch.Tensor,
    image_path: str,
    output_dir: str,
    sample_name: str,
    num_classes: int,
    mean: List[float],
    std: List[float]
) -> None:
    """
    Save mask visualization and overlay for a sample.
    
    Args:
        pred_mask: Predicted mask tensor [H, W]
        image_path: Path to original image
        output_dir: Output directory
        sample_name: Name for the output files
        num_classes: Number of classes
        mean: Normalization mean
        std: Normalization std
    """
    # Create output directories
    mask_dir = os.path.join(output_dir, 'predictions_masks')
    overlay_dir = os.path.join(output_dir, 'predictions_overlays')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    # Convert prediction to numpy
    pred_np = pred_mask.cpu().numpy().astype(np.uint8)
    
    # Visualize mask
    mask_vis = visualize_mask(pred_np, num_classes)
    
    # Save visualized mask
    mask_path = os.path.join(mask_dir, f'{sample_name}.png')
    Image.fromarray(mask_vis).save(mask_path)
    
    # Load original image and create overlay
    original_image = np.array(Image.open(image_path).convert('RGB'))
    
    # Resize mask visualization to match original image size if needed
    if original_image.shape[:2] != mask_vis.shape[:2]:
        mask_vis_resized = np.array(Image.fromarray(mask_vis).resize(
            (original_image.shape[1], original_image.shape[0]), Image.Resampling.NEAREST
        ))
    else:
        mask_vis_resized = mask_vis
    
    # Create and save overlay
    overlay = create_overlay(original_image, mask_vis_resized, alpha=0.5)
    overlay_path = os.path.join(overlay_dir, f'{sample_name}.png')
    Image.fromarray(overlay).save(overlay_path)


def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, 
                             num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    """
    Compute confusion matrix for a single sample.
    
    Args:
        pred: Predicted mask tensor [H, W]
        target: Ground truth mask tensor [H, W]
        num_classes: Number of classes
        ignore_index: Index to ignore
        
    Returns:
        Confusion matrix tensor [num_classes, num_classes]
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # Create mask for valid pixels
    mask = (target != ignore_index)
    pred = pred[mask]
    target = target[mask]
    
    # Compute confusion matrix
    indices = num_classes * target + pred
    cm = torch.bincount(indices, minlength=num_classes**2)
    cm = cm.reshape(num_classes, num_classes)
    
    return cm


def compute_metrics_from_confusion_matrix(confusion_matrix: torch.Tensor, 
                                          num_classes: int) -> Dict[str, float]:
    """
    Compute all segmentation metrics from a confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix tensor [num_classes, num_classes]
        num_classes: Number of classes
        
    Returns:
        Dictionary containing all metrics
    """
    cm = confusion_matrix.float()
    
    # TP, FP, FN
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    
    # IoU
    denominator = tp + fp + fn
    iou = tp / (denominator + 1e-8)
    valid_classes = (denominator > 0)
    iou = iou * valid_classes.float()
    mean_iou = iou[valid_classes].mean() if valid_classes.any() else torch.tensor(0.0)
    
    # Pixel Accuracy
    correct_pixels = torch.diag(cm).sum()
    total_pixels = cm.sum()
    pixel_accuracy = correct_pixels / (total_pixels + 1e-8)
    
    # Mean Accuracy
    total_per_class = cm.sum(dim=1).float()
    class_accuracy = tp / (total_per_class + 1e-8)
    valid_acc = (total_per_class > 0)
    mean_accuracy = class_accuracy[valid_acc].mean() if valid_acc.any() else torch.tensor(0.0)
    
    # Frequency Weighted IoU
    class_frequencies = cm.sum(dim=1).float()
    weights = class_frequencies / (total_pixels + 1e-8)
    fwiou = (weights * iou).sum()
    
    # Dice Score
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    valid_dice = ((tp + fp + fn) > 0)
    dice = dice * valid_dice.float()
    mean_dice = dice[valid_dice].mean() if valid_dice.any() else torch.tensor(0.0)
    
    # Precision and Recall
    precision = tp / (tp + fp + 1e-8)
    valid_precision = ((tp + fp) > 0)
    precision = precision * valid_precision.float()
    mean_precision = precision[valid_precision].mean() if valid_precision.any() else torch.tensor(0.0)
    
    recall = tp / (tp + fn + 1e-8)
    valid_recall = ((tp + fn) > 0)
    recall = recall * valid_recall.float()
    mean_recall = recall[valid_recall].mean() if valid_recall.any() else torch.tensor(0.0)
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    valid_f1 = ((precision + recall) > 0)
    f1 = f1 * valid_f1.float()
    mean_f1 = f1[valid_f1].mean() if valid_f1.any() else torch.tensor(0.0)
    
    # Build metrics dict
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
    for i in range(num_classes):
        metrics[f'IoU_Class_{i}'] = iou[i].item()
        metrics[f'Dice_Class_{i}'] = dice[i].item()
        metrics[f'Precision_Class_{i}'] = precision[i].item()
        metrics[f'Recall_Class_{i}'] = recall[i].item()
        metrics[f'F1_Class_{i}'] = f1[i].item()
    
    return metrics


def collect_per_sample_confusion_matrices_resize(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader,
    device: torch.device, 
    num_classes: int,
    ignore_index: int = 255,
    use_amp: bool = False,
    output_dir: str = None,
    max_save_per_task: int = 20,
    mean: List[float] = None,
    std: List[float] = None
) -> List[torch.Tensor]:
    """
    Collect per-sample confusion matrices using resize-based inference.
    
    Args:
        model: Loaded segmentation model
        dataloader: DataLoader providing input batches
        device: Target device
        num_classes: Number of classes
        ignore_index: Index to ignore
        use_amp: Whether to use mixed precision
        output_dir: Output directory for visualizations
        max_save_per_task: Maximum number of samples to save per task
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        List of confusion matrices, one per sample
    """
    confusion_matrices = []
    saved_count = 0
    total_samples = len(dataloader.dataset)
    samples_to_save = min(max_save_per_task, total_samples)
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Collecting per-sample confusion matrices"):
            images = batch['image'].to(device)
            label_paths = batch['label_path']
            image_paths = batch['image_path']
            
            # Forward pass
            if use_amp:
                with autocast():
                    preds = model(images)['out']
            else:
                preds = model(images)['out']
            
            # Process predictions
            pred_masks = torch.argmax(preds, dim=1)  # [B, H, W]
            
            if None not in label_paths:
                labels = batch['label'].to(device)  # [B, H, W]
                
                # Compute confusion matrix for each sample
                for i in range(pred_masks.shape[0]):
                    cm = compute_confusion_matrix(
                        pred_masks[i], labels[i], num_classes, ignore_index
                    )
                    confusion_matrices.append(cm.cpu())
                    
                    # Save visualization if needed
                    if output_dir is not None and saved_count < samples_to_save:
                        sample_name = os.path.splitext(os.path.basename(image_paths[i]))[0]
                        save_visualization(
                            pred_masks[i],
                            image_paths[i],
                            output_dir,
                            sample_name,
                            num_classes,
                            mean,
                            std
                        )
                        saved_count += 1
    
    if output_dir is not None:
        logging.info(f"Saved {saved_count} visualizations to {output_dir}")
    
    return confusion_matrices


def collect_per_sample_confusion_matrices_slidewindow(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader,
    device: torch.device, 
    num_classes: int,
    window_size: int,
    overlap: float = 0.2,
    ignore_index: int = 255,
    use_amp: bool = False,
    output_dir: str = None,
    max_save_per_task: int = 20,
    mean: List[float] = None,
    std: List[float] = None
) -> List[torch.Tensor]:
    """
    Collect per-sample confusion matrices using sliding window inference.
    
    Args:
        model: Loaded segmentation model
        dataloader: DataLoader providing input batches
        device: Target device
        num_classes: Number of classes
        window_size: Size of sliding window
        overlap: Overlap ratio between windows
        ignore_index: Index to ignore
        use_amp: Whether to use mixed precision
        output_dir: Output directory for visualizations
        max_save_per_task: Maximum number of samples to save per task
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        List of confusion matrices, one per sample
    """
    confusion_matrices = []
    saved_count = 0
    total_samples = len(dataloader.dataset)
    samples_to_save = min(max_save_per_task, total_samples)
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Collecting per-sample confusion matrices"):
            images = batch['image'].to(device)
            batch_size = images.shape[0]
            stride = int(window_size * (1 - overlap))
            
            # Process with sliding window
            patches, coords = slideWindow_preprocess(images, window_size, stride)
            label_paths = batch['label_path']
            image_paths = batch['image_path']
            
            # Predict and merge
            if use_amp:
                with autocast():
                    patches_preds = model(patches)['out']
            else:
                patches_preds = model(patches)['out']
            
            preds = slideWindow_merge(patches_preds, window_size, stride, coords, batch_size)
            pred_masks = torch.argmax(preds, dim=1)  # [B, H, W]
            
            if None not in label_paths:
                labels = torch.stack(
                    [maskPath2tensor(path, device).squeeze(0) for path in label_paths], 
                    dim=0
                )  # [B, H, W]
                
                # Compute confusion matrix for each sample
                for i in range(pred_masks.shape[0]):
                    cm = compute_confusion_matrix(
                        pred_masks[i], labels[i], num_classes, ignore_index
                    )
                    confusion_matrices.append(cm.cpu())
                    
                    # Save visualization if needed
                    if output_dir is not None and saved_count < samples_to_save:
                        sample_name = os.path.splitext(os.path.basename(image_paths[i]))[0]
                        save_visualization(
                            pred_masks[i],
                            image_paths[i],
                            output_dir,
                            sample_name,
                            num_classes,
                            mean,
                            std
                        )
                        saved_count += 1
    
    if output_dir is not None:
        logging.info(f"Saved {saved_count} visualizations to {output_dir}")
    
    return confusion_matrices


def bootstrap_evaluation(confusion_matrices: List[torch.Tensor], 
                         num_classes: int,
                         n_bootstrap: int = 1000,
                         seed: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Perform bootstrap evaluation to compute mean and 95% CI for all metrics.
    
    Args:
        confusion_matrices: List of per-sample confusion matrices
        num_classes: Number of classes
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed
        
    Returns:
        Dictionary with 'mean', 'ci_lower', 'ci_upper' for each metric
    """
    np.random.seed(seed)
    n_samples = len(confusion_matrices)
    
    # Stack confusion matrices for efficient indexing
    cm_stack = torch.stack(confusion_matrices, dim=0)  # [N, num_classes, num_classes]
    
    # Storage for bootstrap metrics
    bootstrap_metrics = {}
    
    logging.info(f"Running {n_bootstrap} bootstrap iterations with {n_samples} samples...")
    
    for i in tqdm.tqdm(range(n_bootstrap), desc="Bootstrap iterations"):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Sum confusion matrices for sampled indices
        sampled_cm = cm_stack[indices].sum(dim=0)
        
        # Compute metrics
        metrics = compute_metrics_from_confusion_matrix(sampled_cm, num_classes)
        
        # Store metrics
        for key, value in metrics.items():
            if key not in bootstrap_metrics:
                bootstrap_metrics[key] = []
            bootstrap_metrics[key].append(value)
    
    # Compute mean and 95% CI
    results = {}
    for key, values in bootstrap_metrics.items():
        values = np.array(values)
        results[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'ci_lower': float(np.percentile(values, 2.5)),
            'ci_upper': float(np.percentile(values, 97.5))
        }
    
    return results


def format_results(results: Dict[str, Dict[str, float]], num_classes: int) -> str:
    """
    Format bootstrap results for logging.
    
    Args:
        results: Bootstrap evaluation results
        num_classes: Number of classes
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("Bootstrap Evaluation Results (Mean [95% CI])")
    lines.append("=" * 80)
    
    # Main metrics
    main_metrics = ['mIoU', 'Pixel_Accuracy', 'Mean_Accuracy', 'Frequency_Weighted_IoU',
                    'Mean_Dice', 'Mean_Precision', 'Mean_Recall', 'Mean_F1']
    
    lines.append("\nOverall Metrics:")
    lines.append("-" * 60)
    for metric in main_metrics:
        if metric in results:
            r = results[metric]
            lines.append(f"  {metric:<25}: {r['mean']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")
    
    # Per-class metrics
    lines.append("\nPer-Class Metrics:")
    lines.append("-" * 60)
    
    for i in range(num_classes):
        lines.append(f"\n  Class {i}:")
        for metric_type in ['IoU', 'Dice', 'Precision', 'Recall', 'F1']:
            key = f'{metric_type}_Class_{i}'
            if key in results:
                r = results[key]
                lines.append(f"    {metric_type:<12}: {r['mean']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


def main() -> None:
    """Main execution function for bootstrap evaluation script."""
    args = parse_args()
    
    # Setup output directory and logging
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Bootstrap Evaluation for Semantic Segmentation")
    logger.info("=" * 60)
    
    # Load config and model
    config = load_config(args.config)
    device = get_device(args.device)
    
    logger.info("Loading model...")
    model = load_model(config, args.checkpoint, device)
    
    # Setup transforms
    logger.info("Loading transforms...")
    pfm_name = config['model'].get('pfm_name', 'unet')
    mean, std = get_model_normalization(pfm_name)
    
    if args.resize_or_windowslide == 'resize':
        test_transforms = SegmentationTransforms.get_validation_transforms(
            img_size=args.input_size,
            mean=mean,
            std=std
        )
    else:
        test_transforms = SegmentationTransforms.get_validation_transforms(
            img_size=None,
            mean=mean,
            std=std
        )
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    test_dataset = JSONSegmentationDataset(
        json_file=args.input_json, split='test', transform=test_transforms
    )
    
    # Adjust batch size for sliding window if needed
    infer_batch_size = args.batch_size
    if args.resize_or_windowslide == 'windowslide' and not test_dataset.fixed_size:
        infer_batch_size = 1
    
    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=infer_batch_size,
        shuffle=False,
        num_workers=config['system'].get('num_workers', 4),
        pin_memory=config['system'].get('pin_memory', True),
        drop_last=False
    )
    
    # Get config values
    num_classes = config['model']['num_classes']
    ignore_index = config['dataset'].get('ignore_index', 255)
    use_amp = config.get('training', {}).get('use_amp', False)
    
    logger.info(f"Number of test samples: {len(test_dataset)}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Inference mode: {args.resize_or_windowslide}")
    logger.info(f"Bootstrap iterations: {args.n_bootstrap}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Mixed precision: {use_amp}")
    logger.info(f"Save visualizations: {args.save_vis}")
    if args.save_vis:
        logger.info(f"Max samples to save: {args.max_save_per_task}")
    
    # Determine output directory for visualizations
    vis_output_dir = args.output_dir if args.save_vis else None
    
    # Collect per-sample confusion matrices
    logger.info("Step 1: Collecting per-sample confusion matrices...")
    if args.resize_or_windowslide == 'resize':
        confusion_matrices = collect_per_sample_confusion_matrices_resize(
            model, test_dataloader, device, num_classes, ignore_index, use_amp,
            output_dir=vis_output_dir,
            max_save_per_task=args.max_save_per_task,
            mean=mean,
            std=std
        )
    else:
        confusion_matrices = collect_per_sample_confusion_matrices_slidewindow(
            model, test_dataloader, device, num_classes, args.input_size, 
            overlap=0.2, ignore_index=ignore_index, use_amp=use_amp,
            output_dir=vis_output_dir,
            max_save_per_task=args.max_save_per_task,
            mean=mean,
            std=std
        )
    
    logger.info(f"Collected {len(confusion_matrices)} per-sample confusion matrices")
    
    # Compute original metrics (without bootstrap) for reference
    logger.info("Computing original metrics (all samples)...")
    total_cm = torch.stack(confusion_matrices, dim=0).sum(dim=0)
    original_metrics = compute_metrics_from_confusion_matrix(total_cm, num_classes)
    
    # Run bootstrap evaluation
    logger.info("Step 2: Running bootstrap evaluation...")
    bootstrap_results = bootstrap_evaluation(
        confusion_matrices, num_classes, args.n_bootstrap, args.seed
    )
    
    # Format and log results
    results_str = format_results(bootstrap_results, num_classes)
    logger.info(results_str)
    
    # Prepare output data
    output_data = {
        'config': {
            'n_bootstrap': args.n_bootstrap,
            'seed': args.seed,
            'n_samples': len(confusion_matrices),
            'inference_mode': args.resize_or_windowslide,
            'input_size': args.input_size
        },
        'original_metrics': original_metrics,
        'bootstrap_results': bootstrap_results
    }
    
    # Save results
    output_file = os.path.join(args.output_dir, 'bootstrap_metrics.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info("Bootstrap evaluation completed successfully!")


if __name__ == '__main__':
    main()

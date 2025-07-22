#!/usr/bin/env python3
"""
Inference Script for Semantic Segmentation with two modes:
1. Resize-based inference (for fixed-size inputs)
2. Sliding window inference (for large or variable-size inputs)

Features:
- Supports batch processing with DataLoader
- Handles both resizing and sliding window approaches
- Includes visualization utilities for predictions

Author: @Toby
Function: Inference for semantic segmentation models
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json
import logging
from PIL import Image
from typing import Dict, Any, List, Tuple
import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.seg_dataset import JSONSegmentationDataset
from data.utils import create_dataloader
from data.transforms import SegmentationTransforms
from utils.metrics import SegmentationMetrics
from models.pfm_seg_models import create_pfm_segmentation_model
from utils.visualization import apply_color_map, create_color_palette, put_text_with_bg
from utils.logs import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for inference configuration."""
    parser = argparse.ArgumentParser(description='Semantic Segmentation Inference Script')
    parser.add_argument('--config', type=str, 
                       default='/mnt/sdb/lxt/PFM_Seg/logs/crag_conch_v1_5_v2/config.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, 
                       default='/mnt/sdb/lxt/PFM_Seg/logs/crag_conch_v1_5_v2/checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--input_json', type=str, 
                       default='/mnt/sdb/lxt/PFM_Seg/CRAG/CRAG_no_mask.json',
                       help='Path to JSON file containing input data')
    parser.add_argument('--output_dir', type=str, 
                       default='/mnt/sdb/lxt/PFM_Seg/logs/crag_conch_v1_5_v2/inference_slidewindow',
                       help='Directory to save inference results')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device for inference (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--input_size', type=int, default=512,
                       help='Input size for resize or window size for sliding window')
    parser.add_argument('--resize_or_windowslide', type=str, 
                       choices=['resize', 'windowslide'], default='windowslide',
                       help='Inference mode: resize or sliding window')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for inference')
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_device(device_str: str) -> torch.device:
    """Get PyTorch device from string descriptor."""
    return torch.device(device_str if torch.cuda.is_available() else 'cpu')


def load_model(config: Dict[str, Any], checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load model from checkpoint with configuration.
    
    Args:
        config: Model configuration dictionary
        checkpoint_path: Path to model checkpoint
        device: Target device for model
        
    Returns:
        Loaded and configured model in evaluation mode
    """
    model = create_pfm_segmentation_model(config['model']).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.eval()
    return model


def postprocess(image_paths: List[str], pred_masks: List[np.ndarray], 
               label_paths: List[str], preds_dir: str, overlap_dir: str, 
               palette: np.ndarray) -> None:
    """
    Post-process and visualize inference results.
    
    Args:
        image_paths: List of input image paths
        pred_masks: List of predicted masks (2D numpy arrays)
        label_paths: List of ground truth label paths
        preds_dir: Directory to save prediction masks
        overlap_dir: Directory to save visualization overlays
        palette: Color palette for visualization
    """
    for i in range(len(image_paths)):
        # Process predicted mask
        pred_mask = pred_masks[i]

        # Apply color mapping
        pred_colored = apply_color_map(pred_mask, palette)


        # Save prediction mask
        Image.fromarray(pred_mask.astype(np.uint8)).save(
            os.path.join(preds_dir, os.path.basename(image_paths[i])))

        if label_paths[i] is not None:
            # Load and process original image
            original_image = Image.open(image_paths[i]).convert('RGB')
            original_np = np.array(original_image)

            # Create overlays
            label_mask = np.array(Image.open(label_paths[i]))
            label_colored = apply_color_map(label_mask, palette)
            overlay_label = cv2.addWeighted(original_np, 0.5, label_colored, 0.5, 0)
            overlay_pred = cv2.addWeighted(original_np, 0.5, pred_colored, 0.5, 0)

            # Add annotations
            put_text_with_bg(overlay_label, "Label", position=(10, 40))
            put_text_with_bg(overlay_pred, "Prediction", position=(10, 40))

            # Combine side-by-side
            combined = np.concatenate([overlay_label, overlay_pred], axis=1)

            # Save visualization
            Image.fromarray(combined).save(
                os.path.join(overlap_dir, os.path.basename(image_paths[i])))


def resizeMode_inference(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                        device: torch.device, output_dir: str, palette: np.ndarray, seg_metrics: SegmentationMetrics) -> None:
    """
    Perform inference using resize-based approach.
    
    Args:
        model: Loaded segmentation model
        dataloader: DataLoader providing input batches
        device: Target device for computation
        output_dir: Base directory for saving results
        palette: Color palette for visualization
        seg_metrics: Segmentation metrics object for evaluation
    """
    preds_dir = os.path.join(output_dir, 'predictions_masks')
    overlap_dir = os.path.join(output_dir, 'predictions_overlays')
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(overlap_dir, exist_ok=True)
    seg_metrics.reset()
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Inference Progress"):
            images = batch['image'].to(device)
            image_paths = batch['image_path']
            label_paths = batch['label_path']
            ori_sizes = batch['ori_size']
            # Forward pass
            preds = model(images)['out']
            
            # Process predictions
            pred_masks = [torch.argmax(pred, dim=0).cpu().numpy() for pred in preds]
            pred_masks = [cv2.resize(pred_mask, (ori_sizes[i][0], ori_sizes[i][1]), 
                          interpolation=cv2.INTER_NEAREST) for i, pred_mask in enumerate(pred_masks)]
            _pred_masks = [torch.tensor(mask) for mask in pred_masks]
            if None not in label_paths:
                labels = torch.stack([maskPath2tensor(path, device) for path in label_paths], dim=0)  # [B, H, W]
                seg_metrics.update(torch.stack(_pred_masks, dim=0).to(device), labels)

            # Save results
            postprocess(image_paths, pred_masks, label_paths, preds_dir, overlap_dir, palette)
    return seg_metrics.compute()


def slideWindow_preprocess(image: torch.Tensor, window_size: int, stride: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split image into sliding window patches.
    
    Args:
        image: Input tensor of shape [B, 3, H, W]
        window_size: Size of sliding window (square)
        stride: Step size between windows
        
    Returns:
        patches: Tensor of patches [A, 3, window_size, window_size]
        coords: Tensor of patch coordinates [A, 2] (x, y)
    """
    B, C, H, W = image.shape
    all_patches = []
    all_coords = []

    # Calculate unique window positions
    y_positions = []
    for y in range(0, H, stride):
        if y + window_size > H:
            y = H - window_size
        if y not in y_positions:
            y_positions.append(y)

    x_positions = []
    for x in range(0, W, stride):
        if x + window_size > W:
            x = W - window_size
        if x not in x_positions:
            x_positions.append(x)

    # Extract patches
    for b in range(B):
        for y in y_positions:
            for x in x_positions:
                patch = image[b, :, y:y+window_size, x:x+window_size]
                all_patches.append(patch.unsqueeze(0))
                all_coords.append([x, y])

    patches = torch.cat(all_patches, dim=0)
    coords = torch.tensor(all_coords, dtype=torch.int)

    return patches, coords


def slideWindow_merge(patches_pred: torch.Tensor, window_size: int, stride: int,
                     coords: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Merge sliding window predictions into full-size output.
    
    Args:
        patches_pred: Patch predictions [A, num_classes, window_size, window_size]
        window_size: Size of sliding window
        stride: Step size used between windows
        coords: Patch coordinates [A, 2] (x, y)
        batch_size: Original number of images in batch
        
    Returns:
        merged: Reconstructed predictions [B, num_classes, H, W]
    """
    A, num_classes, _, _ = patches_pred.shape
    device = patches_pred.device
    patches_per_image = A // batch_size
    coords = coords.to(device)

    # Calculate output dimensions
    max_x = coords[:, 0].max().item() + window_size
    max_y = coords[:, 1].max().item() + window_size
    H, W = max_y, max_x

    # Initialize output buffers
    merged = torch.zeros((batch_size, num_classes, H, W), 
                        dtype=patches_pred.dtype, device=device)
    count = torch.zeros((batch_size, 1, H, W), 
                       dtype=patches_pred.dtype, device=device)

    # Accumulate predictions
    for idx in range(A):
        b = idx // patches_per_image
        x, y = coords[idx]
        merged[b, :, y:y+window_size, x:x+window_size] += patches_pred[idx]
        count[b, :, y:y+window_size, x:x+window_size] += 1

    # Normalize overlapping regions
    count = torch.clamp(count, min=1.0)
    merged = merged / count

    return merged

def maskPath2tensor(mask_path: str, device: torch.device) -> torch.Tensor:
    """
    Load a mask image from path and convert to tensor.
    
    Args:
        mask_path: Path to the mask image
        device: Target device for tensor
        
    Returns:
        Tensor of shape [1, H, W] with mask values
    """
    mask = Image.open(mask_path).convert('L')
    mask_tensor = torch.tensor(np.array(mask), dtype=torch.long, device=device)
    return mask_tensor.unsqueeze(0)  # Add batch dimension

def slideWindowMode_inference(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                             device: torch.device, output_dir: str, palette: np.ndarray,
                             seg_metrics: SegmentationMetrics,
                             window_size: int, overlap: float = 0.2) -> SegmentationMetrics:
    """
    Perform inference using sliding window approach.
    
    Args:
        model: Loaded segmentation model
        dataloader: DataLoader providing input batches
        device: Target device for computation
        output_dir: Base directory for saving results
        palette: Color palette for visualization
        seg_metrics: Segmentation metrics object for evaluation
        window_size: Size of sliding window
        overlap: Overlap ratio between windows (0-1)
    """
    preds_dir = os.path.join(output_dir, 'predictions_masks')
    overlap_dir = os.path.join(output_dir, 'predictions_overlays')
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(overlap_dir, exist_ok=True)
    seg_metrics.reset()
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader,desc="Inference Progress"):
            images = batch['image'].to(device)
            batch_size = images.shape[0]
            stride = int(window_size * (1 - overlap))
            
            # Process with sliding window
            patches, coords = slideWindow_preprocess(images, window_size, stride)
            image_paths = batch['image_path']
            label_paths = batch['label_path']
            # Predict and merge
            patches_preds = model(patches)['out']
            preds = slideWindow_merge(patches_preds, window_size, stride, coords, batch_size)
            # Process results
            pred_masks = [torch.argmax(pred, dim=0) for pred in preds]
            _pred_masks = torch.stack(pred_masks, dim=0)
            if None not in label_paths:
                labels = torch.stack([maskPath2tensor(path, device) for path in label_paths], dim=0) # [B, H, W]
                seg_metrics.update(_pred_masks, labels)
            pred_masks = [pred_mask.cpu().numpy() for pred_mask in pred_masks]
            postprocess(image_paths, pred_masks, label_paths, preds_dir, overlap_dir, palette)
    return seg_metrics.compute()


def run_inference(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                 output_dir: str, num_classes: int, device: torch.device,
                 resize_or_windowslide: str, input_size: int, ignore_index: int = 255) -> Dict[str, float]:
    """
    Main inference runner that dispatches to appropriate mode.
    
    Args:
        model: Loaded segmentation model
        dataloader: DataLoader providing input batches
        output_dir: Directory to save results
        num_classes: Number of segmentation classes
        device: Target device for computation
        resize_or_windowslide: Inference mode ('resize' or 'windowslide')
        input_size: Size parameter (resize dim or window size)
    """
    palette = create_color_palette(num_classes)
    os.makedirs(output_dir, exist_ok=True)
    seg_metrics = SegmentationMetrics(num_classes, device=device, ignore_index = ignore_index)
    
    if resize_or_windowslide == 'resize':
        metrics = resizeMode_inference(model, dataloader, device, output_dir, palette, seg_metrics)
    elif resize_or_windowslide == 'windowslide':
        metrics = slideWindowMode_inference(model, dataloader, device, output_dir, palette, seg_metrics, input_size)
    return metrics

def main() -> None:
    """Main execution function for inference script."""
    args = parse_args()
    log_dir = args.output_dir
    setup_logging(log_dir)
    logger = logging.getLogger(__name__)
    config = load_config(args.config)
    device = get_device(args.device)
    
    logger.info("Loading model...")
    model = load_model(config, args.checkpoint, device)
    
    logger.info("Loading transforms...")
    if args.resize_or_windowslide == 'resize':
        test_transforms = SegmentationTransforms.get_validation_transforms(
            img_size=args.input_size,
            mean=config['model']['mean'],
            std=config['model']['std']
        )
    elif args.resize_or_windowslide == 'windowslide':
        test_transforms = SegmentationTransforms.get_validation_transforms(
            img_size=None,
            mean=config['model']['mean'],
            std=config['model']['std']
        )

    logger.info("Preparing dataset...")
    test_dataset = JSONSegmentationDataset(
        json_file=args.input_json, split='test', transform=test_transforms)

    # Adjust batch size for sliding window if needed
    infer_batch_size = args.batch_size
    if args.resize_or_windowslide == 'windowslide' and not test_dataset.fixed_size:
        infer_batch_size = 1  # Force batch size 1 for variable size inputs

    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=infer_batch_size,
        shuffle=False,
        num_workers=config['system'].get('num_workers', 4),
        pin_memory=config['system'].get('pin_memory', True),
        drop_last=False
    )

    logger.info("Running inference...")
    metrics = run_inference(
        model, test_dataloader, args.output_dir, 
        config['model']['num_classes'], device, 
        args.resize_or_windowslide, args.input_size,
        config['dataset'].get('ignore_index')  
    )
    logger.info("Inference completed successfully.")
    logger.info(f'Metrics:{metrics}')
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    main()
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
from models import create_segmentation_model
from utils.visualization import apply_color_map, create_color_palette, put_text_with_bg
from utils.logs import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for inference configuration."""
    parser = argparse.ArgumentParser(description='Semantic Segmentation Inference Script')
    parser.add_argument('--config', type=str, 
                       default='/mnt/sdb/chenwm/PFM_Segmentation/configs/test.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, 
                       default='/mnt/sdb/chenwm/PFM_Segmentation/logs/test/checkpoints',
                       help='Path to model checkpoint file or checkpoint directory. '
                            'For LoRA/DoRA mode, will automatically load both base model and LoRA/DoRA weights.')
    parser.add_argument('--input_json', type=str, 
                       default='/mnt/sdb/chenwm/PFM_Segmentation/dataset_json/TNBC.json',
                       help='Path to JSON file containing input data')
    parser.add_argument('--output_dir', type=str, 
                       default='/mnt/sdb/chenwm/PFM_Segmentation/inference_slidewindow',
                       help='Directory to save inference results')
    parser.add_argument('--device', type=str, default='cuda:7',
                       help='Device for inference (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--input_size', type=int, default=224,
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


def resolve_checkpoint_paths(checkpoint_path: str, finetune_mode: str) -> Tuple[str, str]:
    """
    Resolve checkpoint paths for model weights and LoRA/DoRA weights based on finetune mode.
    
    Args:
        checkpoint_path: Path to checkpoint file or checkpoint directory
        finetune_mode: Finetune mode ('full', 'frozen', 'lora', 'dora')
        
    Returns:
        Tuple of (model_path, lora_dora_path)
        lora_dora_path will be None if not in LoRA/DoRA mode or file doesn't exist
    """
    if finetune_mode == 'full':
        # Full mode: use best_full_model.pth
        expected_filename = 'best_full_model.pth'
    else:
        # Non-full mode: use best_decoder_head.pth
        expected_filename = 'best_decoder_head.pth'
    
    if os.path.isdir(checkpoint_path):
        # If it's a directory, look for model file and LoRA/DoRA file
        model_path = os.path.join(checkpoint_path, expected_filename)
        lora_dora_path = os.path.join(checkpoint_path, 'best_lora_dora_weights.pth')
        if not os.path.exists(lora_dora_path):
            lora_dora_path = None
    else:
        # If it's a file, use it as model path and look for LoRA/DoRA in same directory
        model_path = checkpoint_path
        checkpoint_dir = os.path.dirname(checkpoint_path)
        # Verify filename matches expected pattern
        filename = os.path.basename(checkpoint_path)
        if finetune_mode == 'full' and 'best_full_model' not in filename:
            logging.warning(f'Expected filename containing "best_full_model" for full mode, got: {filename}')
        elif finetune_mode != 'full' and 'best_decoder_head' not in filename:
            logging.warning(f'Expected filename containing "best_decoder_head" for {finetune_mode} mode, got: {filename}')
        
        lora_dora_path = os.path.join(checkpoint_dir, 'best_lora_dora_weights.pth')
        if not os.path.exists(lora_dora_path):
            lora_dora_path = None
    
    return model_path, lora_dora_path


def load_model(config: Dict[str, Any], checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load model from checkpoint with configuration.
    For full mode: loads entire model from best_full_model.pth
    For non-full modes (frozen, lora, dora): loads only decoder+head from best_decoder_head.pth
    For LoRA/DoRA mode: additionally loads LoRA/DoRA weights from best_lora_dora_weights.pth
    
    Args:
        config: Model configuration dictionary
        checkpoint_path: Path to checkpoint file or checkpoint directory
        device: Target device for model
        
    Returns:
        Loaded and configured model in evaluation mode
    """
    # Get finetune mode from config
    finetune_mode = config.get('model', {}).get('finetune_mode', {}).get('type', None)
    if finetune_mode is None:
        logging.warning('Finetune mode not specified in config, defaulting to full mode')
        finetune_mode = 'full'
    
    is_lora_dora_mode = finetune_mode in ['lora', 'dora']
    is_full_mode = finetune_mode == 'full'
    
    # Create model (PFM weights are loaded during model creation)
    model = create_segmentation_model(config['model']).to(device)
    
    # Resolve checkpoint paths based on finetune mode
    model_path, lora_dora_path = resolve_checkpoint_paths(checkpoint_path, finetune_mode)
    
    # Verify checkpoint file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'Model checkpoint not found: {model_path}\n'
            f'Expected filename for {finetune_mode} mode: '
            f'{"best_full_model.pth" if is_full_mode else "best_decoder_head.pth"}'
        )
    
    # Load checkpoint
    logging.info(f'Loading checkpoint from: {model_path}')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Verify checkpoint matches finetune mode
    checkpoint_finetune_mode = checkpoint.get('finetune_mode', None)
    if checkpoint_finetune_mode is not None and checkpoint_finetune_mode != finetune_mode:
        raise ValueError(
            f'Checkpoint finetune mode mismatch: checkpoint has "{checkpoint_finetune_mode}", '
            f'but config specifies "{finetune_mode}"'
        )
    
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    if is_full_mode:
        # Full mode: load entire model
        model.load_state_dict(model_state_dict, strict=True)
        logging.info('Full model loaded successfully')
    else:
        # Non-full mode: load only decoder and segmentation_head
        model_state = model.state_dict()
        loaded_params = 0
        missing_params = []
        
        for name, param in model_state_dict.items():
            if name.startswith('decoder.') or name.startswith('segmentation_head.'):
                if name in model_state:
                    model_state[name] = param.to(device)
                    loaded_params += 1
                else:
                    missing_params.append(name)
            elif not name.startswith('pfm.'):
                # Allow loading other non-PFM parameters (e.g., if there are any)
                if name in model_state:
                    model_state[name] = param.to(device)
                    loaded_params += 1
        
        if missing_params:
            logging.warning(f'Some decoder/head parameters not found in model: {missing_params}')
        
        model.load_state_dict(model_state, strict=False)
        logging.info(f'Decoder+head loaded successfully: {loaded_params} parameters')
    
    # Load LoRA/DoRA weights if in LoRA/DoRA mode
    if is_lora_dora_mode and lora_dora_path and os.path.exists(lora_dora_path):
        logging.info(f'Loading LoRA/DoRA weights from: {lora_dora_path}')
        lora_checkpoint = torch.load(lora_dora_path, map_location=device)
        
        if 'lora_dora_state_dict' in lora_checkpoint:
            lora_dora_params = lora_checkpoint['lora_dora_state_dict']
            model_state = model.state_dict()
            
            # Update only LoRA/DoRA parameters
            loaded_params = 0
            for name, param in lora_dora_params.items():
                if name in model_state:
                    model_state[name] = param.to(device)
                    loaded_params += 1
                else:
                    logging.warning(f'LoRA/DoRA parameter {name} not found in model')
            
            model.load_state_dict(model_state, strict=False)
            logging.info(f'LoRA/DoRA weights loaded successfully: {loaded_params} parameters')
        else:
            logging.warning('LoRA/DoRA checkpoint does not contain lora_dora_state_dict')
    elif is_lora_dora_mode:
        logging.warning(f'LoRA/DoRA weights not found at {lora_dora_path}, using base model only')
    
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
    # Get normalization values based on model name
    from data.transforms import get_model_normalization
    pfm_name = config['model'].get('pfm_name', 'unet')
    mean, std = get_model_normalization(pfm_name)
    
    if args.resize_or_windowslide == 'resize':
        test_transforms = SegmentationTransforms.get_validation_transforms(
            img_size=args.input_size,
            mean=mean,
            std=std
        )
    elif args.resize_or_windowslide == 'windowslide':
        test_transforms = SegmentationTransforms.get_validation_transforms(
            img_size=None,
            mean=mean,
            std=std
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
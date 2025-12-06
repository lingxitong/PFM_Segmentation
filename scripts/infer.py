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
from torch.cuda.amp import autocast
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
                       default='/mnt/sdb/chenwm/PFM_Segmentation/logs/test_12-07/checkpoints',
                       help='Path to model checkpoint file or checkpoint directory')
    parser.add_argument('--input_json', type=str, 
                       default='/mnt/sdb/chenwm/PFM_Segmentation/dataset_json/GlaS.json',
                       help='Path to JSON file containing input data')
    parser.add_argument('--output_dir', type=str, 
                       default='/mnt/sdb/chenwm/PFM_Segmentation/inference_slidewindow',
                       help='Directory to save inference results')
    parser.add_argument('--device', type=str, default='cuda:6',
                       help='Device for inference (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--input_size', type=int, default=416,
                       help='Input size for resize or window size for sliding window')
    parser.add_argument('--resize_or_windowslide', type=str, 
                       choices=['resize', 'windowslide'], default='resize',
                       help='Inference mode: resize or sliding window')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for inference')
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_device(device_str: str) -> torch.device:
    """Get PyTorch device from string descriptor."""
    return torch.device(device_str if torch.cuda.is_available() else 'cpu')


def resolve_checkpoint_paths(checkpoint_path: str, finetune_mode: str) -> str:
    """
    Resolve checkpoint path for model weights based on finetune mode.
    
    Args:
        checkpoint_path: Path to checkpoint file or checkpoint directory
        finetune_mode: Finetune mode ('full', 'frozen', 'lora', 'dora', 'cnn_adapter', 'transformer_adapter')
        
    Returns:
        str: Path to model checkpoint file
    """
    if finetune_mode == 'full':
        # Full mode: use best_full_model.pth
        expected_filename = 'best_full_model.pth'
    elif finetune_mode == 'cnn_adapter':
        # CNN adapter mode: use best_cnn_adapter_and_decoder_head.pth
        expected_filename = 'best_cnn_adapter_and_decoder_head.pth'
    elif finetune_mode == 'transformer_adapter':
        # Transformer adapter mode: use best_transformer_adapter_and_decoder_head.pth
        expected_filename = 'best_transformer_adapter_and_decoder_head.pth'
    elif finetune_mode == 'lora':
        # LoRA mode: use best_lora_and_decoder_head.pth
        expected_filename = 'best_lora_and_decoder_head.pth'
    elif finetune_mode == 'dora':
        # DoRA mode: use best_dora_and_decoder_head.pth
        expected_filename = 'best_dora_and_decoder_head.pth'
    elif finetune_mode == 'frozen':
        # Frozen mode: use best_decoder_head.pth
        expected_filename = 'best_decoder_head.pth'
    else:
        raise ValueError(f"Unknown finetune mode: {finetune_mode}")
    
    if os.path.isdir(checkpoint_path):
        # If it's a directory, look for model file
        model_path = os.path.join(checkpoint_path, expected_filename)
    else:
        # If it's a file, use it as model path
        model_path = checkpoint_path
        # Verify filename matches expected pattern
        filename = os.path.basename(checkpoint_path)
        if finetune_mode == 'full' and 'best_full_model' not in filename:
            logging.warning(f'Expected filename containing "best_full_model" for full mode, got: {filename}')
        elif finetune_mode == 'cnn_adapter' and 'best_cnn_adapter_and_decoder_head' not in filename:
            logging.warning(f'Expected filename containing "best_cnn_adapter_and_decoder_head" for cnn_adapter mode, got: {filename}')
        elif finetune_mode == 'transformer_adapter' and 'best_transformer_adapter_and_decoder_head' not in filename:
            logging.warning(f'Expected filename containing "best_transformer_adapter_and_decoder_head" for transformer_adapter mode, got: {filename}')
        elif finetune_mode == 'lora' and 'best_lora_and_decoder_head' not in filename:
            logging.warning(f'Expected filename containing "best_lora_and_decoder_head" for lora mode, got: {filename}')
        elif finetune_mode == 'dora' and 'best_dora_and_decoder_head' not in filename:
            logging.warning(f'Expected filename containing "best_dora_and_decoder_head" for dora mode, got: {filename}')
        elif finetune_mode == 'frozen' and 'best_decoder_head' not in filename:
            logging.warning(f'Expected filename containing "best_decoder_head" for frozen mode, got: {filename}')
    
    return model_path


def load_model(config: Dict[str, Any], checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load model from checkpoint with configuration.
    For full mode: loads entire model from best_full_model.pth
    For cnn_adapter mode: loads CNN adapter+decoder+head from best_cnn_adapter_and_decoder_head.pth
    For transformer_adapter mode: loads Transformer adapter+decoder+head from best_transformer_adapter_and_decoder_head.pth
    For lora/dora mode: loads LoRA/DoRA+decoder+head from best_lora_dora_and_decoder_head.pth
    For frozen mode: loads only decoder+head from best_decoder_head.pth
    
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
    
    # Create model (PFM weights are loaded during model creation)
    model = create_segmentation_model(config['model']).to(device)
    
    # Resolve checkpoint path based on finetune mode
    model_path = resolve_checkpoint_paths(checkpoint_path, finetune_mode)
    
    # Verify checkpoint file exists
    if not os.path.exists(model_path):
        if finetune_mode=='full':
            expected_file = 'best_full_model.pth'
        elif finetune_mode=='cnn_adapter':
            expected_file = 'best_cnn_adapter_and_decoder_head.pth'
        elif finetune_mode=='transformer_adapter':
            expected_file = 'best_transformer_adapter_and_decoder_head.pth'
        elif finetune_mode == 'lora':
            expected_file = 'best_lora_and_decoder_head.pth'
        elif finetune_mode == 'dora':
            expected_file = 'best_dora_and_decoder_head.pth'
        elif finetune_mode == 'frozen':
            expected_file = 'best_decoder_head.pth'
        else:
            expected_file = 'unknown'
        raise FileNotFoundError(
            f'Model checkpoint not found: {model_path}\n'
            f'Expected filename for {finetune_mode} mode: {expected_file}'
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
    
    if finetune_mode=='full':
        # Full mode: load entire model
        model.load_state_dict(model_state_dict, strict=True)
        logging.info('Full model loaded successfully')
    elif finetune_mode=='cnn_adapter':
        # CNN adapter mode: load cnn_adapter + decoder + segmentation_head
        model_state = model.state_dict()
        loaded_params = 0
        missing_params = []
        
        for name, param in model_state_dict.items():
            if name.startswith('cnn_adapter.') or name.startswith('decoder.') or name.startswith('segmentation_head.'):
                if name in model_state:
                    model_state[name] = param.to(device)
                    loaded_params += 1
                else:
                    missing_params.append(name)
            elif not name.startswith('pfm.'):
                # Allow loading other non-PFM parameters
                if name in model_state:
                    model_state[name] = param.to(device)
                    loaded_params += 1
        
        if missing_params:
            logging.warning(f'Some CNN adapter/decoder/head parameters not found in model: {missing_params}')
        
        model.load_state_dict(model_state, strict=False)
        logging.info(f'CNN adapter+decoder+head loaded successfully: {loaded_params} parameters')
    elif finetune_mode=='transformer_adapter':
        # Transformer adapter mode: load transformer_adapter + decoder + segmentation_head
        model_state = model.state_dict()
        loaded_params = 0
        missing_params = []
        
        for name, param in model_state_dict.items():
            if name.startswith('transformer_adapter.') or name.startswith('decoder.') or name.startswith('segmentation_head.'):
                if name in model_state:
                    model_state[name] = param.to(device)
                    loaded_params += 1
                else:
                    missing_params.append(name)
            elif not name.startswith('pfm.'):
                # Allow loading other non-PFM parameters (e.g., _transformer_adapter_skip_tokens)
                if name in model_state:
                    model_state[name] = param.to(device)
                    loaded_params += 1
        
        if missing_params:
            logging.warning(f'Some Transformer adapter/decoder/head parameters not found in model: {missing_params}')
        
        model.load_state_dict(model_state, strict=False)
        logging.info(f'Transformer adapter+decoder+head loaded successfully: {loaded_params} parameters')
    elif finetune_mode == 'lora':
        # LoRA mode: load lora + decoder + segmentation_head together
        model_state = model.state_dict()
        loaded_params = 0
        missing_params = []
        
        for name, param in model_state_dict.items():
            # Load lora parameters (lora_a, lora_b in pfm module) and decoder/segmentation_head
            if ('lora_a' in name or 'lora_b' in name or 
                name.startswith('decoder.') or name.startswith('segmentation_head.')):
                if name in model_state:
                    model_state[name] = param.to(device)
                    loaded_params += 1
                else:
                    missing_params.append(name)
            elif not name.startswith('pfm.'):
                # Allow loading other non-PFM parameters
                if name in model_state:
                    model_state[name] = param.to(device)
                    loaded_params += 1
        
        if missing_params:
            logging.warning(f'Some LoRA/decoder/head parameters not found in model: {missing_params}')
        
        model.load_state_dict(model_state, strict=False)
        logging.info(f'LoRA+decoder+head loaded successfully: {loaded_params} parameters')
    elif finetune_mode == 'dora':
        # DoRA mode: load dora + decoder + segmentation_head together
        model_state = model.state_dict()
        loaded_params = 0
        missing_params = []
        
        for name, param in model_state_dict.items():
            # Load dora parameters (lora_a, lora_b, m in pfm module) and decoder/segmentation_head
            # Note: use name.endswith('.m') to avoid matching .mlp or other parameters containing 'm'
            if ('lora_a' in name or 'lora_b' in name or name.endswith('.m') or 
                name.startswith('decoder.') or name.startswith('segmentation_head.')):
                if name in model_state:
                    model_state[name] = param.to(device)
                    loaded_params += 1
                else:
                    missing_params.append(name)
            elif not name.startswith('pfm.'):
                # Allow loading other non-PFM parameters
                if name in model_state:
                    model_state[name] = param.to(device)
                    loaded_params += 1
        
        if missing_params:
            logging.warning(f'Some DoRA/decoder/head parameters not found in model: {missing_params}')
        
        model.load_state_dict(model_state, strict=False)
        logging.info(f'DoRA+decoder+head loaded successfully: {loaded_params} parameters')
    elif finetune_mode == 'frozen':
        # Frozen mode: load only decoder and segmentation_head
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
    else:
        raise ValueError(f"Unknown finetune mode: {finetune_mode}")
    
    model.eval()
    return model


def postprocess(image_paths: List[str], pred_masks: List[np.ndarray], 
               label_paths: List[str], preds_dir: str, overlap_dir: str, 
               palette: np.ndarray, resize_to_pred_size: bool = False) -> None:
    """
    Post-process and visualize inference results.
    
    Args:
        image_paths: List of input image paths
        pred_masks: List of predicted masks (2D numpy arrays)
        label_paths: List of ground truth label paths
        preds_dir: Directory to save prediction masks
        overlap_dir: Directory to save visualization overlays
        palette: Color palette for visualization
        resize_to_pred_size: If True, resize original images and labels to match prediction mask size (for resize mode)
    """
    for i in range(len(image_paths)):
        # Process predicted mask
        pred_mask = pred_masks[i]
        pred_h, pred_w = pred_mask.shape[:2]

        # Apply color mapping
        pred_colored = apply_color_map(pred_mask, palette)


        # Save prediction mask
        Image.fromarray(pred_mask.astype(np.uint8)).save(
            os.path.join(preds_dir, os.path.basename(image_paths[i])))

        if label_paths[i] is not None:
            # Load and process original image
            original_image = Image.open(image_paths[i]).convert('RGB')
            original_np = np.array(original_image)
            
            # Resize original image to match prediction mask size if needed (only in resize mode)
            if resize_to_pred_size and original_np.shape[:2] != (pred_h, pred_w):
                original_np = cv2.resize(original_np, (pred_w, pred_h), interpolation=cv2.INTER_LINEAR)

            # Create overlays
            label_mask = np.array(Image.open(label_paths[i]))
            # Resize label mask to match prediction mask size if needed (only in resize mode)
            if resize_to_pred_size and label_mask.shape[:2] != (pred_h, pred_w):
                label_mask = cv2.resize(label_mask, (pred_w, pred_h), interpolation=cv2.INTER_NEAREST)
            
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
                        device: torch.device, output_dir: str, palette: np.ndarray, seg_metrics: SegmentationMetrics,
                        use_amp: bool = False) -> None:
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
            # Get resize后的尺寸 (H, W)
            _, _, H, W = images.shape
            # Forward pass with optional mixed precision
            if use_amp:
                with autocast():
                    preds = model(images)['out']
            else:
                preds = model(images)['out']
            
            # Process predictions - 直接使用resize后的尺寸，不缩放回原图
            pred_masks = [torch.argmax(pred, dim=0).cpu().numpy() for pred in preds]
            _pred_masks = [torch.tensor(mask) for mask in pred_masks]
            if None not in label_paths:
                # 使用dataloader中已经resize好的标签，确保与训练时一致
                # 注意：dataloader返回的label已经是resize后的尺寸
                labels = batch['label'].to(device)  # [B, H, W]
                seg_metrics.update(torch.stack(_pred_masks, dim=0).to(device), labels)

            # Save results - 在resize模式下，需要resize原始图像和标签到预测尺寸
            postprocess(image_paths, pred_masks, label_paths, preds_dir, overlap_dir, palette, resize_to_pred_size=True)
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
                             window_size: int, overlap: float = 0.2, use_amp: bool = False) -> SegmentationMetrics:
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
            # Predict and merge with optional mixed precision
            if use_amp:
                with autocast():
                    patches_preds = model(patches)['out']
            else:
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
                 resize_or_windowslide: str, input_size: int, ignore_index: int = 255,
                 use_amp: bool = False) -> Dict[str, float]:
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
        metrics = resizeMode_inference(model, dataloader, device, output_dir, palette, seg_metrics, use_amp)
    elif resize_or_windowslide == 'windowslide':
        metrics = slideWindowMode_inference(model, dataloader, device, output_dir, palette, seg_metrics, input_size, use_amp=use_amp)
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
    # Get use_amp from config (default to training.use_amp if available, otherwise False)
    use_amp = config.get('training', {}).get('use_amp', False)
    logger.info(f'Mixed precision inference: {use_amp}')
    metrics = run_inference(
        model, test_dataloader, args.output_dir, 
        config['model']['num_classes'], device, 
        args.resize_or_windowslide, args.input_size,
        config['dataset'].get('ignore_index'),
        use_amp=use_amp
    )
    logger.info("Inference completed successfully.")
    logger.info(f'Metrics:{metrics}')
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    main()
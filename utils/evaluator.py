"""
Model Evaluator for Semantic Segmentation

This module contains comprehensive evaluation utilities including
model testing, inference with TTA, and detailed analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm import tqdm
import json
import cv2
from PIL import Image

from .metrics import SegmentationMetrics, StreamingMetrics
from .visualization import visualize_prediction, apply_color_map, create_color_palette


class SegmentationEvaluator:
    """
    Comprehensive evaluator for semantic segmentation models.
    
    Args:
        model (nn.Module): Segmentation model
        device (str): Device for evaluation
        num_classes (int): Number of classes
        ignore_index (int): Index to ignore in evaluation
        class_names (Optional[List[str]]): Names of classes
        class_colors (Optional[List[List[int]]]): Colors for each class
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda',
                 num_classes: int = 19, ignore_index: int = 255,
                 class_names: Optional[List[str]] = None,
                 class_colors: Optional[List[List[int]]] = None):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        
        # Create color palette
        if class_colors is not None:
            self.color_palette = np.array(class_colors, dtype=np.uint8)
        else:
            self.color_palette = create_color_palette(num_classes)
        
        # Initialize metrics
        self.metrics = SegmentationMetrics(num_classes, ignore_index, device)
        
    def evaluate_dataset(self, data_loader: DataLoader, 
                        use_tta: bool = False,
                        tta_scales: List[float] = [0.75, 1.0, 1.25],
                        tta_flip: bool = True,
                        save_predictions: bool = False,
                        output_dir: str = "eval_results") -> Dict[str, Any]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader (DataLoader): Data loader for evaluation
            use_tta (bool): Whether to use Test Time Augmentation
            tta_scales (List[float]): Scales for TTA
            tta_flip (bool): Whether to use horizontal flip in TTA
            save_predictions (bool): Whether to save predictions
            output_dir (str): Output directory for results
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        self.model.eval()
        self.metrics.reset()
        
        if save_predictions:
            os.makedirs(output_dir, exist_ok=True)
            pred_dir = os.path.join(output_dir, 'predictions')
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(pred_dir, exist_ok=True)
            os.makedirs(vis_dir, exist_ok=True)
        
        total_time = 0
        num_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc='Evaluating')
            
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                start_time = time.time()
                
                if use_tta:
                    predictions = self._predict_with_tta(images, tta_scales, tta_flip)
                else:
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        predictions = outputs['out']
                    else:
                        predictions = outputs
                
                inference_time = time.time() - start_time
                total_time += inference_time
                num_samples += len(images)
                
                # Update metrics
                self.metrics.update(predictions, labels)
                
                # Save predictions and visualizations
                if save_predictions:
                    self._save_batch_predictions(
                        batch, predictions, batch_idx, pred_dir, vis_dir
                    )
                
                # Update progress bar
                current_metrics = self.metrics.compute()
                pbar.set_postfix({
                    'mIoU': f'{current_metrics["mIoU"]:.4f}',
                    'Pixel Acc': f'{current_metrics["Pixel_Accuracy"]:.4f}'
                })
        
        # Compute final metrics
        final_metrics = self.metrics.compute()
        
        # Add timing information
        final_metrics['inference_time_per_sample'] = total_time / num_samples
        final_metrics['fps'] = num_samples / total_time
        
        # Save metrics
        if save_predictions:
            self._save_evaluation_results(final_metrics, output_dir)
        
        return final_metrics
    
    def _predict_with_tta(self, images: torch.Tensor, 
                         scales: List[float], use_flip: bool) -> torch.Tensor:
        """
        Perform prediction with Test Time Augmentation.
        
        Args:
            images (torch.Tensor): Input images
            scales (List[float]): Scale factors
            use_flip (bool): Whether to use horizontal flip
            
        Returns:
            torch.Tensor: Averaged predictions
        """
        b, c, h, w = images.shape
        
        # Initialize aggregated predictions
        aggregated_preds = torch.zeros(b, self.num_classes, h, w, device=self.device)
        num_predictions = 0
        
        for scale in scales:
            # Resize images
            scaled_h, scaled_w = int(h * scale), int(w * scale)
            scaled_images = F.interpolate(
                images, size=(scaled_h, scaled_w), 
                mode='bilinear', align_corners=False
            )
            
            # Normal prediction
            outputs = self.model(scaled_images)
            if isinstance(outputs, dict):
                preds = outputs['out']
            else:
                preds = outputs
            
            # Resize back to original size
            preds = F.interpolate(
                preds, size=(h, w), 
                mode='bilinear', align_corners=False
            )
            aggregated_preds += preds
            num_predictions += 1
            
            # Flipped prediction
            if use_flip:
                flipped_images = torch.flip(scaled_images, dims=[3])
                outputs = self.model(flipped_images)
                if isinstance(outputs, dict):
                    preds = outputs['out']
                else:
                    preds = outputs
                
                # Flip back and resize
                preds = torch.flip(preds, dims=[3])
                preds = F.interpolate(
                    preds, size=(h, w), 
                    mode='bilinear', align_corners=False
                )
                aggregated_preds += preds
                num_predictions += 1
        
        # Average predictions
        averaged_preds = aggregated_preds / num_predictions
        
        return averaged_preds
    
    def _save_batch_predictions(self, batch: Dict[str, torch.Tensor],
                               predictions: torch.Tensor, batch_idx: int,
                               pred_dir: str, vis_dir: str) -> None:
        """Save batch predictions and visualizations."""
        batch_size = len(batch['image'])
        
        for i in range(batch_size):
            sample_idx = batch_idx * batch_size + i
            
            # Get data
            image = batch['image'][i].cpu()
            label = batch['label'][i].cpu()
            pred = torch.argmax(predictions[i], dim=0).cpu()
            confidence = torch.max(torch.softmax(predictions[i], dim=0), dim=0)[0].cpu()
            
            # Save prediction mask
            pred_path = os.path.join(pred_dir, f'prediction_{sample_idx:06d}.png')
            pred_image = Image.fromarray(pred.numpy().astype(np.uint8))
            pred_image.save(pred_path)
            
            # Save visualization
            vis_path = os.path.join(vis_dir, f'visualization_{sample_idx:06d}.png')
            visualize_prediction(
                image=image,
                label=label,
                prediction=pred,
                confidence=confidence,
                color_palette=self.color_palette,
                save_path=vis_path
            )
    
    def _save_evaluation_results(self, metrics: Dict[str, float], output_dir: str) -> None:
        """Save evaluation results to files."""
        # Save metrics as JSON
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save detailed per-class metrics
        detailed_metrics = {}
        for i in range(self.num_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
            detailed_metrics[class_name] = {
                'IoU': metrics.get(f'IoU_Class_{i}', 0.0),
                'Dice': metrics.get(f'Dice_Class_{i}', 0.0),
                'Precision': metrics.get(f'Precision_Class_{i}', 0.0),
                'Recall': metrics.get(f'Recall_Class_{i}', 0.0),
                'F1': metrics.get(f'F1_Class_{i}', 0.0)
            }
        
        detailed_path = os.path.join(output_dir, 'detailed_metrics.json')
        with open(detailed_path, 'w') as f:
            json.dump(detailed_metrics, f, indent=2)
        
        # Save confusion matrix
        confusion_matrix = self.metrics.get_confusion_matrix()
        np.save(os.path.join(output_dir, 'confusion_matrix.npy'), confusion_matrix)
        
        print(f"Evaluation results saved to: {output_dir}")
    
    def evaluate_single_image(self, image_path: str, 
                             use_tta: bool = False,
                             save_result: bool = True,
                             output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate model on a single image.
        
        Args:
            image_path (str): Path to input image
            use_tta (bool): Whether to use TTA
            save_result (bool): Whether to save result
            output_path (Optional[str]): Output path for result
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        self.model.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Convert to tensor (assuming normalization is handled in transforms)
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            start_time = time.time()
            
            if use_tta:
                predictions = self._predict_with_tta(
                    image_tensor, scales=[0.75, 1.0, 1.25], use_flip=True
                )
            else:
                outputs = self.model(image_tensor)
                if isinstance(outputs, dict):
                    predictions = outputs['out']
                else:
                    predictions = outputs
            
            inference_time = time.time() - start_time
        
        # Process predictions
        pred_mask = torch.argmax(predictions[0], dim=0).cpu().numpy()
        confidence_map = torch.max(torch.softmax(predictions[0], dim=0), dim=0)[0].cpu().numpy()
        
        # Create colored prediction
        colored_pred = apply_color_map(pred_mask, self.color_palette, self.ignore_index)
        
        results = {
            'prediction_mask': pred_mask,
            'confidence_map': confidence_map,
            'colored_prediction': colored_pred,
            'inference_time': inference_time,
            'original_size': original_size
        }
        
        # Save results
        if save_result:
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = f"{base_name}_prediction.png"
            
            # Save colored prediction
            colored_pred_image = Image.fromarray(colored_pred)
            colored_pred_image.save(output_path)
            
            # Save raw prediction mask
            mask_path = output_path.replace('.png', '_mask.png')
            mask_image = Image.fromarray(pred_mask.astype(np.uint8))
            mask_image.save(mask_path)
            
            print(f"Results saved to: {output_path}")
        
        return results
    
    def benchmark_model(self, data_loader: DataLoader,
                       num_warmup: int = 10,
                       num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark model performance.
        
        Args:
            data_loader (DataLoader): Data loader for benchmarking
            num_warmup (int): Number of warmup iterations
            num_iterations (int): Number of benchmark iterations
            
        Returns:
            Dict[str, float]: Benchmark results
        """
        self.model.eval()
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_warmup:
                    break
                
                images = batch['image'].to(self.device, non_blocking=True)
                outputs = self.model(images)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
        
        # Benchmark
        print("Benchmarking...")
        times = []
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_iterations:
                    break
                
                images = batch['image'].to(self.device, non_blocking=True)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                outputs = self.model(images)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        # Compute statistics
        times = np.array(times)
        batch_size = len(batch['image'])
        
        results = {
            'avg_batch_time': float(np.mean(times)),
            'std_batch_time': float(np.std(times)),
            'min_batch_time': float(np.min(times)),
            'max_batch_time': float(np.max(times)),
            'avg_sample_time': float(np.mean(times) / batch_size),
            'fps': float(batch_size / np.mean(times)),
            'throughput_samples_per_sec': float(num_iterations * batch_size / np.sum(times))
        }
        
        return results
    
    def analyze_failure_cases(self, data_loader: DataLoader,
                             iou_threshold: float = 0.3,
                             max_cases: int = 50,
                             output_dir: str = "failure_analysis") -> List[Dict[str, Any]]:
        """
        Analyze failure cases where model performs poorly.
        
        Args:
            data_loader (DataLoader): Data loader
            iou_threshold (float): IoU threshold below which samples are considered failures
            max_cases (int): Maximum number of failure cases to analyze
            output_dir (str): Output directory for analysis
            
        Returns:
            List[Dict[str, Any]]: List of failure case information
        """
        self.model.eval()
        failure_cases = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc='Analyzing failure cases')
            
            for batch_idx, batch in enumerate(pbar):
                if len(failure_cases) >= max_cases:
                    break
                
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    predictions = outputs['out']
                else:
                    predictions = outputs
                
                # Compute per-sample IoU
                batch_size = len(images)
                for i in range(batch_size):
                    sample_pred = predictions[i:i+1]
                    sample_label = labels[i:i+1]
                    
                    # Compute sample metrics
                    sample_metrics = SegmentationMetrics(self.num_classes, self.ignore_index, self.device)
                    sample_metrics.update(sample_pred, sample_label)
                    metrics = sample_metrics.compute()
                    
                    if metrics['mIoU'] < iou_threshold:
                        # Save failure case
                        sample_idx = len(failure_cases)
                        
                        case_info = {
                            'sample_index': sample_idx,
                            'batch_index': batch_idx,
                            'sample_in_batch': i,
                            'miou': metrics['mIoU'],
                            'pixel_accuracy': metrics['Pixel_Accuracy'],
                            'image_path': batch.get('image_path', [''])[i],
                            'label_path': batch.get('label_path', [''])[i]
                        }
                        
                        # Save visualization
                        vis_path = os.path.join(output_dir, f'failure_case_{sample_idx:03d}.png')
                        visualize_prediction(
                            image=images[i].cpu(),
                            label=labels[i].cpu(),
                            prediction=torch.argmax(predictions[i], dim=0).cpu(),
                            confidence=torch.max(torch.softmax(predictions[i], dim=0), dim=0)[0].cpu(),
                            color_palette=self.color_palette,
                            save_path=vis_path
                        )
                        
                        failure_cases.append(case_info)
        
        # Save failure case summary
        summary_path = os.path.join(output_dir, 'failure_cases_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(failure_cases, f, indent=2)
        
        print(f"Found {len(failure_cases)} failure cases. Analysis saved to: {output_dir}")
        
        return failure_cases


if __name__ == "__main__":
    # Test evaluator functionality
    print("Testing evaluator module...")
    
    # This would require actual model, data loaders, etc.
    # For now, just test imports
    print("Evaluator module loaded successfully!")

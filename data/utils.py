"""
Data Utilities for Semantic Segmentation

This module contains utility functions for data loading, preprocessing,
and dataset management.
"""

import torch
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os


def create_dataloader(dataset, batch_size: int = 8, shuffle: bool = True,
                     num_workers: int = 4, pin_memory: bool = True,
                     drop_last: bool = True, distributed: bool = False, generator = None, worker_init_fn = None) -> DataLoader:
    """
    Create DataLoader with appropriate settings.
    
    Args:
        dataset: PyTorch dataset
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of worker processes
        pin_memory (bool): Whether to pin memory
        drop_last (bool): Whether to drop last incomplete batch
        distributed (bool): Whether to use distributed training
        generator: Random number generator for reproducibility
        worker_init_fn (Callable): Function to initialize workers
        
    Returns:
        DataLoader: Configured data loader
    """
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Disable shuffle when using sampler
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler=sampler,
        generator=generator,
        collate_fn=segmentation_collate_fn,
        worker_init_fn=worker_init_fn,
    )


def segmentation_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for segmentation data.
    
    Args:
        batch (List[Dict[str, Any]]): List of sample dictionaries
        
    Returns:
        Dict[str, torch.Tensor]: Batched data
    """
    images = []
    labels = []
    ori_sizes = []
    image_paths = []
    label_paths = []
    
    for sample in batch:
        images.append(sample['image'])
        labels.append(sample['label'])
        image_paths.append(sample['image_path'])
        label_paths.append(sample['label_path'])
        ori_sizes.append(sample['ori_size'])
    
    # Stack images and labels
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    
    return {
        'image': images,
        'label': labels,
        'ori_size': ori_sizes,
        'image_path': image_paths,
        'label_path': label_paths
    }


def compute_class_distribution(dataset, num_classes: int, ignore_index: int = 255) -> Dict[str, Any]:
    """
    Compute class distribution statistics for a dataset.
    
    Args:
        dataset: Segmentation dataset
        num_classes (int): Number of classes
        ignore_index (int): Index to ignore in calculations
        
    Returns:
        Dict[str, Any]: Class distribution statistics
    """
    class_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0
    
    print("Computing class distribution...")
    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            print(f"Processed {i}/{len(dataset)} samples")
            
        label = sample['label'].numpy()
        mask = (label != ignore_index)
        
        for c in range(num_classes):
            class_counts[c] += np.sum(label == c)
        total_pixels += np.sum(mask)
    
    # Compute statistics
    class_frequencies = class_counts / total_pixels
    class_weights = 1.0 / (class_frequencies + 1e-8)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    return {
        'class_counts': class_counts,
        'class_frequencies': class_frequencies,
        'class_weights': class_weights,
        'total_pixels': total_pixels
    }


def visualize_class_distribution(class_stats: Dict[str, Any], class_names: Optional[List[str]] = None,
                               save_path: Optional[str] = None) -> None:
    """
    Visualize class distribution statistics.
    
    Args:
        class_stats (Dict[str, Any]): Class statistics from compute_class_distribution
        class_names (Optional[List[str]]): Names of classes
        save_path (Optional[str]): Path to save the plot
    """
    num_classes = len(class_stats['class_counts'])
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot class counts
    bars1 = ax1.bar(range(num_classes), class_stats['class_counts'])
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Pixel Count')
    ax1.set_title('Class Distribution (Pixel Counts)')
    ax1.set_xticks(range(num_classes))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=8)
    
    # Plot class frequencies
    bars2 = ax2.bar(range(num_classes), class_stats['class_frequencies'])
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Class Distribution (Frequencies)')
    ax2.set_xticks(range(num_classes))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to: {save_path}")
    
    plt.show()


def visualize_sample(sample: Dict[str, torch.Tensor], class_colors: Optional[List[List[int]]] = None,
                    class_names: Optional[List[str]] = None, save_path: Optional[str] = None) -> None:
    """
    Visualize a single sample with image and label overlay.
    
    Args:
        sample (Dict[str, torch.Tensor]): Sample containing image and label
        class_colors (Optional[List[List[int]]]): RGB colors for each class
        class_names (Optional[List[str]]): Names of classes
        save_path (Optional[str]): Path to save the visualization
    """
    image = sample['image']
    label = sample['label']
    
    # Convert tensors to numpy arrays
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:  # C, H, W
            image = image.permute(1, 2, 0)
        image = image.numpy()
        
    if isinstance(label, torch.Tensor):
        label = label.numpy()
    
    # Denormalize image if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Create colored label map
    if class_colors is None:
        # Generate random colors
        num_classes = int(label.max()) + 1
        class_colors = plt.cm.tab20(np.linspace(0, 1, num_classes))[:, :3] * 255
        class_colors = class_colors.astype(np.uint8)
    
    colored_label = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(class_colors):
        colored_label[label == class_id] = color
    
    # Create overlay
    alpha = 0.6
    overlay = cv2.addWeighted(image, 1 - alpha, colored_label, alpha, 0)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Label map
    axes[1].imshow(colored_label)
    axes[1].set_title('Label Map')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample visualization saved to: {save_path}")
    
    plt.show()


def create_color_map(num_classes: int) -> np.ndarray:
    """
    Create a color map for visualization.
    
    Args:
        num_classes (int): Number of classes
        
    Returns:
        np.ndarray: Color map of shape (num_classes, 3)
    """
    colors = []
    for i in range(num_classes):
        # Generate distinct colors using HSV space
        hue = i / num_classes
        saturation = 0.7 + 0.3 * (i % 2)  # Alternate between high and higher saturation
        value = 0.8 + 0.2 * ((i // 2) % 2)  # Alternate brightness
        
        # Convert HSV to RGB
        hsv = np.array([hue, saturation, value]).reshape(1, 1, 3)
        rgb = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
        colors.append(rgb)
    
    return np.array(colors)


def analyze_dataset_quality(dataset, sample_ratio: float = 0.1) -> Dict[str, Any]:
    """
    Analyze dataset quality metrics.
    
    Args:
        dataset: Segmentation dataset
        sample_ratio (float): Ratio of samples to analyze
        
    Returns:
        Dict[str, Any]: Quality analysis results
    """
    num_samples = int(len(dataset) * sample_ratio)
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    image_sizes = []
    label_coverage = []  # Percentage of labeled pixels
    class_diversity = []  # Number of unique classes per image
    
    print(f"Analyzing dataset quality on {num_samples} samples...")
    
    for i, idx in enumerate(indices):
        if i % 50 == 0:
            print(f"Processed {i}/{num_samples} samples")
            
        sample = dataset[idx]
        image = sample['image']
        label = sample['label']
        
        # Image size
        if isinstance(image, torch.Tensor):
            h, w = image.shape[-2:]
        else:
            h, w = image.shape[:2]
        image_sizes.append((h, w))
        
        # Label coverage
        if isinstance(label, torch.Tensor):
            label_np = label.numpy()
        else:
            label_np = label
            
        valid_pixels = np.sum(label_np != 255)  # Assuming 255 is ignore_index
        total_pixels = label_np.size
        coverage = valid_pixels / total_pixels
        label_coverage.append(coverage)
        
        # Class diversity
        unique_classes = len(np.unique(label_np[label_np != 255]))
        class_diversity.append(unique_classes)
    
    # Compute statistics
    unique_sizes = list(set(image_sizes))
    size_consistency = len(unique_sizes) == 1
    
    return {
        'num_samples_analyzed': num_samples,
        'unique_image_sizes': unique_sizes,
        'size_consistency': size_consistency,
        'avg_label_coverage': np.mean(label_coverage),
        'std_label_coverage': np.std(label_coverage),
        'avg_class_diversity': np.mean(class_diversity),
        'std_class_diversity': np.std(class_diversity),
        'label_coverage_distribution': label_coverage,
        'class_diversity_distribution': class_diversity
    }


def save_dataset_info(dataset, output_dir: str, dataset_name: str = "dataset") -> None:
    """
    Save comprehensive dataset information to files.
    
    Args:
        dataset: Segmentation dataset
        output_dir (str): Output directory
        dataset_name (str): Name of the dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic info
    info = {
        'dataset_name': dataset_name,
        'num_samples': len(dataset),
        'num_classes': getattr(dataset, 'num_classes', 'unknown'),
        'ignore_index': getattr(dataset, 'ignore_index', 255)
    }
    
    # Save basic info
    import json
    with open(os.path.join(output_dir, f'{dataset_name}_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    # Compute and save class distribution
    if hasattr(dataset, 'num_classes'):
        class_stats = compute_class_distribution(dataset, dataset.num_classes)
        
        # Save class statistics
        np.save(os.path.join(output_dir, f'{dataset_name}_class_counts.npy'), 
                class_stats['class_counts'])
        np.save(os.path.join(output_dir, f'{dataset_name}_class_weights.npy'), 
                class_stats['class_weights'])
        
        # Save class distribution plot
        visualize_class_distribution(
            class_stats, 
            save_path=os.path.join(output_dir, f'{dataset_name}_class_distribution.png')
        )
    
    # Dataset quality analysis
    quality_stats = analyze_dataset_quality(dataset)
    with open(os.path.join(output_dir, f'{dataset_name}_quality.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        quality_stats_serializable = {}
        for k, v in quality_stats.items():
            if isinstance(v, np.ndarray):
                quality_stats_serializable[k] = v.tolist()
            elif isinstance(v, np.float64):
                quality_stats_serializable[k] = float(v)
            else:
                quality_stats_serializable[k] = v
        json.dump(quality_stats_serializable, f, indent=2)
    
    print(f"Dataset information saved to: {output_dir}")


def create_data_split(image_dir: str, label_dir: str, 
                     train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1,
                     output_dir: str = "splits", seed: int = 42) -> None:
    """
    Create train/val/test splits for a custom dataset.
    
    Args:
        image_dir (str): Directory containing images
        label_dir (str): Directory containing labels
        train_ratio (float): Ratio for training set
        val_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
        output_dir (str): Output directory for split files
        seed (int): Random seed for reproducibility
    """
    import random
    import shutil
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Filter files that have corresponding labels
    valid_files = []
    for img_file in image_files:
        label_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
        if os.path.exists(os.path.join(label_dir, label_file)):
            valid_files.append(img_file)
    
    print(f"Found {len(valid_files)} valid image-label pairs")
    
    # Shuffle files
    random.seed(seed)
    random.shuffle(valid_files)
    
    # Compute split indices
    num_files = len(valid_files)
    train_end = int(num_files * train_ratio)
    val_end = train_end + int(num_files * val_ratio)
    
    # Split files
    train_files = valid_files[:train_end]
    val_files = valid_files[train_end:val_end]
    test_files = valid_files[val_end:]
    
    print(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
    
    # Copy files to respective directories
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for img_file in files:
            label_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
            
            # Copy image
            shutil.copy2(
                os.path.join(image_dir, img_file),
                os.path.join(output_dir, 'images', split, img_file)
            )
            
            # Copy label
            shutil.copy2(
                os.path.join(label_dir, label_file),
                os.path.join(output_dir, 'labels', split, label_file)
            )
    
    print(f"Data split created in: {output_dir}")


if __name__ == "__main__":
    # Test utilities
    print("Testing data utilities...")
    
    # Test color map creation
    color_map = create_color_map(19)
    print(f"Created color map with shape: {color_map.shape}")
    
    # Test other functions would require actual dataset
    print("Data utilities module loaded successfully!")

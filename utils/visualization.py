"""
Visualization Tools for Semantic Segmentation

This module contains comprehensive visualization tools for training progress,
predictions, metrics, and model analysis.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import os
from typing import List, Dict, Optional, Tuple, Any, Union
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def put_text_with_bg(img: np.ndarray, text: str, position: Tuple[int, int], 
                    font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2, 
                    text_color=(255, 255, 255), bg_color=(0, 0, 0)) -> None:
    """
    Draw text with background rectangle on image.
    
    Args:
        img: Input image (modified in-place)
        text: Text string to draw
        position: (x, y) coordinates of text bottom-left corner
        font: OpenCV font type
        font_scale: Font size scale factor
        thickness: Text thickness
        text_color: Text color in RGB
        bg_color: Background color in RGB
    """
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(img, (x, y - text_height - 10), (x + text_width + 10, y + 5), bg_color, -1)
    cv2.putText(img, text, (x + 5, y - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)


def apply_color_map(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Apply color palette to segmentation mask.
    
    Args:
        mask: 2D numpy array of shape (H, W) containing class indices
        palette: Color palette array of shape (num_classes, 3)
        
    Returns:
        Colorized mask as 3-channel RGB image
    """
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_mask[mask == label] = color
    return color_mask

def create_color_palette(num_classes: int, colormap: str = 'tab20') -> np.ndarray:
    """
    Create a color palette for segmentation visualization.
    
    Args:
        num_classes (int): Number of classes
        colormap (str): Matplotlib colormap name
        
    Returns:
        np.ndarray: Color palette of shape (num_classes, 3)
    """
    if num_classes <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:num_classes, :3]
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes))[:, :3]
    
    return (colors * 255).astype(np.uint8)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to image array.
    
    Args:
        tensor (torch.Tensor): Input tensor
        
    Returns:
        np.ndarray: Image array
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    if tensor.ndim == 3:  # C, H, W
        tensor = tensor.transpose(1, 2, 0)
    
    
    return tensor.astype(np.uint8)


def apply_color_map(label_map: np.ndarray, color_palette: np.ndarray, 
                   ignore_index: int = 255) -> np.ndarray:
    """
    Apply color mapping to label map.
    
    Args:
        label_map (np.ndarray): Label map
        color_palette (np.ndarray): Color palette
        ignore_index (int): Index to ignore
        
    Returns:
        np.ndarray: Colored label map
    """
    h, w = label_map.shape
    colored_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(len(color_palette)):
        mask = (label_map == class_id)
        colored_map[mask] = color_palette[class_id]
    
    # Set ignore pixels to black
    ignore_mask = (label_map == ignore_index)
    colored_map[ignore_mask] = [0, 0, 0]
    
    return colored_map


def visualize_prediction(image: Union[torch.Tensor, np.ndarray],
                        label: Union[torch.Tensor, np.ndarray],
                        prediction: Union[torch.Tensor, np.ndarray],
                        class_names: Optional[List[str]] = None,
                        color_palette: Optional[np.ndarray] = None,
                        save_path: Optional[str] = None,
                        alpha: float = 0.6) -> None:
    """
    Visualize prediction results.
    
    Args:
        image (Union[torch.Tensor, np.ndarray]): Input image
        label (Union[torch.Tensor, np.ndarray]): Ground truth label
        prediction (Union[torch.Tensor, np.ndarray]): Model prediction
        class_names (Optional[List[str]]): Class names
        color_palette (Optional[np.ndarray]): Color palette
        save_path (Optional[str]): Path to save visualization
        alpha (float): Overlay transparency
    """
    # setup_matplotlib_for_plotting()
    
    # Convert tensors to numpy arrays
    image = tensor_to_image(image)
    
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    
    # Create color palette if not provided
    if color_palette is None:
        num_classes = max(int(label.max()) + 1, int(prediction.max()) + 1)
        color_palette = create_color_palette(num_classes)
    
    # Apply color mapping
    colored_label = apply_color_map(label, color_palette)
    colored_prediction = apply_color_map(prediction, color_palette)
    
    # Create overlays
    label_overlay = cv2.addWeighted(image, 1 - alpha, colored_label, alpha, 0)
    pred_overlay = cv2.addWeighted(image, 1 - alpha, colored_prediction, alpha, 0)
    
    # Create visualization
    fig_width = 15
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, 5))
    
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # Ground truth overlay
    axes[1].imshow(label_overlay)
    axes[1].set_title('Ground Truth', fontsize=14)
    axes[1].axis('off')
    
    # Prediction overlay
    axes[2].imshow(pred_overlay)
    axes[2].set_title('Prediction', fontsize=14)
    axes[2].axis('off')
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 注释掉 plt.show() 以避免阻塞训练（在后台运行时不需要显示图像）
    # plt.show()
    plt.close()  # 关闭图形以释放内存


def save_predictions(predictions_data: List[Dict[str, torch.Tensor]], 
                    save_dir: str, epoch: int,
                    class_colors: Optional[List[List[int]]] = None) -> None:
    """
    Save prediction visualizations.
    
    Args:
        predictions_data (List[Dict[str, torch.Tensor]]): List of prediction data
        save_dir (str): Directory to save visualizations
        epoch (int): Current epoch
        class_colors (Optional[List[List[int]]]): Class colors
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create color palette
    if class_colors is not None:
        color_palette = np.array(class_colors, dtype=np.uint8)
    else:
        max_classes = max(int(data['label'].max()) + 1 for data in predictions_data)
        color_palette = create_color_palette(max_classes)
    
    for i, data in enumerate(predictions_data):
        save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_sample_{i:02d}.png')
        
        visualize_prediction(
            image=data['image'],
            label=data['label'],
            prediction=data['prediction'],
            color_palette=color_palette,
            save_path=save_path
        )
        plt.close()  # Close figure to free memory


def plot_training_history(train_losses: List[float], val_losses: List[float],
                         val_metrics: List[float], metric_name: str = 'mIoU',
                         save_path: Optional[str] = None) -> None:
    """
    Plot training history.
    
    Args:
        train_losses (List[float]): Training losses
        val_losses (List[float]): Validation losses
        val_metrics (List[float]): Validation metrics
        metric_name (str): Name of the metric
        save_path (Optional[str]): Path to save plot
    """
    # setup_matplotlib_for_plotting()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    # Align validation losses with training epochs
    if len(val_losses) > 0:
        val_epochs = np.linspace(1, len(train_losses), len(val_losses))
        ax1.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot metrics
    if len(val_metrics) > 0:
        val_epochs = np.linspace(1, len(train_losses), len(val_metrics))
        ax2.plot(val_epochs, val_metrics, 'g-', label=f'Validation {metric_name}', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric_name)
        ax2.set_title(f'Validation {metric_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add best score annotation
        best_idx = np.argmax(val_metrics)
        best_score = val_metrics[best_idx]
        best_epoch = val_epochs[best_idx]
        ax2.annotate(f'Best: {best_score:.4f}', 
                    xy=(best_epoch, best_score),
                    xytext=(best_epoch + len(train_losses) * 0.1, best_score),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=12, color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 注释掉 plt.show() 以避免阻塞训练（在后台运行时不需要显示图像）
    # plt.show()
    plt.close()  # 关闭图形以释放内存


def plot_confusion_matrix(confusion_matrix: np.ndarray, 
                         class_names: Optional[List[str]] = None,
                         normalize: bool = True, save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix (np.ndarray): Confusion matrix
        class_names (Optional[List[str]]): Class names
        normalize (bool): Whether to normalize the matrix
        save_path (Optional[str]): Path to save plot
    """
    # setup_matplotlib_for_plotting()
    
    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        cm = confusion_matrix
        title = 'Confusion Matrix'
        fmt = 'd'
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(confusion_matrix))]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 注释掉 plt.show() 以避免阻塞训练（在后台运行时不需要显示图像）
    # plt.show()
    plt.close()  # 关闭图形以释放内存


def plot_class_metrics(metrics_dict: Dict[str, float], 
                      class_names: Optional[List[str]] = None,
                      save_path: Optional[str] = None) -> None:
    """
    Plot per-class metrics.
    
    Args:
        metrics_dict (Dict[str, float]): Dictionary containing per-class metrics
        class_names (Optional[List[str]]): Class names
        save_path (Optional[str]): Path to save plot
    """
    # setup_matplotlib_for_plotting()
    
    # Extract per-class metrics
    iou_metrics = {k: v for k, v in metrics_dict.items() if k.startswith('IoU_Class_')}
    dice_metrics = {k: v for k, v in metrics_dict.items() if k.startswith('Dice_Class_')}
    
    if not iou_metrics:
        print("No per-class metrics found")
        return
    
    num_classes = len(iou_metrics)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    # Prepare data
    classes = range(num_classes)
    iou_values = [iou_metrics[f'IoU_Class_{i}'] for i in classes]
    dice_values = [dice_metrics[f'Dice_Class_{i}'] for i in classes] if dice_metrics else None
    
    # Create plot
    fig, axes = plt.subplots(1, 2 if dice_values else 1, figsize=(15, 6))
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    
    # IoU plot
    bars1 = axes[0].bar(classes, iou_values, color='skyblue', alpha=0.8)
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('IoU Score')
    axes[0].set_title('Per-Class IoU Scores')
    axes[0].set_xticks(classes)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Dice plot
    if dice_values and len(axes) > 1:
        bars2 = axes[1].bar(classes, dice_values, color='lightcoral', alpha=0.8)
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Dice Score')
        axes[1].set_title('Per-Class Dice Scores')
        axes[1].set_xticks(classes)
        axes[1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 注释掉 plt.show() 以避免阻塞训练（在后台运行时不需要显示图像）
    # plt.show()
    plt.close()  # 关闭图形以释放内存


def create_interactive_training_dashboard(training_stats: Dict[str, List[float]],
                                        save_path: Optional[str] = None) -> None:
    """
    Create an interactive training dashboard using Plotly.
    
    Args:
        training_stats (Dict[str, List[float]]): Training statistics
        save_path (Optional[str]): Path to save HTML file
    """
    train_losses = training_stats.get('train_losses', [])
    val_losses = training_stats.get('val_losses', [])
    val_mious = training_stats.get('val_mious', [])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Loss', 'Validation Loss', 'Validation mIoU', 'Loss Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": True}]]
    )
    
    epochs = list(range(1, len(train_losses) + 1))
    
    # Training loss
    fig.add_trace(
        go.Scatter(x=epochs, y=train_losses, mode='lines', name='Training Loss',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Validation loss
    if val_losses:
        val_epochs = np.linspace(1, len(train_losses), len(val_losses))
        fig.add_trace(
            go.Scatter(x=val_epochs, y=val_losses, mode='lines+markers', name='Validation Loss',
                      line=dict(color='red', width=2)),
            row=1, col=2
        )
    
    # Validation mIoU
    if val_mious:
        val_epochs = np.linspace(1, len(train_losses), len(val_mious))
        fig.add_trace(
            go.Scatter(x=val_epochs, y=val_mious, mode='lines+markers', name='Validation mIoU',
                      line=dict(color='green', width=2)),
            row=2, col=1
        )
    
    # Loss comparison
    fig.add_trace(
        go.Scatter(x=epochs, y=train_losses, mode='lines', name='Training Loss',
                  line=dict(color='blue', width=2)),
        row=2, col=2
    )
    
    if val_losses:
        val_epochs = np.linspace(1, len(train_losses), len(val_losses))
        fig.add_trace(
            go.Scatter(x=val_epochs, y=val_losses, mode='lines', name='Validation Loss',
                      line=dict(color='red', width=2)),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text="Training Dashboard",
        title_x=0.5,
        showlegend=True,
        height=800,
        template="plotly_white"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_yaxes(title_text="mIoU", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=2)
    
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to: {save_path}")
    
    # 注释掉 fig.show() 以避免阻塞训练（在后台运行时不需要显示图像）
    # fig.show()


def visualize_feature_maps(feature_maps: torch.Tensor, 
                          save_path: Optional[str] = None,
                          max_maps: int = 16) -> None:
    """
    Visualize feature maps from intermediate layers.
    
    Args:
        feature_maps (torch.Tensor): Feature maps of shape (B, C, H, W)
        save_path (Optional[str]): Path to save visualization
        max_maps (int): Maximum number of feature maps to visualize
    """
    # setup_matplotlib_for_plotting()
    
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = feature_maps.cpu().numpy()
    
    # Take first batch and limit number of maps
    if feature_maps.ndim == 4:
        feature_maps = feature_maps[0]  # Take first batch
    
    num_maps = min(feature_maps.shape[0], max_maps)
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_maps)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(num_maps):
        axes[i].imshow(feature_maps[i], cmap='viridis')
        axes[i].set_title(f'Feature Map {i}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_maps, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 注释掉 plt.show() 以避免阻塞训练（在后台运行时不需要显示图像）
    # plt.show()
    plt.close()  # 关闭图形以释放内存


if __name__ == "__main__":
    # Test visualization functions
    # setup_matplotlib_for_plotting()
    
    # Create dummy data
    height, width = 256, 256
    num_classes = 19
    
    # Dummy image, label, and prediction
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    label = np.random.randint(0, num_classes, (height, width))
    prediction = np.random.randint(0, num_classes, (height, width))
    confidence = np.random.rand(height, width)
    
    # Test color palette creation
    color_palette = create_color_palette(num_classes)
    print(f"Created color palette with shape: {color_palette.shape}")
    
    # Test prediction visualization
    visualize_prediction(
        image=image,
        label=label,
        prediction=prediction,
        confidence=confidence,
        color_palette=color_palette,
        save_path='/workspace/semantic_segmentation_project/test_prediction.png'
    )
    
    # Test training history plot
    train_losses = np.random.rand(100) * 2 + 0.5
    val_losses = np.random.rand(20) * 1.5 + 0.3
    val_mious = np.random.rand(20) * 0.5 + 0.4
    
    plot_training_history(
        train_losses=train_losses.tolist(),
        val_losses=val_losses.tolist(),
        val_metrics=val_mious.tolist(),
        save_path='/workspace/semantic_segmentation_project/test_history.png'
    )
    
    print("Visualization module test completed successfully!")

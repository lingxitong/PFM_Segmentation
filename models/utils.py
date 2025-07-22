"""
Model Utilities for Semantic Segmentation

This module contains utility functions for model management, including
model creation, weight initialization, and checkpoint handling.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import os



def count_parameters(model: nn.Module) -> Dict[str, float]:
    """
    Count the number of parameters in a model and return in millions (M).
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        Dict[str, float]: Dictionary containing parameter counts in millions (M)
                         with 2 decimal places precision
    """
    total_params = sum(p.numel() for p in model.parameters()) / 1e6  # Convert to millions
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    
    return {
        'total_parameters(M)': round(total_params, 2),
        'trainable_parameters(M)': round(trainable_params, 2),
        'non_trainable_parameters(M)': round(total_params - trainable_params, 2)
    }


def initialize_weights(model: nn.Module, init_type: str = 'kaiming') -> None:
    """
    Initialize model weights with specified initialization method.
    
    Args:
        model (nn.Module): PyTorch model
        init_type (str): Initialization type ('kaiming', 'xavier', 'normal', 'zero')
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, 0, 0.01)
            elif init_type == 'zero':
                nn.init.zeros_(m.weight)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
        elif isinstance(m, nn.Linear):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, 0, 0.01)
            elif init_type == 'zero':
                nn.init.zeros_(m.weight)
                
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   epoch: int, loss: float, metrics: Dict[str, float],
                   checkpoint_path: str, is_best: bool = False) -> None:
    """
    Save model checkpoint.
    
    Args:
        model (nn.Module): PyTorch model
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler
        epoch (int): Current epoch
        loss (float): Current loss value
        metrics (Dict[str, float]): Evaluation metrics
        checkpoint_path (str): Path to save checkpoint
        is_best (bool): Whether this is the best checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'metrics': metrics
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(os.path.dirname(checkpoint_path), 'best_model.pth')
        torch.save(checkpoint, best_path)


def load_checkpoint(model: nn.Module, checkpoint_path: str,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   device: str = 'cpu') -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model (nn.Module): PyTorch model
        checkpoint_path (str): Path to checkpoint file
        optimizer (Optional[torch.optim.Optimizer]): Optimizer to load state
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Scheduler to load state
        device (str): Device to load checkpoint on
        
    Returns:
        Dict[str, Any]: Checkpoint information
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'metrics': checkpoint.get('metrics', {})
    }


def get_model_complexity(model: nn.Module, input_size: tuple = (1, 3, 512, 512)) -> Dict[str, Any]:
    """
    Analyze model complexity including parameters, FLOPs, and memory usage.
    
    Args:
        model (nn.Module): PyTorch model
        input_size (tuple): Input tensor size
        
    Returns:
        Dict[str, Any]: Model complexity metrics
    """
    # Count parameters
    param_stats = count_parameters(model)
    
    # Estimate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    # Create dummy input for memory estimation
    dummy_input = torch.randn(input_size)
    model.eval()
    
    with torch.no_grad():
        # Estimate memory usage
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
            
            # Measure memory before forward pass
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
            
            # Forward pass
            _ = model(dummy_input)
            
            # Measure memory after forward pass
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated()
            
            memory_usage_mb = (mem_after - mem_before) / (1024 ** 2)
        else:
            # CPU memory estimation (approximate)
            output = model(dummy_input)
            memory_usage_mb = sum(
                tensor.numel() * tensor.element_size() 
                for tensor in [dummy_input, output['out']]
            ) / (1024 ** 2)
    
    return {
        'parameters': param_stats,
        'model_size_mb': model_size_mb,
        'memory_usage_mb': memory_usage_mb,
        'input_size': input_size
    }


def convert_to_onnx(model: nn.Module, output_path: str, 
                   input_size: tuple = (1, 3, 512, 512),
                   opset_version: int = 11) -> None:
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model (nn.Module): PyTorch model
        output_path (str): Path to save ONNX model
        input_size (tuple): Input tensor size
        opset_version (int): ONNX opset version
    """
    try:
        import onnx
        import onnxruntime
    except ImportError:
        raise ImportError("ONNX and ONNXRuntime are required for ONNX conversion")
    
    model.eval()
    dummy_input = torch.randn(input_size)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"Model successfully converted to ONNX: {output_path}")


def print_model_summary(model: nn.Module, input_size: tuple = (1, 3, 512, 512)) -> None:
    """
    Print a comprehensive model summary.
    
    Args:
        model (nn.Module): PyTorch model
        input_size (tuple): Input tensor size
    """
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    
    # Model architecture
    print(f"Model: {model.__class__.__name__}")
    print(f"Input size: {input_size}")
    
    # Parameter statistics
    param_stats = count_parameters(model)
    print(f"Total parameters: {param_stats['total_parameters']:,}")
    print(f"Trainable parameters: {param_stats['trainable_parameters']:,}")
    print(f"Non-trainable parameters: {param_stats['non_trainable_parameters']:,}")
    
    # Model complexity
    complexity = get_model_complexity(model, input_size)
    print(f"Model size: {complexity['model_size_mb']:.2f} MB")
    print(f"Memory usage: {complexity['memory_usage_mb']:.2f} MB")
    
    print("=" * 80)




if __name__ == "__main__":
    # Test model utilities
    from .PFM import create_pfm_model
    
    # Create a test model
    model = create_pfm_model(num_classes=19, img_size=512)
    
    # Print model summary
    print_model_summary(model)
    
    # Test other utilities
    initialize_weights(model, 'kaiming')
    print("Model weights initialized successfully")

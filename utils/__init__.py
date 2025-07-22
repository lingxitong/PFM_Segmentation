"""
Utils package for semantic segmentation.

This package contains training utilities, evaluation metrics, visualization tools,
and other helper functions for semantic segmentation.
"""

from .trainer import SegmentationTrainer
from .logs import setup_logging
from .evaluator import SegmentationEvaluator
from .metrics import SegmentationMetrics, StreamingMetrics
from .scheduler import (
    CosineAnnealingWithWarmup, PolynomialLR, WarmupMultiStepLR,
    OneCycleLR, CyclicLR, get_scheduler, WarmupScheduler
)
from .visualization import (
    create_color_palette, tensor_to_image,
    apply_color_map, visualize_prediction, save_predictions, 
    plot_training_history, plot_confusion_matrix, plot_class_metrics,
    create_interactive_training_dashboard, visualize_feature_maps
)

__all__ = [
    # Training
    'SegmentationTrainer',
    
    # Evaluation
    'SegmentationEvaluator',
    
    # Metrics
    'SegmentationMetrics',
    'StreamingMetrics',
    
    # Schedulers
    'CosineAnnealingWithWarmup',
    'PolynomialLR',
    'WarmupMultiStepLR',
    'OneCycleLR',
    'CyclicLR',
    'get_scheduler',
    'WarmupScheduler',
    
    # Visualization
    'setup_matplotlib_for_plotting',
    'create_color_palette',
    'tensor_to_image',
    'apply_color_map',
    'visualize_prediction',
    'save_predictions',
    'plot_training_history',
    'plot_confusion_matrix', 
    'plot_class_metrics',
    'create_interactive_training_dashboard',
    'visualize_feature_maps'

    # Utility functions
    'setup_logging'
]

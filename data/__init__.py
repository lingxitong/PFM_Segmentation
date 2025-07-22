"""
Data package for semantic segmentation.

This package contains dataset classes, data transforms, and utilities
for loading and preprocessing segmentation data.
"""

from data.seg_dataset import (
    JSONSegmentationDataset,
)

from .transforms import (
    SegmentationTransforms, parse_transform_config, get_transforms,
    MixUp, CutMix, Mosaic, AdvancedAugmentationPipeline
)
from .utils import (
    create_dataloader, segmentation_collate_fn, compute_class_distribution,
    visualize_class_distribution, visualize_sample, create_color_map,
    analyze_dataset_quality, save_dataset_info, create_data_split
)

__all__ = [
    # Datasets
    'BaseSegmentationDataset',
    'CityscapesDataset',
    'ADE20KDataset', 
    'PascalVOCDataset',
    'CustomDataset',
    'get_dataset',
    'DatasetStatistics',
    
    # Transforms
    'SegmentationTransforms',
    'parse_transform_config',
    'get_transforms',
    'MixUp',
    'CutMix', 
    'Mosaic',
    'AdvancedAugmentationPipeline',
    
    # Utils
    'create_dataloader',
    'segmentation_collate_fn',
    'compute_class_distribution',
    'visualize_class_distribution',
    'visualize_sample',
    'create_color_map',
    'analyze_dataset_quality',
    'save_dataset_info',
    'create_data_split'
]

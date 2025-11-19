"""
Data Transforms for Semantic Segmentation

This module contains data augmentation and preprocessing transforms
using albumentations library for robust training.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
import torch


def get_model_normalization(pfm_name: str) -> Tuple[List[float], List[float]]:
    """
    Get normalization mean and std values for a given PFM model.
    
    Args:
        pfm_name (str): Name of the PFM model
        
    Returns:
        Tuple[List[float], List[float]]: (mean, std) values for normalization
        
    Note:
        - conch_v1 uses CLIP normalization values
        - Other models use ImageNet normalization values
    """
    pfm_name = pfm_name.lower()
    
    # Conch v1 uses CLIP normalization
    if pfm_name == 'conch_v1':
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    elif pfm_name == 'conch_v1_5':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif pfm_name == 'virchow_v1':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif pfm_name == 'virchow_v2':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif pfm_name == 'gigapath':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif pfm_name == 'patho3dmatrix-vision':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif pfm_name == 'uni_v2':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif pfm_name == 'uni_v1':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif pfm_name == 'phikon' or pfm_name == 'phikon_v2':
        # Phikon uses ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif pfm_name == 'hoptimus_0' or pfm_name == 'hoptimus_1':
        mean=(0.707223, 0.578729, 0.703617)
        std=(0.211883, 0.230117, 0.177517)
    elif pfm_name == 'musk':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif pfm_name == 'midnight12k':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif pfm_name.startswith('kaiko-'):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif pfm_name == 'hibou_l':
        mean = [0.7068,0.5755,0.722]
        std = [0.195,0.2316,0.1816]
    else:
        # Default ImageNet normalization for other models
        # (uni_v1, uni_v2, virchow_v1, virchow_v2, gigapath, 
        #  patho3dmatrix-vision, conch_v1_5, unet, phikon)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    return mean, std

class SegmentationTransforms:
    """
    Collection of segmentation-specific transforms.
    """
    
    @staticmethod
    def get_training_transforms(img_size: int = 512, 
                              mean: List[float] = [0.485, 0.456, 0.406],
                              std: List[float] = [0.229, 0.224, 0.225],
                              seed: int = 42) -> A.Compose:
        """
        Get training transforms with strong augmentations.
        
        Args:
            img_size (int): Target image size
            mean (List[float]): Normalization mean
            std (List[float]): Normalization standard deviation
            
        Returns:
            A.Compose: Composed transforms
        """
        return A.Compose([
            # Geometric transforms
            A.RandomResizedCrop(size=(img_size,img_size), scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.3),
            A.Transpose(p=0.3),
            
            # Spatial transforms
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.OpticalDistortion(distort_limit=0.2, p=1.0),
            ], p=0.3),
            
            # Color transforms
            A.OneOf([
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
            ], p=0.5),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=1.0),
            ], p=0.3),
            
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
            ], p=0.2),
            
            # Weather effects
            A.OneOf([
                A.RandomRain(brightness_coefficient=0.7, p=1.0),
                A.RandomSnow(brightness_coeff=2.5, p=1.0),
                A.RandomFog(alpha_coef=0.08, p=1.0),
            ], p=0.2),
            
            # Lighting
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            
            # Cutout and mixing
            A.CoarseDropout(p=0.3),
            
            # Normalization
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], seed=seed)

    @staticmethod
    def get_validation_transforms(img_size: int = 512,
                                mean: List[float] = [0.485, 0.456, 0.406],
                                std: List[float] = [0.229, 0.224, 0.225]) -> A.Compose:
        """
        Get validation transforms with minimal augmentation.
        
        Args:
            img_size (int): Target image size
            mean (List[float]): Normalization mean
            std (List[float]): Normalization standard deviation
            
        Returns:
            A.Compose: Composed transforms
        """
        if img_size != None:
            return A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
    

def parse_transform_config(config: Dict[str, Any]) -> A.Compose:
    """
    Parse transform configuration and create albumentations transforms.
    
    Args:
        config (Dict[str, Any]): Transform configuration
        
    Returns:
        A.Compose: Composed transforms
    """
    transforms = []
    
    for transform_config in config:
        transform_type = transform_config['type']
        transform_params = {k: v for k, v in transform_config.items() if k != 'type'}
        
        # Get transform class from albumentations
        if hasattr(A, transform_type):
            transform_class = getattr(A, transform_type)
            transforms.append(transform_class(**transform_params))
        elif transform_type == 'ToTensorV2':
            transforms.append(ToTensorV2())
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
    
    return A.Compose(transforms)


def get_transforms(transform_config: List[Dict[str, Any]]) -> A.Compose:
    """
    Factory function to create transforms from configuration.
    
    Args:
        transform_config (List[Dict[str, Any]]): List of transform configurations
        
    Returns:
        A.Compose: Composed transforms
    """
    if isinstance(transform_config, list):
        return parse_transform_config(transform_config)
    else:
        raise ValueError("Transform config must be a list of dictionaries")


class MixUp:
    """
    MixUp augmentation for semantic segmentation.
    """
    
    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        self.alpha = alpha
        self.p = p
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply MixUp to a batch of data.
        
        Args:
            batch (Dict[str, torch.Tensor]): Batch containing images and labels
            
        Returns:
            Dict[str, torch.Tensor]: Mixed batch
        """
        if np.random.random() > self.p:
            return batch
            
        images = batch['image']
        labels = batch['label']
        
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # For segmentation, we need to handle labels differently
        # We can either use the original labels or create mixed labels
        mixed_labels = labels  # Keep original labels for simplicity
        
        return {
            'image': mixed_images,
            'label': mixed_labels,
            'lambda': lam,
            'indices': indices
        }


class CutMix:
    """
    CutMix augmentation for semantic segmentation.
    """
    
    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        self.alpha = alpha
        self.p = p
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply CutMix to a batch of data.
        
        Args:
            batch (Dict[str, torch.Tensor]): Batch containing images and labels
            
        Returns:
            Dict[str, torch.Tensor]: Cut-mixed batch
        """
        if np.random.random() > self.p:
            return batch
            
        images = batch['image']
        labels = batch['label']
        
        batch_size, _, height, width = images.shape
        indices = torch.randperm(batch_size)
        
        # Sample lambda and bounding box
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Generate random bounding box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)
        
        cx = np.random.randint(width)
        cy = np.random.randint(height)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)
        
        # Apply CutMix
        mixed_images = images.clone()
        mixed_labels = labels.clone()
        
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
        mixed_labels[:, bby1:bby2, bbx1:bbx2] = labels[indices, bby1:bby2, bbx1:bbx2]
        
        return {
            'image': mixed_images,
            'label': mixed_labels,
            'lambda': lam,
            'indices': indices,
            'bbox': (bbx1, bby1, bbx2, bby2)
        }


class Mosaic:
    """
    Mosaic augmentation for semantic segmentation.
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply Mosaic to a batch of data.
        
        Args:
            batch (Dict[str, torch.Tensor]): Batch containing images and labels
            
        Returns:
            Dict[str, torch.Tensor]: Mosaic batch
        """
        if np.random.random() > self.p or batch['image'].size(0) < 4:
            return batch
            
        images = batch['image']
        labels = batch['label']
        
        batch_size, channels, height, width = images.shape
        
        # Create mosaic for first sample
        mosaic_image = torch.zeros(channels, height, width)
        mosaic_label = torch.zeros(height, width, dtype=labels.dtype)
        
        # Divide image into 4 quadrants
        h_mid = height // 2
        w_mid = width // 2
        
        indices = torch.randperm(batch_size)[:4]
        
        # Top-left
        mosaic_image[:, :h_mid, :w_mid] = images[indices[0], :, :h_mid, :w_mid]
        mosaic_label[:h_mid, :w_mid] = labels[indices[0], :h_mid, :w_mid]
        
        # Top-right
        mosaic_image[:, :h_mid, w_mid:] = images[indices[1], :, :h_mid, w_mid:]
        mosaic_label[:h_mid, w_mid:] = labels[indices[1], :h_mid, w_mid:]
        
        # Bottom-left
        mosaic_image[:, h_mid:, :w_mid] = images[indices[2], :, h_mid:, :w_mid]
        mosaic_label[h_mid:, :w_mid] = labels[indices[2], h_mid:, :w_mid]
        
        # Bottom-right
        mosaic_image[:, h_mid:, w_mid:] = images[indices[3], :, h_mid:, w_mid:]
        mosaic_label[h_mid:, w_mid:] = labels[indices[3], h_mid:, w_mid:]
        
        # Replace first sample with mosaic
        new_images = images.clone()
        new_labels = labels.clone()
        new_images[0] = mosaic_image
        new_labels[0] = mosaic_label
        
        return {
            'image': new_images,
            'label': new_labels
        }


class AdvancedAugmentationPipeline:
    """
    Advanced augmentation pipeline combining multiple techniques.
    """
    
    def __init__(self, mixup_p: float = 0.3, cutmix_p: float = 0.3, mosaic_p: float = 0.2):
        self.mixup = MixUp(p=mixup_p)
        self.cutmix = CutMix(p=cutmix_p)
        self.mosaic = Mosaic(p=mosaic_p)
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply advanced augmentations to batch.
        
        Args:
            batch (Dict[str, torch.Tensor]): Input batch
            
        Returns:
            Dict[str, torch.Tensor]: Augmented batch
        """
        # Apply augmentations in random order
        augmentations = [self.mixup, self.cutmix, self.mosaic]
        np.random.shuffle(augmentations)
        
        for aug in augmentations:
            batch = aug(batch)
            
        return batch


if __name__ == "__main__":
    # Test transforms
    from PIL import Image
    import numpy as np
    
    # Create dummy data
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mask = np.random.randint(0, 19, (512, 512), dtype=np.uint8)
    
    # Test training transforms
    train_transforms = SegmentationTransforms.get_training_transforms()
    transformed = train_transforms(image=image, mask=mask)
    
    print(f"Original image shape: {image.shape}")
    print(f"Transformed image shape: {transformed['image'].shape}")
    print(f"Transformed mask shape: {transformed['mask'].shape}")
    
    # Test validation transforms
    val_transforms = SegmentationTransforms.get_validation_transforms()
    val_transformed = val_transforms(image=image, mask=mask)
    
    print(f"Validation image shape: {val_transformed['image'].shape}")
    print(f"Validation mask shape: {val_transformed['mask'].shape}")
    
    # Test TTA transforms
    tta_transforms = SegmentationTransforms.get_test_time_augmentation_transforms()
    print(f"Number of TTA transforms: {len(tta_transforms)}")

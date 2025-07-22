"""
Simplified JSON-based dataset class

Supports basic img_path and mask_path format, designed for semantic segmentation tasks.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Callable


class JSONSegmentationDataset(Dataset):
    """
    Semantic segmentation dataset based on a JSON file.
    
    Expected JSON format:
    {
        "num_classes": 3,
        "data": {
            "train": [
                {"image_path": "/path/to/image1.jpg", "mask_path": "/path/to/mask1.png"},
                {"image_path": "/path/to/image2.jpg", "mask_path": "/path/to/mask2.png"}
            ],
            "val": [...],
            "test": [...]
        }
    }
    
    Args:
        json_file (str): Path to the JSON config file.
        split (str): Dataset split ('train', 'val', or 'test').
        transform (Optional[Callable]): Data transformation/augmentation function.
    """
    
    def __init__(self, json_file: str, split: str = 'train',
                 transform: Optional[Callable] = None):
        self.json_file = json_file
        self.split = split
        self.transform = transform
        
        # Load JSON configuration
        self.config = self._load_json_config()
        
        # Extract basic info
        self.num_classes = self.config.get('num_classes')
        self.ignore_index = 255  # fixed ignore label
        # Load data entries
        self.data_items = self._load_data_items()
        self.fixed_size = self._check_fixed_size()        
        self.has_mask = self._check_has_mask()
        if not self.has_mask:
            self._reset_mask()
        print(f"Dataset loaded: split = {split}, samples = {len(self.data_items)}, classes = {self.num_classes}")
        
    def _load_json_config(self) -> Dict:
        """Load the JSON config file."""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON config file not found: {self.json_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

    def _check_has_mask(self) -> bool:
        """Check if the dataset has mask paths."""
        for item in self.data_items:
            mask_path = item.get('mask_path')
            if mask_path == None:
                return False
            if not os.path.exists(mask_path):
                return False
        return True

    def _reset_mask(self) -> None:
        """Reset mask paths to None if they are not present."""
        new_items = []
        for item in self.data_items:
            item['mask_path'] = None
            new_items.append(item)
        self.data_items = new_items
            
        
    def _check_fixed_size(self) -> bool:
        """Check if the dataset has a fixed image size."""
        _img_size = None
        for item in self.data_items:
            img_path = item.get('img_path', '')
            with Image.open(img_path) as img:
                if _img_size is None:
                    _img_size = img.size
                elif _img_size != img.size:
                    return False
        return True
    
    def _load_data_items(self) -> List[Dict]:
        """Load the data entries for the given split."""
        data_config = self.config.get('data')
        split_data = data_config.get(self.split)
        
        if not split_data:
            raise ValueError(f"No data found for split '{self.split}'")
        
        processed_items = []
        for item in split_data:
            processed_item = self._process_data_item(item)
            if processed_item:
                processed_items.append(processed_item)
        
        if not processed_items:
            raise ValueError(f"No valid items found in split '{self.split}'")
            
        return processed_items
    
    def _process_data_item(self, item: Dict) -> Optional[Dict]:
        """Process a single data entry."""
        img_path = item.get('image_path', '')
        mask_path = item.get('mask_path', None)
        
        if not img_path or not mask_path:
            if self.split == 'train' or self.split == 'val':
                print(f"Missing image or mask path: {item}")
                return None
        
        return {
            'img_path': img_path,
            'mask_path': mask_path
        }
    
    def __len__(self) -> int:
        """Return the dataset size."""
        return len(self.data_items)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a single data entry.
        
        Args:
            index (int): Index of the data item
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing image and label tensors
        """
        item = self.data_items[index]
        
        image = Image.open(item['img_path']).convert('RGB')
        ori_size = image.size
        
        if self.has_mask:
            mask = Image.open(item['mask_path'])
            if mask.mode != 'L':
                mask = mask.convert('L')
            mask = np.array(mask, dtype=np.int64)
        else:
            mask = np.ones((ori_size[1],ori_size[0]), dtype=np.int64) * (-1)
        
        # Validate mask values (should be within [0, num_classes-1] or 255 as ignore index)
        unique_values = np.unique(mask)
        valid_values = set(range(self.num_classes)) | {self.ignore_index}
        invalid_values = set(unique_values) - valid_values
        
        if invalid_values and self.has_mask:
            print(f"Invalid label values {invalid_values} found in {item['mask_path']}")
            for invalid_val in invalid_values:
                mask[mask == invalid_val] = self.ignore_index
        
        # Apply transformation
        if self.transform:
            transformed = self.transform(image=np.array(image), mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'label': mask,
            'ori_size': ori_size,
            'image_path': item['img_path'],
            'label_path': item['mask_path']
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights to handle class imbalance.
        
        Returns:
            torch.Tensor: Computed class weights
        """
        print("Computing class weights...")
        
        class_counts = np.zeros(self.num_classes)
        total_pixels = 0
        
        for item in self.data_items:
            mask = Image.open(item['mask_path'])
            if mask.mode != 'L':
                mask = mask.convert('L')
            mask_array = np.array(mask)
            
            for class_id in range(self.num_classes):
                class_counts[class_id] += np.sum(mask_array == class_id)
            
            valid_pixels = mask_array != self.ignore_index
            total_pixels += np.sum(valid_pixels)
        
        class_counts = np.maximum(class_counts, 1)
        weights = total_pixels / (self.num_classes * class_counts)
        weights = weights / weights.sum() * self.num_classes
        
        print(f"Class weights: {weights}")
        return torch.from_numpy(weights).float()


def get_dataset(data_configs, transforms, split):
    json_file = data_configs.get('json_file')
    return JSONSegmentationDataset(
        json_file=json_file,
        split=split,
        transform=transforms
    )


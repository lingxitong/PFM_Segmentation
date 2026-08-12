import os
import json
import base64
import zlib
import numpy as np
from PIL import Image
from pathlib import Path
import io # for handling binary data from bitmap decoding


def decode_supervisely_bitmap(bitmap_data, origin, target_size):
    """
    Decode Supervisely bitmap annotation to a 2D binary mask.
    
    Args:
        bitmap_data (str): Base64-encoded zlib-compressed bitmap.
        origin (tuple): (x, y) top-left corner in the full image.
        target_size (tuple): (height, width) of the full image.
    
    Returns:
        np.ndarray: Binary mask of shape (H, W), dtype=np.uint8.
    """
    try:
        # Decode base64
        compressed = base64.b64decode(bitmap_data)
        # Decompress with zlib
        decompressed = zlib.decompress(compressed)
        # Treat decompressed data as 1-bit PNG
        img_obj = Image.open(io.BytesIO(decompressed)).convert('1')
        obj_mask = np.array(img_obj, dtype=np.uint8)
        H_obj, W_obj = obj_mask.shape
        full_mask = np.zeros(target_size, dtype=np.uint8)
        x0, y0 = origin
        x1, y1 = x0 + W_obj, y0 + H_obj
        # Ensure within bounds
        x1 = min(x1, target_size[1])
        y1 = min(y1, target_size[0])
        full_mask[y0:y1, x0:x1] = obj_mask[:(y1 - y0), :(x1 - x0)]
        return full_mask
    except Exception as e:
        print(f"Warning: Failed to decode bitmap as PNG: {e}")
        # Fallback: create empty mask
        return np.zeros(target_size, dtype=np.uint8)


def process_glas_dataset(input_root, output_root, num_classes=2):
    """
    Process GLAS dataset (with train, test_a, test_b) to generate PNG images and mask PNGs.
    Supports images in any format (e.g., .bmp, .jpg, .png). Masks are generated from JSON annotations.
    The JSON files are expected to have the format: <image_name>.bmp.json (or similar).
    
    Args:
        input_root (str): Path to root folder containing 'train', 'test_a', 'test_b'.
        output_root (str): Output directory.
        num_classes (int): Number of classes (default: 2 -> 0=background, 1=gland).
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    img_out_dir = output_root / 'images'
    mask_out_dir = output_root / 'masks'
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'test_a', 'test_b']:
        img_dir = input_root / split / 'img'
        ann_dir = input_root / split / 'ann'
        
        if not img_dir.exists() or not ann_dir.exists():
            print(f"Skipping {split}: img or ann folder missing.")
            continue
        
        # Get all image files (any extension)
        img_files = [f for f in img_dir.iterdir() if f.is_file()]
        
        for img_path in img_files:
            # Load image with PIL
            try:
                img_pil = Image.open(img_path).convert('RGB')
                img_array = np.array(img_pil)
                h, w = img_array.shape[:2]
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
            
            # Determine the stem name for matching JSON
            # e.g., train_1.bmp -> train_1
            stem = img_path.stem
            
            # Save image as PNG (with the same stem name)
            output_img_path = img_out_dir / (stem + '.png')
            Image.fromarray(img_array).save(output_img_path)
            
            # Find the corresponding JSON file in the ann directory
            # It's likely named <stem>.<original_ext>.json (e.g., train_1.bmp.json)
            # Search for the first JSON file that starts with the stem
            matching_json_files = [f for f in ann_dir.iterdir() if f.is_file() and f.suffix == '.json' and f.name.startswith(stem + '.')]
            
            mask = np.zeros((h, w), dtype=np.uint8)
            
            if matching_json_files:
                # If multiple JSON files match the stem, use the first one
                ann_path = matching_json_files[0]
                try:
                    with open(ann_path, 'r') as f:
                        ann = json.load(f)
                    # Use size from annotation if available
                    ann_h = ann.get('size', {}).get('height', h)
                    ann_w = ann.get('size', {}).get('width', w)
                    # Reinitialize mask if size differs
                    if ann_h != h or ann_w != w:
                        mask = np.zeros((ann_h, ann_w), dtype=np.uint8)
                        h, w = ann_h, ann_w
                    
                    for obj in ann.get('objects', []):
                        if obj.get('classTitle', '').lower() == 'gland':
                            if obj.get('geometryType') == 'bitmap':
                                bitmap = obj['bitmap']
                                origin = tuple(bitmap['origin'])  # (x, y)
                                bitmap_data = bitmap['data']
                                try:
                                    obj_mask_full = decode_supervisely_bitmap(bitmap_data, origin, (h, w))
                                    mask = np.maximum(mask, obj_mask_full)
                                except Exception as e:
                                    print(f"Error decoding object in {ann_path}: {e}")
                except Exception as e:
                    print(f"Error parsing annotation {ann_path}: {e}")
            else:
                print(f"Annotation not found for image {img_path} (expected a JSON file starting with '{stem}.'). Using empty mask.")
            
            # Map gland (1) and background (0)
            mask[mask != 0] = 1
            
            # Validate
            assert mask.max() < num_classes, f"Mask contains invalid label >= {num_classes}"
            
            # Save mask (with the same stem name as the original image)
            output_mask_path = mask_out_dir / (stem + '.png')
            Image.fromarray(mask).save(output_mask_path)
    
    print(f"Processing completed. Output saved to: {output_root}")


# Example usage:
process_glas_dataset(
    input_root='/path/to/PFM_Segmentation_Data/GlaS/',
    output_root='/path/to/PFM_Segmentation_Data/GlaS/',
    num_classes=2
)
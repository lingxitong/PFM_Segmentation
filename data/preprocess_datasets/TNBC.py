import os
import cv2
from tqdm import tqdm
import numpy as np

# Input root directory
input_root = "/path/to/PFM_Segmentation_Data/TNBC/"

# Output paths
output_images = os.path.join(input_root, "images")
output_masks = os.path.join(input_root, "masks")

# Create output folders
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)

# Get Slide and GT folder lists (sorted numerically)
slide_folders = sorted([f for f in os.listdir(input_root) if f.startswith("Slide_")])
gt_folders = sorted([f for f in os.listdir(input_root) if f.startswith("GT_")])

# Ensure counts match
assert len(slide_folders) == len(gt_folders), "Slide and GT folder counts do not match!"

print(f"Found {len(slide_folders)} Slide/GT folder pairs.")

# Iterate over each folder pair
for slide_folder, gt_folder in zip(slide_folders, gt_folders):
    slide_path = os.path.join(input_root, slide_folder)
    gt_path = os.path.join(input_root, gt_folder)

    # Get all png files under the folder (case-insensitive)
    slide_files = [f for f in os.listdir(slide_path) if f.lower().endswith('.png')]
    gt_files = [f for f in os.listdir(gt_path) if f.lower().endswith('.png')]

    # Sort to keep order consistent
    slide_files.sort()
    gt_files.sort()

    print(f"\nProcessing {slide_folder} <-> {gt_folder} ...")

    for img_name, mask_name in tqdm(zip(slide_files, gt_files), total=len(slide_files)):
        # Ensure filenames match (optional check)
        if img_name != mask_name:
            print(f"Warning: filename mismatch: {img_name} vs {mask_name}")
            # Optionally skip or rename to align; here we proceed with the paired names
            # Adjust as needed

        # Read original image
        img_path = os.path.join(slide_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read image: {img_path}")
            continue

        # Read mask (single-channel)
        mask_path = os.path.join(gt_path, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to read mask: {mask_path}")
            continue

        # Convert mask values from {0, 255} to {0, 1}
        mask = (mask > 0).astype(np.uint8)  # Convert all non-zero values to 1

        # Save image to images/
        output_img_path = os.path.join(output_images, img_name)
        cv2.imwrite(output_img_path, image)

        # Save mask to masks/
        output_mask_path = os.path.join(output_masks, mask_name)
        cv2.imwrite(output_mask_path, mask)

        # Optional: progress display (already using tqdm)

print("\nProcessing completed.")
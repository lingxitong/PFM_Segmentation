import os
import shutil
from PIL import Image
import numpy as np

# ========== Configuration ==========
# Input root directory (contains 6 class subfolders)
input_root = "/path/to/PFM_Segmentation_Data/EBHI/"

# Output directory
output_root = "/path/to/PFM_Segmentation_Data/EBHI/"

# Class list (in order; mask values start from 1)
categories = [
    "Adenocarcinoma",
    "High-grade IN",
    "Low-grade IN",
    "Normal",
    "Polyp",
    "Serrated adenoma"
]

# ========== Create output directories ==========
os.makedirs(os.path.join(output_root, "images"), exist_ok=True)
os.makedirs(os.path.join(output_root, "masks"), exist_ok=True)

# ========== Iterate over each class subfolder ==========
for idx, category in enumerate(categories):
    category_dir = os.path.join(input_root, category)
    image_dir = os.path.join(category_dir, "image")
    label_dir = os.path.join(category_dir, "label")

    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"Warning: skip {category}: missing image or label subfolder")
        continue

    # Get all image files (assume extensions like .png/.jpg/.jpeg/.tiff)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    label_files = [f for f in os.listdir(label_dir) if f.lower().endswith('.png')]

    # Match images and labels with the same stem
    matched_files = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + ".png"
        if label_file in label_files:
            matched_files.append((img_file, label_file))
        else:
            print(f"Warning: no matching label for: {img_file}")

    print(f"Processing class '{category}' ({idx+1}/6), {len(matched_files)} pairs...")

    for img_file, label_file in matched_files:
        src_img_path = os.path.join(image_dir, img_file)
        src_label_path = os.path.join(label_dir, label_file)
        dst_img_path = os.path.join(output_root, "images", img_file)
        dst_mask_path = os.path.join(output_root, "masks", label_file)

        # Copy image
        shutil.copyfile(src_img_path, dst_img_path)

        # Read label image and re-encode
        label_img = Image.open(src_label_path).convert('L')  # Convert to grayscale
        label_np = np.array(label_img, dtype=np.uint8)

        # Map binary labels (0,1) -> (0, class_id); class_id starts from 1
        class_id = idx + 1
        label_np[label_np == 1] = class_id  # Original 1 becomes the current class id

        # Save the new mask
        new_label_img = Image.fromarray(label_np, mode='L')
        new_label_img.save(dst_mask_path)

print("All classes processed.")
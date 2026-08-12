import os
import numpy as np
from PIL import Image
import glob

# Configuration
base_dir = "/path/to/PFM_Segmentation_Data/PanNuke"
output_dir = "/path/to/PFM_Segmentation_Data/PanNuke"  # Specified output directory

os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

# Process three folds
for fold_name in ["fold_1", "fold_2", "fold_3"]:
    # Build paths (note mixed casing in the original PanNuke layout)
    img_path = os.path.join(base_dir, fold_name, f"Fold {fold_name[-1]}", "images", f"Fold{fold_name[-1]}", "images.npy")
    mask_path = os.path.join(base_dir, fold_name, f"Fold {fold_name[-1]}", "masks", f"Fold{fold_name[-1]}", "masks.npy")
    
    print(f"Loading {img_path} and {mask_path}...")
    
    images = np.load(img_path)      # shape: (N, 256, 256, 3)
    masks = np.load(mask_path)      # shape: (N, 256, 256, 6)

    assert images.shape[0] == masks.shape[0], "Number of images and masks mismatch!"

    num_samples = images.shape[0]
    start_index = len(glob.glob(os.path.join(output_dir, "images", "*.png")))  # Ensure filenames are unique

    for i in range(num_samples):
        img = images[i]  # (256, 256, 3), uint8
        mask_onehot = masks[i]  # (256, 256, 6)

        # Convert to semantic labels via argmax -> classes 0~5
        semantic_mask = np.argmax(mask_onehot, axis=-1).astype(np.uint8)  # (256, 256)

        # Assign unified sequential filenames
        file_id = start_index + i
        filename = f"{file_id:05d}.png"

        # Save image
        img_pil = Image.fromarray(img.astype(np.uint8))
        img_pil.save(os.path.join(output_dir, "images", filename))

        # Save mask (single-channel, values 0~5)
        mask_pil = Image.fromarray(semantic_mask, mode='L')
        mask_pil.save(os.path.join(output_dir, "masks", filename))

    print(f"Processed {num_samples} samples from {fold_name}.")

print("All folds processed. Data saved to:", output_dir)
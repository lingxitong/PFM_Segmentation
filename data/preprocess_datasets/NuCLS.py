import os
import numpy as np
from PIL import Image
from glob import glob

# ====== Hard-coded path settings ======
input_folder = "/path/to/PFM_Segmentation_Data/NuCLS/mask"  # Input folder (multi-channel PNGs)
output_root = "/path/to/PFM_Segmentation_Data/NuCLS/"            # Output root directory
output_mask_dir = os.path.join(output_root, "masks")
os.makedirs(output_mask_dir, exist_ok=True)

# Get all PNG files
png_files = glob(os.path.join(input_folder, "*.png"))

if not png_files:
    print(f"Warning: no .png files found in {input_folder}!")
else:
    print(f"Found {len(png_files)} PNG files, processing...")

for png_path in png_files:
    try:
        # Read image
        img = Image.open(png_path)
        img_array = np.array(img)

        # Extract the first channel
        if img_array.ndim == 2:
            channel0 = img_array
        elif img_array.ndim == 3:
            channel0 = img_array[:, :, 0]  # Take the R channel
        else:
            print(f"Skip invalid image: {png_path} (unexpected shape: {img_array.shape})")
            continue

        # Cast to uint8 for consistent dtype
        channel0 = channel0.astype(np.uint8)

        # Set pixels with values 253 and 99 to 0
        channel0 = np.where((channel0 == 253) | (channel0 == 99), 0, channel0)

        # Save as a single-channel PNG
        filename = os.path.basename(png_path)
        output_path = os.path.join(output_mask_dir, filename)
        Image.fromarray(channel0, mode='L').save(output_path)
        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error processing {png_path}: {e}")

print("All images processed.")
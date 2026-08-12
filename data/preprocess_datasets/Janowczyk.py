import os
import re
from PIL import Image
import numpy as np

def convert_dataset(input_folder, output_folder):
    images_out = os.path.join(output_folder, 'images')
    masks_out = os.path.join(output_folder, 'masks')
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(masks_out, exist_ok=True)

    tif_files = [f for f in os.listdir(input_folder) if f.endswith('_original.tif')]
    
    print(f"Found {len(tif_files)} original images.")

    for tif_file in tif_files:
        base_name = tif_file.replace('_original.tif', '')
        mask_file = base_name + '_mask.png'
        
        tif_path = os.path.join(input_folder, tif_file)
        mask_path = os.path.join(input_folder, mask_file)

        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {tif_file}, skipping.")
            continue

        # Process image
        try:
            img = Image.open(tif_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(os.path.join(images_out, f"{base_name}.png"))
        except Exception as e:
            print(f"Error processing image {tif_file}: {e}")
            continue

        # Process mask
        try:
            mask = Image.open(mask_path)
            if mask.mode != 'L':
                mask = mask.convert('L')
            mask_arr = np.array(mask)
            binary_mask = (mask_arr > 0).astype(np.uint8)  # Convert to binary (0 or 1)
            binary_mask_img = Image.fromarray(binary_mask, mode='L')
            binary_mask_img.save(os.path.join(masks_out, f"{base_name}.png"))
        except Exception as e:
            print(f"Error processing mask {mask_file}: {e}")
            continue

    print(f"Conversion complete! Output saved to '{output_folder}'")


if __name__ == "__main__":
 
    input_folder = "/path/to/PFM_Segmentation_Data/Janowczyk/"     
    output_folder = "/path/to/PFM_Segmentation_Data/Janowczyk/"    

    convert_dataset(input_folder, output_folder)
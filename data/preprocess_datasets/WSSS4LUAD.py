import os
from PIL import Image
import numpy as np

def convert_rgb_mask_to_label(mask_path, output_mask_path, color_to_label_map):
    """
    Convert an RGB mask to a single-channel label mask.

    Args:
        mask_path (str): Path to the input RGB mask.
        output_mask_path (str): Path to the output single-channel label mask.
        color_to_label_map (dict): Mapping from RGB colors to label values.
    """
    # Open and convert the mask image
    mask_img = Image.open(mask_path).convert('RGB')
    mask_array = np.array(mask_img)

    # Initialize the output mask array
    h, w = mask_array.shape[:2]
    label_mask = np.zeros((h, w), dtype=np.uint8) + 255  # Initialize as 255 (invalid value)

    # Iterate over each color-label pair to fill the mask
    for rgb_color, label in color_to_label_map.items():
        # Create a boolean mask for the current color
        color_mask = np.all(mask_array == rgb_color, axis=-1)
        # Assign the label value to matching locations
        label_mask[color_mask] = label

    # Check for unmapped pixels (still 255)
    if np.any(label_mask == 255):
        print(f"Warning: Found unmapped pixels in {mask_path}. Assigning them to label 3 (background). Please verify input masks.")
        # Force unmapped pixels to background label 3 so outputs stay in 0-3
        label_mask[label_mask == 255] = 3

    # Save as a single-channel PNG
    label_mask_image = Image.fromarray(label_mask, mode='L') # 'L' mode for single channel
    label_mask_image.save(output_mask_path)

def get_unique_filename(output_dir, base_name, extension):
    """
    Generate a unique filename to avoid overwriting.

    Args:
        output_dir (str): Output directory path.
        base_name (str): Base name of the file.
        extension (str): File extension (e.g., '.png').

    Returns:
        str: Unique file path.
    """
    counter = 0
    while True:
        if counter == 0:
            filename = f"{base_name}{extension}"
        else:
            filename = f"{base_name}_{counter}{extension}"
        full_path = os.path.join(output_dir, filename)
        if not os.path.exists(full_path):
            return full_path
        counter += 1

def process_datasets(input_dirs, output_base_dir, color_to_label_map):
    """
    Process a list of input directories, organize images/masks into the output directory, and ensure unique filenames.

    Args:
        input_dirs (list of str): List of input dataset directories.
        output_base_dir (str): Root directory of the output dataset.
        color_to_label_map (dict): Mapping from RGB colors to label values.
    """
    # Create output directories
    images_output_dir = os.path.join(output_base_dir, "images")
    masks_output_dir = os.path.join(output_base_dir, "masks")
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(masks_output_dir, exist_ok=True)

    processed_count = 0  # Used to generate a globally unique index

    for input_dir in input_dirs:
        print(f"Processing directory: {input_dir}")
        img_dir = os.path.join(input_dir, "img")
        mask_dir = os.path.join(input_dir, "mask")

        if not os.path.exists(img_dir):
            print(f"Warning: Image directory {img_dir} does not exist. Skipping.")
            continue
        if not os.path.exists(mask_dir):
            print(f"Warning: Mask directory {mask_dir} does not exist. Skipping.")
            continue

        # Get image and mask file lists
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.png')])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith('.png')])

        # Ensure the number of images and masks match
        if len(img_files) != len(mask_files):
            print(f"Warning: Number of images ({len(img_files)}) and masks ({len(mask_files)}) do not match in {input_dir}. Using minimum count.")

        # Iterate over image and mask files
        min_count = min(len(img_files), len(mask_files))
        for i in range(min_count):
            img_file = img_files[i]
            mask_file = mask_files[i]

            # Ensure image and mask names match (assume the same filename stem)
            img_name_base = os.path.splitext(img_file)[0]
            mask_name_base = os.path.splitext(mask_file)[0]

            if img_name_base != mask_name_base:
                print(f"Warning: Image name '{img_name_base}' does not match mask name '{mask_name_base}' in {input_dir}. Skipping pair.")
                continue

            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            # Use a global counter to generate a unique base filename
            unique_base_name = f"sample_{processed_count:06d}"

            # Define output paths and ensure filenames are unique
            output_img_path = get_unique_filename(images_output_dir, unique_base_name, '.png')
            output_mask_path = get_unique_filename(masks_output_dir, unique_base_name, '.png')

            # Copy image
            try:
                img_to_copy = Image.open(img_path)
                img_to_copy.save(output_img_path)
                print(f"Copied image: {img_file} -> {os.path.basename(output_img_path)}")
            except Exception as e:
                print(f"Error copying image {img_path}: {e}")
                continue # If copy fails, skip the corresponding mask

            # Convert and save mask
            try:
                convert_rgb_mask_to_label(mask_path, output_mask_path, color_to_label_map)
                print(f"Converted and saved mask: {mask_file} -> {os.path.basename(output_mask_path)}")
            except Exception as e:
                print(f"Error processing mask {mask_path}: {e}")
                # If mask processing fails after the image was copied, optionally delete the image to keep pairs, or keep it.
                # Here we keep it, since the main goal is to process all valid pairs.
            
            processed_count += 1 # Increment global counter


if __name__ == "__main__":
    # Define RGB color to label mapping
    COLOR_TO_LABEL = {
        (0, 64, 128): 0,  # Tumor epithelial
        (64, 128, 0):  1,  # Tumor-associated stroma
        (243, 152, 0): 2,  # Normal
        (255, 255, 255): 3 # White background
    }

    # Define input directory list
    INPUT_DIRS = [
        "/path/to/PFM_Segmentation_Data/WSSS4LUAD/2.validation/",
        "/path/to/PFM_Segmentation_Data/WSSS4LUAD/3.testing/"
    ]

    # Define output directory
    OUTPUT_BASE_DIR = "/path/to/PFM_Segmentation_Data/WSSS4LUAD/" # Modify this path as needed

    print("Starting dataset processing...")
    process_datasets(INPUT_DIRS, OUTPUT_BASE_DIR, COLOR_TO_LABEL)
    print("Dataset processing completed.")
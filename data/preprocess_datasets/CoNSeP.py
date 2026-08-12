import os
import scipy.io
import numpy as np
from PIL import Image

def convert_mat_to_semantic_mask_from_type_map(mat_path, output_mask_path):
    """
    Load 'type_map' from a .mat file and save it as a single-channel PNG semantic mask.

    Args:
        mat_path (str): Path to the input .mat file.
        output_mask_path (str): Path to the output single-channel PNG semantic mask.
    """
    try:
        # Load .mat file
        mat_data = scipy.io.loadmat(mat_path, squeeze_me=True, matlab_compatible=False)

        # Check whether 'type_map' exists
        if 'type_map' not in mat_data:
            print(f"Warning: 'type_map' not found in {mat_path}. Skipping.")
            return

        type_map = mat_data['type_map']

        # Ensure type_map is a numpy array
        if not isinstance(type_map, np.ndarray):
            print(f"Warning: 'type_map' in {mat_path} is not a numpy array. Got {type_map}. Skipping.")
            return

        # Check dimensions; usually expected to be 2D (H, W)
        if type_map.ndim != 2:
             print(f"Warning: 'type_map' in {mat_path} is not 2D. Shape is {type_map.shape}. Attempting to handle...")
             type_map = np.squeeze(type_map)
             if type_map.ndim != 2:
                 print(f"Error: Could not reshape 'type_map' in {mat_path} to 2D. Final shape is {type_map.shape}. Skipping.")
                 return

        # Check data type
        if type_map.dtype.kind not in ['i', 'u']:
            print(f"Warning: 'type_map' in {mat_path} is not integer type ({type_map.dtype}). Attempting conversion...")
            try:
                type_map = type_map.astype(np.int32)
            except (ValueError, TypeError) as e:
                print(f"Error: Could not convert 'type_map' in {mat_path} to integer. Error: {e}. Skipping.")
                return

        # Validate value range (optional, for debugging)
        unique_vals = np.unique(type_map)
        min_val = type_map.min()
        max_val = type_map.max()
        print(f"  - 'type_map' values range from {min_val} to {max_val}. Unique values: {unique_vals}")

        # Convert numpy array to a PIL image and save
        # Use 'L' mode because type_map values are in 0-7, within uint8 range
        mask_image = Image.fromarray(type_map.astype(np.uint8), mode='L') # 'L' mode for single channel 0-255
        mask_image.save(output_mask_path)

    except FileNotFoundError:
        print(f"Error: .mat file not found - {mat_path}")
    except Exception as e:
        print(f"Error processing .mat file {mat_path}: {e}")

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

def process_consep_dataset_unified(input_base_dir, output_base_dir):
    """
    Process CoNSeP Train and Test splits, read 'type_map' to build semantic masks, and write unified outputs.

    Args:
        input_base_dir (str): Root directory of the input dataset (e.g., /path/to/.../CoNSeP/CoNSeP/).
        output_base_dir (str): Root directory of the output dataset.
    """
    # Define subdirectories to process
    subdirs_to_process = ['Train', 'Test'] # Note: CoNSeP subdirectory names are capitalized 'Train' and 'Test'

    # Create output directories
    images_output_dir = os.path.join(output_base_dir, "images")
    masks_output_dir = os.path.join(output_base_dir, "masks")
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(masks_output_dir, exist_ok=True)

    processed_count = 0  # Used to generate a globally unique index

    print(f"Processing CoNSeP dataset from subdirectories: {subdirs_to_process}")
    print(f"Output will be saved to: {output_base_dir}")
    print("Reading 'type_map' from .mat files for semantic segmentation.")

    for subdir_name in subdirs_to_process:
        print(f"\n--- Processing {subdir_name} ---")
        current_input_dir = os.path.join(input_base_dir, subdir_name)
        
        if not os.path.exists(current_input_dir):
            print(f"Warning: Directory {current_input_dir} does not exist. Skipping.")
            continue

        images_input_dir = os.path.join(current_input_dir, "Images")
        labels_input_dir = os.path.join(current_input_dir, "Labels")

        if not os.path.exists(images_input_dir):
            print(f"Warning: Images directory {images_input_dir} does not exist. Skipping {subdir_name}.")
            continue
        if not os.path.exists(labels_input_dir):
            print(f"Warning: Labels directory {labels_input_dir} does not exist. Skipping {subdir_name}.")
            continue

        # Get image and label file lists
        # Based on the dataset layout, images are usually .png
        img_files = sorted([f for f in os.listdir(images_input_dir) if f.lower().endswith('.png')])
        mat_files = sorted([f for f in os.listdir(labels_input_dir) if f.lower().endswith('.mat')])

        # Ensure the number of images and labels match
        if len(img_files) != len(mat_files):
            print(f"Warning in {subdir_name}: Number of images ({len(img_files)}) and .mat labels ({len(mat_files)}) do not match. Using minimum count.")
            min_count = min(len(img_files), len(mat_files))
        else:
            min_count = len(img_files)

        # Iterate over image and label files
        for i in range(min_count):
            img_file = img_files[i]
            mat_file = mat_files[i]

            # Try to match image and label names (usually only the extension differs)
            img_name_base = os.path.splitext(img_file)[0]
            mat_name_base = os.path.splitext(mat_file)[0]

            if img_name_base != mat_name_base:
                print(f"Warning in {subdir_name}: Image name '{img_name_base}' does not match .mat name '{mat_name_base}'. Processing based on list order.")

            img_path = os.path.join(images_input_dir, img_file)
            mat_path = os.path.join(labels_input_dir, mat_file)

            # Use a global counter to generate a unique base filename
            unique_base_name = f"consep_{subdir_name}_sample_{processed_count:06d}"

            # Define output paths and ensure filenames are unique
            output_img_path = get_unique_filename(images_output_dir, unique_base_name, '.png')
            output_mask_path = get_unique_filename(masks_output_dir, unique_base_name, '.png')

            # Copy and convert image (optionally unify to RGB PNG here)
            try:
                img_to_copy = Image.open(img_path).convert('RGB') # Ensure RGB mode
                img_to_copy.save(output_img_path, format='PNG') # Save as PNG
                print(f"Copied image: {img_file} -> {os.path.basename(output_img_path)}")
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue # If image processing fails, skip the corresponding mask

            # Convert and save semantic mask (from type_map)
            try:
                convert_mat_to_semantic_mask_from_type_map(mat_path, output_mask_path)
                print(f"Converted .mat label to semantic mask (from type_map): {mat_file} -> {os.path.basename(output_mask_path)}")
            except Exception as e:
                print(f"Error converting .mat label {mat_path} to semantic mask: {e}")
                # If mask processing fails after the image was copied, optionally delete the image to keep pairs, or keep it.
                # Here we keep it.

            processed_count += 1 # Increment global counter

    print("\nProcessing completed. All images and masks are saved in the same output directories.")


if __name__ == "__main__":
    # Define input and output directories
    INPUT_BASE_DIR = "/path/to/PFM_Segmentation_Data/CoNSeP/CoNSeP/"
    OUTPUT_BASE_DIR = "/path/to/PFM_Segmentation_Data/CoNSeP/" # Modify this path as needed

    print("Starting unified CoNSeP dataset processing for semantic segmentation (Train & Test) using 'type_map'...")
    process_consep_dataset_unified(INPUT_BASE_DIR, OUTPUT_BASE_DIR)
    print("Unified CoNSeP dataset processing for semantic segmentation completed.")
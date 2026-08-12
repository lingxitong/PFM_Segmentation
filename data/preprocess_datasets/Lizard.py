import os
import scipy.io
import numpy as np
from PIL import Image

def convert_mat_to_semantic_mask_from_lizard(mat_path, output_mask_path):
    """
    Load 'inst_map', 'id', and 'class' from a Lizard .mat file and convert to a single-channel PNG semantic mask.
    Handle cases where 'id' and 'class' may have shape (N,) or (N, 1).

    Args:
        mat_path (str): Path to the input .mat file.
        output_mask_path (str): Path to the output single-channel PNG semantic mask.
    """
    try:
        # Load .mat file
        mat_data = scipy.io.loadmat(mat_path, squeeze_me=True, matlab_compatible=False)

        # Check that required keys exist
        required_keys = ['inst_map', 'id', 'class']
        for key in required_keys:
            if key not in mat_data:
                print(f"Warning: Required key '{key}' not found in {mat_path}. Skipping.")
                return

        inst_map = mat_data['inst_map']
        nuclei_id = mat_data['id'] # Shape: (N,) or (N, 1)
        classes = mat_data['class'] # Shape: (N,) or (N, 1)

        # Ensure inst_map is a numpy array
        if not isinstance(inst_map, np.ndarray):
            print(f"Warning: 'inst_map' in {mat_path} is not a numpy array. Got {type(inst_map)}. Skipping.")
            return

        # Check dimensions; usually expected to be 2D (H, W)
        if inst_map.ndim != 2:
             print(f"Warning: 'inst_map' in {mat_path} is not 2D. Shape is {inst_map.shape}. Attempting to handle...")
             inst_map = np.squeeze(inst_map)
             if inst_map.ndim != 2:
                 print(f"Error: Could not reshape 'inst_map' in {mat_path} to 2D. Final shape is {inst_map.shape}. Skipping.")
                 return

        # Ensure nuclei_id and classes are numpy arrays
        if not isinstance(nuclei_id, np.ndarray):
            print(f"Warning: 'id' in {mat_path} is not a numpy array. Got {type(nuclei_id)}. Skipping.")
            return
        if not isinstance(classes, np.ndarray):
            print(f"Warning: 'class' in {mat_path} is not a numpy array. Got {type(classes)}. Skipping.")
            return

        # Normalize possible shapes (N,) or (N, 1)
        if nuclei_id.ndim == 2 and nuclei_id.shape[1] == 1:
            nuclei_id_flat = np.squeeze(nuclei_id, axis=1) # Shape: (N,)
        elif nuclei_id.ndim == 1:
            nuclei_id_flat = nuclei_id # Shape: (N,)
        else:
            print(f"Warning: 'id' in {mat_path} has unexpected shape {nuclei_id.shape}. Skipping.")
            return

        if classes.ndim == 2 and classes.shape[1] == 1:
            classes_flat = np.squeeze(classes, axis=1) # Shape: (N,)
        elif classes.ndim == 1:
            classes_flat = classes # Shape: (N,)
        else:
            print(f"Warning: 'class' in {mat_path} has unexpected shape {classes.shape}. Skipping.")
            return

        # Get all unique values in inst_map (excluding background 0)
        unique_inst_ids = np.unique(inst_map)
        # Create an array with the same shape as inst_map, initialized as background (0)
        semantic_mask = np.zeros_like(inst_map, dtype=np.uint8)

        # Iterate over each non-background instance id
        for inst_id in unique_inst_ids:
            if inst_id == 0: # Skip background
                continue
            # Find the index of this instance id in nuclei_id_flat
            try:
                idx = np.where(nuclei_id_flat == inst_id)[0][0]
                # Get the corresponding class
                class_val = int(classes_flat[idx])
                # Assign class_val to all pixels equal to inst_id in inst_map
                semantic_mask[inst_map == inst_id] = class_val
            except IndexError:
                print(f"Warning: Instance ID {inst_id} from 'inst_map' not found in 'id' array in {mat_path}. Assigning to background (0).")
                # If the id is missing, treat it as background
                semantic_mask[inst_map == inst_id] = 0

        # Validate value range (optional, for debugging)
        unique_vals = np.unique(semantic_mask)
        min_val = semantic_mask.min()
        max_val = semantic_mask.max()
        print(f"  - Generated semantic mask values range from {min_val} to {max_val}. Unique values: {unique_vals}")

        # Convert numpy array to a PIL image and save
        # Use 'L' mode because class values are in 1-6, within uint8 range
        mask_image = Image.fromarray(semantic_mask, mode='L') # 'L' mode for single channel 0-255
        mask_image.save(output_mask_path)

    except FileNotFoundError:
        print(f"Error: .mat file not found - {mat_path}")
    except Exception as e:
        print(f"Error processing .mat file {mat_path}: {e}")

def process_lizard_dataset_unified(input_base_dir, output_base_dir):
    """
    Process the Lizard dataset, build semantic masks from 'inst_map'/'id'/'class', and write unified outputs.
    Match images and labels by filename; keep output filenames the same as inputs.

    Args:
        input_base_dir (str): Root directory of the input dataset (e.g., /path/to/.../Lizard/).
        output_base_dir (str): Root directory of the output dataset.
    """
    # Define parent directories of image subfolders
    image_parent_subdirs = ['lizard_images1', 'lizard_images2']

    # Define label directory
    labels_input_dir = os.path.join(input_base_dir, "lizard_labels", "Lizard_Labels", "Labels")

    # Create output directories
    images_output_dir = os.path.join(output_base_dir, "images")
    masks_output_dir = os.path.join(output_base_dir, "masks")
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(masks_output_dir, exist_ok=True)

    processed_count = 0  # Count of successfully processed image-label pairs

    print(f"Processing Lizard dataset from image parent directories: {image_parent_subdirs}")
    print(f"Labels are located in: {labels_input_dir}")
    print(f"Output will be saved to: {output_base_dir}")
    print("Generating semantic segmentation masks from instance maps and class labels.")
    print("Matching images and labels by filename.")
    print("Output filenames will match input filenames.")

    # Collect all image filenames and full paths
    all_img_files = []
    img_full_paths = [] # Store full image paths

    for parent_subdir_name in image_parent_subdirs:
        # Build parent directory path
        parent_input_dir = os.path.join(input_base_dir, parent_subdir_name)
        if not os.path.exists(parent_input_dir):
            print(f"Warning: Parent image directory {parent_input_dir} does not exist. Skipping.")
            continue

        # Look for child directories named "Lizard_Images1" or "Lizard_Images2"
        # Determine the child directory name from the parent name
        if parent_subdir_name == 'lizard_images1':
            child_dir_name = 'Lizard_Images1'
        elif parent_subdir_name == 'lizard_images2':
            child_dir_name = 'Lizard_Images2'
        else:
            print(f"Warning: Unknown parent directory name: {parent_subdir_name}. Skipping.")
            continue

        child_input_dir = os.path.join(parent_input_dir, child_dir_name)
        if not os.path.exists(child_input_dir):
            print(f"Warning: Child image directory {child_input_dir} does not exist. Skipping {parent_subdir_name}.")
            continue

        # Get all PNG files under the current child directory
        img_files_in_child_dir = [f for f in os.listdir(child_input_dir) if f.lower().endswith('.png')]
        # Append to the global lists
        for img_file in img_files_in_child_dir:
            all_img_files.append(img_file)
            img_full_paths.append(os.path.join(child_input_dir, img_file))

    # Get all label files
    if not os.path.exists(labels_input_dir):
        print(f"Error: Labels directory {labels_input_dir} does not exist. Aborting.")
        return

    all_mat_files = [f for f in os.listdir(labels_input_dir) if f.lower().endswith('.mat')]

    # Build a dict mapping label base names to full paths
    mat_file_map = {}
    for mat_file in all_mat_files:
        mat_base_name = os.path.splitext(mat_file)[0]
        mat_file_map[mat_base_name] = os.path.join(labels_input_dir, mat_file)

    print(f"Found {len(all_img_files)} images and {len(all_mat_files)} label files.")
    print(f"Found {len(mat_file_map)} unique label base names.")

    # Iterate over image files
    for i, img_file in enumerate(all_img_files):
        img_name_base = os.path.splitext(img_file)[0]
        img_path = img_full_paths[i]

        # Try to find the matching label file in mat_file_map
        if img_name_base in mat_file_map:
            mat_path = mat_file_map[img_name_base]
            mat_file = os.path.basename(mat_path) # Get the actual .mat filename
            print(f"Found matching label for image '{img_file}': '{mat_file}'")
        else:
            print(f"Warning: No matching label found for image '{img_file}'. Skipping.")
            continue

        # Use the original filename as the output filename
        output_img_path = os.path.join(images_output_dir, img_file) # Keep the original image filename
        output_mask_path = os.path.join(masks_output_dir, img_file) # Keep the original name; extension will be .png

        # Check whether the output filename already exists; with no renaming, conflicts are unlikely unless old files remain
        # Add overwrite checks here if needed; we keep original names as requested
        # (If names collide, PIL save will overwrite)

        # Copy and convert image
        try:
            img_to_copy = Image.open(img_path).convert('RGB') # Ensure RGB mode
            img_to_copy.save(output_img_path, format='PNG') # Save as PNG, overwrites if exists
            print(f"Copied image: {img_file} -> {os.path.basename(output_img_path)}")
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue # If image processing fails, skip the corresponding mask

        # Convert and save semantic mask (from inst_map, id, class)
        try:
            convert_mat_to_semantic_mask_from_lizard(mat_path, output_mask_path)
            # Make the printed mask output name explicitly end with .png
            mask_output_name = os.path.splitext(img_file)[0] + '.png'
            print(f"Converted .mat label to semantic mask: {mat_file} -> {mask_output_name}")
             # output_mask_path is already .png, so mask_output_name matches the saved filename
             # For clarity, we can directly print the basename of output_mask_path
            print(f"Saved semantic mask: {os.path.basename(output_mask_path)}")
        except Exception as e:
            print(f"Error converting .mat label {mat_path} to semantic mask: {e}")
            # If mask processing fails after the image was copied, optionally delete the image to keep pairs, or keep it.
            # Here we keep it.

        processed_count += 1 # Increment processed counter

    print(f"\nProcessing completed. Successfully processed {processed_count} image-label pairs.")
    print(f"All images and masks are saved in the same output directories: {images_output_dir} and {masks_output_dir}")


if __name__ == "__main__":
    # Define input and output directories
    INPUT_BASE_DIR = "/path/to/PFM_Segmentation_Data/Lizard/"
    OUTPUT_BASE_DIR = "/path/to/PFM_Segmentation_Data/Lizard/" # Modify this path as needed

    print("Starting unified Lizard dataset processing for semantic segmentation...")
    process_lizard_dataset_unified(INPUT_BASE_DIR, OUTPUT_BASE_DIR)
    print("Unified Lizard dataset processing for semantic segmentation completed.")
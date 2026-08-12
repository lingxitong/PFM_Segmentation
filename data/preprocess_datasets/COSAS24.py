import os
import shutil
from pathlib import Path

def copy_images_masks(source_dir, destination_dir):
    """
    Collect all image and mask files from the source directory tree and copy them into images/ and masks/ under the destination.

    Assumed directory structure:
    source_dir/
    ├── SubFolderA/
    │   ├── Folder1/
    │   │   ├── images/
    │   │   │   └── ...
    │   │   └── masks/
    │   │       └── ...
    │   ├── Folder2/
    │   │   ├── images/
    │   │   └── masks/
    │   └── Folder3/
    │       ├── images/
    │       └── masks/
    └── SubFolderB/
        ├── Folder1/
        │   ├── images/
        │   └── masks/
        ├── Folder2/
        │   ├── images/
        │   └── masks/
        └── Folder3/
            ├── images/
            └── masks/

    Args:
        source_dir (str or Path): Path to the source folder.
        destination_dir (str or Path): Path to the destination folder.
    """
    source_path = Path(source_dir)
    dest_path = Path(destination_dir)

    # Create destination directories
    images_dest = dest_path / 'images'
    masks_dest = dest_path / 'masks'
    images_dest.mkdir(parents=True, exist_ok=True)
    masks_dest.mkdir(parents=True, exist_ok=True)

    print(f"Collecting files from '{source_path}'...")
    print(f"Images will be copied to: '{images_dest}'")
    print(f"Masks will be copied to: '{masks_dest}'")

    # Iterate over top-level subfolders under the source directory (SubFolderA, SubFolderB, etc.)
    for sub_folder in source_path.iterdir():
        if not sub_folder.is_dir():
            continue

        # Iterate over nested folders under each subfolder (Folder1, Folder2, Folder3, etc.)
        for folder in sub_folder.iterdir():
            if not folder.is_dir():
                continue

            # Build paths to image and mask subdirectories
            images_src = folder / 'image'
            masks_src = folder / 'mask'

            # Check whether the source image directory exists
            if images_src.is_dir():
                for image_file in images_src.iterdir():
                    if image_file.is_file():
                        # Build destination path; prefix with source path parts to avoid name collisions
                        # Example: original_name.png -> SubFolderA_Folder1_original_name.png
                        unique_name = f"{sub_folder.name}_{folder.name}_{image_file.name}"
                        dest_file_path = images_dest / unique_name
                        shutil.copyfile(image_file, dest_file_path)
                        print(f"Copied image: {image_file} -> {dest_file_path}")  # Optional: print copy info

            # Check whether the source mask directory exists
            if masks_src.is_dir():
                for mask_file in masks_src.iterdir():
                    if mask_file.is_file():
                        unique_name = f"{sub_folder.name}_{folder.name}_{mask_file.name}"
                        dest_file_path = masks_dest / unique_name
                        shutil.copyfile(mask_file, dest_file_path)
                        print(f"Copied mask: {mask_file} -> {dest_file_path}")  # Optional: print copy info

    print(f"File collection and copy finished! Please check '{dest_path}'.")

# --- Example usage ---
if __name__ == "__main__":
    # Replace with your actual source directory path
    source_directory = "/path/to/PFM_Segmentation_Data/COSAS24/COSAS24-TrainingSet/"
    # Replace with your desired destination directory path
    destination_directory = "/path/to/PFM_Segmentation_Data/COSAS24/"

    # Run the conversion
    copy_images_masks(source_directory, destination_directory)

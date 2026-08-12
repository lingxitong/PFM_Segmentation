import os
from PIL import Image
import numpy as np


def convert_mask_to_binary(mask_arr):
    """
    Convert an original CRAG mask to a single-channel semantic mask.
    Non-zero pixels (e.g., 255 in masks_ori) become class 1 (gland);
    zeros remain background (0).
    """
    return (mask_arr > 0).astype(np.uint8)


def process_crag_dataset(input_root, output_root=None, mask_src_dirname="masks_ori"):
    """
    Process CRAG into the unified images/ + masks/ layout.

    Expected input layout (after initial organization):
        input_root/
            images/       # RGB images, e.g. train_1.png, test_1.png
            masks_ori/    # original masks with values {0, 255}

    Output layout:
        output_root/
            images/       # copied RGB images (PNG)
            masks/        # semantic masks with values {0, 1}

    Args:
        input_root (str): Root directory containing images/ and masks_ori/.
        output_root (str): Output root. Defaults to input_root.
        mask_src_dirname (str): Source mask folder name. Default: "masks_ori".
    """
    if output_root is None:
        output_root = input_root

    images_input_dir = os.path.join(input_root, "images")
    masks_input_dir = os.path.join(input_root, mask_src_dirname)

    if not os.path.isdir(images_input_dir):
        raise FileNotFoundError(f"Images directory not found: {images_input_dir}")
    if not os.path.isdir(masks_input_dir):
        raise FileNotFoundError(f"Mask directory not found: {masks_input_dir}")

    images_output_dir = os.path.join(output_root, "images")
    masks_output_dir = os.path.join(output_root, "masks")
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(masks_output_dir, exist_ok=True)

    image_files = sorted(
        [f for f in os.listdir(images_input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
    )
    print(f"Found {len(image_files)} images under {images_input_dir}")
    print(f"Reading original masks from: {masks_input_dir}")
    print(f"Writing semantic masks (0: background, 1: gland) to: {masks_output_dir}")

    processed_count = 0
    skipped_count = 0

    for img_file in image_files:
        stem = os.path.splitext(img_file)[0]
        # Prefer a same-stem .png mask; fall back to same filename
        mask_candidates = [f"{stem}.png", img_file]
        mask_file = None
        for cand in mask_candidates:
            cand_path = os.path.join(masks_input_dir, cand)
            if os.path.exists(cand_path):
                mask_file = cand
                break

        if mask_file is None:
            print(f"Warning: no matching mask for image {img_file}. Skipping.")
            skipped_count += 1
            continue

        img_path = os.path.join(images_input_dir, img_file)
        mask_path = os.path.join(masks_input_dir, mask_file)

        try:
            image = Image.open(img_path).convert("RGB")
            out_img_name = f"{stem}.png"
            image.save(os.path.join(images_output_dir, out_img_name), format="PNG")
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            skipped_count += 1
            continue

        try:
            mask = Image.open(mask_path).convert("L")
            mask_arr = np.array(mask)
            binary_mask = convert_mask_to_binary(mask_arr)
            Image.fromarray(binary_mask, mode="L").save(
                os.path.join(masks_output_dir, f"{stem}.png")
            )
        except Exception as e:
            print(f"Error processing mask {mask_path}: {e}")
            skipped_count += 1
            continue

        processed_count += 1
        if processed_count % 50 == 0:
            print(f"Processed {processed_count}/{len(image_files)} samples...")

    print(
        f"Processing completed. Success: {processed_count}, skipped: {skipped_count}. "
        f"Output saved to: {output_root}"
    )


if __name__ == "__main__":
    # CRAG root should contain images/ and masks_ori/ ({0, 255}).
    # This script writes masks/ with values {0, 1}.
    INPUT_ROOT = "/path/to/PFM_Segmentation_Data/CRAG/"
    OUTPUT_ROOT = "/path/to/PFM_Segmentation_Data/CRAG/"

    print("Starting CRAG dataset processing for semantic segmentation...")
    process_crag_dataset(INPUT_ROOT, OUTPUT_ROOT, mask_src_dirname="masks_ori")
    print("CRAG dataset processing completed.")

import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

def merge_masks(gland_path, tumor_path, num_classes=3):
    """
    Read 1-bit PNG masks for gland and tumor, merge into a single-channel mask,
    pixel values: 0=background, 1=gland, 2=tumor.
    If gland and tumor overlap, priority is configurable (here tumor overrides gland).
    """
    if gland_path and os.path.exists(gland_path):
        gland = np.array(Image.open(gland_path).convert('L'))
        gland = (gland > 0).astype(np.uint8)  # Binarize to 0/1
    else:
        gland = None

    if tumor_path and os.path.exists(tumor_path):
        tumor = np.array(Image.open(tumor_path).convert('L'))
        tumor = (tumor > 0).astype(np.uint8)
    else:
        tumor = None

    # Initialize merged mask as all zeros (background)
    if gland is not None:
        merged = np.zeros_like(gland, dtype=np.uint8)
    elif tumor is not None:
        merged = np.zeros_like(tumor, dtype=np.uint8)
    else:
        raise ValueError("Both gland and tumor masks are missing.")

    # Assign gland first (label=1)
    if gland is not None:
        merged[gland == 1] = 1

    # Then assign tumor (label=2), overriding gland on overlaps
    if tumor is not None:
        merged[tumor == 1] = 2

    # Ensure values are in [0, num_classes)
    assert merged.max() < num_classes, f"Mask contains value >= {num_classes}"
    return merged

def process_dataset(input_dir, output_dir, num_classes=3):
    splits = ['TRAIN', 'TEST']
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / 'images').mkdir(exist_ok=True)
    (Path(output_dir) / 'masks').mkdir(exist_ok=True)

    for split in splits:
        split_path = Path(input_dir) / split
        if not split_path.exists():
            print(f"Split {split} not found, skipping.")
            continue

        images_dir = split_path / 'IMAGES'
        glands_dir = split_path / 'MANUAL GLANDS'
        tumors_dir = split_path / 'MANUAL TUMOR'

        if not images_dir.exists():
            print(f"Images folder not found in {split}, skipping.")
            continue

        image_files = sorted([f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
        for img_path in image_files:
            stem = img_path.stem

            # Build possible annotation paths (supports .png)
            gland_path = glands_dir / f"{stem}.png"
            tumor_path = tumors_dir / f"{stem}.png"

            # Read and merge masks
            try:
                mask = merge_masks(
                    gland_path if gland_path.exists() else None,
                    tumor_path if tumor_path.exists() else None,
                    num_classes=num_classes
                )
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

            # Copy original image (keep original format)
            dst_img = Path(output_dir) / 'images' / img_path.name
            shutil.copyfile(img_path, dst_img)

            # Save merged mask (single-channel PNG with values 0,1,2,...)
            mask_img = Image.fromarray(mask, mode='L')
            dst_mask = Path(output_dir) / 'masks' / f"{stem}.png"
            mask_img.save(dst_mask)

    print(f"Dataset processed and saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge gland and tumor masks and organize dataset.")
    parser.add_argument("--input_dir", default="/path/to/PFM_Segmentation_Data/RINGS/RINGS_algorithm_dataset/",help="Path to input folder containing TRAIN/TEST")
    parser.add_argument("--output_dir", default="/path/to/PFM_Segmentation_Data/RINGS/RINGS_algorithm_dataset/", help="Path to output folder")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes (including background)")
    args = parser.parse_args()

    process_dataset(args.input_dir, args.output_dir, args.num_classes)
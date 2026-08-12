import h5py
import numpy as np
import os
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

def analyze_classes(hdf5_path):
    """
    Analyze class distribution in the CoCaHis.hdf5 dataset
    """
    print("Analyzing dataset classes...")
    
    with h5py.File(hdf5_path, 'r') as f:
        gt = f["GT/GT_majority_vote"][()]
        
        print(f"Ground truth shape: {gt.shape}")
        
        # Collect all unique values
        all_unique_values = set()
        class_counts = {}
        
        for i in range(len(gt)):
            mask = gt[i].astype(np.uint8)
            unique_values = np.unique(mask)
            
            # Update the set of all unique values
            all_unique_values.update(unique_values)
            
            # Count pixels for each class
            flat_mask = mask.flatten()
            counter = Counter(flat_mask)
            
            for class_id, count in counter.items():
                if class_id not in class_counts:
                    class_counts[class_id] = 0
                class_counts[class_id] += count
        
        # Sort and display class information
        sorted_classes = sorted(list(all_unique_values))
        
        print(f"\nFound {len(sorted_classes)} classes: {sorted_classes}")
        print("\nClass distribution:")
        for class_id in sorted_classes:
            print(f"  Class {class_id}: {class_counts[class_id]} pixels")
        
        return sorted_classes, class_counts

def convert_cocahis_dataset(hdf5_path, output_dir, num_classes=None):
    """
    Convert the CoCaHis.hdf5 dataset to a standard format
    - images: store original images
    - masks: store PNG masks
    
    Args:
        hdf5_path: Path to the CoCaHis.hdf5 file
        output_dir: Output directory path
        num_classes: Number of classes (auto-analyzed if None)
    """
    
    # Automatically analyze classes
    if num_classes is None:
        classes, class_counts = analyze_classes(hdf5_path)
        num_classes = max(classes) + 1  # Classes start from 0, so add 1
        print(f"\nAutomatically determined number of classes: {num_classes}")
        print(f"Classes found: {classes}")
    else:
        classes, class_counts = analyze_classes(hdf5_path)
        print(f"\nUsing provided number of classes: {num_classes}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    
    print(f"\nReading dataset from: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Read data
        raw_images = f["HE/raw"][()]
        gt = f["GT/GT_majority_vote"][()]
        
        print(f"Raw images shape: {raw_images.shape}")
        print(f"Ground truth shape: {gt.shape}")
        
        # Get dataset metadata
        img_num = f["HE/"].attrs["image_num"]
        patients = f["HE/"].attrs["patient_num"]
        trtst = f["HE/"].attrs["train_test_split"]
        
        print(f"Total images: {img_num}")
        print(f"Total patients: {patients}")
        print(f"Train/test split: {trtst}")
    
    # Process each image
    total_pixels = 0
    class_pixel_counts = {class_id: 0 for class_id in classes}
    
    for i in range(len(raw_images)):
        # Use the same filename prefix so images and masks match
        filename_prefix = f"sample_{i:04d}"
        
        # Process original image
        image = raw_images[i]
        
        # Ensure the image has the correct dtype
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Save the image using the same prefix as the mask
        image_path = os.path.join(output_dir, "images", f"{filename_prefix}.png")
        Image.fromarray(image).save(image_path)
        
        # Process mask
        mask = gt[i]
        
        # Ensure the mask is an integer type
        mask = mask.astype(np.uint8)
        
        # Validate the mask value range
        unique_values = np.unique(mask)
        print(f"Image {i}: mask values range from {unique_values.min()} to {unique_values.max()}")
        
        # Count class distribution for the current image
        for val in unique_values:
            if val in class_pixel_counts:
                class_pixel_counts[val] += np.sum(mask == val)
        total_pixels += mask.size
        
        # Clip mask values if they exceed the valid range
        mask = np.clip(mask, 0, num_classes - 1)
        
        # Ensure the mask is single-channel
        if len(mask.shape) == 3:
            mask = mask.squeeze(-1) if mask.shape[-1] == 1 else mask
        elif len(mask.shape) == 4:
            mask = mask.squeeze()
        
        # Save the mask using the same prefix as the image
        mask_path = os.path.join(output_dir, "masks", f"{filename_prefix}.png")
        Image.fromarray(mask, mode='L').save(mask_path)
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(raw_images)} images")
    
    # Display class statistics
    print(f"\nFinal class distribution across all masks:")
    for class_id in sorted(classes):
        percentage = (class_pixel_counts[class_id] / total_pixels) * 100
        print(f"  Class {class_id}: {class_pixel_counts[class_id]} pixels ({percentage:.2f}%)")
    
    print(f"\nDataset conversion completed!")
    print(f"Output saved to: {output_dir}")
    print(f"Images saved to: {os.path.join(output_dir, 'images')}")
    print(f"Masks saved to: {os.path.join(output_dir, 'masks')}")
    
    # Verify conversion results
    verify_conversion(output_dir, num_classes)

def verify_conversion(output_dir, num_classes):
    """Verify conversion results."""
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
    
    print(f"\nVerification:")
    print(f"Number of images: {len(image_files)}")
    print(f"Number of masks: {len(mask_files)}")
    
    if len(image_files) != len(mask_files):
        print("Warning: Number of images and masks don't match!")
        return
    
    # Verify that filenames match one-to-one
    image_basenames = {os.path.splitext(f)[0] for f in image_files}
    mask_basenames = {os.path.splitext(f)[0] for f in mask_files}
    
    if image_basenames != mask_basenames:
        print("Warning: Image and mask filenames don't match!")
        missing_in_masks = image_basenames - mask_basenames
        missing_in_images = mask_basenames - image_basenames
        if missing_in_masks:
            print(f"Missing masks for: {missing_in_masks}")
        if missing_in_images:
            print(f"Missing images for: {missing_in_images}")
    else:
        print("Success: All image and mask filenames match!")
    
    # Check a few samples
    for i in range(min(5, len(image_files))):
        mask_path = os.path.join(masks_dir, mask_files[i])
        mask = np.array(Image.open(mask_path))
        
        unique_values = np.unique(mask)
        print(f"Sample {i}: mask values {unique_values} (max allowed: {num_classes-1})")
        
        if unique_values.max() >= num_classes:
            print(f"Warning: Mask {mask_files[i]} contains values > {num_classes-1}")
        if unique_values.min() < 0:
            print(f"Warning: Mask {mask_files[i]} contains negative values")

def visualize_sample(output_dir, sample_idx=0):
    """Visualize one sample to verify the conversion."""
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
    
    if sample_idx >= len(image_files):
        print(f"Sample index {sample_idx} exceeds available samples ({len(image_files)})")
        return
    
    # Get the corresponding filename (same stem for image and mask)
    base_filename = os.path.splitext(image_files[sample_idx])[0]
    
    # Verify that image and mask filenames match
    expected_mask_filename = f"{base_filename}.png"
    if expected_mask_filename != mask_files[sample_idx]:
        print(f"Warning: Image and mask filenames don't match!")
        print(f"Image: {image_files[sample_idx]}")
        print(f"Mask: {mask_files[sample_idx]}")
        return
    
    # Load image and mask
    image_path = os.path.join(images_dir, image_files[sample_idx])
    mask_path = os.path.join(masks_dir, mask_files[sample_idx])
    
    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path))
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image - {base_filename}")
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='tab20', vmin=0, vmax=20)  # Display mask with a color map
    axes[1].set_title(f"Mask - {base_filename}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample_visualization_{base_filename}.png"), dpi=150)
    plt.show()
    
    print(f"Sample visualization saved as sample_visualization_{base_filename}.png")

# Example usage
if __name__ == "__main__":
    # Set paths
    hdf5_path = "/path/to/PFM_Segmentation_Data/CoCaHis/CoCaHis.hdf5"
    output_dir = "/path/to/PFM_Segmentation_Data/CoCaHis/"  # Specified output directory
    
    # Auto-analyze classes and convert the dataset
    convert_cocahis_dataset(hdf5_path, output_dir)
    
    # Visualize one sample for verification
    visualize_sample(output_dir, sample_idx=0)
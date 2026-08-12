import numpy as np
import os
from PIL import Image

images_npy_path = '/path/to/PFM_Segmentation_Data/CoNIC2022/images.npy'
labels_npy_path = '/path/to/PFM_Segmentation_Data/CoNIC2022/labels.npy'
output_dir = 'dataset'
images_out_dir = os.path.join(output_dir, 'images')
masks_out_dir = os.path.join(output_dir, 'masks')

os.makedirs(images_out_dir, exist_ok=True)
os.makedirs(masks_out_dir, exist_ok=True)

print("Loading images...")
images = np.load(images_npy_path)  # (N, 256, 256, 3), uint8
print("Loading labels...")
labels = np.load(labels_npy_path)  # (N, 256, 256, 2), uint16

assert images.shape[0] == labels.shape[0], "Number of images and labels must match!"
N = images.shape[0]

class_maps = labels[:, :, :, 1]  # shape: (N, 256, 256)
unique_vals = np.unique(class_maps)
print(f"Unique values in classification map: {unique_vals}")
assert np.all((class_maps >= 0) & (class_maps <= 6)), "Class labels should be in 0-6"

for i in range(N):
  
    img = images[i]  # (256, 256, 3), uint8
    img_pil = Image.fromarray(img, mode='RGB')
    img_pil.save(os.path.join(images_out_dir, f'{i:04d}.png'))

    mask = class_maps[i].astype(np.uint8)  # (256, 256)
    mask_pil = Image.fromarray(mask, mode='L')  # 'L' for grayscale
    mask_pil.save(os.path.join(masks_out_dir, f'{i:04d}.png'))

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{N} samples...")

print(f"Conversion complete! Saved to '{output_dir}/'")
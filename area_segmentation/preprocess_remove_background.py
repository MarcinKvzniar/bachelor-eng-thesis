"""
Removes background-dominant images based on mask analysis.
Deletes both mask and corresponding clean images if the mask indicates >95% background.
"""

import os
import cv2
import numpy as np

clean_dir = os.path.abspath("")
mask_dir = os.path.abspath("")
THRESHOLD = 0.95

removed_count = 0
processed_count = 0

print(f"Processing clean directory: {clean_dir}")
print(f"Processing mask directory: {mask_dir}")

for folder_num in range(1, 14):
    mask_subdir = os.path.join(mask_dir, str(folder_num))
    clean_subdir = os.path.join(clean_dir, str(folder_num))

    if not os.path.isdir(mask_subdir) or not os.path.isdir(clean_subdir):
        print(f"Subdirectory {folder_num} does not exist in either mask or clean folder, skipping.")
        continue

    print(f"Processing subdirectory {folder_num}...")
    files_in_folder = 0

    for filename in os.listdir(mask_subdir):
        if not filename.endswith("_mask.png"):
            continue  

        mask_path = os.path.join(mask_subdir, filename)

        clean_filename = filename.replace("_mask.png", ".png")
        clean_path = os.path.join(clean_subdir, clean_filename)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Could not read {mask_path}, skipping.")
            continue
            
        processed_count += 1
        files_in_folder += 1

        total_pixels = mask.size
        
        if total_pixels == 0:
            print(f"Warning: {mask_path} has 0 pixels, skipping.")
            continue

        background_pixels = np.sum(mask == 0)
        background_ratio = background_pixels / total_pixels
        
        is_mostly_background = background_ratio > THRESHOLD
        
        if is_mostly_background:
            print(f"Removing (>{THRESHOLD*100:.0f}% background): {os.path.join(str(folder_num), filename)}")
            os.remove(mask_path)
            
            if os.path.exists(clean_path):
                os.remove(clean_path)
                removed_count += 1
            else:
                print(f"Corresponding clean image not found: {clean_path}")
    
    print(f"Processed {files_in_folder} files in folder {folder_num}")

print(f"\nTotal files processed: {processed_count}")
print(f"\nTotal files removed: {removed_count}")
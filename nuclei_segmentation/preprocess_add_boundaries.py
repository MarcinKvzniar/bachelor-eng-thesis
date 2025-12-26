"""
Utility functions to add a boundary class to annotated nuclei segmentation masks.
This script processes RGB masks, converts them to indexed format and adds a boundary class.
"""

import numpy as np
from skimage.morphology import disk, dilation
from PIL import Image
import os
from tqdm import tqdm

RGB_TO_CLASS = {
    (255, 255, 255): 0,  # Background
    (112, 112, 225): 1,  # Negative
    (250, 62, 62):   2   # Positive
}

OUTPUT_COLORMAP = [
    (255, 255, 255),  # 0: Background
    (112, 112, 225),  # 1: Negative (Blue)
    (250, 62, 62),    # 2: Positive (Red)
    (0, 0, 0),        # 3: Boundary (Black)
]

def rgb_to_indices(mask_rgb):
    mask_rgb = np.array(mask_rgb)
    
    if mask_rgb.shape[-1] == 4:
        mask_rgb = mask_rgb[..., :3]
        
    h, w, _ = mask_rgb.shape
    mask_indices = np.zeros((h, w), dtype=np.uint8)
    
    for color, class_idx in RGB_TO_CLASS.items():
        matches = np.all(mask_rgb == color, axis=-1)
        mask_indices[matches] = class_idx
        
    return mask_indices

def apply_colormap(pil_img, colormap):
    flat_colormap = []
    for color in colormap:
        flat_colormap.extend(color)
    flat_colormap.extend([0] * (768 - len(flat_colormap)))
    pil_img.putpalette(flat_colormap)
    return pil_img

def add_boundary_class_to_mask(mask_indices, boundary_class=3, thickness=2):
    objects_mask = (mask_indices > 0).astype(np.uint8)
    
    selem = disk(thickness)
    dilated_objects = dilation(objects_mask, selem)
    
    boundaries = dilated_objects.astype(bool) & (~objects_mask.astype(bool))
    
    mask_with_boundaries = mask_indices.copy()
    mask_with_boundaries[boundaries] = boundary_class
    
    return mask_with_boundaries

def process_folder(input_folder, output_folder, thickness=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    print(f"Found {len(files)} images to process.")

    for filename in tqdm(files, desc="Processing masks"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            with Image.open(input_path).convert('RGB') as img:
                
                mask_indices = rgb_to_indices(img)

            mask_with_boundaries = add_boundary_class_to_mask(mask_indices, boundary_class=3, thickness=thickness)
            
            mask_img = Image.fromarray(mask_with_boundaries.astype(np.uint8), mode='P')
            mask_img = apply_colormap(mask_img, OUTPUT_COLORMAP)
            mask_img.save(output_path)
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

INPUT_DIR = ""
OUTPUT_DIR = ""

process_folder(INPUT_DIR, OUTPUT_DIR, thickness=4)
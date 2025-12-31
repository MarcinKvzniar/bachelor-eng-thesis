"""
Deletes mask images that are mostly white and also deletes the corresponding
clean images from a parallel folder with the same structure.
"""

import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def count_non_white_pixels(image_path, white_threshold):
    """Count non-white pixels in image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    is_white = (
        (image_rgb[:, :, 0] >= white_threshold) &
        (image_rgb[:, :, 1] >= white_threshold) &
        (image_rgb[:, :, 2] >= white_threshold)
    )

    total_pixels = image_rgb.shape[0] * image_rgb.shape[1]
    non_white_pixels = np.sum(~is_white)

    return non_white_pixels, total_pixels

def remove_white_images_and_clean(mask_folder, clean_folder, min_content_percent=1.0, white_threshold=250, dry_run=True, verbose=True):
    """Remove mask images that are mostly white and corresponding clean images."""
    mask_path = Path(mask_folder)
    clean_path = Path(clean_folder)

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask folder does not exist: {mask_folder}")
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean folder does not exist: {clean_folder}")

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    image_files = []

    for ext in image_extensions:
        image_files.extend(mask_path.rglob(f"*{ext}"))
        image_files.extend(mask_path.rglob(f"*{ext.upper()}"))

    image_files = sorted(set(image_files))

    if verbose:
        print(f"\nProcessing {len(image_files)} mask images")
        if dry_run:
            print("DRY RUN MODE - nothing will be deleted")

    total = len(image_files)
    removed_masks = 0
    removed_cleans = 0
    kept = 0

    for mask_img in tqdm(image_files, desc="Processing masks", disable=not verbose):
        try:
            non_white, total_pixels = count_non_white_pixels(str(mask_img), white_threshold)
            pct = (non_white / total_pixels) * 100

            if pct < min_content_percent:
                removed_masks += 1

                mask_rel = mask_img.relative_to(mask_path)
                clean_img = clean_path / mask_rel
                clean_img = Path(str(clean_img).replace("_mask", ""))

                if verbose:
                    tqdm.write(f"Removing: {mask_img.name} ({pct:.2f}% content)")

                if not dry_run:
                    os.remove(mask_img)

                if clean_img.exists():
                    if verbose:
                        tqdm.write(f"Removing clean: {clean_img.name}")

                    if not dry_run:
                        os.remove(clean_img)
                    removed_cleans += 1

            else:
                kept += 1

        except Exception as e:
            if verbose:
                tqdm.write(f" Error on {mask_img.name}: {e}")

    if verbose:
        print(f"\nTotal: {total}, Kept: {kept}, Removed: {removed_masks} masks + {removed_cleans} cleans")
        if dry_run:
            print("DRY RUN - set dry_run=False to delete")

    return total, kept, removed_masks, removed_cleans


def main():
    MASK_FOLDER = ""
    CLEAN_FOLDER = ""

    MIN_CONTENT_PERCENT = 10.0
    WHITE_THRESHOLD = 250
    DRY_RUN = True

    remove_white_images_and_clean(
        mask_folder=MASK_FOLDER,
        clean_folder=CLEAN_FOLDER,
        min_content_percent=MIN_CONTENT_PERCENT,
        white_threshold=WHITE_THRESHOLD,
        dry_run=DRY_RUN,
        verbose=True
    )


if __name__ == "__main__":
    main()

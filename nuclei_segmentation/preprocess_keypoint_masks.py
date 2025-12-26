"""
Preprocess Annotated Masks Based on Keypoint Annotations

This script processes nuclei segmentation masks based on keypoint annotations:
1. Removes objects that do not have a keypoint inside them
2. Creates new nuclei objects using binary dilation where keypoints exist but no object is present
3. Saves all processed masks in a separate folder
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.morphology import disk, dilation
from skimage.segmentation import watershed
from tqdm import tqdm

IMG_HEIGHT, IMG_WIDTH = 512, 512

CLASS_MAPPING_4_CLASS = {
    (255, 255, 255): 0,  # White - Background
    (112, 112, 225): 1,  # Blue - Negative
    (250, 62, 62): 2,    # Red - Positive
    (0, 0, 0): 3,        # Black - Cell boundaries
}
INVERSE_CLASS_MAPPING_4_CLASS = {v: k for k, v in CLASS_MAPPING_4_CLASS.items()}


def load_keypoints_from_annotations(case_dir: Path) -> dict:
    """
    Load keypoints from annotations.json file for a single case.
    
    Args:
        case_dir: Path to the case directory containing annotations.json
        
    Returns:
        Dictionary mapping image_id to list of keypoints [(x, y, category_id)]
    """
    annotations_file = case_dir / "annotations.json"
    
    if not annotations_file.exists():
        return {}
    
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    images = {img['id']: img['file_name'] for img in data.get('images', [])}
    
    keypoints_by_image = defaultdict(list)
    
    for ann in data.get('annotations', []):
        image_id = ann['image_id']
        keypoints = ann.get('keypoints', [])
        category_id = ann.get('category_id', 0)
        
        for i in range(0, len(keypoints), 3):
            x, y, visibility = keypoints[i], keypoints[i+1], keypoints[i+2]
            if visibility > 0:
                keypoints_by_image[image_id].append((int(x), int(y), category_id))
    
    result = {}
    for image_id, kp_list in keypoints_by_image.items():
        if image_id in images:
            filename = images[image_id]
            result[filename] = kp_list
    
    return result


def load_keypoints_from_central_annotations(annotations_path: Path, case_id: str) -> dict:
    """
    Load keypoints from a central annotations.json file for a specific case.
    
    Args:
        annotations_path: Path to the central annotations.json file
        case_id: Case ID to filter annotations for
        
    Returns:
        Dictionary mapping filename to list of keypoints [(x, y, category_id)]
    """
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    images = {}
    for img in data.get('images', []):
        file_name = img['file_name']
        img_case_id = file_name.split('/')[0] if '/' in file_name else None
        
        if img_case_id == case_id:
            images[img['id']] = '/'.join(file_name.split('/')[1:]) if '/' in file_name else file_name
    
    keypoints_by_image = defaultdict(list)
    
    for ann in data.get('annotations', []):
        image_id = ann['image_id']
        
        if image_id not in images:
            continue
        
        keypoints = ann.get('keypoints', [])
        category_id = ann.get('category_id', 0)
        
        for i in range(0, len(keypoints), 3):
            x, y, visibility = keypoints[i], keypoints[i+1], keypoints[i+2]
            if visibility > 0:
                keypoints_by_image[image_id].append((int(x), int(y), category_id))
    
    result = {}
    for image_id, kp_list in keypoints_by_image.items():
        if image_id in images:
            filename = images[image_id]
            result[filename] = kp_list
    
    return result


def rgb_to_class_indices(mask_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB mask to class indices.
    
    Args:
        mask_rgb: RGB mask array (H, W, 3)
        
    Returns:
        Class indices array (H, W)
    """
    h, w = mask_rgb.shape[:2]
    mask_indices = np.zeros((h, w), dtype=np.uint8)
    
    for rgb, class_idx in CLASS_MAPPING_4_CLASS.items():
        matches = np.all(mask_rgb[:, :, :3] == rgb, axis=-1)
        mask_indices[matches] = class_idx
    
    return mask_indices


def class_indices_to_rgb(mask_indices: np.ndarray) -> np.ndarray:
    """
    Convert class indices to RGB mask.
    
    Args:
        mask_indices: Class indices array (H, W)
        
    Returns:
        RGB mask array (H, W, 3)
    """
    h, w = mask_indices.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx, color in INVERSE_CLASS_MAPPING_4_CLASS.items():
        mask_rgb[mask_indices == class_idx] = color
    
    return mask_rgb


def process_mask_with_keypoints(mask_path: Path, 
                                keypoints: list,
                                nucleus_radius: int = 8) -> tuple:
    """
    Process a mask based on keypoint annotations:
    1. Remove nuclei without keypoints
    2. Split nuclei with multiple keypoints using watershed
    3. Create circular nuclei for keypoints outside any mask
    4. Keep boundaries around kept nuclei
    
    Args:
        mask_path: Path to the mask image
        keypoints: List of (x, y, category_id) tuples
        nucleus_radius: Radius for creating circular nuclei around orphan keypoints
        
    Returns:
        Tuple of (corrected_mask_rgb, statistics_dict)
    """
    mask = plt.imread(mask_path)
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        mask = (mask * 255).astype(np.uint8)
    
    if mask.ndim == 3 and mask.shape[2] == 4:
        mask = mask[:, :, :3]
    elif mask.ndim == 2:
        mask = np.stack([mask] * 3, axis=-1)
    
    h, w = mask.shape[:2]
    
    mask_indices = rgb_to_class_indices(mask)
    
    positive_mask = mask_indices == 2
    negative_mask = mask_indices == 1
    
    labeled_positive = label(positive_mask, connectivity=2)
    labeled_negative = label(negative_mask, connectivity=2)
    
    original_positive_count = labeled_positive.max()
    original_negative_count = labeled_negative.max()
    
    from collections import defaultdict
    positive_nuclei_keypoints = defaultdict(list) 
    negative_nuclei_keypoints = defaultdict(list)
    orphan_positive_keypoints = []
    orphan_negative_keypoints = []
    
    for x, y, category_id in keypoints:
        x_int = max(0, min(w - 1, int(x)))
        y_int = max(0, min(h - 1, int(y)))
        
        if category_id == 1:
            nucleus_id = labeled_positive[y_int, x_int]
            if nucleus_id > 0:
                positive_nuclei_keypoints[nucleus_id].append((x_int, y_int))
            else:
                orphan_positive_keypoints.append((x_int, y_int))
        else:
            nucleus_id = labeled_negative[y_int, x_int]
            if nucleus_id > 0:
                negative_nuclei_keypoints[nucleus_id].append((x_int, y_int))
            else:
                orphan_negative_keypoints.append((x_int, y_int))
    
    corrected_mask_indices = np.zeros_like(mask_indices)
    
    nuclei_split_count = 0
    
    for nucleus_id, kp_list in positive_nuclei_keypoints.items():
        nucleus_mask = labeled_positive == nucleus_id
        
        if len(kp_list) == 1:
            corrected_mask_indices[nucleus_mask] = 2
        else:
            nuclei_split_count += 1
            
            markers = np.zeros((h, w), dtype=np.int32)
            for idx, (kp_x, kp_y) in enumerate(kp_list, start=1):
                markers[kp_y, kp_x] = idx
            
            distance = distance_transform_edt(nucleus_mask)
            
            watershed_labels = watershed(-distance, markers, mask=nucleus_mask)
            
            for split_id in range(1, len(kp_list) + 1):
                split_region = watershed_labels == split_id
                corrected_mask_indices[split_region] = 2
    
    for nucleus_id, kp_list in negative_nuclei_keypoints.items():
        nucleus_mask = labeled_negative == nucleus_id
        
        if len(kp_list) == 1:
            corrected_mask_indices[nucleus_mask] = 1
        else:
            nuclei_split_count += 1
            
            markers = np.zeros((h, w), dtype=np.int32)
            for idx, (kp_x, kp_y) in enumerate(kp_list, start=1):
                markers[kp_y, kp_x] = idx
            
            distance = distance_transform_edt(nucleus_mask)
            
            watershed_labels = watershed(-distance, markers, mask=nucleus_mask)
            
            for split_id in range(1, len(kp_list) + 1):
                split_region = watershed_labels == split_id
                corrected_mask_indices[split_region] = 1
    
    def create_circular_nucleus(x: int, y: int, class_idx: int, radius: int):
        """Create a circular nucleus mask around a keypoint"""
        selem = disk(radius)
        
        point_mask = np.zeros((h, w), dtype=bool)
        point_mask[y, x] = True
        
        dilated = dilation(point_mask, selem)
        
        available_space = dilated & (corrected_mask_indices == 0)
        corrected_mask_indices[available_space] = class_idx
    
    for x, y in orphan_positive_keypoints:
        create_circular_nucleus(x, y, class_idx=2, radius=nucleus_radius)
    
    for x, y in orphan_negative_keypoints:
        create_circular_nucleus(x, y, class_idx=1, radius=nucleus_radius)
    
    boundary_thickness = 4
    selem_boundary = disk(boundary_thickness)
    
    all_nuclei_mask = (corrected_mask_indices == 1) | (corrected_mask_indices == 2)
    labeled_nuclei = label(all_nuclei_mask, connectivity=2)
    
    all_nuclei_mask = (corrected_mask_indices == 1) | (corrected_mask_indices == 2)
    
    all_nuclei_dilated = dilation(all_nuclei_mask, selem_boundary)
    
    boundaries = all_nuclei_dilated & (~all_nuclei_mask)
    
    corrected_mask_indices[boundaries] = 3
    
    corrected_mask_rgb = class_indices_to_rgb(corrected_mask_indices)
    
    removed_positive = original_positive_count - len(positive_nuclei_keypoints)
    removed_negative = original_negative_count - len(negative_nuclei_keypoints)
    created_positive = len(orphan_positive_keypoints)
    created_negative = len(orphan_negative_keypoints)
    
    stats = {
        'original_positive_nuclei': original_positive_count,
        'original_negative_nuclei': original_negative_count,
        'kept_positive_nuclei': len(positive_nuclei_keypoints),
        'kept_negative_nuclei': len(negative_nuclei_keypoints),
        'removed_positive_nuclei': removed_positive,
        'removed_negative_nuclei': removed_negative,
        'created_positive_nuclei': created_positive,
        'created_negative_nuclei': created_negative,
        'orphan_positive_keypoints': created_positive,
        'orphan_negative_keypoints': created_negative,
        'nuclei_split': nuclei_split_count,
        'total_removed': removed_positive + removed_negative,
        'total_created': created_positive + created_negative,
    }
    
    return corrected_mask_rgb, stats


def process_flat_dataset(input_folder: str,
                        output_folder: str,
                        annotations_file: str,
                        nucleus_radius: int = 8,
                        verbose: bool = True) -> dict:
    """
    Process all masks in a FLAT dataset structure (all images in one folder).
    
    Args:
        input_folder: Folder containing all mask images (flat structure)
        output_folder: Output folder for processed masks
        annotations_file: Path to central annotations.json file (required)
        nucleus_radius: Radius for creating nuclei around orphan keypoints
        verbose: Show progress and details
        
    Returns:
        Dictionary with processing statistics
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    
    annotations_path = Path(annotations_file)
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    image_info = {}
    images = {img['id']: img for img in data.get('images', [])}
    
    for ann in data.get('annotations', []):
        image_id = ann['image_id']
        if image_id not in images:
            continue
            
        img_data = images[image_id]
        file_name = img_data['file_name'] 
        basename = Path(file_name).name
        case_id = file_name.split('/')[0] if '/' in file_name else 'unknown'
        
        keypoints = ann.get('keypoints', [])
        category_id = ann.get('category_id', 0)
        
        if basename not in image_info:
            image_info[basename] = {
                'keypoints': [],
                'case_id': case_id,
                'full_path': file_name
            }
        
        for i in range(0, len(keypoints), 3):
            x, y, visibility = keypoints[i], keypoints[i+1], keypoints[i+2]
            if visibility > 0:
                image_info[basename]['keypoints'].append((int(x), int(y), category_id))
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"PREPROCESSING ANNOTATED MASKS (FLAT STRUCTURE)")
        print(f"{'='*80}")
        print(f"Input folder: {input_folder}")
        print(f"Output folder: {output_folder}")
        print(f"Annotations file: {annotations_file}")
        print(f"Nucleus radius: {nucleus_radius}")
        print(f"Found {len(image_info)} images with keypoints in annotations")
        print(f"{'='*80}\n")
    
    all_stats = []
    total_processed = 0
    total_skipped = 0
    
    for basename, info in tqdm(image_info.items(), desc="Processing images", disable=not verbose):
        mask_path = input_path / basename
        
        if not mask_path.exists():
            if verbose:
                tqdm.write(f"   Mask not found: {basename}")
            total_skipped += 1
            continue
        
        keypoints = info['keypoints']
        if not keypoints:
            total_skipped += 1
            continue
        
        try:
            corrected_mask, stats = process_mask_with_keypoints(
                mask_path,
                keypoints,
                nucleus_radius=nucleus_radius
            )
            
            output_mask_path = output_path / basename
            plt.imsave(output_mask_path, corrected_mask)
            
            stats['case_id'] = info['case_id']
            stats['image_filename'] = basename
            stats['num_keypoints'] = len(keypoints)
            all_stats.append(stats)
            
            total_processed += 1
            
        except Exception as e:
            if verbose:
                tqdm.write(f"Error processing {basename}: {e}")
            total_skipped += 1
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total masks processed: {total_processed}")
        print(f"Total masks skipped: {total_skipped}")
        
        if all_stats:
            total_removed = sum(s['total_removed'] for s in all_stats)
            total_created = sum(s['total_created'] for s in all_stats)
            print(f"Total nuclei removed (no keypoints): {total_removed}")
            print(f"Total nuclei created (orphan keypoints): {total_created}")
        
        print(f"{'='*80}")
        print(f"\n Processed masks saved to: {output_folder}")
    
    return {
        'total_processed': total_processed,
        'total_skipped': total_skipped,
        'all_stats': all_stats,
    }


def process_dataset(input_folder: str, 
                   output_folder: str,
                   annotations_file: str = None,
                   nucleus_radius: int = 8,
                   verbose: bool = True) -> dict:
    """
    Process all masks in the dataset based on keypoint annotations.
    
    Args:
        input_folder: Root folder with case subdirectories
        output_folder: Output folder for processed masks
        annotations_file: Path to annotations.json (if None, searches for it in each case folder)
        nucleus_radius: Radius for creating nuclei around orphan keypoints
        verbose: Show progress and details
        
    Returns:
        Dictionary with processing statistics
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    
    # Determine annotation strategy
    if annotations_file:
        # Single annotations.json for all cases
        annotations_path = Path(annotations_file)
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
        use_single_annotations = True
    else:
        # Each case has its own annotations.json
        use_single_annotations = False
    
    # Find all case directories
    if use_single_annotations:
        case_dirs = [d for d in input_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    else:
        case_dirs = [d for d in input_path.iterdir() 
                    if d.is_dir() and (d / "annotations.json").exists()]
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"PREPROCESSING ANNOTATED MASKS")
        print(f"{'='*80}")
        print(f"Input folder: {input_folder}")
        print(f"Output folder: {output_folder}")
        if use_single_annotations:
            print(f"Annotations file: {annotations_file}")
        else:
            print(f"Annotations: per-case files")
        print(f"Nucleus radius: {nucleus_radius}")
        print(f"Found {len(case_dirs)} cases")
        print(f"{'='*80}\n")
    
    all_stats = []
    total_processed = 0
    total_skipped = 0
    
    # Process each case
    for case_dir in tqdm(case_dirs, desc="Processing cases", disable=not verbose):
        case_id = case_dir.name
        
        # Load keypoints for this case
        if use_single_annotations:
            keypoints_by_image = load_keypoints_from_central_annotations(annotations_path, case_id)
        else:
            keypoints_by_image = load_keypoints_from_annotations(case_dir)
        
        if not keypoints_by_image:
            if verbose:
                tqdm.write(f"   No keypoints found for case: {case_id}")
            total_skipped += 1
            continue
        
        # Create output directory for this case
        case_output_dir = output_path / case_id
        case_output_dir.mkdir(exist_ok=True)
        
        # Process each mask with keypoints
        masks_dir = case_dir / "patches_512_30"
        if not masks_dir.exists():
            if verbose:
                tqdm.write(f"   No masks directory found for case: {case_id}")
            total_skipped += 1
            continue
        
        case_processed = 0
        case_skipped = 0
        
        for image_filename, keypoints in keypoints_by_image.items():
            mask_path = masks_dir / image_filename
            
            if not mask_path.exists():
                case_skipped += 1
                continue
            
            try:
                # Process mask
                corrected_mask, stats = process_mask_with_keypoints(
                    mask_path, 
                    keypoints,
                    nucleus_radius=nucleus_radius
                )
                
                # Save corrected mask
                output_mask_path = case_output_dir / image_filename
                plt.imsave(output_mask_path, corrected_mask)
                
                # Store statistics
                stats['case_id'] = case_id
                stats['image_filename'] = image_filename
                stats['num_keypoints'] = len(keypoints)
                all_stats.append(stats)
                
                case_processed += 1
                total_processed += 1
                
            except Exception as e:
                if verbose:
                    tqdm.write(f"   Error processing {case_id}/{image_filename}: {e}")
                case_skipped += 1
                total_skipped += 1
        
        if verbose and case_processed > 0:
            tqdm.write(f"   {case_id}: processed {case_processed} masks, skipped {case_skipped}")
    
    # Summary statistics
    if verbose:
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total masks processed: {total_processed}")
        print(f"Total masks skipped: {total_skipped}")
        
        if all_stats:
            total_removed = sum(s['total_removed'] for s in all_stats)
            total_created = sum(s['total_created'] for s in all_stats)
            print(f"Total nuclei removed (no keypoints): {total_removed}")
            print(f"Total nuclei created (orphan keypoints): {total_created}")
        
        print(f"{'='*80}")
        print(f"\n Processed masks saved to: {output_folder}")
    
    return {
        'total_processed': total_processed,
        'total_skipped': total_skipped,
        'all_stats': all_stats,
    }


def main():
    """Main execution."""
    INPUT_FOLDER = ""
    OUTPUT_FOLDER = ""
    ANNOTATIONS_FILE = ""  
    NUCLEUS_RADIUS = 5  
    USE_FLAT_STRUCTURE = True  
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"ERROR: Folder not found: {INPUT_FOLDER}")
        return
    
    if USE_FLAT_STRUCTURE:
        if not ANNOTATIONS_FILE:
            print("ERROR: ANNOTATIONS_FILE is required for flat structure processing")
            return
        
        results = process_flat_dataset(
            input_folder=INPUT_FOLDER,
            output_folder=OUTPUT_FOLDER,
            annotations_file=ANNOTATIONS_FILE,
            nucleus_radius=NUCLEUS_RADIUS,
            verbose=True
        )
    else:
        results = process_dataset(
            input_folder=INPUT_FOLDER,
            output_folder=OUTPUT_FOLDER,
            annotations_file=ANNOTATIONS_FILE,
            nucleus_radius=NUCLEUS_RADIUS,
            verbose=True
        )
    
    if results['all_stats']:
        import pandas as pd
        
        df = pd.DataFrame(results['all_stats'])
        csv_path = Path(OUTPUT_FOLDER) / "preprocessing_statistics.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n Detailed statistics saved to: {csv_path}")


if __name__ == "__main__":
    main()

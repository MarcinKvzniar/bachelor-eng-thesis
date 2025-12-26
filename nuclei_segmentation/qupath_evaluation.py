"""
QuPath Segmentation Evaluation Script
Evaluates QuPath predictions against keypoint annotations using the same metrics as model evaluation.
"""

import os
import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import label as scipy_label

warnings.filterwarnings('ignore')

DATA_ROOT = ""
QUPATH_SEG_DIR = ""
OUTPUT_DIR = Path('')

# Parameters
IMG_HEIGHT, IMG_WIDTH = 512, 512
NUM_CLASSES = 4

CLASS_NAMES = ['Background', 'Negative', 'Positive']
CLASS_MAPPING = {
    (255, 255, 255): 0,  # Background
    (112, 112, 225): 1,  # Negative
    (250, 62, 62): 2,    # Positive
}
INVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

KEYPOINT_TO_CLASS = {
    'negative': 1,
    'positive': 2,
}

JSON_CATEGORY_ID_TO_NAME = {
    0: 'negative',
    1: 'positive'
}

TISSUE_CLASS_NAMES = ['Background', 'Cancer', 'Other Tissue']
CANCER_CLASS_ID = 1

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"QuPath Evaluation Script")
print(f"Output directory: {OUTPUT_DIR}")

def decode_mask_to_colors(mask):
    """Convert class indices to RGB colors."""
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_index, color in INVERSE_CLASS_MAPPING.items():
        color_mask[mask == class_index] = color
    return color_mask


def rgb_to_class_mask(rgb_mask):
    """Convert RGB mask to class indices."""
    class_mask = np.zeros((rgb_mask.shape[0], rgb_mask.shape[1]), dtype=np.uint8)
    
    for color, class_id in CLASS_MAPPING.items():
        mask = np.all(rgb_mask == color, axis=-1)
        class_mask[mask] = class_id
    
    return class_mask


def load_slide_data(data_root):
    """Load data from slide directories."""
    data_root = Path(data_root)
    slide_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    
    all_data = []
    print(f"Scanning {len(slide_dirs)} slide directories...")
    
    for slide_dir in sorted(slide_dirs):
        annotations_file = slide_dir / 'annotations.json'
        patches_dir = slide_dir / 'patches_512_30'
        metadata_file = patches_dir / 'patches_metadata.json'
        
        if not annotations_file.exists() or not metadata_file.exists():
            continue
        
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        with open(metadata_file, 'r') as f:
            patches_metadata = json.load(f)
        
        keypoint_lookup = defaultdict(list)
        for ann in annotations.get('annotations', []):
            if ann.get('type') == 'point':
                bbox_data = ann.get('bbox', [0, 0, 0, 0])
                x_global, y_global = bbox_data[0], bbox_data[1]
                category_id = ann.get('category_id', 0)
                label_name = JSON_CATEGORY_ID_TO_NAME.get(category_id, 'unknown')
                
                keypoint_lookup[(x_global, y_global)].append({
                    'label': label_name,
                    'global_x': x_global,
                    'global_y': y_global
                })
        
        for patch_info in patches_metadata:
            patch_keypoints_raw = patch_info.get('keypoints', [])
            
            if len(patch_keypoints_raw) == 0:
                continue
            
            x_offset = patch_info['x']
            y_offset = patch_info['y']
            patch_w = patch_info['width']
            patch_h = patch_info['height']
            patch_id = patch_info['patch_id']
            primary_tissue = patch_info.get('primary_tissue', 'unknown')
            
            patch_filename = f"patch_{patch_id:06d}_{primary_tissue}_{x_offset}_{y_offset}.png"
            patch_path = patches_dir / patch_filename
            
            if not patch_path.exists():
                continue
            
            patch_keypoints = []
            for kp in patch_keypoints_raw:
                label = kp.get('label', 'unknown')
                if label not in ['negative', 'positive']:
                    continue
                
                local_x = kp.get('relative_x', 0)
                local_y = kp.get('relative_y', 0)
                
                patch_keypoints.append({
                    'label': label,
                    'local_x': local_x,
                    'local_y': local_y,
                    'global_x': kp.get('absolute_x', 0),
                    'global_y': kp.get('absolute_y', 0)
                })
            
            if len(patch_keypoints) > 0:
                all_data.append({
                    'patch_path': patch_path,
                    'patch_info': {
                        'filename': patch_filename,
                        'slide_id': slide_dir.name,
                        'keypoints': patch_keypoints,
                        'x_offset': x_offset,
                        'y_offset': y_offset
                    }
                })
    
    total_found = len(all_data)
    print(f"Found {total_found} valid patches with keypoints.")
    
    return all_data


def find_connected_components(mask, class_id):
    """Find connected components for a specific class."""
    binary_mask = (mask == class_id).astype(np.uint8)
    labeled_mask, num_features = scipy_label(binary_mask)
    components = []
    for i in range(1, num_features + 1):
        component_mask = (labeled_mask == i).astype(np.uint8)
        area = np.sum(component_mask)
        components.append({
            'label': i,
            'area': area,
            'mask': component_mask
        })
    return labeled_mask, num_features, components


def check_component_has_keypoint(component, keypoints, expected_class, proximity_threshold=3):
    """Check if a component has a keypoint inside or nearby."""
    component_mask = component['mask']
    
    for kp in keypoints:
        if kp['label'] = expected_class:
            continue
        
        x_int = int(round(kp['local_x']))
        y_int = int(round(kp['local_y']))
        
        if not (0 <= x_int < component_mask.shape[1] and 0 <= y_int < component_mask.shape[0]):
            continue
        
        if component_mask[y_int, x_int] == 1:
            return True
        
        y_coords, x_coords = np.where(component_mask == 1)
        if len(x_coords) > 0:
            distances = np.sqrt((x_coords - x_int)**2 + (y_coords - y_int)**2)
            min_distance = np.min(distances)
            if min_distance <= proximity_threshold:
                return True
    
    return False


def evaluate_patch_ki67(keypoints, pred_mask):
    """Calculate Ki-67 metrics comparing GT Keypoints vs Prediction Blobs."""
    gt_pos = sum(1 for k in keypoints if k['label'] == 'positive')
    gt_neg = sum(1 for k in keypoints if k['label'] == 'negative')
    gt_total = gt_pos + gt_neg
    gt_index = (gt_pos / gt_total) if gt_total > 0 else 0.0
    
    _, pred_neg_count, blobs_neg = find_connected_components(pred_mask, 1)
    _, pred_pos_count, blobs_pos = find_connected_components(pred_mask, 2)
    
    pred_total = pred_pos_count + pred_neg_count
    pred_index = (pred_pos_count / pred_total) if pred_total > 0 else 0.0
    
    return {
        'gt_pos': gt_pos, 'gt_neg': gt_neg, 'gt_total': gt_total, 'gt_index': gt_index,
        'pred_pos': pred_pos_count, 'pred_neg': pred_neg_count, 'pred_total': pred_total, 'pred_index': pred_index,
        'ki67_abs_error': abs(gt_index - pred_index)
    }


def calculate_correlations(ki67_results, all_coverage_results):
    """Calculate Pearson correlation coefficients."""
    from scipy.stats import pearsonr
    
    gt_pos_counts = [r['gt_pos_count'] for r in ki67_results]
    gt_neg_counts = [r['gt_neg_count'] for r in ki67_results]
    gt_total_counts = [r['gt_total'] for r in ki67_results]
    
    pred_pos_counts = [r['pred_pos_count'] for r in ki67_results]
    pred_neg_counts = [r['pred_neg_count'] for r in ki67_results]
    pred_total_counts = [r['pred_total'] for r in ki67_results]
    
    correlations = {}
    
    if len(gt_total_counts) > 1:
        corr, _ = pearsonr(gt_total_counts, pred_total_counts)
        correlations['total_pearson'] = corr
    else:
        correlations['total_pearson'] = 0.0
    
    if len(gt_pos_counts) > 1 and sum(gt_pos_counts) > 0:
        corr, _ = pearsonr(gt_pos_counts, pred_pos_counts)
        correlations['positive_pearson'] = corr
    else:
        correlations['positive_pearson'] = 0.0
    
    if len(gt_neg_counts) > 1 and sum(gt_neg_counts) > 0:
        corr, _ = pearsonr(gt_neg_counts, pred_neg_counts)
        correlations['negative_pearson'] = corr
    else:
        correlations['negative_pearson'] = 0.0
    
    return correlations


def calculate_ki67_metrics(ki67_results):
    """Calculate global and aggregated Ki-67 metrics."""
    errors = [r['mae'] for r in ki67_results]
    
    ki67_metrics = {
        'mae': np.mean(errors) if errors else 0.0,
        'patch_details': ki67_results
    }

    total_gt_pos = sum(r['gt_pos_count'] for r in ki67_results)
    total_gt_all = sum(r['gt_total'] for r in ki67_results)
    ki67_metrics['global_gt'] = (total_gt_pos / total_gt_all * 100) if total_gt_all else 0.0
    
    total_pred_pos = sum(r['pred_pos_count'] for r in ki67_results)
    total_pred_all = sum(r['pred_total'] for r in ki67_results)
    ki67_metrics['global_pred'] = (total_pred_pos / total_pred_all * 100) if total_pred_all else 0.0
    
    ki67_metrics['global_error'] = abs(ki67_metrics['global_gt'] - ki67_metrics['global_pred'])
    
    return ki67_metrics


def evaluate_patch_fast(keypoints, pred_mask, components_neg, components_pos):
    """
    Evaluate patch using pre-computed connected components.
    
    Args:
        keypoints: List of keypoints with annotations
        pred_mask: Predicted segmentation mask
        components_neg: Tuple of (labeled_mask, num_features, components_list) for Ki-67-
        components_pos: Tuple of (labeled_mask, num_features, components_list) for Ki-67+
    """
    results = {
        'coverage_rate': 0.0,
        'mask_validity_rate': 0.0,
        'false_positive_rate': 0.0,
        'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
        'pos_coverage_rate': 0.0,
        'neg_coverage_rate': 0.0,
        'pos_mask_validity': 0.0,
        'neg_mask_validity': 0.0,
        'pos_total': 0,
        'neg_total': 0,
        'pos_covered': 0,
        'neg_covered': 0
    }
    
    if len(keypoints) == 0:
        return results
    
    labeled_mask_neg, num_nuclei_neg, _ = components_neg
    labeled_mask_pos, num_nuclei_pos, _ = components_pos
    
    matched_nuclei_neg = set()
    matched_nuclei_pos = set()
    
    total_keypoints = len(keypoints)
    covered_keypoints = 0
    valid_masks = 0
    
    pos_total = sum(1 for kp in keypoints if kp['label'] == 'positive')
    neg_total = sum(1 for kp in keypoints if kp['label'] == 'negative')
    pos_covered = 0
    neg_covered = 0
    pos_valid = 0
    neg_valid = 0
    
    for kp in keypoints:
        label = kp['label']
        if label == 'negative':
            gt_category = 1
        elif label == 'positive':
            gt_category = 2
        else:
            continue
        
        x, y = int(kp['local_x']), int(kp['local_y'])
        
        if not (0 <= y < pred_mask.shape[0] and 0 <= x < pred_mask.shape[1]):
            continue
        
        pred_category = pred_mask[y, x]
        
        if pred_category == gt_category:
            covered_keypoints += 1
            
            if gt_category == 2:
                pos_covered += 1
            else:
                neg_covered += 1
            
            if gt_category == 1:
                nucleus_id = labeled_mask_neg[y, x]
                if nucleus_id > 0:
                    matched_nuclei_neg.add(nucleus_id)
                    valid_masks += 1
                    neg_valid += 1
            elif gt_category == 2:
                nucleus_id = labeled_mask_pos[y, x]
                if nucleus_id > 0:
                    matched_nuclei_pos.add(nucleus_id)
                    valid_masks += 1
                    pos_valid += 1
    
    results['coverage_rate'] = covered_keypoints / total_keypoints if total_keypoints > 0 else 0.0
    results['mask_validity_rate'] = valid_masks / covered_keypoints if covered_keypoints > 0 else 0.0
    
    results['pos_coverage_rate'] = pos_covered / pos_total if pos_total > 0 else 0.0
    results['neg_coverage_rate'] = neg_covered / neg_total if neg_total > 0 else 0.0
    results['pos_mask_validity'] = pos_valid / pos_covered if pos_covered > 0 else 0.0
    results['neg_mask_validity'] = neg_valid / neg_covered if neg_covered > 0 else 0.0
    results['pos_total'] = pos_total
    results['neg_total'] = neg_total
    results['pos_covered'] = pos_covered
    results['neg_covered'] = neg_covered
    
    total_nuclei = num_nuclei_neg + num_nuclei_pos
    matched_nuclei = len(matched_nuclei_neg) + len(matched_nuclei_pos)
    false_positives = total_nuclei - matched_nuclei
    
    results['false_positive_rate'] = false_positives / total_nuclei if total_nuclei > 0 else 0.0
    
    results['tp'] = covered_keypoints
    results['fn'] = total_keypoints - covered_keypoints
    results['fp'] = false_positives
    results['tn'] = 0
    
    return results


def evaluate_patch_ki67_fast(keypoints, components_neg, components_pos):
    """
    Evaluate Ki-67 proliferation index using pre-computed components.
    
    Args:
        keypoints: List of keypoints with annotations
        components_neg: Tuple of (labeled_mask, num_features, components_list) for Ki-67-
        components_pos: Tuple of (labeled_mask, num_features, components_list) for Ki-67+
    """
    gt_neg_count = sum(1 for kp in keypoints if kp['label'] == 'negative')
    gt_pos_count = sum(1 for kp in keypoints if kp['label'] == 'positive')
    gt_total = gt_neg_count + gt_pos_count
    
    pred_neg_count = components_neg[1]  # num_nuclei_neg
    pred_pos_count = components_pos[1]  # num_nuclei_pos
    pred_total = pred_neg_count + pred_pos_count
    
    gt_ki67_index = (gt_pos_count / gt_total * 100) if gt_total > 0 else 0.0
    pred_ki67_index = (pred_pos_count / pred_total * 100) if pred_total > 0 else 0.0
    
    mae = abs(gt_ki67_index - pred_ki67_index)
    
    return {
        'gt_ki67_index': gt_ki67_index,
        'pred_ki67_index': pred_ki67_index,
        'mae': mae,
        'gt_neg_count': gt_neg_count,
        'gt_pos_count': gt_pos_count,
        'gt_total': gt_total,
        'pred_neg_count': pred_neg_count,
        'pred_pos_count': pred_pos_count,
        'pred_total': pred_total
    }


def evaluate_patch(keypoints, pred_mask):
    """Evaluate coverage and mask validity."""
    results = {
        'keypoint_metrics': defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'background': 0}),
        'mask_metrics': defaultdict(lambda: {'with_keypoint': 0, 'without_keypoint': 0, 'total_area': 0}),
        'total_keypoints': defaultdict(int),
        'confusion': defaultdict(int)
    }
    
    for kp in keypoints:
        label = kp['label']
        expected_class_id = KEYPOINT_TO_CLASS[label]
        
        x_int = int(round(kp['local_x']))
        y_int = int(round(kp['local_y']))
        
        results['total_keypoints'][label] += 1
        
        if 0 <= x_int < pred_mask.shape[1] and 0 <= y_int < pred_mask.shape[0]:
            predicted_class = pred_mask[y_int, x_int]
            
            if predicted_class == expected_class_id:
                results['keypoint_metrics'][label]['correct'] += 1
            elif predicted_class == 0:
                results['keypoint_metrics'][label]['background'] += 1
            else:
                results['keypoint_metrics'][label]['incorrect'] += 1
            
            if label == 'negative':
                if predicted_class == 1:
                    results['confusion']['tn'] += 1
                elif predicted_class == 2:
                    results['confusion']['fp'] += 1
            elif label == 'positive':
                if predicted_class == 2:
                    results['confusion']['tp'] += 1
                elif predicted_class == 1:
                    results['confusion']['fn'] += 1
    
    for class_name, class_id in [('negative', 1), ('positive', 2)]:
        _, _, components = find_connected_components(pred_mask, class_id)
        
        for comp in components:
            has_kp = check_component_has_keypoint(comp, keypoints, class_name, proximity_threshold=3)
            
            if has_kp:
                results['mask_metrics'][class_name]['with_keypoint'] += 1
            else:
                results['mask_metrics'][class_name]['without_keypoint'] += 1
            
            results['mask_metrics'][class_name]['total_area'] += comp['area']
    
    return results


def aggregate_results(all_results):
    """Aggregate results from all patches."""
    aggregated = {
        'total_patches': len(all_results),
        'total_tp': 0,
        'total_tn': 0,
        'total_fp': 0,
        'total_fn': 0,
        'coverage_rates': [],
        'mask_validity_rates': [],
        'false_positive_rates': [],
        'pos_coverage_rates': [],
        'neg_coverage_rates': [],
        'pos_mask_validities': [],
        'neg_mask_validities': []
    }
    
    for result in all_results:
        aggregated['total_tp'] += result['tp']
        aggregated['total_tn'] += result['tn']
        aggregated['total_fp'] += result['fp']
        aggregated['total_fn'] += result['fn']
        aggregated['coverage_rates'].append(result['coverage_rate'])
        aggregated['mask_validity_rates'].append(result['mask_validity_rate'])
        aggregated['false_positive_rates'].append(result['false_positive_rate'])
        aggregated['pos_coverage_rates'].append(result['pos_coverage_rate'])
        aggregated['neg_coverage_rates'].append(result['neg_coverage_rate'])
        aggregated['pos_mask_validities'].append(result['pos_mask_validity'])
        aggregated['neg_mask_validities'].append(result['neg_mask_validity'])
    
    return aggregated


def calculate_metrics(aggregated):
    """Calculate final metrics."""
    metrics = {
        'coverage_rate_mean': np.mean(aggregated['coverage_rates']) if aggregated['coverage_rates'] else 0.0,
        'coverage_rate_std': np.std(aggregated['coverage_rates']) if aggregated['coverage_rates'] else 0.0,
        'mask_validity_rate_mean': np.mean(aggregated['mask_validity_rates']) if aggregated['mask_validity_rates'] else 0.0,
        'mask_validity_rate_std': np.std(aggregated['mask_validity_rates']) if aggregated['mask_validity_rates'] else 0.0,
        'false_positive_rate_mean': np.mean(aggregated['false_positive_rates']) if aggregated['false_positive_rates'] else 0.0,
        'false_positive_rate_std': np.std(aggregated['false_positive_rates']) if aggregated['false_positive_rates'] else 0.0,
        'tp': aggregated['total_tp'],
        'tn': aggregated['total_tn'],
        'fp': aggregated['total_fp'],
        'fn': aggregated['total_fn'],
        'total_patches': aggregated['total_patches'],
        'pos_coverage_mean': np.mean(aggregated['pos_coverage_rates']) if aggregated['pos_coverage_rates'] else 0.0,
        'pos_coverage_std': np.std(aggregated['pos_coverage_rates']) if aggregated['pos_coverage_rates'] else 0.0,
        'neg_coverage_mean': np.mean(aggregated['neg_coverage_rates']) if aggregated['neg_coverage_rates'] else 0.0,
        'neg_coverage_std': np.std(aggregated['neg_coverage_rates']) if aggregated['neg_coverage_rates'] else 0.0,
        'pos_validity_mean': np.mean(aggregated['pos_mask_validities']) if aggregated['pos_mask_validities'] else 0.0,
        'pos_validity_std': np.std(aggregated['pos_mask_validities']) if aggregated['pos_mask_validities'] else 0.0,
        'neg_validity_mean': np.mean(aggregated['neg_mask_validities']) if aggregated['neg_mask_validities'] else 0.0,
        'neg_validity_std': np.std(aggregated['neg_mask_validities']) if aggregated['neg_mask_validities'] else 0.0
    }
    
    tp = metrics['tp']
    tn = metrics['tn']
    fp = metrics['fp']
    fn = metrics['fn']
    
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0.0
    
    return metrics


def run_qupath_evaluation(data_list, qupath_seg_dir):
    """Run evaluation on QuPath segmentation results."""
    print("\nRunning QuPath evaluation...")
    print(f"Total patches to process: {len(data_list)}")
    
    all_coverage_results = []
    ki67_results = []
    samples_for_vis = []
    
    qupath_seg_path = Path(qupath_seg_dir)
    matched_count = 0
    unmatched_count = 0
    
    print("Building QuPath files index and matching patches...")
    qupath_files_dict = {f.name: f for f in qupath_seg_path.glob("*.png")}
    print(f"Found {len(qupath_files_dict)} QuPath segmentation files")
    
    matched_pairs = []
    for data in data_list:
        patch_info = data['patch_info']
        patch_filename = patch_info['filename']
        keypoints = patch_info['keypoints']
        
        parts = patch_filename.replace('.png', '').split('_')
        if len(parts) >= 4:
            patch_num = parts[1]
            x_coord = parts[-2]
            y_coord = parts[-1]
            
            qupath_filename = f"patch_{patch_num}_unknown_{x_coord}_{y_coord}.png"
            
            if qupath_filename in qupath_files_dict:
                matched_pairs.append({
                    'qupath_path': qupath_files_dict[qupath_filename],
                    'original_path': data['patch_path'],
                    'keypoints': keypoints,
                    'filename': patch_filename
                })
            else:
                unmatched_count += 1
    
    matched_count = len(matched_pairs)
    print(f"Matched {matched_count} patches, {unmatched_count} unmatched")
    print("\nProcessing matches...")
    
    for i, match_data in enumerate(matched_pairs):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing: {i+1}/{matched_count}", end='\r')
        
        qupath_rgb = cv2.imread(str(match_data['qupath_path']))
        if qupath_rgb is None:
            print(f"\nWarning: Failed to load {match_data['qupath_path'].name}")
            continue
            
        qupath_rgb = cv2.cvtColor(qupath_rgb, cv2.COLOR_BGR2RGB)
        
        pred_mask = rgb_to_class_mask(qupath_rgb)
        
        components_neg = find_connected_components(pred_mask, 1)
        components_pos = find_connected_components(pred_mask, 2)
        
        coverage_result = evaluate_patch_fast(match_data['keypoints'], pred_mask, components_neg, components_pos)
        all_coverage_results.append(coverage_result)
        
        ki67_result = evaluate_patch_ki67_fast(match_data['keypoints'], components_neg, components_pos)
        ki67_results.append(ki67_result)
        
        if len(samples_for_vis) < 5:
            img = cv2.imread(str(match_data['original_path']))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                samples_for_vis.append({
                    'image': img,
                    'pred_mask': pred_mask,
                    'keypoints': match_data['keypoints'],
                    'filename': match_data['filename']
                })
    
    print(f"\n\n Evaluation complete")
    print(f"  Total patches: {len(data_list)}")
    print(f"  Matched with QuPath results: {matched_count}")
    print(f"  Unmatched: {unmatched_count}")
    
    return all_coverage_results, ki67_results, samples_for_vis


def visualize_samples(samples):
    """Visualize sample predictions."""
    print("\nGenerating visualization...")
    
    num_samples = min(5, len(samples))
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples[:num_samples]):
        img = sample['image']
        pred_mask = sample['pred_mask']
        keypoints = sample['keypoints']
        filename = sample['filename']
        
        # Panel 1: Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original Image\n{filename}", fontsize=10)
        axes[i, 0].axis('off')
        
        # Panel 2: QuPath prediction
        pred_rgb = decode_mask_to_colors(pred_mask)
        axes[i, 1].imshow(pred_rgb)
        axes[i, 1].set_title("QuPath Segmentation", fontsize=10)
        axes[i, 1].axis('off')
        
        # Panel 3: Prediction with keypoints
        axes[i, 2].imshow(pred_rgb)
        for kp in keypoints:
            color = 'red' if kp['label'] == 'positive' else 'blue'
            axes[i, 2].plot(kp['local_x'], kp['local_y'], 'o', color=color, 
                          markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        
        gt_pos = sum(1 for kp in keypoints if kp['label'] == 'positive')
        gt_neg = sum(1 for kp in keypoints if kp['label'] == 'negative')
        axes[i, 2].set_title(f"Prediction + Keypoints\nPos={gt_pos}, Neg={gt_neg}", fontsize=10)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'qupath_visualization.png', dpi=150)
    plt.close()
    print(" Visualization saved.")


def generate_report(metrics, ki67_metrics, correlations, num_patches):
    """Generate evaluation report."""
    report_path = OUTPUT_DIR / 'qupath_evaluation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("QUPATH SEGMENTATION EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total patches evaluated: {num_patches}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("KI-67 INDEX ESTIMATION\n")
        f.write("-"*80 + "\n\n")
        f.write(f"Global Ground Truth Index:      {ki67_metrics['global_gt']:.3f}\n")
        f.write(f"Global Predicted Index:         {ki67_metrics['global_pred']:.3f}\n")
        f.write(f"MAE (Patch-level):              {ki67_metrics['mae']:.3f}\n")
        f.write(f"Global Error:                   {ki67_metrics['global_error']:.3f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("CORRELATION WITH GROUND TRUTH\n")
        f.write("-"*80 + "\n\n")
        f.write(f"Total Pearson Correlation:      {correlations['total_pearson']:.3f}\n")
        f.write(f"Positive Class Correlation:     {correlations['positive_pearson']:.3f}\n")
        f.write(f"Negative Class Correlation:     {correlations['negative_pearson']:.3f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("KEYPOINT COVERAGE\n")
        f.write("-"*80 + "\n\n")
        f.write(f"Overall Coverage Rate:          {metrics['coverage_rate_mean']:.2%} ± {metrics['coverage_rate_std']:.2%}\n")
        f.write(f"Positive Nuclei Coverage:       {metrics['pos_coverage_mean']:.2%} ± {metrics['pos_coverage_std']:.2%}\n")
        f.write(f"Negative Nuclei Coverage:       {metrics['neg_coverage_mean']:.2%} ± {metrics['neg_coverage_std']:.2%}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("MASK QUALITY\n")
        f.write("-"*80 + "\n\n")
        f.write(f"Positive Mask Validity:         {metrics['pos_validity_mean']:.2%} ± {metrics['pos_validity_std']:.2%}\n")
        f.write(f"Negative Mask Validity:         {metrics['neg_validity_mean']:.2%} ± {metrics['neg_validity_std']:.2%}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("CONFUSION MATRIX & CLASSIFICATION METRICS\n")
        f.write("-"*80 + "\n\n")
        f.write(f"True Positives (TP):            {metrics['tp']}\n")
        f.write(f"True Negatives (TN):            {metrics['tn']}\n")
        f.write(f"False Positives (FP):           {metrics['fp']}\n")
        f.write(f"False Negatives (FN):           {metrics['fn']}\n\n")
        f.write(f"Precision:                      {metrics['precision']:.4f}\n")
        f.write(f"Recall:                         {metrics['recall']:.4f}\n")
        f.write(f"F1 Score:                       {metrics['f1']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f" Report saved to {report_path}")
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY (Matching Model Format)")
    print("="*80)
    print("\nKi-67 Index Estimation:")
    print(f"  Global GT Index:           {ki67_metrics['global_gt']:.3f}")
    print(f"  Global Predicted Index:    {ki67_metrics['global_pred']:.3f}")
    print(f"  MAE (Patch-level):         {ki67_metrics['mae']:.3f}")
    print(f"  Global Error:              {ki67_metrics['global_error']:.3f}")
    print("\nCorrelation with Ground Truth:")
    print(f"  Total Pearson:             {correlations['total_pearson']:.3f}")
    print(f"  Positive Class:            {correlations['positive_pearson']:.3f}")
    print(f"  Negative Class:            {correlations['negative_pearson']:.3f}")
    print("\nKeypoint Coverage:")
    print(f"  Overall:                   {metrics['coverage_rate_mean']:.2%} ± {metrics['coverage_rate_std']:.2%}")
    print(f"  Positive Nuclei:           {metrics['pos_coverage_mean']:.2%} ± {metrics['pos_coverage_std']:.2%}")
    print(f"  Negative Nuclei:           {metrics['neg_coverage_mean']:.2%} ± {metrics['neg_coverage_std']:.2%}")
    print("\nMask Quality:")
    print(f"  Positive Mask Validity:    {metrics['pos_validity_mean']:.2%} ± {metrics['pos_validity_std']:.2%}")
    print(f"  Negative Mask Validity:    {metrics['neg_validity_mean']:.2%} ± {metrics['neg_validity_std']:.2%}")
    print(f"\nClassification Metrics:")
    print(f"  Precision:                 {metrics['precision']:.4f}")
    print(f"  Recall:                    {metrics['recall']:.4f}")
    print(f"  F1 Score:                  {metrics['f1']:.4f}")
    print("="*80 + "\n")


def main():
    print("="*80)
    print("QUPATH SEGMENTATION EVALUATION")
    print("="*80)
    
    print("\nLoading test data...")
    data_list = load_slide_data(DATA_ROOT)
    
    if len(data_list) == 0:
        print("ERROR: No data found")
        return
    
    all_coverage_results, ki67_results, samples_for_vis = run_qupath_evaluation(
        data_list, QUPATH_SEG_DIR
    )
    
    if len(all_coverage_results) == 0:
        print("ERROR: No QuPath segmentation results were matched")
        return
    
    print("\nAggregating results...")
    aggregated = aggregate_results(all_coverage_results)
    
    print("Calculating metrics...")
    metrics = calculate_metrics(aggregated)
    
    ki67_metrics = calculate_ki67_metrics(ki67_results)
    
    print("Calculating correlations...")
    correlations = calculate_correlations(ki67_results, all_coverage_results)
    
    if len(samples_for_vis) > 0:
        visualize_samples(samples_for_vis)
    
    generate_report(metrics, ki67_metrics, correlations, len(all_coverage_results))
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - qupath_evaluation_report.txt")
    print("  - qupath_visualization.png")
    print("="*80)


if __name__ == "__main__":
    main()

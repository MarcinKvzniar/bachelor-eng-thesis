"""
Nuclei Segmentation Evaluation Script using SegFormer Model
This script evaluates a trained SegFormer model for nuclei segmentation
"""

import os
import json
import time
import warnings
from pathlib import Path
from collections import defaultdict
import random

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import label as scipy_label
from PIL import Image

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import segmentation_models as sm

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import albumentations as A
from albumentations.pytorch import ToTensorV2

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

DATA_ROOT = ""

MODEL_PATH = ''
OUTPUT_DIR = Path('')

TISSUE_MODEL_SPEC_PATH = ''
TISSUE_BACKBONE = 'seresnet50'
TISSUE_NUM_CLASSES = 3
CANCER_CLASS_ID = 1

# Evaluation Config
MAX_PATCHES = None  # None = all patches
IMG_HEIGHT, IMG_WIDTH = 512, 512
NUM_CLASSES = 4
BATCH_SIZE = 4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = ['Background', 'Negative', 'Positive', 'Boundaries']
CLASS_MAPPING = {
    (255, 255, 255): 0,  # White - Background
    (112, 112, 225): 1,  # Blue - Negative
    (250, 62, 62): 2,    # Red - Positive
    (0, 0, 0): 3,        # Black - Boundaries
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"PyTorch version: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
print(f"Device: {DEVICE}")
print(f"Output directory: {OUTPUT_DIR}")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def convert_mask_to_classes(mask_image):
    """Converts a color-coded mask image to a 2D array of class indices."""
    mask_classes = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    for color, class_index in CLASS_MAPPING.items():
        match = np.all(mask_image == color, axis=-1)
        mask_classes[match] = class_index
    return mask_classes

def decode_mask_to_colors(mask):
    """Convert class indices back to RGB colors for visualization."""
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_index, color in INVERSE_CLASS_MAPPING.items():
        color_mask[mask == class_index] = color
    return color_mask

def blend_image_mask(image, mask, class_id, color=(255, 255, 0), alpha=0.4):
    """Blends a binary mask (specific class) onto an image."""
    if image.dtype != np.uint8:
        img_vis = (image * 255).astype(np.uint8)
    else:
        img_vis = image.copy()
        
    binary_mask = (mask == class_id).astype(np.uint8)
    overlay = np.zeros_like(img_vis)
    overlay[binary_mask == 1] = color
    
    output = img_vis.copy()
    mask_indices = binary_mask == 1
    output[mask_indices] = cv2.addWeighted(img_vis[mask_indices], 1-alpha, overlay[mask_indices], alpha, 0)
    return output

def load_slide_data(data_root, max_patches=None):
    """Load data and optionally sample a subset of patches."""
    data_root = Path(data_root)
    slide_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    
    all_data = []
    print(f"Scanning {len(slide_dirs)} slide directories...")
    
    for slide_dir in sorted(slide_dirs):
        slide_name = slide_dir.name
        annotations_file = slide_dir / 'annotations.json'
        patches_dir = slide_dir / 'patches_512_30'
        
        if not annotations_file.exists():
            continue
                
        if not patches_dir.exists():
            continue
        
        try:
            with open(annotations_file, 'r') as f:
                coco_data = json.load(f)
        except json.JSONDecodeError:
            continue
        
        print(f"  Processing {slide_name}...")
        print(f"    Images: {len(coco_data.get('images', []))}, Annotations: {len(coco_data.get('annotations', []))}")
            
        img_id_to_kps = defaultdict(list)
        for ann in coco_data.get('annotations', []):
            image_id = ann['image_id']
            category_id = ann['category_id']
            category_name = JSON_CATEGORY_ID_TO_NAME.get(category_id, 'unknown')
            
            if 'keypoints' in ann and ann['keypoints']:
                kps = ann['keypoints']
                for i in range(0, len(kps), 3):
                    x, y, visibility = kps[i], kps[i+1], kps[i+2]
                    if visibility > 0:
                        img_id_to_kps[image_id].append({
                            'x': x,
                            'y': y,
                            'label': category_name
                        })
        
        print(f"    Images with keypoints: {len(img_id_to_kps)}")
        
        images_with_kps = 0
        for img_entry in coco_data.get('images', []):
            img_id = img_entry['id']
            filename = img_entry['file_name']
            
            if '/' in filename:
                filename = Path(filename).name
            
            patch_path = patches_dir / filename
            
            if patch_path.exists() and img_id in img_id_to_kps:
                all_data.append({
                    'slide': slide_name,
                    'image_path': str(patch_path),
                    'keypoints': img_id_to_kps[img_id]
                })
                images_with_kps += 1
        
        if images_with_kps > 0:
            print(f"  {slide_name}: Found {images_with_kps} patches with keypoints")

    total_found = len(all_data)
    print(f"Found {total_found} valid patches.")
    
    if max_patches is not None and total_found > max_patches:
        all_data = random.sample(all_data, max_patches)
        print(f"Sampled {max_patches} patches for evaluation.")
    
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
            'mask': component_mask,
            'area': area
        })
    return components

def check_component_has_keypoint(component, keypoints, expected_class, proximity_threshold=3):
    """Check if a component has a keypoint inside or nearby."""
    component_mask = component['mask']
    
    for kp in keypoints:
        if kp['label'] != expected_class:
            continue
        
        x_int = int(round(kp['x']))
        y_int = int(round(kp['y']))
        
        if 0 <= y_int < component_mask.shape[0] and 0 <= x_int < component_mask.shape[1]:
            if component_mask[y_int, x_int] == 1:
                return True
            
            y_min = max(0, y_int - proximity_threshold)
            y_max = min(component_mask.shape[0], y_int + proximity_threshold + 1)
            x_min = max(0, x_int - proximity_threshold)
            x_max = min(component_mask.shape[1], x_int + proximity_threshold + 1)
            
            region = component_mask[y_min:y_max, x_min:x_max]
            if np.any(region == 1):
                return True
    
    return False

def evaluate_patch_coverage(keypoints, pred_mask, excluded_keypoints=None):
    """Evaluate keypoint coverage and mask validity."""
    results = {
        'keypoint_metrics': defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'background': 0}),
        'mask_metrics': defaultdict(lambda: {'with_keypoint': 0, 'without_keypoint': 0, 'total_area': 0}),
        'total_keypoints': defaultdict(int),
        'excluded_keypoints': defaultdict(int),
        'confusion': defaultdict(int)
    }
    
    if excluded_keypoints:
        for kp in excluded_keypoints:
            results['excluded_keypoints'][kp['label']] += 1
    
    for kp in keypoints:
        label = kp['label']
        expected_class_id = KEYPOINT_TO_CLASS.get(label)
        
        if expected_class_id is None:
            continue
        
        results['total_keypoints'][label] += 1
        
        x_int = int(round(kp['x']))
        y_int = int(round(kp['y']))
        
        if 0 <= y_int < pred_mask.shape[0] and 0 <= x_int < pred_mask.shape[1]:
            pred_class = pred_mask[y_int, x_int]
            
            if pred_class == expected_class_id:
                results['keypoint_metrics'][label]['correct'] += 1
            elif pred_class == 0:
                results['keypoint_metrics'][label]['background'] += 1
            else:
                results['keypoint_metrics'][label]['incorrect'] += 1
                results['confusion'][f"{label}_as_{CLASS_NAMES[pred_class]}"] += 1
    
    for class_name, class_id in [('negative', 1), ('positive', 2)]:
        components = find_connected_components(pred_mask, class_id)
        
        for component in components:
            results['mask_metrics'][class_name]['total_area'] += component['area']
            
            if check_component_has_keypoint(component, keypoints, class_name):
                results['mask_metrics'][class_name]['with_keypoint'] += 1
            else:
                results['mask_metrics'][class_name]['without_keypoint'] += 1
    
    return results

def evaluate_patch_ki67(keypoints, pred_mask):
    """Calculate Ki-67 metrics."""
    gt_pos = sum(1 for k in keypoints if k['label'] == 'positive')
    gt_neg = sum(1 for k in keypoints if k['label'] == 'negative')
    gt_total = gt_pos + gt_neg
    gt_index = (gt_pos / gt_total) if gt_total > 0 else 0.0
    
    blobs_neg = find_connected_components(pred_mask, 1)
    blobs_pos = find_connected_components(pred_mask, 2)
    
    pred_pos_count = len(blobs_pos)
    pred_neg_count = len(blobs_neg)
    pred_total = pred_pos_count + pred_neg_count
    pred_index = (pred_pos_count / pred_total) if pred_total > 0 else 0.0
    
    return {
        'gt_pos': gt_pos,
        'gt_neg': gt_neg,
        'gt_total': gt_total,
        'gt_index': gt_index,
        'pred_pos': pred_pos_count,
        'pred_neg': pred_neg_count,
        'pred_total': pred_total,
        'pred_index': pred_index,
        'ki67_abs_error': abs(gt_index - pred_index)
    }

def extract_cancer_region_ensemble(image, model_spec):
    """Extract cancer region using tissue segmentation model."""
    input_tensor = np.expand_dims(image, axis=0)
    pred_spec = model_spec.predict(input_tensor, verbose=0)[0]
    
    mask_spec = np.argmax(pred_spec, axis=-1)
    
    cancer_mask = (mask_spec == CANCER_CLASS_ID).astype(np.uint8)
    cancer_image = image.copy()
    cancer_image[cancer_mask == 0] = 1.0
    
    return cancer_mask, cancer_image

def filter_keypoints_by_cancer_mask(keypoints, cancer_mask):
    """Filter keypoints: Keep only those inside the cancer mask (value=1)."""
    filtered_keypoints = []
    excluded_keypoints = []
    
    for kp in keypoints:
        x_int = int(round(kp['x']))
        y_int = int(round(kp['y']))
        
        if 0 <= y_int < cancer_mask.shape[0] and 0 <= x_int < cancer_mask.shape[1]:
            if cancer_mask[y_int, x_int] == 1:
                filtered_keypoints.append(kp)
            else:
                excluded_keypoints.append(kp)
        else:
            excluded_keypoints.append(kp)
    
    return filtered_keypoints, excluded_keypoints

def remove_boundaries_from_nuclei(nuclei_mask):
    """Remove boundary class (class 3) by setting to background."""
    processed_mask = nuclei_mask.copy()
    boundary_count = np.sum(processed_mask == 3)
    processed_mask[processed_mask == 3] = 0
    return processed_mask, boundary_count

def remove_small_objects_from_mask(nuclei_mask, min_size=50):
    """Remove small connected components from nuclei classes."""
    processed_mask = nuclei_mask.copy()
    removed_counts = {}
    
    for class_id in [1, 2]:
        binary_mask = (nuclei_mask == class_id).astype(np.uint8)
        labeled_mask, num_features = scipy_label(binary_mask)
        
        removed_count = 0
        for i in range(1, num_features + 1):
            component_mask = (labeled_mask == i)
            area = np.sum(component_mask)
            
            if area < min_size:
                processed_mask[component_mask] = 0
                removed_count += 1
        
        removed_counts[CLASS_NAMES[class_id]] = removed_count
    
    return processed_mask, removed_counts

def apply_post_processing(pred_mask, min_object_size=50):
    """Apply full post-processing pipeline."""
    stats = {}
    
    mask_no_boundaries, boundary_count = remove_boundaries_from_nuclei(pred_mask)
    stats['boundaries_removed'] = boundary_count
    
    processed_mask, removed_counts = remove_small_objects_from_mask(mask_no_boundaries, min_size=min_object_size)
    stats['small_objects_removed'] = removed_counts
    
    return processed_mask, stats

def compute_ece(y_true, y_pred, y_conf, n_bins=10):
    """Calculate Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        in_bin = (y_conf > bin_lower) & (y_conf <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == y_pred[in_bin])
            avg_confidence_in_bin = np.mean(y_conf[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def benchmark_inference_speed(model, num_warmup=10, num_runs=50, batch_size=1):
    """Benchmark inference speed."""
    print(f"\nBenchmarking inference speed (Batch Size: {batch_size})...")
    
    model = model.to(DEVICE)
    model.eval()
    
    dummy_input = np.random.randint(0, 256, size=(batch_size, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    dummy_batch = []
    for i in range(batch_size):
        transformed = transform(image=dummy_input[i])
        dummy_batch.append(transformed['image'])
    dummy_tensor = torch.stack(dummy_batch).to(DEVICE)
    
    print(" Warming up GPU/CPU...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(pixel_values=dummy_tensor)
    
    print(f" Running {num_runs} iterations...")
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(pixel_values=dummy_tensor)
        end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_runs
    avg_time_per_image = avg_time_per_batch / batch_size
    fps = 1.0 / avg_time_per_image
    
    print(f" Result: {avg_time_per_image*1000:.2f} ms/image | {fps:.2f} FPS")
    
    return {
        'ms_per_image': avg_time_per_image * 1000,
        'fps': fps,
        'batch_size': batch_size
    }

def build_and_load_model():
    """Build and load the trained SegFormer model."""
    print("\nBuilding SegFormer model architecture...")
    
    model_name = "nvidia/mit-b3"
    config = SegformerConfig.from_pretrained(model_name)
    config.num_labels = NUM_CLASSES
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    )
    
    print("Loading trained weights...")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(" Model loaded successfully")
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise
    
    model = model.to(DEVICE)
    model.eval()
    
    return model

def load_tissue_model():
    """Load TensorFlow tissue segmentation model."""
    print(f"\nLoading Tissue Specialist Model (Backbone: {TISSUE_BACKBONE})...")
    model_spec = sm.Unet(backbone_name=TISSUE_BACKBONE, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), 
                         classes=TISSUE_NUM_CLASSES, activation='softmax', encoder_weights=None)
    model_spec.load_weights(TISSUE_MODEL_SPEC_PATH)
    print("Tissue model loaded successfully.")
    return model_spec

def run_evaluation(model, tissue_model, data_list):
    """Run evaluation on all patches."""
    print("\nRunning evaluation...")
    
    all_coverage_results = []
    ki67_results = []
    samples_for_vis = []
    
    ece_true_sample = []
    ece_pred_sample = []
    ece_conf_sample = []
    
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    with torch.no_grad():
        for i, data in enumerate(data_list):
            image_path = data['image_path']
            keypoints = data['keypoints']
            
            img_raw = img_to_array(load_img(str(image_path), target_size=(512, 512)))
            img_norm = img_raw / 255.0
            
            cancer_mask, cancer_image = extract_cancer_region_ensemble(img_norm, tissue_model)
            
            filtered_kps, excluded_kps = filter_keypoints_by_cancer_mask(keypoints, cancer_mask)
            
            image = Image.open(image_path).convert('RGB')
            image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
            image_np = np.array(image)
            
            transformed = transform(image=image_np)
            input_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
            
            outputs = model(pixel_values=input_tensor)
            logits = outputs.logits
            
            logits = nn.functional.interpolate(
                logits,
                size=(IMG_HEIGHT, IMG_WIDTH),
                mode='bilinear',
                align_corners=False
            )
            
            probs = torch.softmax(logits, dim=1)
            pred_mask = torch.argmax(probs, dim=1).cpu().numpy()[0]
            confidences = torch.max(probs, dim=1)[0].cpu().numpy()[0]
            
            pred_mask_processed, stats = apply_post_processing(pred_mask, min_object_size=50)
            
            coverage_result = evaluate_patch_coverage(filtered_kps, pred_mask_processed, excluded_kps)
            coverage_result['post_processing'] = stats
            all_coverage_results.append(coverage_result)
            
            ki67_result = evaluate_patch_ki67(filtered_kps, pred_mask_processed)
            ki67_result['filename'] = Path(image_path).name
            ki67_results.append(ki67_result)
            
            if len(ece_true_sample) < 100000:
                gt_mask = np.zeros_like(pred_mask_processed)
                for kp in filtered_kps:
                    x_int = int(round(kp['x']))
                    y_int = int(round(kp['y']))
                    if 0 <= y_int < gt_mask.shape[0] and 0 <= x_int < gt_mask.shape[1]:
                        gt_mask[y_int, x_int] = KEYPOINT_TO_CLASS.get(kp['label'], 0)
                
                true_flat = gt_mask.flatten()
                pred_flat = pred_mask_processed.flatten()
                conf_flat = confidences.flatten()
                
                sample_size = min(10000, len(true_flat), 100000 - len(ece_true_sample))
                if sample_size > 0:
                    indices = np.random.choice(len(true_flat), size=sample_size, replace=False)
                    ece_true_sample.extend(true_flat[indices].tolist())
                    ece_pred_sample.extend(pred_flat[indices].tolist())
                    ece_conf_sample.extend(conf_flat[indices].tolist())
            
            if len(samples_for_vis) < 5:
                samples_for_vis.append({
                    'image': img_raw.astype(np.uint8),
                    'cancer_mask': cancer_mask,
                    'pred_mask': pred_mask_processed,
                    'keypoints': filtered_kps,
                    'ki67_data': ki67_result,
                    'post_proc_stats': stats
                })
            elif random.random() < 0.1 and len(samples_for_vis) < 5:
                samples_for_vis[random.randint(0, len(samples_for_vis)-1)] = {
                    'image': img_raw.astype(np.uint8),
                    'cancer_mask': cancer_mask,
                    'pred_mask': pred_mask_processed,
                    'keypoints': filtered_kps,
                    'ki67_data': ki67_result,
                    'post_proc_stats': stats
                }
            
            if (i + 1) % 50 == 0 or (i + 1) == len(data_list):
                print(f"Processed {i + 1}/{len(data_list)} patches...")
    
    ece_true_sample = np.array(ece_true_sample)
    ece_pred_sample = np.array(ece_pred_sample)
    ece_conf_sample = np.array(ece_conf_sample)
    
    return all_coverage_results, ki67_results, samples_for_vis, (ece_true_sample, ece_pred_sample, ece_conf_sample)

def aggregate_results(all_results):
    """Aggregate coverage results from all patches."""
    aggregated = {
        'keypoint_coverage': defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'background': 0, 'total': 0}),
        'mask_validity': defaultdict(lambda: {'with_keypoint': 0, 'without_keypoint': 0, 'total': 0, 'total_area': 0}),
        'excluded_keypoints': defaultdict(int),
        'confusion': defaultdict(int),
        'total_patches': len(all_results)
    }
    
    for result in all_results:
        for label, count in result.get('excluded_keypoints', {}).items():
            aggregated['excluded_keypoints'][label] += count
        for label, counts in result['keypoint_metrics'].items():
            aggregated['keypoint_coverage'][label]['correct'] += counts['correct']
            aggregated['keypoint_coverage'][label]['incorrect'] += counts['incorrect']
            aggregated['keypoint_coverage'][label]['background'] += counts['background']
        
        for label in result['total_keypoints']:
            aggregated['keypoint_coverage'][label]['total'] += result['total_keypoints'][label]
        
        for class_name, counts in result['mask_metrics'].items():
            aggregated['mask_validity'][class_name]['with_keypoint'] += counts['with_keypoint']
            aggregated['mask_validity'][class_name]['without_keypoint'] += counts['without_keypoint']
            aggregated['mask_validity'][class_name]['total_area'] += counts['total_area']
            aggregated['mask_validity'][class_name]['total'] += (counts['with_keypoint'] + counts['without_keypoint'])
        
        for key, count in result['confusion'].items():
            aggregated['confusion'][key] += count
    
    return aggregated

def calculate_metrics(aggregated):
    """Calculate final metrics."""
    metrics = {
        'keypoint_coverage': {},
        'mask_validity': {},
        'overall': {}
    }
    
    total_correct = 0
    total_incorrect = 0
    total_background = 0
    total_keypoints = 0
    
    for label in ['negative', 'positive']:
        total = aggregated['keypoint_coverage'][label]['total']
        correct = aggregated['keypoint_coverage'][label]['correct']
        incorrect = aggregated['keypoint_coverage'][label]['incorrect']
        background = aggregated['keypoint_coverage'][label]['background']
        
        metrics['keypoint_coverage'][label] = {
            'total_keypoints': total,
            'correct': correct,
            'incorrect': incorrect,
            'background': background,
            'coverage_rate': correct / total if total > 0 else 0.0,
            'incorrect_rate': incorrect / total if total > 0 else 0.0,
            'background_rate': background / total if total > 0 else 0.0
        }
        
        total_correct += correct
        total_incorrect += incorrect
        total_background += background
        total_keypoints += total
    
    if total_keypoints > 0:
        metrics['overall']['coverage_rate'] = total_correct / total_keypoints
        metrics['overall']['incorrect_rate'] = total_incorrect / total_keypoints
        metrics['overall']['background_rate'] = total_background / total_keypoints
    
    metrics['excluded_keypoints'] = dict(aggregated['excluded_keypoints'])
    
    for class_name in ['negative', 'positive']:
        total_masks = aggregated['mask_validity'][class_name]['total']
        with_kp = aggregated['mask_validity'][class_name]['with_keypoint']
        without_kp = aggregated['mask_validity'][class_name]['without_keypoint']
        total_area = aggregated['mask_validity'][class_name]['total_area']
        
        metrics['mask_validity'][class_name] = {
            'total_blobs': total_masks,
            'with_keypoint': with_kp,
            'without_keypoint': without_kp,
            'with_keypoint_rate': with_kp / total_masks if total_masks > 0 else 0.0,
            'without_keypoint_rate': without_kp / total_masks if total_masks > 0 else 0.0,
            'total_area': total_area
        }
    
    metrics['confusion'] = dict(aggregated['confusion'])
    return metrics

def calculate_ki67_metrics(ki67_results):
    """Calculate global Ki-67 metrics."""
    errors = [r['ki67_abs_error'] for r in ki67_results]
    
    ki67_metrics = {
        'mae': np.mean(errors) if errors else 0.0,
        'patch_details': ki67_results
    }

    total_gt_pos = sum(r['gt_pos'] for r in ki67_results)
    total_gt_all = sum(r['gt_total'] for r in ki67_results)
    ki67_metrics['global_gt'] = total_gt_pos / total_gt_all if total_gt_all else 0.0
    
    total_pred_pos = sum(r['pred_pos'] for r in ki67_results)
    total_pred_all = sum(r['pred_total'] for r in ki67_results)
    ki67_metrics['global_pred'] = total_pred_pos / total_pred_all if total_pred_all else 0.0
    
    ki67_metrics['global_error'] = abs(ki67_metrics['global_gt'] - ki67_metrics['global_pred'])
    
    return ki67_metrics

def visualize_3_panel(samples):
    """Generate 3-panel visualization."""
    print("\nGenerating 3-panel visualization...")
    
    num_samples = min(10, len(samples))
    samples = samples[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i, sample in enumerate(samples):
        image = sample['image']
        pred_mask = sample['pred_mask']
        keypoints = sample['keypoints']
        
        # Original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Prediction mask
        axes[i, 1].imshow(decode_mask_to_colors(pred_mask))
        axes[i, 1].set_title('Prediction Mask', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
        
        # Overlay with keypoints
        overlay = image.copy()
        for kp in keypoints:
            x, y = int(kp['x']), int(kp['y'])
            color = (255, 0, 0) if kp['label'] == 'positive' else (0, 0, 255)
            cv2.circle(overlay, (x, y), 5, color, -1)
        
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title('Image + Keypoints', fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'visualization_3_panel.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Visualization saved.")

def plot_keypoint_coverage_metrics(metrics):
    """Generate keypoint coverage plots."""
    print("\nGenerating keypoint coverage plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    labels = ['Negative', 'Positive']
    
    coverage_rates = [
        metrics['keypoint_coverage']['negative']['coverage_rate'],
        metrics['keypoint_coverage']['positive']['coverage_rate']
    ]
    background_rates = [
        metrics['keypoint_coverage']['negative']['background_rate'],
        metrics['keypoint_coverage']['positive']['background_rate']
    ]
    incorrect_rates = [
        metrics['keypoint_coverage']['negative']['incorrect_rate'],
        metrics['keypoint_coverage']['positive']['incorrect_rate']
    ]
    
    x = np.arange(len(labels))
    width = 0.25
    
    axes[0].bar(x - width, coverage_rates, width, label='Correct', color='#4CAF50')
    axes[0].bar(x, background_rates, width, label='Background', color='#FFC107')
    axes[0].bar(x + width, incorrect_rates, width, label='Incorrect', color='#F44336')
    axes[0].set_title('Keypoint Coverage Rates', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(axis='y', alpha=0.3)
    
    counts = [
        metrics['keypoint_coverage']['negative']['total_keypoints'],
        metrics['keypoint_coverage']['positive']['total_keypoints']
    ]
    axes[1].bar(labels, counts, color=['#2196F3', '#E91E63'])
    axes[1].set_title('Total Keypoints', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'keypoint_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Keypoint metrics plot saved.")

def generate_full_report(metrics, ki67_metrics, num_patches):
    """Generate full text report matching U-Net format."""
    print("\nGenerating full text report...")
    
    report_lines = []
    report_lines.append("NUCLEI SEGMENTATION FULL REPORT")
    report_lines.append("=================================")
    report_lines.append(f"Nuclei Model: {MODEL_PATH}")
    report_lines.append("Tissue Model: Specialist")
    report_lines.append(f"Total Patches Evaluated: {num_patches}")
    report_lines.append("Post-Processing: Boundary Removal + Small Object Removal (min_size=50)")
    
    report_lines.append("\n--- 1. KI-67 INDEX ACCURACY ---")
    report_lines.append(f"  Mean Absolute Error (Patch-level): {ki67_metrics['mae']:.4f}")
    report_lines.append(f"  Global GT Ki-67 Index:             {ki67_metrics['global_gt']:.4f}")
    report_lines.append(f"  Global Pred Ki-67 Index:           {ki67_metrics['global_pred']:.4f}")
    report_lines.append(f"  Global Error:                      {ki67_metrics['global_error']:.4f}")
    
    report_lines.append("\n--- 2. KEYPOINT COVERAGE & VALIDITY ---")
    
    total_kps_in_cancer = sum(metrics['keypoint_coverage'][label]['total_keypoints'] for label in ['negative', 'positive'])
    total_excluded = sum(metrics.get('excluded_keypoints', {}).values())
    overall_coverage = metrics['overall']['coverage_rate']
    overall_background = metrics['overall']['background_rate']
    
    report_lines.append("  OVERALL COVERAGE (Keypoint-wise):")
    report_lines.append(f"    Total Keypoints in Cancer Region: {total_kps_in_cancer}")
    report_lines.append(f"    Keypoint Coverage Rate (Correctly predicted): {overall_coverage:.2%}")
    report_lines.append(f"    Predicted as Background: {overall_background:.2%}")
    report_lines.append(f"    Keypoints outside Cancer Region (Excluded): {total_excluded}")
    
    report_lines.append("  CLASS COVERAGE:")
    
    for class_label in ['positive', 'negative']:
        class_display = class_label.upper()
        m_kp = metrics['keypoint_coverage'][class_label]
        m_mask = metrics['mask_validity'][class_label]
        
        report_lines.append(f"    {class_display} (Keypoints: {m_kp['total_keypoints']}):")
        report_lines.append(f"      Coverage Rate: {m_kp['coverage_rate']:.2%}")
        report_lines.append(f"      Background Rate: {m_kp['background_rate']:.2%}")
        
        report_lines.append(f"    {class_display} MASKS (Total Blobs: {m_mask['total_blobs']}):")
        report_lines.append(f"      Mask Validity Rate (w/ KP): {m_mask['with_keypoint_rate']:.2%}")
        report_lines.append(f"      False Positive Rate (w/o KP): {m_mask['without_keypoint_rate']:.2%}")
    
    report_text = "\n".join(report_lines)
    
    with open(OUTPUT_DIR / 'evaluation_report.txt', 'w') as f:
        f.write(report_text)
    print(f" Text report saved to {OUTPUT_DIR / 'evaluation_report.txt'}")
    
    print("\n" + report_text)

def save_json_results_old(metrics, ki67_metrics, time_metrics, ece_metrics):
    """Save detailed results to JSON."""
    print("\nSaving JSON results...")
    
    results = {
        'model_info': {
            'path': str(MODEL_PATH),
            'architecture': 'SegFormer (nvidia/mit-b3)',
            'framework': f'PyTorch {torch.__version__}',
            'device': str(DEVICE)
        },
        'keypoint_coverage': metrics['keypoint_coverage'],
        'mask_validity': metrics['mask_validity'],
        'overall': metrics['overall'],
        'confusion': metrics['confusion'],
        'ki67_metrics': ki67_metrics,
        'inference_performance': time_metrics,
        'ece': float(ece_metrics)
    }
    
    with open(OUTPUT_DIR / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f" JSON results saved to {OUTPUT_DIR / 'evaluation_results.json'}")

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print(" " * 15 + "SEGFORMER NUCLEI MODEL EVALUATION")
    print("=" * 80)
    
    # Load data
    data_list = load_slide_data(DATA_ROOT, max_patches=MAX_PATCHES)
    if not data_list:
        print("ERROR: No data found")
        return
    
    model = build_and_load_model()
    tissue_model = load_tissue_model()
    
    all_coverage_results, ki67_results, samples, ece_data = run_evaluation(model, tissue_model, data_list)
    
    aggregated = aggregate_results(all_coverage_results)
    metrics = calculate_metrics(aggregated)
    ki67_metrics = calculate_ki67_metrics(ki67_results)
    
    visualize_3_panel(samples)
    plot_keypoint_coverage_metrics(metrics)
    
    generate_full_report(metrics, ki67_metrics, len(data_list))
    
    print("\n" + "=" * 80)
    print(" ALL EVALUATION TASKS COMPLETED SUCCESSFULLY")
    print(f" All results saved to: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()

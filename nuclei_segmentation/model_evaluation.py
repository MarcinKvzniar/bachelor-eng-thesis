"""
Nuclei Segmentation Model Evaluation Script
Evaluates model predictions against keypoint annotations by checking mask coverage.
"""

import os
import json
import warnings
import random  
from pathlib import Path
from collections import defaultdict
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label as scipy_label
import cv2

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D, Add, Multiply, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import segmentation_models as sm

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

DATA_ROOT = ""

MODEL_PATH = ''
MODEL_TYPE = 'unet' # 'alternatives': 'attention_unet', 'unet'
BACKBONE = 'seresnet50' # 'alternatives': 'efficientnetb4', 'seresnet50'  

OUTPUT_DIR = Path('')
TISSUE_MODEL_SPEC_PATH = ''
TISSUE_BACKBONE = 'seresnet50' 

MAX_PATCHES = None  # patch limit for evaluation (None = all patches)
IMG_HEIGHT, IMG_WIDTH = 512, 512
NUM_CLASSES = 4 
TISSUE_NUM_CLASSES = 3

# Class definitions
CLASS_NAMES = ['Background', 'Negative', 'Positive', 'Boundaries']
CLASS_MAPPING = {
    (255, 255, 255): 0, 
    (112, 112, 225): 1, 
    (250, 62, 62): 2,   
    (0, 0, 0): 3,       
}
INVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

# Label mapping for keypoints
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

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"Output: {OUTPUT_DIR}")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def attention_gate(gating_signal, skip_connection, inter_channels):
    theta_g = Conv2D(inter_channels, kernel_size=1, strides=1, padding="same")(gating_signal)
    theta_g = BatchNormalization()(theta_g)

    phi_x = Conv2D(inter_channels, kernel_size=1, strides=1, padding="same")(skip_connection)
    phi_x = BatchNormalization()(phi_x)

    add_xg = Add()([theta_g, phi_x])
    act_xg = Activation("relu")(add_xg)

    psi = Conv2D(1, kernel_size=1, strides=1, padding="same")(act_xg)
    psi = BatchNormalization()(psi)
    psi = Activation("sigmoid")(psi)

    return Multiply()([skip_connection, psi])

def conv_block(x, filters, kernel_size=3):
    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    return Activation("relu")(x)

def decoder_block(x, skip, filters, use_attention=True, dropout_rate=0.1):
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters, kernel_size=2, padding="same")(x)

    if use_attention:
        skip = attention_gate(gating_signal=x, skip_connection=skip, inter_channels=filters // 2)

    x = concatenate([x, skip], axis=-1)

    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    return conv_block(x, filters)

def build_attention_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES, backbone="efficientnetb4"):
    keras.backend.set_image_data_format("channels_last")
    print("Creating EfficientNetB4 model with 3-channel input")
    input_layer = Input(shape=(512, 512, 3), name="input_layer")

    base_model = EfficientNetB4(include_top=False, weights=None, input_tensor=input_layer)
    try:
        weights_path = tf.keras.utils.get_file(
            "efficientnetb4_notop.h5",
            "https://storage.googleapis.com/keras-applications/efficientnetb4_notop.h5",
            cache_subdir="models",
        )
        base_model.load_weights(weights_path, skip_mismatch=True, by_name=True)
    except:
        print("Warning: Could not load ImageNet weights locally. Proceeding without (will load trained weights anyway).")

    skip_names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation"]
    skip_connections = [base_model.get_layer(name).output for name in skip_names]
    encoder_model = Model(inputs=base_model.input, outputs=skip_connections + [base_model.output])

    inputs = Input(shape=input_shape, name="input_layer")
    all_outputs = encoder_model(inputs, training=False)
    skip_connections = all_outputs[:-1]
    bottleneck = all_outputs[-1]

    skip4, skip3, skip2, skip1 = skip_connections[3], skip_connections[2], skip_connections[1], skip_connections[0]

    dec4 = decoder_block(bottleneck, skip4, 512, use_attention=True)
    dec3 = decoder_block(dec4, skip3, 256, use_attention=True)
    dec2 = decoder_block(dec3, skip2, 128, use_attention=True)
    dec1 = decoder_block(dec2, skip1, 64, use_attention=True)

    final_up = UpSampling2D(size=(2, 2))(dec1)
    final_conv = Conv2D(64, kernel_size=3, padding="same")(final_up)
    final_conv = BatchNormalization()(final_conv)
    final_conv = Activation("relu")(final_conv)

    outputs = Conv2D(num_classes, kernel_size=1, padding="same", activation="softmax")(final_conv)

    return Model(inputs=inputs, outputs=outputs, name=f"attention_unet_{backbone}")

def decode_mask_to_colors(mask):
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
    """Load data and optionally sample patches."""
    data_root = Path(data_root)
    slide_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    
    all_data = []
    print(f"Scanning {len(slide_dirs)} slide directories")
    
    for slide_dir in sorted(slide_dirs):
        annotations_file = slide_dir / "annotations.json"
        patches_dir = slide_dir / "patches_512_30"
        
        if not annotations_file.exists():
            if (patches_dir / "annotations.json").exists():
                annotations_file = patches_dir / "annotations.json"
            else:
                continue
                
        if not patches_dir.exists():
            continue
        
        try:
            with open(annotations_file, 'r') as f:
                coco_data = json.load(f)
        except json.JSONDecodeError:
            continue
            
        img_id_to_kps = defaultdict(list)
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            cat_id = ann['category_id']
            kp_x, kp_y = ann['keypoints'][0], ann['keypoints'][1]
            label_name = JSON_CATEGORY_ID_TO_NAME.get(cat_id)
            if label_name:
                img_id_to_kps[img_id].append({
                    'label': label_name,
                    'local_x': kp_x,
                    'local_y': kp_y
                })
        
        for img_entry in coco_data.get('images', []):
            img_id = img_entry['id']
            file_name = Path(img_entry['file_name']).name 
            patch_path = patches_dir / file_name
            keypoints = img_id_to_kps.get(img_id, [])
            
            if patch_path.exists() and len(keypoints) > 0:
                all_data.append({
                    'slide_dir': slide_dir,
                    'patch_path': patch_path,
                    'patch_info': {
                        'filename': file_name,
                        'keypoints': keypoints
                    }
                })

    total_found = len(all_data)
    print(f"Found {total_found} valid patches.")
    
    if max_patches is not None and total_found > max_patches:
        print(f"Sampling {max_patches} random patches for evaluation...")
        all_data = random.sample(all_data, max_patches)
    
    return all_data

def get_mask_at_keypoint(mask, x, y, class_id):
    x_int = int(round(x))
    y_int = int(round(y))
    if 0 <= x_int < mask.shape[1] and 0 <= y_int < mask.shape[0]:
        return mask[y_int, x_int]
    return None

def find_connected_components(mask, class_id):
    binary_mask = (mask == class_id).astype(np.uint8)
    labeled_mask, num_features = scipy_label(binary_mask)
    components = []
    for i in range(1, num_features + 1):
        component_mask = (labeled_mask == i)
        components.append({
            'id': i,
            'mask': component_mask,
            'area': component_mask.sum(),
            'centroid': np.array(np.where(component_mask)).mean(axis=1)[::-1]
        })
    return components

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
        'gt_pos': gt_pos, 'gt_neg': gt_neg, 'gt_total': gt_total, 'gt_index': gt_index,
        'pred_pos': pred_pos_count, 'pred_neg': pred_neg_count, 'pred_total': pred_total, 'pred_index': pred_index,
        'ki67_abs_error': abs(gt_index - pred_index)
    }

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
        for kp in excluded_keypoints: results['excluded_keypoints'][kp['label']] += 1
    
    for kp in keypoints:
        label = kp['label']
        expected_class_id = KEYPOINT_TO_CLASS[label]
        x, y = kp['local_x'], kp['local_y']
        
        predicted_class = get_mask_at_keypoint(pred_mask, x, y, expected_class_id)
        results['total_keypoints'][label] += 1
        
        if predicted_class is None or predicted_class == 0:
            results['keypoint_metrics'][label]['background'] += 1
        elif predicted_class == expected_class_id:
            results['keypoint_metrics'][label]['correct'] += 1
        else:
            results['keypoint_metrics'][label]['incorrect'] += 1
            if label == 'negative' and predicted_class == 2: results['confusion']['negative_as_positive'] += 1
            elif label == 'positive' and predicted_class == 1: results['confusion']['positive_as_negative'] += 1
    
    for class_name, class_id in [('negative', 1), ('positive', 2)]:
        components = find_connected_components(pred_mask, class_id)
        for component in components:
            if check_component_has_keypoint(component, keypoints, class_name):
                results['mask_metrics'][class_name]['with_keypoint'] += 1
            else:
                results['mask_metrics'][class_name]['without_keypoint'] += 1
            results['mask_metrics'][class_name]['total_area'] += component['area']
    
    return results

def check_component_has_keypoint(component, keypoints, expected_class, proximity_threshold=3):
    """Check if component has keypoint inside or nearby."""
    component_mask = component['mask']
    
    for kp in keypoints:
        if kp['label'] == expected_class:
            x_int = int(round(kp['local_x']))
            y_int = int(round(kp['local_y']))
            
            if 0 <= y_int < component_mask.shape[0] and 0 <= x_int < component_mask.shape[1]:
                if component_mask[y_int, x_int]:
                    return True
                
                y_min = max(0, y_int - proximity_threshold)
                y_max = min(component_mask.shape[0], y_int + proximity_threshold + 1)
                x_min = max(0, x_int - proximity_threshold)
                x_max = min(component_mask.shape[1], x_int + proximity_threshold + 1)
                
                window = component_mask[y_min:y_max, x_min:x_max]
                if np.any(window):
                    return True
    
    return False


def extract_cancer_region_ensemble(image, model_spec):
    input_tensor = np.expand_dims(image, axis=0)
    pred_spec = model_spec.predict(input_tensor, verbose=0)[0]
    
    mask_spec = np.argmax(pred_spec, axis=-1)
    
    cancer_mask = (mask_spec == CANCER_CLASS_ID).astype(np.uint8)
    cancer_image = image.copy()
    cancer_image[cancer_mask == 0] = 1.0 
    
    return cancer_mask, cancer_image

def filter_keypoints_by_cancer_mask(keypoints, cancer_mask):
    """Filter keypoints by cancer mask."""
    filtered_keypoints = []
    excluded_keypoints = []
    
    for kp in keypoints:
        x_int = int(round(kp['local_x']))
        y_int = int(round(kp['local_y']))
        
        if 0 <= y_int < cancer_mask.shape[0] and 0 <= x_int < cancer_mask.shape[1]:
            if cancer_mask[y_int, x_int] == 1:
                filtered_keypoints.append(kp)
            else:
                excluded_keypoints.append(kp)
        else:
            excluded_keypoints.append(kp)
    
    return filtered_keypoints, excluded_keypoints

def remove_boundaries_from_nuclei(nuclei_mask):
    """Remove boundary class from mask."""
    processed_mask = nuclei_mask.copy()
    
    boundary_count = np.sum(processed_mask == 3)
    
    processed_mask[processed_mask == 3] = 0
    
    return processed_mask, boundary_count

def remove_small_objects_from_mask(nuclei_mask, min_size=50):
    """Remove small connected components from mask."""
    processed_mask = nuclei_mask.copy()
    removed_counts = {}
    
    for class_id in [1, 2]:
        class_name = CLASS_NAMES[class_id]
        
        binary_mask = (processed_mask == class_id).astype(np.uint8)
        
        labeled_mask, num_objects = scipy_label(binary_mask)
        
        objects_before = num_objects
        
        for obj_id in range(1, num_objects + 1):
            obj_pixels = np.sum(labeled_mask == obj_id)
            if obj_pixels < min_size:
                processed_mask[labeled_mask == obj_id] = 0
        
        binary_mask_after = (processed_mask == class_id).astype(np.uint8)
        _, objects_after = scipy_label(binary_mask_after)
        
        removed_counts[class_name] = objects_before - objects_after
    
    return processed_mask, removed_counts

def apply_post_processing(pred_mask, min_object_size=50):
    """Apply post-processing pipeline to mask."""
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

def benchmark_inference_speed(model, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_warmup=10, num_runs=50, batch_size=1):
    """Benchmark inference speed."""
    print(f"\nBenchmarking inference (batch_size={batch_size})")
    
    dummy_input = np.random.rand(batch_size, *input_shape).astype(np.float32)
    
    for _ in range(num_warmup):
        _ = model.predict(dummy_input, verbose=0)
        
    start_time = time.time()
    for _ in range(num_runs):
        _ = model.predict(dummy_input, verbose=0)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_runs
    avg_time_per_image = avg_time_per_batch / batch_size
    fps = 1.0 / avg_time_per_image
    
    print(f"Result: {avg_time_per_image*1000:.2f} ms/image | {fps:.2f} FPS")
    
    return {
        'ms_per_image': avg_time_per_image * 1000,
        'fps': fps,
        'batch_size': batch_size
    }

def build_and_load_nuclei_model():
    print(f"\nBuilding model: {MODEL_TYPE} with {BACKBONE}")
    model = None
    try:
        if 'attention' in MODEL_TYPE.lower():
            model = build_attention_unet(
                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                num_classes=NUM_CLASSES,
                backbone=BACKBONE
            )
        else:
            model = sm.Unet(
                backbone_name=BACKBONE,
                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                classes=NUM_CLASSES,
                activation='softmax',
                encoder_weights=None 
            )
        print("Architecture built")
    except Exception as e:
        print(f"CRITICAL ERROR building architecture: {e}")
        return None

    print(f"Loading weights: {MODEL_PATH}")
    try:
        model.load_weights(MODEL_PATH)
        print("Weights loaded")
    except Exception as e:
        print(f"Standard load failed ({e}). Trying skip_mismatch...")
        try:
            model.load_weights(MODEL_PATH, skip_mismatch=True, by_name=True)
            print("Weights loaded with skip_mismatch")
        except Exception as e2:
            print(f"CRITICAL ERROR loading weights: {e2}")
            return None
    return model

def load_tissue_model():
    print(f"\nLoading tissue model ({TISSUE_BACKBONE})")
    model_spec = sm.Unet(backbone_name=TISSUE_BACKBONE, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), classes=TISSUE_NUM_CLASSES, activation='softmax', encoder_weights=None)
    model_spec.load_weights(TISSUE_MODEL_SPEC_PATH)
    print("Tissue model loaded")
    return model_spec

def evaluate_patch(image, keypoints, pred_mask, excluded_keypoints=None):
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
        expected_class_id = KEYPOINT_TO_CLASS[label]
        x, y = kp['local_x'], kp['local_y']
        
        predicted_class = get_mask_at_keypoint(pred_mask, x, y, expected_class_id)
        results['total_keypoints'][label] += 1
        
        if predicted_class is None:
            results['keypoint_metrics'][label]['incorrect'] += 1
        elif predicted_class == expected_class_id:
            results['keypoint_metrics'][label]['correct'] += 1
        elif predicted_class == 0:
            results['keypoint_metrics'][label]['background'] += 1
        else:
            results['keypoint_metrics'][label]['incorrect'] += 1
            if label == 'negative' and predicted_class == 2:
                results['confusion']['negative_as_positive'] += 1
            elif label == 'positive' and predicted_class == 1:
                results['confusion']['positive_as_negative'] += 1
    
    for class_name, class_id in [('negative', 1), ('positive', 2)]:
        components = find_connected_components(pred_mask, class_id)
        for component in components:
            if check_component_has_keypoint(component, keypoints, class_name):
                results['mask_metrics'][class_name]['with_keypoint'] += 1
            else:
                results['mask_metrics'][class_name]['without_keypoint'] += 1
            results['mask_metrics'][class_name]['total_area'] += component['area']
    
    return results

def aggregate_results(all_results):
    """Aggregate results from all patches."""
    aggregated = {
        'keypoint_coverage': defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'background': 0, 'total': 0}),
        'mask_validity': defaultdict(lambda: {'with_keypoint': 0, 'without_keypoint': 0, 'total': 0, 'total_area': 0}),
        'excluded_keypoints': defaultdict(int),
        'confusion': defaultdict(int),
        'total_patches': len(all_results)
    }
    
    for result in all_results:
        for label, metrics in result['keypoint_metrics'].items():
            aggregated['keypoint_coverage'][label]['correct'] += metrics['correct']
            aggregated['keypoint_coverage'][label]['incorrect'] += metrics['incorrect']
            aggregated['keypoint_coverage'][label]['background'] += metrics['background']
            aggregated['keypoint_coverage'][label]['total'] += result['total_keypoints'][label]
        
        for class_name, metrics in result['mask_metrics'].items():
            aggregated['mask_validity'][class_name]['with_keypoint'] += metrics['with_keypoint']
            aggregated['mask_validity'][class_name]['without_keypoint'] += metrics['without_keypoint']
            aggregated['mask_validity'][class_name]['total'] += metrics['with_keypoint'] + metrics['without_keypoint']
            aggregated['mask_validity'][class_name]['total_area'] += metrics['total_area']
        
        for conf_type, count in result['confusion'].items():
            aggregated['confusion'][conf_type] += count
        
        for label, count in result['excluded_keypoints'].items():
            aggregated['excluded_keypoints'][label] += count
    
    return aggregated

def calculate_metrics(aggregated):
    """Calculate final metrics."""
    metrics = {
        'keypoint_coverage': {},
        'mask_validity': {},
        'overall': {}
    }
    
    total_correct = 0; total_incorrect = 0; total_background = 0; total_keypoints = 0
    
    for label in ['negative', 'positive']:
        kp_metrics = aggregated['keypoint_coverage'][label]
        total = kp_metrics['total']
        coverage_rate = kp_metrics['correct'] / total if total > 0 else 0.0
        background_rate = kp_metrics['background'] / total if total > 0 else 0.0
        incorrect_rate = kp_metrics['incorrect'] / total if total > 0 else 0.0
        
        metrics['keypoint_coverage'][label] = {
            'total_keypoints': total, 'correctly_covered': kp_metrics['correct'], 'coverage_rate': coverage_rate,
            'predicted_as_background': kp_metrics['background'], 'background_rate': background_rate,
            'incorrectly_covered': kp_metrics['incorrect'], 'incorrect_rate': incorrect_rate
        }
        total_correct += kp_metrics['correct']; total_incorrect += kp_metrics['incorrect']; 
        total_background += kp_metrics['background']; total_keypoints += total
    
    if total_keypoints > 0:
        metrics['overall']['keypoint_coverage_rate'] = total_correct / total_keypoints
        metrics['overall']['keypoint_background_rate'] = total_background / total_keypoints
        metrics['overall']['keypoint_incorrect_rate'] = total_incorrect / total_keypoints
    
    total_excluded = sum(aggregated['excluded_keypoints'].values())
    metrics['overall']['excluded_keypoints'] = dict(aggregated['excluded_keypoints'])
    metrics['overall']['total_evaluated_keypoints'] = total_keypoints
    metrics['overall']['total_original_keypoints'] = total_keypoints + total_excluded
    if total_keypoints + total_excluded > 0:
        metrics['overall']['cancer_region_keypoint_rate'] = total_keypoints / (total_keypoints + total_excluded)
    
    for class_name in ['negative', 'positive']:
        mask_metrics = aggregated['mask_validity'][class_name]
        total_masks = mask_metrics['total']
        validity_rate = mask_metrics['with_keypoint'] / total_masks if total_masks > 0 else 0.0
        false_positive_rate = mask_metrics['without_keypoint'] / total_masks if total_masks > 0 else 0.0
        
        metrics['mask_validity'][class_name] = {
            'total_masks': total_masks, 'masks_with_keypoint': mask_metrics['with_keypoint'],
            'masks_without_keypoint': mask_metrics['without_keypoint'], 'validity_rate': validity_rate,
            'false_positive_rate': false_positive_rate, 'total_area': mask_metrics['total_area']
        }
    
    metrics['confusion'] = dict(aggregated['confusion'])
    return metrics

def run_evaluation(nuclei_model, model_spec, data_list):
    print("\nRunning evaluation")
    
    all_coverage_results = [] 
    ki67_results = []        
    samples_for_vis = []
    
    ece_true_sample = []
    ece_pred_sample = []
    ece_conf_sample = []
    
    for i, data in enumerate(data_list):
        img_raw = img_to_array(load_img(str(data['patch_path']), target_size=(512, 512)))
        img_norm = img_raw / 255.0
        
        cancer_mask, cancer_image = extract_cancer_region_ensemble(img_norm, model_spec)
        
        filtered_kps, excluded_kps = filter_keypoints_by_cancer_mask(data['patch_info']['keypoints'], cancer_mask)
        
        if 'efficientnet' in BACKBONE:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = (cancer_image - mean) / std
        else:
            inp = cancer_image
        
        pred = nuclei_model.predict(np.expand_dims(inp, axis=0), verbose=0)[0]
        pred_mask = np.argmax(pred, axis=-1)
        pred_confidences = np.max(pred, axis=-1)
        
        pred_mask_processed, post_proc_stats = apply_post_processing(pred_mask, min_object_size=50)
        
        coverage_data = evaluate_patch_coverage(filtered_kps, pred_mask_processed, excluded_kps)
        coverage_data['patch_info'] = data['patch_info']
        coverage_data['post_processing'] = post_proc_stats
        all_coverage_results.append(coverage_data)
        
        ki67_data = evaluate_patch_ki67(filtered_kps, pred_mask_processed)
        ki67_data['filename'] = data['patch_info']['filename']
        ki67_results.append(ki67_data)
        
        if len(ece_true_sample) < 100000:
            gt_mask = np.zeros_like(pred_mask_processed)
            for kp in filtered_kps:
                x_int = int(round(kp['local_x']))
                y_int = int(round(kp['local_y']))
                if 0 <= y_int < gt_mask.shape[0] and 0 <= x_int < gt_mask.shape[1]:
                    gt_mask[y_int, x_int] = KEYPOINT_TO_CLASS.get(kp['label'], 0)
            
            true_flat = gt_mask.flatten()
            pred_flat = pred_mask_processed.flatten()
            conf_flat = pred_confidences.flatten()
            
            sample_size = min(10000, len(true_flat), 100000 - len(ece_true_sample))
            if sample_size > 0:
                indices = np.random.choice(len(true_flat), size=sample_size, replace=False)
                ece_true_sample.extend(true_flat[indices].tolist())
                ece_pred_sample.extend(pred_flat[indices].tolist())
                ece_conf_sample.extend(conf_flat[indices].tolist())
        
        if len(samples_for_vis) < 5:
            samples_for_vis.append({'image': img_raw.astype(np.uint8), 'cancer_mask': cancer_mask, 'pred_mask': pred_mask_processed, 'keypoints': filtered_kps, 'ki67_data': ki67_data, 'post_proc_stats': post_proc_stats})
        elif random.random() < 0.1 and len(samples_for_vis) < 5:
             samples_for_vis[random.randint(0, len(samples_for_vis)-1)] = {'image': img_raw.astype(np.uint8), 'cancer_mask': cancer_mask, 'pred_mask': pred_mask_processed, 'keypoints': filtered_kps, 'ki67_data': ki67_data, 'post_proc_stats': post_proc_stats}

        if (i+1) % 10 == 0 or (i+1) == len(data_list): print(f"Processed {i+1}/{len(data_list)}...")
    
    ece_true_sample = np.array(ece_true_sample)
    ece_pred_sample = np.array(ece_pred_sample)
    ece_conf_sample = np.array(ece_conf_sample)
    
    print(f"\nCollected {len(ece_true_sample):,} pixels for ECE calculation")
    
    return all_coverage_results, ki67_results, samples_for_vis, (ece_true_sample, ece_pred_sample, ece_conf_sample)

def visualize_3_panel(samples):
    print("\nGenerating 3-panel visualization")
    
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6 * num_samples))
    if num_samples == 1: axes = [axes]
    
    for i, sample in enumerate(samples):
        img = sample['image']
        c_mask = sample['cancer_mask']
        p_mask = sample['pred_mask']
        kps = sample['keypoints']
        stats = sample['ki67_data']
        
        overlay = np.zeros_like(img)
        overlay[c_mask == 1] = [255, 255, 0]
        
        panel1 = img.copy()
        mask_indices = c_mask == 1
        alpha = 0.3
        panel1[mask_indices] = cv2.addWeighted(img[mask_indices], 1-alpha, overlay[mask_indices], alpha, 0)
        
        axes[i, 0].imshow(panel1)
        axes[i, 0].set_title("Input + Cancer Region (Yellow)", fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img)
        for kp in kps:
            c = 'red' if kp['label'] == 'positive' else 'blue'
            axes[i, 1].plot(kp['local_x'], kp['local_y'], 'o', color=c, markersize=6, markeredgecolor='white')
        axes[i, 1].set_title(f"GT Keypoints (Cancer Only)\nKi-67: {stats['gt_index']:.2%}", fontsize=10, fontweight='bold')
        axes[i, 1].axis('off')
        
        pred_rgb = decode_mask_to_colors(p_mask)
        panel3 = cv2.addWeighted(img, 0.6, pred_rgb, 0.4, 0)
        
        axes[i, 3-1].imshow(panel3)
        for kp in kps:
            c = 'white' if kp['label'] == 'positive' else 'cyan'
            marker = 'o' if kp['label'] == 'positive' else 'x'
            axes[i, 2].plot(kp['local_x'], kp['local_y'], marker, color=c, markersize=5, alpha=0.8)
            
        axes[i, 2].set_title(f"Pred Masks + GT Keypoints\nPred Ki-67: {stats['pred_index']:.2%} (Err: {stats['ki67_abs_error']:.2f})", fontsize=10, fontweight='bold')
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'visualization_3_panel.png', dpi=150)
    plt.close()
    print("Visualization saved.")

def plot_keypoint_coverage_metrics(metrics):
    print("\nGenerating keypoint coverage plots")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    labels = ['Negative', 'Positive']
    coverage_rates = [metrics['keypoint_coverage']['negative']['coverage_rate'], metrics['keypoint_coverage']['positive']['coverage_rate']]
    background_rates = [metrics['keypoint_coverage']['negative']['background_rate'], metrics['keypoint_coverage']['positive']['background_rate']]
    incorrect_rates = [metrics['keypoint_coverage']['negative']['incorrect_rate'], metrics['keypoint_coverage']['positive']['incorrect_rate']]
    
    x = np.arange(len(labels)); width = 0.25
    axes[0].bar(x - width, coverage_rates, width, label='Correct', color='#4CAF50')
    axes[0].bar(x, background_rates, width, label='Background', color='#FFC107')
    axes[0].bar(x + width, incorrect_rates, width, label='Incorrect', color='#F44336')
    axes[0].set_title('Keypoint Coverage Rates'); axes[0].set_xticks(x); axes[0].set_xticklabels(labels); axes[0].legend()
    
    counts = [metrics['keypoint_coverage']['negative']['total_keypoints'], metrics['keypoint_coverage']['positive']['total_keypoints']]
    axes[1].bar(labels, counts, color=['#2196F3', '#E91E63'])
    axes[1].set_title('Total Keypoints')
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'keypoint_metrics.png'); plt.close()

def generate_full_report(metrics, ki67_metrics, time_metrics, ece_value, num_patches):
    """Generate full text report."""
    print("\nGenerating report")
    
    with open(OUTPUT_DIR / 'evaluation_report.txt', 'w') as f:
        f.write("NUCLEI SEGMENTATION FULL REPORT\n=================================\n")
        f.write(f"Nuclei Model: {MODEL_PATH}\n")
        f.write(f"Tissue Model: Specialist\n")
        f.write(f"Total Patches Evaluated: {num_patches}\n")
        f.write(f"Post-Processing: Boundary Removal + Small Object Removal (min_size=50)\n")
        
        f.write("\n--- INFERENCE PERFORMANCE ---\n")
        f.write(f"  Inference Time: {time_metrics['ms_per_image']:.2f} ms/image\n")
        f.write(f"  Throughput:     {time_metrics['fps']:.2f} FPS\n")
        f.write(f"  ECE (Calibration): {ece_value:.4f}\n")
        
        f.write("\n--- 1. KI-67 INDEX ACCURACY ---\n")
        f.write(f"  Mean Absolute Error (Patch-level): {ki67_metrics['mae']:.4f}\n")
        f.write(f"  Global GT Ki-67 Index:             {ki67_metrics['global_gt']:.4f}\n")
        f.write(f"  Global Pred Ki-67 Index:           {ki67_metrics['global_pred']:.4f}\n")
        f.write(f"  Global Error:                      {ki67_metrics['global_error']:.4f}\n")
        
        f.write("\n--- 2. KEYPOINT COVERAGE & VALIDITY ---\n")
        f.write(f"  OVERALL COVERAGE (Keypoint-wise):\n")
        f.write(f"    Total Keypoints in Cancer Region: {metrics['overall']['total_evaluated_keypoints']}\n")
        f.write(f"    Keypoint Coverage Rate (Correctly predicted): {metrics['overall'].get('keypoint_coverage_rate', 0):.2%}\n")
        f.write(f"    Predicted as Background: {metrics['overall'].get('keypoint_background_rate', 0):.2%}\n")
        f.write(f"    Keypoints outside Cancer Region (Excluded): {sum(metrics['overall']['excluded_keypoints'].values())}\n")
        
        f.write(f"  CLASS COVERAGE:\n")
        for cls in ['positive', 'negative']:
            m = metrics['keypoint_coverage'][cls]
            f.write(f"    {cls.upper()} (Keypoints: {m['total_keypoints']}):\n")
            f.write(f"      Coverage Rate: {m['coverage_rate']:.2%}\n")
            f.write(f"      Background Rate: {m['background_rate']:.2%}\n")
            
            mask_m = metrics['mask_validity'][cls]
            f.write(f"    {cls.upper()} MASKS (Total Blobs: {mask_m['total_masks']}):\n")
            f.write(f"      Mask Validity Rate (w/ KP): {mask_m['validity_rate']:.2%}\n")
            f.write(f"      False Positive Rate (w/o KP): {mask_m['false_positive_rate']:.2%}\n")

def main():
    data_list = load_slide_data(DATA_ROOT, max_patches=MAX_PATCHES)
    if not data_list: return
    
    print("\n" + "="*80)
    print("TEST SET KEYPOINT STATISTICS")
    print("="*80)
    total_positive = 0
    total_negative = 0
    for data in data_list:
        keypoints = data['patch_info']['keypoints']
        total_positive += sum(1 for kp in keypoints if kp['label'] == 'positive')
        total_negative += sum(1 for kp in keypoints if kp['label'] == 'negative')
    
    total_keypoints = total_positive + total_negative
    print(f"Total Patches: {len(data_list)}")
    print(f"Total Keypoints: {total_keypoints:,}")
    print(f"  Positive (Ki-67+): {total_positive:,} ({total_positive/total_keypoints*100:.2f}%)")
    print(f"  Negative (Ki-67-): {total_negative:,} ({total_negative/total_keypoints*100:.2f}%)")
    print(f"Ground Truth Ki-67 Index: {total_positive/total_keypoints*100:.2f}%")
    print("="*80 + "\n")

    nuclei_model = build_and_load_nuclei_model()
    model_spec = load_tissue_model()
    
    time_metrics = benchmark_inference_speed(nuclei_model)
    
    all_coverage_results, ki67_results, samples, ece_data = run_evaluation(nuclei_model, model_spec, data_list)

    aggregated = aggregate_results(all_coverage_results)
    metrics = calculate_metrics(aggregated)

    ki67_metrics = calculate_ki67_metrics(ki67_results)
    
    ece_true, ece_pred, ece_conf = ece_data
    if len(ece_true) > 0:
        ece_value = compute_ece(ece_true, ece_pred, ece_conf, n_bins=10)
        print(f"\nECE (Expected Calibration Error): {ece_value:.4f}")
    else:
        ece_value = 0.0
        print("\nWarning: No data collected for ECE calculation")

    visualize_3_panel(samples)

    plot_keypoint_coverage_metrics(metrics)
    
    generate_full_report(metrics, ki67_metrics, time_metrics, ece_value, len(data_list))
    
    full_output = {
        'coverage_metrics': metrics,
        'ki67_metrics': ki67_metrics
    }
    with open(OUTPUT_DIR / 'full_metrics_dump.json', 'w') as f:
        json.dump(full_output, f, indent=4, cls=NumpyEncoder)
        
    print("\nEvaluation complete")

if __name__ == "__main__":
    main()
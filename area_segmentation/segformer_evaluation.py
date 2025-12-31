"""
SegFormer model evaluation script for tissue segmentation task.
Calculates various metrics, generates visualizations, and saves reports.
"""

import os
import json
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import albumentations as A
from albumentations.pytorch import ToTensorV2

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

CLEAN_PATH = ""
MASK_PATH = ""
MODEL_PATH = ''
OUTPUT_DIR = Path('')

# Model parameters
IMG_HEIGHT, IMG_WIDTH = 512, 512
NUM_CLASSES = 3
BATCH_SIZE = 4

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class definitions
CLASS_NAMES = ['Background', 'Cancer', 'Other Tissue']
CLASS_MAPPING = {
    (0, 0, 0): 0,          # Black - Background
    (245, 66, 66): 1,      # Red - Cancer
    (66, 135, 245): 2,     # Blue - Other tissue types
}
INVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"PyTorch version: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
print(f"Device: {DEVICE}")
print(f"Output directory: {OUTPUT_DIR}")

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

def load_image_and_mask(image_path, mask_path):
    """Load and preprocess a single image-mask pair."""
    img = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')
    
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    mask = mask.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)
    
    img = np.array(img)
    mask = convert_mask_to_classes(np.array(mask))
    
    return img, mask

def get_data_paths(clean_dir, mask_dir):
    """Find and return lists of image and corresponding mask file paths."""
    image_files = []
    mask_files = []
    
    clean_dir = Path(clean_dir)
    mask_path = Path(mask_dir)
    
    for case_dir in sorted(clean_dir.iterdir()):
        if case_dir.is_dir():
            case_id = case_dir.name
            case_images = sorted(case_dir.glob('*.png'))
            
            mask_case_dir = mask_path / case_id
            if mask_case_dir.exists():
                for img_file in case_images:
                    mask_file = mask_case_dir / img_file.name
                    
                    if mask_file.exists():
                        image_files.append(str(img_file))
                        mask_files.append(str(mask_file))
    
    print(f"Found {len(image_files)} valid image-mask pairs")
    return image_files, mask_files

def data_generator(image_paths, mask_paths, batch_size, transform):
    """Yields batches of images and masks."""
    for i in range(0, len(image_paths), batch_size):
        batch_end = min(i + batch_size, len(image_paths))
        
        batch_imgs_raw = []
        batch_imgs_tensor = []
        batch_masks = []
        
        for j in range(i, batch_end):
            img, mask = load_image_and_mask(image_paths[j], mask_paths[j])
            batch_imgs_raw.append(img)
            batch_masks.append(mask)
            
            transformed = transform(image=img)
            batch_imgs_tensor.append(transformed['image'])
        
        yield np.array(batch_imgs_raw), torch.stack(batch_imgs_tensor), batch_masks

def compute_ece(y_true, y_pred, y_conf, n_bins=10):
    """
    Calculates Expected Calibration Error (ECE).
    y_true: Flattened array of true labels
    y_pred: Flattened array of predicted labels
    y_conf: Flattened array of confidence scores (max probability)
    """
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
    """Benchmarks inference speed without data loading overhead."""
    print(f"\nBenchmarking inference speed (Batch Size: {batch_size})...")
    
    model = model.to(DEVICE)
    model.eval()
    
    dummy_input = np.random.rand(batch_size, *input_shape).astype(np.float32)
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    dummy_batch = []
    for i in range(batch_size):
        transformed = transform(image=dummy_input[i])
        dummy_batch.append(transformed['image'])
    dummy_tensor = torch.stack(dummy_batch).to(DEVICE)
    
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
    
    print(f"Result: {avg_time_per_image*1000:.2f} ms/image | {fps:.2f} FPS")
    
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
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise
    
    model = model.to(DEVICE)
    model.eval()
    
    return model

def generate_predictions_and_metrics(model, image_paths, mask_paths):
    """Generate predictions and calculate metrics incrementally."""
    print("\nGenerating predictions and calculating metrics...")
    print("Processing in batches to avoid memory issues...")
    
    total_correct = 0
    total_pixels = 0
    class_tp = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_fp = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_fn = np.zeros(NUM_CLASSES, dtype=np.int64)
    
    ece_true_sample = []
    ece_pred_sample = []
    ece_conf_sample = []
    sample_data = []
    
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    test_generator = data_generator(image_paths, mask_paths, BATCH_SIZE, transform)
    
    with torch.no_grad():
        for i, (batch_imgs_raw, batch_imgs_tensor, batch_true_masks) in enumerate(test_generator):
            batch_imgs_tensor = batch_imgs_tensor.to(DEVICE)
            
            outputs = model(pixel_values=batch_imgs_tensor)
            logits = outputs.logits
            
            logits = nn.functional.interpolate(
                logits, 
                size=(IMG_HEIGHT, IMG_WIDTH), 
                mode='bilinear', 
                align_corners=False
            )
            
            probs = torch.softmax(logits, dim=1)
            batch_pred_masks = torch.argmax(probs, dim=1).cpu().numpy()
            batch_confidences = torch.max(probs, dim=1)[0].cpu().numpy()
            
            for true_mask, pred_mask, conf_map in zip(batch_true_masks, batch_pred_masks, batch_confidences):
                true_flat = true_mask.flatten()
                pred_flat = pred_mask.flatten()
                conf_flat = conf_map.flatten()
                
                total_correct += np.sum(true_flat == pred_flat)
                total_pixels += len(true_flat)
                
                for c in range(NUM_CLASSES):
                    true_c = (true_flat == c)
                    pred_c = (pred_flat == c)
                    class_tp[c] += np.sum(true_c & pred_c)
                    class_fp[c] += np.sum(~true_c & pred_c)
                    class_fn[c] += np.sum(true_c & ~pred_c)
                
                if len(ece_true_sample) < 100000:
                    sample_size = min(10000, len(true_flat), 100000 - len(ece_true_sample))
                    if sample_size > 0:
                        indices = np.random.choice(len(true_flat), size=sample_size, replace=False)
                        ece_true_sample.extend(true_flat[indices].tolist())
                        ece_pred_sample.extend(pred_flat[indices].tolist())
                        ece_conf_sample.extend(conf_flat[indices].tolist())
            
            if len(sample_data) < 20:
                for img, true_mask, pred_mask in zip(batch_imgs_raw, batch_true_masks, batch_pred_masks):
                    if len(sample_data) < 20:
                        sample_data.append({
                            'image': img,
                            'true_mask': true_mask,
                            'pred_mask': pred_mask,
                            'accuracy': (true_mask == pred_mask).mean()
                        })
            
            processed = min((i + 1) * BATCH_SIZE, len(image_paths))
            if processed % 50 == 0 or processed == len(image_paths):
                print(f"Processed {processed}/{len(image_paths)} images...")
    
    ece_true_sample = np.array(ece_true_sample)
    ece_pred_sample = np.array(ece_pred_sample)
    ece_conf_sample = np.array(ece_conf_sample)
    
    print(f" Metrics calculated for {len(image_paths)} images")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Overall accuracy: {total_correct / total_pixels * 100:.2f}%")
    
    metrics_data = {
        'total_correct': total_correct,
        'total_pixels': total_pixels,
        'class_tp': class_tp,
        'class_fp': class_fp,
        'class_fn': class_fn,
        'ece_true': ece_true_sample,
        'ece_pred': ece_pred_sample,
        'ece_conf': ece_conf_sample
    }
    
    return metrics_data, sample_data

def calculate_and_save_metrics(metrics_data, time_metrics):
    """Calculate all metrics from incremental counters and save to JSON."""
    print("\nCalculating final metrics from counters...")
    
    total_correct = metrics_data['total_correct']
    total_pixels = metrics_data['total_pixels']
    class_tp = metrics_data['class_tp']
    class_fp = metrics_data['class_fp']
    class_fn = metrics_data['class_fn']
    
    accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0
    
    per_class_metrics = {}
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    weighted_iou = 0.0
    
    class_distribution = {}
    
    for i, class_name in enumerate(CLASS_NAMES):
        tp = class_tp[i]
        fp = class_fp[i]
        fn = class_fn[i]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        per_class_metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'iou': float(iou)
        }
        
        class_count = tp + fn
        class_distribution[class_name] = {
            'pixel_count': int(class_count),
            'percentage': float(class_count / total_pixels * 100) if total_pixels > 0 else 0.0
        }
        
        weight = class_count / total_pixels if total_pixels > 0 else 0.0
        weighted_precision += precision * weight
        weighted_recall += recall * weight
        weighted_f1 += f1 * weight
        weighted_iou += iou * weight
    
    ece = float(compute_ece(metrics_data['ece_true'], metrics_data['ece_pred'], metrics_data['ece_conf']))
    
    overall_metrics = {
        'accuracy': float(accuracy),
        'precision': float(weighted_precision),
        'recall': float(weighted_recall),
        'f1_score': float(weighted_f1),
        'iou': float(weighted_iou),
        'ece': ece,
        'inference_time_ms': float(time_metrics['ms_per_image']),
        'inference_fps': float(time_metrics['fps'])
    }
    
    all_metrics = {
        'overall_metrics': overall_metrics,
        'per_class_metrics': per_class_metrics,
        'class_distribution': class_distribution,
        'total_pixels': int(total_pixels),
        'framework': 'PyTorch',
        'model': 'SegFormer (nvidia/mit-b3)',
        'device_info': "GPU" if torch.cuda.is_available() else "CPU"
    }
    
    metrics_file = OUTPUT_DIR / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metrics saved to {metrics_file}")
    
    return overall_metrics, per_class_metrics, class_distribution

def plot_confusion_matrix(metrics_data):
    """Generate and save confusion matrix plots from counters."""
    print("\nGenerating confusion matrix...")
    
    class_tp = metrics_data['class_tp']
    class_fp = metrics_data['class_fp']
    class_fn = metrics_data['class_fn']
    
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for i in range(NUM_CLASSES):
        cm[i][i] = class_tp[i]
        total_other_fp = sum(class_fp[j] for j in range(NUM_CLASSES) if j != i)
        if class_fn[i] > 0 and total_other_fp > 0:
            for j in range(NUM_CLASSES):
                if i != j:
                    cm[i][j] = int(class_fn[i] * class_fp[j] / total_other_fp)
    
    cm_normalized = cm.astype(float)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm_normalized, row_sums, where=row_sums > 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_per_class_metrics(per_class_metrics):
    """Generate and save per-class metrics bar plots."""
    print("\nGenerating per-class metrics plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metric_names = ['precision', 'recall', 'f1_score', 'iou']
    display_names = ['Precision', 'Recall', 'F1-Score', 'IoU']
    colors = ['#808080', '#e83e3e', '#4287f5']
    
    for idx, (metric_key, metric_display) in enumerate(zip(metric_names, display_names)):
        ax = axes[idx // 2, idx % 2]
        values = [per_class_metrics[cls][metric_key] for cls in CLASS_NAMES]
        
        bars = ax.bar(CLASS_NAMES, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric_display, fontsize=12, fontweight='bold')
        ax.set_title(f'Per-Class {metric_display}', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / 'per_class_metrics.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Per-class metrics plot saved to {save_path}")

def visualize_sample_predictions(sample_data, num_samples=10):
    """Generate and save sample prediction visualizations."""
    print("\nGenerating sample predictions visualization...")
    
    num_samples = min(num_samples, len(sample_data))
    sample_indices = np.random.choice(len(sample_data), size=num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, num_samples * 5))
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(sample_indices):
        sample = sample_data[idx]
        img = sample['image']
        true_mask = sample['true_mask']
        pred_mask = sample['pred_mask']
        
        # Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(decode_mask_to_colors(true_mask))
        axes[i, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(decode_mask_to_colors(pred_mask))
        axes[i, 2].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')
        
        # Difference map
        diff = (true_mask != pred_mask).astype(np.uint8)
        error_map = np.zeros((*diff.shape, 3), dtype=np.uint8)
        error_map[diff == 1] = [255, 0, 0]  # Red for errors
        error_map[diff == 0] = [0, 255, 0]  # Green for correct
        
        accuracy = (diff == 0).sum() / diff.size * 100
        axes[i, 3].imshow(error_map)
        axes[i, 3].set_title(f'Difference (Acc: {accuracy:.2f}%)', fontsize=12, fontweight='bold')
        axes[i, 3].axis('off')
    
    plt.suptitle('Sample Predictions: Original | Ground Truth | Prediction | Difference', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_path = OUTPUT_DIR / 'sample_predictions.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Sample predictions saved to {save_path}")

def visualize_best_worst(sample_data):
    """Generate and save best/worst predictions."""
    print("\nGenerating best/worst predictions visualization...")
    
    sorted_samples = sorted(sample_data, key=lambda x: x['accuracy'], reverse=True)
    best_samples = sorted_samples[:5]
    worst_samples = sorted_samples[-5:]
    
    fig, axes = plt.subplots(5, 3, figsize=(15, 25))
    fig.suptitle('Top 5 Best Predictions', fontsize=16, fontweight='bold')
    
    for i, sample in enumerate(best_samples):
        axes[i, 0].imshow(sample['image'])
        axes[i, 0].set_title(f"Acc: {sample['accuracy']*100:.2f}%", fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(decode_mask_to_colors(sample['true_mask']))
        axes[i, 1].set_title('Ground Truth', fontsize=10)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(decode_mask_to_colors(sample['pred_mask']))
        axes[i, 2].set_title('Prediction', fontsize=10)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / 'best_predictions.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Best predictions saved to {save_path}")
    
    fig, axes = plt.subplots(5, 3, figsize=(15, 25))
    fig.suptitle('Top 5 Worst Predictions', fontsize=16, fontweight='bold')
    
    for i, sample in enumerate(worst_samples):
        axes[i, 0].imshow(sample['image'])
        axes[i, 0].set_title(f"Acc: {sample['accuracy']*100:.2f}%", fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(decode_mask_to_colors(sample['true_mask']))
        axes[i, 1].set_title('Ground Truth', fontsize=10)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(decode_mask_to_colors(sample['pred_mask']))
        axes[i, 2].set_title('Prediction', fontsize=10)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / 'worst_predictions.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Worst predictions saved to {save_path}")

def generate_text_report(overall_metrics, per_class_metrics, class_distribution, num_images, total_pixels):
    """Generate and save text summary report."""
    print("\nGenerating text report...")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(" " * 25 + "MODEL EVALUATION SUMMARY REPORT")
    report_lines.append("=" * 80)
    
    report_lines.append("\nMODEL INFORMATION:")
    report_lines.append(f"  Model Path:     {MODEL_PATH}")
    report_lines.append(f"  Architecture:   SegFormer")
    report_lines.append(f"  Framework:      PyTorch {torch.__version__}")
    report_lines.append(f"  Input Shape:    {IMG_HEIGHT}x{IMG_WIDTH}x3")
    report_lines.append(f"  Output Classes: {NUM_CLASSES}")
    report_lines.append(f"  Device:         {DEVICE}")
    
    report_lines.append("\nDATASET INFORMATION:")
    report_lines.append(f"  Data Path:      {CLEAN_PATH}")
    report_lines.append(f"  Total Images:   {num_images}")
    report_lines.append(f"  Total Pixels:   {total_pixels:,}")
    
    report_lines.append("\nOVERALL PERFORMANCE:")
    report_lines.append(f"  Pixel Accuracy:    {overall_metrics['accuracy']:.4f} ({overall_metrics['accuracy']*100:.2f}%)")
    report_lines.append(f"  Mean IoU:          {overall_metrics['iou']:.4f}")
    report_lines.append(f"  Mean F1-Score:     {overall_metrics['f1_score']:.4f}")
    report_lines.append(f"  ECE (Calibration): {overall_metrics['ece']:.4f}")
    report_lines.append(f"  Inference Time:    {overall_metrics['inference_time_ms']:.2f} ms/image")
    report_lines.append(f"  Throughput:        {overall_metrics['inference_fps']:.2f} FPS")
    
    report_lines.append("\nPER-CLASS PERFORMANCE:")
    for class_name in CLASS_NAMES:
        metrics = per_class_metrics[class_name]
        report_lines.append(f"\n  {class_name}:")
        report_lines.append(f"    Precision: {metrics['precision']:.4f}")
        report_lines.append(f"    Recall:    {metrics['recall']:.4f}")
        report_lines.append(f"    F1-Score:  {metrics['f1_score']:.4f}")
        report_lines.append(f"    IoU:       {metrics['iou']:.4f}")
    
    report_lines.append("\nCLASS DISTRIBUTION:")
    for class_name in CLASS_NAMES:
        dist = class_distribution[class_name]
        report_lines.append(f"  {class_name:<20}: {dist['pixel_count']:>12,} pixels ({dist['percentage']:>6.2f}%)")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("Evaluation complete!")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    report_file = OUTPUT_DIR / 'evaluation_report.txt'
    with open(report_file, 'w') as f:
        f.write(report_text)
    print(f" Text report saved to {report_file}")
    
    print("\n" + report_text)

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print(" " * 15 + "SEGFORMER MODEL EVALUATION")
    print("=" * 80)
    
    print("\nLoading data paths...")
    image_paths, mask_paths = get_data_paths(CLEAN_PATH, MASK_PATH)
    
    if not image_paths:
        print("ERROR: No images found!")
        return
    
    model = build_and_load_model()
    
    time_metrics = benchmark_inference_speed(model)
    
    metrics_data, sample_data = generate_predictions_and_metrics(model, image_paths, mask_paths)
    
    overall_metrics, per_class_metrics, class_distribution = calculate_and_save_metrics(metrics_data, time_metrics)
    
    plot_confusion_matrix(metrics_data)
    plot_per_class_metrics(per_class_metrics)
    visualize_sample_predictions(sample_data, num_samples=10)
    visualize_best_worst(sample_data)
    
    generate_text_report(overall_metrics, per_class_metrics, class_distribution, 
                        len(image_paths), metrics_data['total_pixels'])
    
    print("\n" + "=" * 80)
    print(f" All results saved to: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()

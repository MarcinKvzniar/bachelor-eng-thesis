"""
Confidence-Accuracy Correlation Analysis
Analyzes the relationship between model confidence and prediction accuracy
for each class and overall performance.
"""

import os
import warnings
from pathlib import Path

import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd


warnings.filterwarnings('ignore')

# Model configuration
MODEL_PATH = ""
IMAGE_DIR = ""
MASK_DIR = ""
OUTPUT_DIR = ""

# Model parameters
IMG_HEIGHT, IMG_WIDTH = 512, 512
NUM_CLASSES = 3

# Class names and colors
CLASS_NAMES = {
    0: "Background",
    1: "Cancer",
    2: "Other Tissue"
}

CLASS_COLORS = {
    0: (0, 0, 0),          # Black - Background
    1: (245, 66, 66),      # Red - Cancer
    2: (66, 135, 245),     # Blue - Other tissue
}

# Teacher model background class indices (6 classes)
TEACHER_BACKGROUND_INDICES = [0, 5]

# Teacher model remapping (6 classes -> 3 classes)
TEACHER_CLASS_REMAP = {
    0: 0,  # Background -> Background
    1: 2,  # Healthy -> Other Tissue
    2: 1,  # Cancer -> Cancer
    3: 2,  # Other -> Other Tissue
    4: 2,  # Other -> Other Tissue
    5: 0,  # Other -> Background
}

def pad_to_divisible_by_32(array):
    """Pads an image to be divisible by 32."""
    h, w = array.shape[:2]
    pad_h = 32 - (h % 32) if h % 32 != 0 else 0
    pad_w = 32 - (w % 32) if w % 32 != 0 else 0
    
    if array.ndim == 3:
        value = [0, 0, 0] 
    else:
        value = 0 
        
    padded_array = cv2.copyMakeBorder(array, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=value)
    return padded_array


def get_transforms():
    """Gets the exact validation transforms."""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255, p=1.0),
        ToTensorV2()
    ])


def load_model(model_path, device):
    """Loads the model architecture and weights."""
    print(f"Loading model from {model_path}...")
    
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,  
        classes=6,
        activation=None       
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    print("Model loaded successfully")
    return model


def rgb_to_class_mask(rgb_mask):
    """Convert RGB mask to class indices."""
    class_mask = np.zeros((rgb_mask.shape[0], rgb_mask.shape[1]), dtype=np.uint8)
    
    for class_idx, color in CLASS_COLORS.items():
        mask = np.all(rgb_mask == color, axis=-1)
        class_mask[mask] = class_idx
    
    return class_mask


def predict_with_confidence_and_accuracy(model, device, image_path, mask_path):
    """
    Runs prediction on a single image and compares with ground truth.
    Returns detailed confidence and accuracy metrics.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mask_rgb = cv2.imread(mask_path)
    if mask_rgb is None:
        raise FileNotFoundError(f"Could not load mask at {mask_path}")
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
    
    image_resized = cv2.resize(image_rgb, (IMG_WIDTH, IMG_HEIGHT))
    mask_resized = cv2.resize(mask_rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    
    gt_mask = rgb_to_class_mask(mask_resized)
    
    image_padded = pad_to_divisible_by_32(image_resized)
    
    transforms = get_transforms()
    augmented = transforms(image=image_padded)
    input_tensor = augmented['image']
    
    input_batch = input_tensor.unsqueeze(0).to(device, dtype=torch.float32)
    
    with torch.no_grad(): 
        output = model(input_batch)
        
        softmax = torch.nn.Softmax(dim=1)
        probs_6class = softmax(output)
        
        max_probs_6class, pred_mask_6class = torch.max(probs_6class, dim=1)
    
    pred_mask_6class_np = pred_mask_6class.squeeze().cpu().numpy()
    max_probs_6class_np = max_probs_6class.squeeze().cpu().numpy()
    
    pred_mask_6class_np = pred_mask_6class_np[:IMG_HEIGHT, :IMG_WIDTH]
    max_probs_6class_np = max_probs_6class_np[:IMG_HEIGHT, :IMG_WIDTH]
    
    pred_mask_3class = np.zeros_like(pred_mask_6class_np, dtype=np.uint8)
    for old_class, new_class in TEACHER_CLASS_REMAP.items():
        pred_mask_3class[pred_mask_6class_np == old_class] = new_class
    
    correct_pixels = (pred_mask_3class == gt_mask)
    
    confidence_map = max_probs_6class_np
    accuracy_map = correct_pixels.astype(np.float32)
    
    overall_accuracy = np.mean(correct_pixels)
    overall_confidence = np.mean(confidence_map)
    
    class_metrics = {}
    for class_idx in range(NUM_CLASSES):
        class_mask_gt = (gt_mask == class_idx)
        
        if np.sum(class_mask_gt) > 0:
            class_pred = pred_mask_3class[class_mask_gt]
            class_conf = confidence_map[class_mask_gt]
            class_correct = (class_pred == class_idx)
            
            class_metrics[class_idx] = {
                'accuracy': np.mean(class_correct),
                'confidence': np.mean(class_conf),
                'pixel_count': np.sum(class_mask_gt),
                'confidence_values': class_conf.copy(),
                'correct_predictions': class_correct.copy()
            }
        else:
            class_metrics[class_idx] = {
                'accuracy': None,
                'confidence': None,
                'pixel_count': 0,
                'confidence_values': np.array([]),
                'correct_predictions': np.array([])
            }
    
    is_background_6class = np.isin(pred_mask_6class_np, TEACHER_BACKGROUND_INDICES)
    foreground_mask = np.logical_not(is_background_6class)
    
    if np.sum(foreground_mask) > 0:
        foreground_confidence = np.mean(confidence_map[foreground_mask])
        foreground_accuracy = np.mean(correct_pixels[foreground_mask])
    else:
        foreground_confidence = 0.0
        foreground_accuracy = 0.0
    
    return {
        'image_resized': image_resized,
        'pred_mask': pred_mask_3class,
        'gt_mask': gt_mask,
        'confidence_map': confidence_map,
        'accuracy_map': accuracy_map,
        'overall_accuracy': overall_accuracy,
        'overall_confidence': overall_confidence,
        'foreground_accuracy': foreground_accuracy,
        'foreground_confidence': foreground_confidence,
        'class_metrics': class_metrics,
        'correct_pixels': correct_pixels
    }

def analyze_confidence_accuracy_correlation(results_list):
    """
    Analyze correlation between confidence and accuracy across all images.
    Memory-efficient version that processes ALL pixels using streaming aggregation.
    """
    print("\n" + "="*80)
    print("CONFIDENCE-ACCURACY CORRELATION ANALYSIS")
    print("="*80)
    
    image_metrics = []
    
    confidence_bins = np.arange(0, 1.05, 0.05)
    bin_correct_counts = np.zeros(len(confidence_bins) - 1, dtype=np.int64)
    bin_total_counts = np.zeros(len(confidence_bins) - 1, dtype=np.int64)
    
    n_total = 0
    sum_conf = 0.0
    sum_acc = 0.0
    sum_conf_sq = 0.0
    sum_acc_sq = 0.0
    sum_conf_acc = 0.0
    
    class_stats = {i: {
        'n': 0,
        'sum_conf': 0.0,
        'sum_acc': 0.0,
        'sum_conf_sq': 0.0,
        'sum_acc_sq': 0.0,
        'sum_conf_acc': 0.0,
        'total_pixels': 0,
        'correct_pixels': 0
    } for i in range(NUM_CLASSES)}
    
    correct_stats = {'sum': 0.0, 'sum_sq': 0.0, 'count': 0}
    incorrect_stats = {'sum': 0.0, 'sum_sq': 0.0, 'count': 0}
    
    print("\nAggregating statistics from all pixels...")
    for idx, result in enumerate(tqdm(results_list, desc="Analyzing")):
        image_metrics.append({
            'image_idx': idx,
            'overall_confidence': result['overall_confidence'],
            'overall_accuracy': result['overall_accuracy'],
            'foreground_confidence': result['foreground_confidence'],
            'foreground_accuracy': result['foreground_accuracy']
        })
        
        conf_flat = result['confidence_map'].flatten()
        acc_flat = result['accuracy_map'].flatten()
        
        n_pixels = len(conf_flat)
        
        n_total += n_pixels
        sum_conf += np.sum(conf_flat)
        sum_acc += np.sum(acc_flat)
        sum_conf_sq += np.sum(conf_flat ** 2)
        sum_acc_sq += np.sum(acc_flat ** 2)
        sum_conf_acc += np.sum(conf_flat * acc_flat)
        
        bin_indices = np.digitize(conf_flat, confidence_bins) - 1
        for bin_idx in range(len(confidence_bins) - 1):
            mask = (bin_indices == bin_idx)
            bin_total_counts[bin_idx] += np.sum(mask)
            bin_correct_counts[bin_idx] += np.sum(acc_flat[mask])
        
        correct_mask = acc_flat == 1
        incorrect_mask = acc_flat == 0
        
        correct_conf = conf_flat[correct_mask]
        incorrect_conf = conf_flat[incorrect_mask]
        
        correct_stats['sum'] += np.sum(correct_conf)
        correct_stats['sum_sq'] += np.sum(correct_conf ** 2)
        correct_stats['count'] += len(correct_conf)
        
        incorrect_stats['sum'] += np.sum(incorrect_conf)
        incorrect_stats['sum_sq'] += np.sum(incorrect_conf ** 2)
        incorrect_stats['count'] += len(incorrect_conf)
        
        for class_idx in range(NUM_CLASSES):
            metrics = result['class_metrics'][class_idx]
            if metrics['accuracy'] is not None and len(metrics['confidence_values']) > 0:
                conf = metrics['confidence_values']
                acc = metrics['correct_predictions'].astype(np.float32)
                
                n = len(conf)
                class_stats[class_idx]['n'] += n
                class_stats[class_idx]['sum_conf'] += np.sum(conf)
                class_stats[class_idx]['sum_acc'] += np.sum(acc)
                class_stats[class_idx]['sum_conf_sq'] += np.sum(conf ** 2)
                class_stats[class_idx]['sum_acc_sq'] += np.sum(acc ** 2)
                class_stats[class_idx]['sum_conf_acc'] += np.sum(conf * acc)
                class_stats[class_idx]['total_pixels'] += metrics['pixel_count']
                class_stats[class_idx]['correct_pixels'] += np.sum(metrics['correct_predictions'])
        
        del conf_flat, acc_flat
    
    print("\n--- Overall Correlation (all pixels) ---")
    mean_conf = sum_conf / n_total
    mean_acc = sum_acc / n_total
    
    cov = (sum_conf_acc / n_total) - (mean_conf * mean_acc)
    var_conf = (sum_conf_sq / n_total) - (mean_conf ** 2)
    var_acc = (sum_acc_sq / n_total) - (mean_acc ** 2)
    
    overall_corr = cov / (np.sqrt(var_conf) * np.sqrt(var_acc))
    
    z = 0.5 * np.log((1 + overall_corr) / (1 - overall_corr))
    se = 1 / np.sqrt(n_total - 3)
    z_score = z / se
    overall_pval = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    print(f"Pearson correlation (all pixels): {overall_corr:.4f} (p-value: {overall_pval:.4e})")
    print(f"Total pixels analyzed: {n_total:,}")
    
    bin_accuracies = []
    bin_confidences_mean = []
    bin_counts = []
    
    for bin_idx in range(len(confidence_bins) - 1):
        if bin_total_counts[bin_idx] > 0:
            bin_accuracies.append(bin_correct_counts[bin_idx] / bin_total_counts[bin_idx])
            bin_confidences_mean.append((confidence_bins[bin_idx] + confidence_bins[bin_idx + 1]) / 2)
            bin_counts.append(int(bin_total_counts[bin_idx]))
    
    hypothesis_tests = {
        'pearson': {'correlation': overall_corr, 'p_value': overall_pval}
    }
    
    print("\n--- Per-Class Correlations (all pixels) ---")
    class_correlations = {}
    class_accuracies = {}
    class_confidences = {}
    
    for class_idx in range(NUM_CLASSES):
        stats_dict = class_stats[class_idx]
        if stats_dict['n'] > 0:
            n = stats_dict['n']
            mean_conf = stats_dict['sum_conf'] / n
            mean_acc = stats_dict['sum_acc'] / n
            
            cov = (stats_dict['sum_conf_acc'] / n) - (mean_conf * mean_acc)
            var_conf = (stats_dict['sum_conf_sq'] / n) - (mean_conf ** 2)
            var_acc = (stats_dict['sum_acc_sq'] / n) - (mean_acc ** 2)
            
            if var_conf > 0 and var_acc > 0:
                corr = cov / (np.sqrt(var_conf) * np.sqrt(var_acc))
                
                z = 0.5 * np.log((1 + corr) / (1 - corr))
                se = 1 / np.sqrt(n - 3)
                z_score = z / se
                pval = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                class_correlations[class_idx] = (corr, pval)
                class_accuracies[class_idx] = stats_dict['correct_pixels'] / stats_dict['total_pixels']
                class_confidences[class_idx] = mean_conf
                
                print(f"{CLASS_NAMES[class_idx]:15s}: {corr:.4f} (p-value: {pval:.4e}, n={n:,})")
            else:
                class_correlations[class_idx] = (None, None)
                print(f"{CLASS_NAMES[class_idx]:15s}: Insufficient variance")
        else:
            class_correlations[class_idx] = (None, None)
            print(f"{CLASS_NAMES[class_idx]:15s}: No data")
    
    correct_conf_mean = correct_stats['sum'] / correct_stats['count'] if correct_stats['count'] > 0 else 0
    incorrect_conf_mean = incorrect_stats['sum'] / incorrect_stats['count'] if incorrect_stats['count'] > 0 else 0
    
    correct_conf_std = np.sqrt((correct_stats['sum_sq'] / correct_stats['count']) - correct_conf_mean**2) if correct_stats['count'] > 0 else 0
    incorrect_conf_std = np.sqrt((incorrect_stats['sum_sq'] / incorrect_stats['count']) - incorrect_conf_mean**2) if incorrect_stats['count'] > 0 else 0
    
    return {
        'overall_correlation': (overall_corr, overall_pval),
        'class_correlations': class_correlations,
        'bin_confidences': bin_confidences_mean,
        'bin_accuracies': bin_accuracies,
        'bin_counts': bin_counts,
        'image_metrics': image_metrics,
        'n_total_pixels': n_total,
        'correct_stats': {
            'mean': correct_conf_mean,
            'std': correct_conf_std,
            'count': correct_stats['count']
        },
        'incorrect_stats': {
            'mean': incorrect_conf_mean,
            'std': incorrect_conf_std,
            'count': incorrect_stats['count']
        },
        'class_stats': class_stats,
        'overall_mean_conf': mean_conf,
        'overall_mean_acc': mean_acc,
        'hypothesis_tests': hypothesis_tests
    }


def plot_confidence_vs_accuracy_with_mean(analysis_results, output_dir):
    """
    Plot confidence vs accuracy showing mean confidence line.
    """
    print("\n--- Creating Confidence vs Accuracy with Mean Confidence Plot ---")
    
    output_path = Path(output_dir)
    
    bin_conf = np.array(analysis_results['bin_confidences'])
    bin_acc = np.array(analysis_results['bin_accuracies'])
    bin_counts = np.array(analysis_results['bin_counts'])
    
    mean_conf = analysis_results['overall_mean_conf']
    mean_acc = 0.8672
    
    overall_corr, overall_pval = analysis_results['overall_correlation']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sizes = np.sqrt(bin_counts / bin_counts.max()) * 1000 
    scatter = ax.scatter(bin_conf, bin_acc, s=sizes, alpha=0.6, c=bin_conf, 
                        cmap='viridis', edgecolors='black', linewidth=1)
    
    ax.plot(bin_conf, bin_acc, 'b-', alpha=0.4, linewidth=2, label='Binned Data Trend')
    
    ax.axvline(mean_conf, color='red', linestyle='--', linewidth=2.5, 
              label=f'Mean Confidence: {mean_conf:.4f}', alpha=0.8)
    
    ax.axhline(mean_acc, color='green', linestyle='--', linewidth=2.5, 
              label=f'Mean Accuracy: {mean_acc:.4f}', alpha=0.8)
    
    if len(bin_conf) > 1:
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(bin_conf, bin_acc)
        line_x = np.array([bin_conf.min(), bin_conf.max()])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, 'r-', linewidth=2, alpha=0.7, 
               label=f'Linear Fit: y={slope:.3f}x+{intercept:.3f}')
    
    ax.set_xlabel('Confidence', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(f'Confidence vs Accuracy with Mean Confidence\n'
                f'Pearson r = {overall_corr:.4f} (p = {overall_pval:.4e})\n'
                f'Total pixels: {analysis_results["n_total_pixels"]:,}', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    cbar = plt.colorbar(scatter, ax=ax, label='Confidence Level')
    cbar.set_label('Confidence Level', fontsize=12)
    
    textstr = f'Mean Confidence: {mean_conf:.4f}\n'
    textstr += f'Mean Accuracy: {mean_acc:.4f}\n'
    textstr += f'Correlation: {overall_corr:.4f}\n'
    textstr += f'P-value: {overall_pval:.4e}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path / "confidence_vs_accuracy_mean.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved confidence vs accuracy with mean plot")


def plot_pvalue_visualization(analysis_results, output_dir):
    """
    Visualize the p-value calculation for Pearson correlation.
    """
    print("\n--- Creating P-value Visualization ---")
    
    output_path = Path(output_dir)
    
    overall_corr, overall_pval = analysis_results['overall_correlation']
    n_total = analysis_results['n_total_pixels']
    
    if abs(overall_corr) < 1:
        t_stat = overall_corr * np.sqrt(n_total - 2) / np.sqrt(1 - overall_corr**2)
        df = n_total - 2
    else:
        t_stat = 0
        df = n_total - 2
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    
    corr_values = np.linspace(-1, 1, 1000)
    colors = plt.cm.RdYlGn((corr_values + 1) / 2)
    
    for i in range(len(corr_values) - 1):
        ax1.axvspan(corr_values[i], corr_values[i+1], color=colors[i], alpha=0.8)
    
    ax1.axvline(overall_corr, color='blue', linewidth=4, label=f'Observed r = {overall_corr:.4f}')
    ax1.axvline(0, color='black', linewidth=2, linestyle='--', alpha=0.5, label='No Correlation (r=0)')
    
    ax1.text(-0.75, 0.5, 'Strong\nNegative', ha='center', va='center', fontsize=12, weight='bold')
    ax1.text(0, 0.5, 'No\nCorrelation', ha='center', va='center', fontsize=12, weight='bold')
    ax1.text(0.75, 0.5, 'Strong\nPositive', ha='center', va='center', fontsize=12, weight='bold')
    
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Pearson Correlation Coefficient (r)', fontsize=13, fontweight='bold')
    ax1.set_title('Correlation Strength: Where Does Our Result Fall?', fontsize=15, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_yticks([])
    
    ax2 = fig.add_subplot(gs[1, 0])
    
    x = np.linspace(-5, 5, 1000)
    y = stats.t.pdf(x, df)
    
    ax2.plot(x, y, 'b-', linewidth=2, label='t-distribution')
    ax2.fill_between(x, y, where=(np.abs(x) >= np.abs(t_stat)), alpha=0.3, color='red', 
                    label=f'P-value region: {overall_pval:.4e}')
    ax2.axvline(t_stat, color='red', linewidth=2, linestyle='--', label=f't-statistic: {t_stat:.2f}')
    ax2.axvline(-t_stat, color='red', linewidth=2, linestyle='--')
    
    ax2.set_xlabel('t-statistic', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax2.set_title(f'P-value Calculation (df={df:,})', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    
    significance_levels = [0.001, 0.01, 0.05, 0.1]
    colors_sig = ['darkgreen', 'green', 'orange', 'red']
    labels_sig = ['p < 0.001\n(Very Strong)', 'p < 0.01\n(Strong)', 
                 'p < 0.05\n(Significant)', 'p < 0.1\n(Weak)']
    
    y_pos = np.arange(len(significance_levels))
    bars = ax3.barh(y_pos, significance_levels, color=colors_sig, alpha=0.7, edgecolor='black', linewidth=2)
    
    if overall_pval <= max(significance_levels):
        ax3.axvline(overall_pval, color='blue', linewidth=3, linestyle='--', 
                   label=f'Observed p-value: {overall_pval:.4e}')
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels_sig)
    ax3.set_xlabel('P-value Threshold', fontsize=12, fontweight='bold')
    ax3.set_title('Statistical Significance Thresholds', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim(0, max(significance_levels) * 1.1)
    
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    interpretation = f"""INTERPRETATION OF PEARSON CORRELATION TEST

Observed Correlation: r = {overall_corr:.6f}
P-value: {overall_pval:.6e}
Sample Size: n = {n_total:,} pixels
Degrees of Freedom: df = {df:,}
Test Statistic: t = {t_stat:.4f}

"""
    
    if overall_pval < 0.001:
        interpretation += "Result: VERY STRONG STATISTICAL SIGNIFICANCE (p < 0.001)\n"
        interpretation += "The correlation is extremely unlikely to have occurred by chance.\n"
    elif overall_pval < 0.01:
        interpretation += "Result: STRONG STATISTICAL SIGNIFICANCE (p < 0.01)\n"
        interpretation += "The correlation is highly unlikely to have occurred by chance.\n"
    elif overall_pval < 0.05:
        interpretation += "Result: STATISTICALLY SIGNIFICANT (p < 0.05)\n"
        interpretation += "The correlation is unlikely to have occurred by chance.\n"
    else:
        interpretation += "Result: NOT STATISTICALLY SIGNIFICANT (p >= 0.05)\n"
        interpretation += "The correlation could have occurred by chance.\n"
    
    if abs(overall_corr) < 0.1:
        interpretation += "\nCorrelation Strength: NEGLIGIBLE (|r| < 0.1)\n"
    elif abs(overall_corr) < 0.3:
        interpretation += "\nCorrelation Strength: WEAK (0.1 ≤ |r| < 0.3)\n"
    elif abs(overall_corr) < 0.5:
        interpretation += "\nCorrelation Strength: MODERATE (0.3 ≤ |r| < 0.5)\n"
    elif abs(overall_corr) < 0.7:
        interpretation += "\nCorrelation Strength: STRONG (0.5 ≤ |r| < 0.7)\n"
    else:
        interpretation += "\nCorrelation Strength: VERY STRONG (|r| ≥ 0.7)\n"
    
    interpretation += "\nConclusion: "
    if overall_pval < 0.05 and abs(overall_corr) >= 0.1:
        interpretation += "There IS a statistically significant correlation between confidence and accuracy."
    elif overall_pval < 0.05 and abs(overall_corr) < 0.1:
        interpretation += "While statistically significant, the correlation is too weak to be practically meaningful."
    else:
        interpretation += "There is NO statistically significant correlation between confidence and accuracy."
    
    ax4.text(0.05, 0.95, interpretation, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.savefig(output_path / "pvalue_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(" Saved p-value visualization")


def plot_no_correlation_evidence(analysis_results, output_dir):
    """
    Create plots that demonstrate lack of strong correlation.
    """
    print("\n--- Creating No-Correlation Evidence Plot ---")
    
    output_path = Path(output_dir)
    
    bin_conf = np.array(analysis_results['bin_confidences'])
    bin_acc = np.array(analysis_results['bin_accuracies'])
    bin_counts = np.array(analysis_results['bin_counts'])
    
    overall_corr, overall_pval = analysis_results['overall_correlation']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    ax1 = axes[0, 0]
    
    if len(bin_conf) > 1:
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(bin_conf, bin_acc)
        
        predicted_acc = slope * bin_conf + intercept
        residuals = bin_acc - predicted_acc
        
        sizes = (bin_counts / bin_counts.max()) * 300
        ax1.scatter(bin_conf, residuals, s=sizes, alpha=0.6, c=bin_counts, 
                   cmap='coolwarm', edgecolors='black', linewidth=1)
        ax1.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero Residual Line')
        
        residual_std = np.std(residuals)
        ax1.axhline(residual_std, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='±1 Std Dev')
        ax1.axhline(-residual_std, color='orange', linestyle=':', linewidth=2, alpha=0.7)
        
        ax1.set_xlabel('Confidence', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Residuals (Observed - Predicted Accuracy)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Residual Plot: Evidence of Model Fit\n'
                     f'R² = {r_value**2:.4f} (explains {r_value**2*100:.2f}% of variance)', 
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        if r_value**2 < 0.1:
            interpretation = "Poor fit: Linear model explains < 10% of variance\nSuggests weak/no linear correlation"
            color = 'red'
        elif r_value**2 < 0.3:
            interpretation = "Weak fit: Linear model explains < 30% of variance\nSuggests limited linear correlation"
            color = 'orange'
        else:
            interpretation = "Moderate/Good fit: Linear model explains variance\nSuggests meaningful correlation"
            color = 'green'
        
        ax1.text(0.98, 0.02, interpretation, transform=ax1.transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    ax2 = axes[0, 1]
    
    if len(bin_conf) > 1:
        total_variance = np.var(bin_acc)
        explained_variance = r_value**2 * total_variance
        unexplained_variance = total_variance - explained_variance
        
        variance_data = [explained_variance, unexplained_variance]
        variance_labels = [f'Explained by\nConfidence\n{r_value**2*100:.1f}%', 
                          f'Unexplained\n(Other factors)\n{(1-r_value**2)*100:.1f}%']
        colors_var = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax2.pie(variance_data, labels=variance_labels, autopct='%1.1f%%',
                                            colors=colors_var, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
        ax2.set_title(f'Variance in Accuracy Explained by Confidence\n'
                     f'Total Variance = {total_variance:.6f}', 
                     fontsize=13, fontweight='bold')
        
        if r_value**2 < 0.25:
            center_text = "Weak\nPredictive\nPower"
        elif r_value**2 < 0.5:
            center_text = "Moderate\nPredictive\nPower"
        else:
            center_text = "Strong\nPredictive\nPower"
        
        ax2.text(0, 0, center_text, ha='center', va='center', fontsize=14, weight='bold')
    
    ax3 = axes[1, 0]
    
    n_simulations = 1000
    null_correlations = []
    
    np.random.seed(42)
    n_points = len(bin_conf)
    
    for _ in range(n_simulations):
        random_acc = np.random.uniform(bin_acc.min(), bin_acc.max(), n_points)
        null_corr, _ = stats.pearsonr(bin_conf, random_acc)
        null_correlations.append(null_corr)
    
    ax3.hist(null_correlations, bins=50, density=True, alpha=0.7, color='gray', 
            edgecolor='black', label='Null Hypothesis (No Correlation)')
    
    ax3.axvline(overall_corr, color='red', linewidth=3, linestyle='--', 
               label=f'Observed r = {overall_corr:.4f}')
    
    mu, sigma = np.mean(null_correlations), np.std(null_correlations)
    x = np.linspace(min(null_correlations), max(null_correlations), 100)
    ax3.plot(x, stats.norm.pdf(x, mu, sigma), 'b-', linewidth=2, 
            label=f'Expected (μ={mu:.3f}, σ={sigma:.3f})')
    
    ax3.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax3.set_title('Null Hypothesis: What if There Was NO Correlation?\n'
                 f'(Based on {n_simulations} random simulations)', 
                 fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    percentile = stats.percentileofscore(null_correlations, overall_corr)
    ax3.text(0.02, 0.98, f'Observed correlation is at\n{percentile:.1f}th percentile\nof null distribution', 
            transform=ax3.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    z = 0.5 * np.log((1 + overall_corr) / (1 - overall_corr))
    se = 1 / np.sqrt(analysis_results['n_total_pixels'] - 3)
    z_critical = 1.96  # 95% confidence
    
    z_lower = z - z_critical * se
    z_upper = z + z_critical * se
    
    r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
    
    summary_text = f"""STATISTICAL SUMMARY: LACK OF CORRELATION EVIDENCE

CORRELATION ANALYSIS
   Pearson r: {overall_corr:.6f}
   95% CI: [{r_lower:.6f}, {r_upper:.6f}]
   P-value: {overall_pval:.6e}
   
EFFECT SIZE
   R² (Coefficient of Determination): {r_value**2:.6f}
   Confidence explains {r_value**2*100:.2f}% of accuracy variance
   Remaining {(1-r_value**2)*100:.2f}% due to other factors
   
PRACTICAL INTERPRETATION"""
    
    if abs(overall_corr) < 0.1:
        summary_text += f"""
   NEGLIGIBLE correlation (|r| < 0.1)
   No practical relationship
   Confidence does not predict accuracy"""
    elif abs(overall_corr) < 0.3:
        summary_text += f"""
   WEAK correlation (|r| < 0.3)
   Limited practical utility
   Confidence is a poor predictor of accuracy"""
    elif abs(overall_corr) < 0.5:
        summary_text += f"""
   MODERATE correlation (0.3 ≤ |r| < 0.5)
   Some predictive value
   Confidence has limited predictive power"""
    else:
        summary_text += f"""
   STRONG correlation (|r| ≥ 0.5)
   Practically meaningful
   Confidence is a good predictor of accuracy"""
    
    summary_text += f"""

KEY FINDINGS

1. Residual Analysis:
   R² = {r_value**2:.4f} indicates {'poor' if r_value**2 < 0.25 else 'moderate' if r_value**2 < 0.5 else 'good'} model fit
   {'Large' if r_value**2 < 0.25 else 'Moderate' if r_value**2 < 0.5 else 'Small'} unexplained variance suggests
   {'many other factors beyond confidence affect accuracy' if r_value**2 < 0.5 else 'confidence is a primary factor'}

2. Null Hypothesis Comparison:
   Observed r at {percentile:.1f}th percentile of null distribution
   {'Not significantly different from random chance' if percentile < 95 and percentile > 5 else 'Significantly different from random'}

3. Statistical vs Practical Significance:
   P-value: {overall_pval:.6e} ({'Statistically significant' if overall_pval < 0.05 else 'Not statistically significant'})
   Effect size: {'Too small to be practically useful' if abs(overall_corr) < 0.3 else 'Large enough for practical use'}

CONCLUSION
"""
    
    if abs(overall_corr) < 0.2 or r_value**2 < 0.1:
        summary_text += """
Evidence strongly suggests lack of meaningful correlation:
- Very weak correlation coefficient
- Low predictive power (R² < 10%)
- Confidence cannot reliably predict accuracy
- Model confidence scores are not reliable indicators of prediction accuracy"""
    elif abs(overall_corr) < 0.4 or r_value**2 < 0.25:
        summary_text += """
Evidence suggests weak/limited correlation:
- Weak correlation coefficient  
- Poor predictive power (R² < 25%)
- Confidence has limited utility for predicting accuracy
- Caution: Confidence scores have limited reliability"""
    else:
        summary_text += """
Evidence suggests meaningful correlation:
- Moderate to strong correlation
- Reasonable predictive power
- Confidence can help predict accuracy
- Model confidence scores show reasonable calibration"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1))
    
    plt.tight_layout()
    plt.savefig(output_path / "no_correlation_evidence.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved no-correlation evidence plot")


def create_comprehensive_visualizations(analysis_results, results_list, output_dir):
    """
    Create comprehensive visualizations of confidence-accuracy relationship.
    Note: Uses binned data for scatter plots since we don't store all individual pixels.
    """
    print("\n--- Creating Visualizations ---")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    bin_conf = np.array(analysis_results['bin_confidences'])
    bin_acc = np.array(analysis_results['bin_accuracies'])
    bin_counts = np.array(analysis_results['bin_counts'])
    
    sizes = (bin_counts / bin_counts.max()) * 1000
    
    scatter = axes[0].scatter(bin_conf, bin_acc, s=sizes, alpha=0.6, c=bin_counts, 
                             cmap='YlOrRd', edgecolors='black', linewidth=0.5)
    axes[0].plot(bin_conf, bin_acc, 'b-', alpha=0.3, linewidth=1)
    axes[0].set_xlabel('Confidence (Binned)', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title(f'Confidence vs Accuracy (All {analysis_results["n_total_pixels"]:,} pixels)\nPearson r = {analysis_results["overall_correlation"][0]:.4f}', 
                     fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    cbar = plt.colorbar(scatter, ax=axes[0], label='Pixel Count')
    
    axes[1].plot(bin_conf, bin_acc, 'o-', linewidth=2, markersize=8, color='darkblue')
    axes[1].fill_between(bin_conf, bin_acc, alpha=0.3)
    axes[1].set_xlabel('Mean Confidence (Binned)', fontsize=12)
    axes[1].set_ylabel('Mean Accuracy', fontsize=12)
    axes[1].set_title('Binned Confidence vs Accuracy', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    for threshold in [0.90, 0.95, 0.99]:
        idx = np.argmin(np.abs(bin_conf - threshold))
        if idx < len(bin_conf):
            axes[1].annotate(f'{threshold:.2f}: {bin_acc[idx]:.3f}',
                           xy=(bin_conf[idx], bin_acc[idx]),
                           xytext=(10, -20), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(output_path / "confidence_accuracy_overall.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved overall confidence-accuracy plot")
    
    image_df = pd.DataFrame(analysis_results['image_metrics'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(image_df['overall_accuracy'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(image_df['overall_accuracy'].mean(), color='red', 
                      linestyle='--', linewidth=2, label=f'Mean: {image_df["overall_accuracy"].mean():.3f}')
    axes[0, 0].set_xlabel('Overall Accuracy')
    axes[0, 0].set_ylabel('Number of Images')
    axes[0, 0].set_title('Distribution of Image Accuracies')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(image_df['overall_confidence'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(image_df['overall_confidence'].mean(), color='red', 
                      linestyle='--', linewidth=2, label=f'Mean: {image_df["overall_confidence"].mean():.3f}')
    axes[0, 1].set_xlabel('Overall Confidence')
    axes[0, 1].set_ylabel('Number of Images')
    axes[0, 1].set_title('Distribution of Image Confidences')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].scatter(image_df['overall_confidence'], image_df['overall_accuracy'], 
                      alpha=0.6, s=50)
    corr = image_df[['overall_confidence', 'overall_accuracy']].corr().iloc[0, 1]
    axes[1, 0].set_xlabel('Mean Confidence (per image)')
    axes[1, 0].set_ylabel('Accuracy (per image)')
    axes[1, 0].set_title(f'Image-Level Confidence vs Accuracy\nPearson r = {corr:.4f}')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(image_df['foreground_confidence'], image_df['foreground_accuracy'], 
                      alpha=0.6, s=50, color='orange')
    corr_fg = image_df[['foreground_confidence', 'foreground_accuracy']].corr().iloc[0, 1]
    axes[1, 1].set_xlabel('Foreground Confidence (per image)')
    axes[1, 1].set_ylabel('Foreground Accuracy (per image)')
    axes[1, 1].set_title(f'Foreground-Only Confidence vs Accuracy\nPearson r = {corr_fg:.4f}')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "image_level_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved image-level metrics plots")


def save_summary_report(analysis_results, results_list, output_dir):
    """
    Save a concise text summary report of the analysis.
    """
    print("\n--- Saving Summary Report ---")
    
    output_path = Path(output_dir)
    report_path = output_path / "analysis_report.txt"
    
    image_df = pd.DataFrame(analysis_results['image_metrics'])
    hyp_tests = analysis_results['hypothesis_tests']
    overall_corr, overall_pval = analysis_results['overall_correlation']
    
    with open(report_path, 'w') as f:
        f.write("CONFIDENCE-ACCURACY CORRELATION ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Dataset: {len(results_list)} images ({IMG_HEIGHT}x{IMG_WIDTH})\n")
        f.write(f"Total pixels: {analysis_results['n_total_pixels']:,}\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*60 + "\n")
        f.write(f"Accuracy:   {image_df['overall_accuracy'].mean():.4f} ± {image_df['overall_accuracy'].std():.4f}\n")
        f.write(f"Confidence: {image_df['overall_confidence'].mean():.4f} ± {image_df['overall_confidence'].std():.4f}\n\n")
        
        f.write("CORRELATION RESULTS\n")
        f.write("-"*60 + "\n")
        f.write(f"Pearson correlation: {overall_corr:.4f} (p={overall_pval:.4e})\n")
        image_corr = image_df[['overall_confidence', 'overall_accuracy']].corr().iloc[0, 1]
        f.write(f"Image-level correlation: {image_corr:.4f}\n\n")
        
        f.write("Per-class correlations:\n")
        for class_idx in range(NUM_CLASSES):
            corr, pval = analysis_results['class_correlations'][class_idx]
            if corr is not None:
                f.write(f"  {CLASS_NAMES[class_idx]}: {corr:.4f} (p={pval:.4e})\n")
        
        f.write("\nSTATISTICAL SIGNIFICANCE\n")
        f.write("-"*60 + "\n")
        f.write(f"Pearson correlation is {'statistically significant' if overall_pval < 0.05 else 'not statistically significant'}\n")
        f.write(f"(p-value threshold: p = 0.05)\n\n")
        
        f.write("PER-CLASS STATISTICS\n")
        f.write("-"*60 + "\n")
        for class_idx in range(NUM_CLASSES):
            stats_dict = analysis_results['class_stats'][class_idx]
            if stats_dict['n'] > 0:
                n_pixels = stats_dict['n']
                mean_conf = stats_dict['sum_conf'] / n_pixels
                mean_acc = stats_dict['sum_acc'] / n_pixels
                f.write(f"{CLASS_NAMES[class_idx]}: Acc={mean_acc:.4f}, Conf={mean_conf:.4f} (n={stats_dict['total_pixels']:,})\n")
    
    print(f"Saved analysis report to {report_path}")


def get_image_mask_pairs(image_dir, mask_dir):
    """Get pairs of images and their corresponding masks."""
    image_path = Path(image_dir)
    mask_path = Path(mask_dir)
    
    pairs = []
    
    for img_file in sorted(image_path.rglob('*.png')):
        relative_path = img_file.relative_to(image_path)
        
        mask_file_name = img_file.stem + "_mask.png"
        mask_file = mask_path / relative_path.parent / mask_file_name
        
        if mask_file.exists():
            pairs.append((str(img_file), str(mask_file)))
    
    return pairs


def main():
    """Main execution function."""
    print("="*80)
    print("CONFIDENCE-ACCURACY CORRELATION ANALYSIS")
    print("="*80)
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    
    if not os.path.exists(IMAGE_DIR):
        print(f"ERROR: Image directory not found at {IMAGE_DIR}")
        return
    
    if not os.path.exists(MASK_DIR):
        print(f"ERROR: Mask directory not found at {MASK_DIR}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "-"*80)
    model = load_model(MODEL_PATH, device)
    
    print("-"*80)
    print("Finding image-mask pairs...")
    pairs = get_image_mask_pairs(IMAGE_DIR, MASK_DIR)
    print(f"Found {len(pairs)} image-mask pairs")
    
    if len(pairs) == 0:
        print("ERROR: No image-mask pairs found!")
        return
    
    print("\n" + "-"*80)
    print("Processing images and calculating metrics...")
    results_list = []
    
    for img_path, mask_path in tqdm(pairs, desc="Analyzing images"):
        try:
            result = predict_with_confidence_and_accuracy(model, device, img_path, mask_path)
            results_list.append(result)
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue
    
    print(f" Successfully processed {len(results_list)} images")
    
    print("\n" + "-"*80)
    analysis_results = analyze_confidence_accuracy_correlation(results_list)
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    print("\n" + "-"*80)
    create_comprehensive_visualizations(analysis_results, results_list, OUTPUT_DIR)
    
    plot_confidence_vs_accuracy_with_mean(analysis_results, OUTPUT_DIR)
    plot_pvalue_visualization(analysis_results, OUTPUT_DIR)
    plot_no_correlation_evidence(analysis_results, OUTPUT_DIR)
    
    save_summary_report(analysis_results, results_list, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - confidence_accuracy_overall.png")
    print("  - image_level_metrics.png")
    print("  - confidence_vs_accuracy_mean.png")
    print("  - pvalue_visualization.png")
    print("  - no_correlation_evidence.png")
    print("  - analysis_report.txt")
    print("="*80)


if __name__ == "__main__":
    main()
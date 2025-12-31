"""
Ensemble Model Evaluation Script
Combines predictions from two models:
1. Generalist Model (Base)
2. Specialist Model (Overwrites Cancer predictions)

Features:
- Inference Time Benchmarking
- ECE (Expected Calibration Error) Calculation
- 3-Panel Visualization (Original, GT, Ensemble)
"""

import os
import time
import warnings
import gc
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Memory optimization
tf.config.optimizer.set_jit(False)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU config warning: {e}")


CLEAN_PATH = ""
MASK_PATH = ""

MODEL_A_PATH = ''
MODEL_B_PATH = ''

OUTPUT_DIR = Path('')

IMG_HEIGHT, IMG_WIDTH = 512, 512
NUM_CLASSES = 3
BATCH_SIZE = 2

CLASS_NAMES = ['Background', 'Cancer', 'Other Tissue']

PRIMARY_CLASS_COLORS = {
    0: (0, 0, 0),           # Black - Background
    1: (245, 66, 66),       # Red - Cancer
    2: (66, 135, 245),      # Blue - Other tissue
}

CLASS_MAPPING = {
    (0, 0, 0): 0,           # Background - Black
    (245, 66, 66): 1,       # Cancer - Red variant 1
    (255, 0, 0): 1,         # Cancer - Red variant 2 (pure red)
    (66, 135, 245): 2,      # Other Tissue - Blue variant 1
    (0, 110, 255): 2,       # Other Tissue - Blue variant 2
}

INVERSE_CLASS_MAPPING = {v: PRIMARY_CLASS_COLORS[v] for v in range(len(CLASS_NAMES))}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def convert_mask_to_classes(mask_image):
    mask_classes = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    for color, class_index in CLASS_MAPPING.items():
        match = np.all(mask_image == color, axis=-1)
        mask_classes[match] = class_index
    return mask_classes

def decode_mask_to_colors(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_index, color in INVERSE_CLASS_MAPPING.items():
        color_mask[mask == class_index] = color
    return color_mask

def load_image_and_mask(image_path, mask_path):
    img = img_to_array(load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))) / 255.0
    mask = img_to_array(load_img(mask_path, target_size=(IMG_HEIGHT, IMG_WIDTH)))
    mask = convert_mask_to_classes(mask)
    return img, mask

def get_data_paths(clean_dir, mask_dir):
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
    print(f"Found {len(image_files)} valid pairs")
    return image_files, mask_files

def data_generator(image_paths, mask_paths, batch_size):
    for i in range(0, len(image_paths), batch_size):
        batch_end = min(i + batch_size, len(image_paths))
        batch_imgs = []
        batch_masks = []
        for j in range(i, batch_end):
            img, mask = load_image_and_mask(image_paths[j], mask_paths[j])
            batch_imgs.append(img)
            batch_masks.append(mask)
        yield np.array(batch_imgs), batch_masks

def benchmark_inference(model_a, model_b, input_shape=(512, 512, 3), n_warmup=5, n_runs=50):
    """Benchmarks the combined inference time of the ensemble."""
    print("\nBenchmarking Inference Time...")
    dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
    
    for _ in range(n_warmup):
        _ = model_a.predict(dummy_input, verbose=0)
        _ = model_b.predict(dummy_input, verbose=0)
        
    start_time = time.time()
    for _ in range(n_runs):
        pa = model_a.predict(dummy_input, verbose=0)
        pb = model_b.predict(dummy_input, verbose=0)
        
        ma = np.argmax(pa, axis=-1)
        mb = np.argmax(pb, axis=-1)
        ma[mb == 1] = 1
        
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_ms = (total_time / n_runs) * 1000
    fps = 1.0 / (total_time / n_runs)
    
    print(f"   Ensemble Latency: {avg_ms:.2f} ms/image")
    print(f"   Ensemble FPS:     {fps:.2f}")
    return avg_ms, fps

class ECETracker:
    """Calculates Expected Calibration Error incrementally."""
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_counts = np.zeros(n_bins)
        self.bin_correct = np.zeros(n_bins)
        self.bin_conf_sum = np.zeros(n_bins)

    def update(self, y_true, y_probs):
        labels = y_true.flatten()
        probs = y_probs.reshape(-1, y_probs.shape[-1])
        
        preds = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        is_correct = (preds == labels).astype(float)
        
        bin_indices = np.digitize(confidences, self.bin_boundaries, right=True) - 1
        bin_indices[bin_indices == -1] = 0 
        bin_indices[bin_indices == self.n_bins] = self.n_bins - 1
        
        np.add.at(self.bin_counts, bin_indices, 1)
        np.add.at(self.bin_correct, bin_indices, is_correct)
        np.add.at(self.bin_conf_sum, bin_indices, confidences)

    def compute(self):
        nonzero = self.bin_counts > 0
        acc_bin = np.zeros_like(self.bin_correct)
        acc_bin[nonzero] = self.bin_correct[nonzero] / self.bin_counts[nonzero]
        conf_bin = np.zeros_like(self.bin_conf_sum)
        conf_bin[nonzero] = self.bin_conf_sum[nonzero] / self.bin_counts[nonzero]
        prop_bin = self.bin_counts / np.sum(self.bin_counts)
        return np.sum(np.abs(acc_bin - conf_bin) * prop_bin)

def ensemble_hard_overlay(probs_generalist, probs_specialist):
    """
    Hard Overlay Strategy:
    Base is Generalist. Where Specialist predicts Cancer (Class 1), overwrite.
    """
    mask_gen = np.argmax(probs_generalist, axis=-1)
    mask_spec = np.argmax(probs_specialist, axis=-1)
    
    final_mask = mask_gen.copy()
    
    cancer_indices = (mask_spec == 1)
    final_mask[cancer_indices] = 1
    
    final_probs = probs_generalist.copy()
    final_probs[cancer_indices] = probs_specialist[cancer_indices]
    
    return final_mask, final_probs

def build_model():
    return sm.Unet('seresnet50', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), classes=NUM_CLASSES, activation='softmax', encoder_weights=None)

def calculate_metrics_incremental(confusion_matrices):
    total_cm = np.sum(confusion_matrices, axis=0)
    per_class = {}
    f1_scores, iou_scores = [], []
    
    for i, name in enumerate(CLASS_NAMES):
        tp = total_cm[i, i]
        fp = total_cm[:, i].sum() - tp
        fn = total_cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        per_class[name] = {'precision': precision, 'recall': recall, 'f1_score': f1, 'iou': iou}
        f1_scores.append(f1)
        iou_scores.append(iou)
        
    metrics = {
        'accuracy': np.trace(total_cm) / np.sum(total_cm),
        'f1_macro': np.mean(f1_scores),
        'iou_macro': np.mean(iou_scores),
        'per_class': per_class
    }
    return metrics

def save_report(metrics, ece_score, time_metrics):
    report_path = OUTPUT_DIR / 'report_hard_overlay.txt'
    with open(report_path, 'w') as f:
        f.write("Ensemble Evaluation Report\n")
        f.write("-"*30 + "\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1:         {metrics['f1_macro']:.4f}\n")
        f.write(f"Macro IoU:        {metrics['iou_macro']:.4f}\n")
        f.write(f"ECE (Calibration):{ece_score:.4f}\n")
        f.write("-" * 20 + "\n")
        f.write(f"Inference Time:   {time_metrics[0]:.2f} ms/img\n")
        f.write(f"Throughput:       {time_metrics[1]:.2f} FPS\n\n")
        
        f.write("PER-CLASS PERFORMANCE:\n")
        for cls, scores in metrics['per_class'].items():
            f.write(f"\n  {cls}:\n")
            f.write(f"    Precision: {scores['precision']:.4f}\n")
            f.write(f"    Recall:    {scores['recall']:.4f}\n")
            f.write(f"    F1-Score:  {scores['f1_score']:.4f}\n")
            f.write(f"    IoU:       {scores['iou']:.4f}\n")
    print(f"Report saved: {report_path}")

def main():
    print("Starting ensemble evaluation...")
    
    img_paths, mask_paths = get_data_paths(CLEAN_PATH, MASK_PATH)
    if not img_paths: return

    print("\nLoading models...")
    model_a = build_model(); model_a.load_weights(MODEL_A_PATH)
    model_b = build_model(); model_b.load_weights(MODEL_B_PATH)

    time_metrics = benchmark_inference(model_a, model_b)

    print("\nProcessing images...")
    cm_hard_list = []
    ece_hard = ECETracker()
    
    samples = []
    sample_indices = np.random.choice(len(img_paths), size=min(5, len(img_paths)), replace=False)
    
    gen = data_generator(img_paths, mask_paths, BATCH_SIZE)
    processed_count = 0
    
    for i, (batch_imgs, batch_masks) in enumerate(gen):
        preds_a = model_a.predict(batch_imgs, verbose=0)
        preds_b = model_b.predict(batch_imgs, verbose=0)
        
        for j in range(len(batch_imgs)):
            p_a, p_b, true_mask = preds_a[j], preds_b[j], batch_masks[j]
            
            mask_hard, prob_hard = ensemble_hard_overlay(p_a, p_b)
            
            cm = confusion_matrix(true_mask.flatten(), mask_hard.flatten(), labels=[0, 1, 2])
            cm_hard_list.append(cm)
            ece_hard.update(true_mask, prob_hard)
            
            if processed_count in sample_indices:
                samples.append({'img': batch_imgs[j], 'true': true_mask, 'hard': mask_hard})
            
            processed_count += 1
        
        if (i+1) % 10 == 0:
            print(f"Batch {i+1} processed... ({processed_count}/{len(img_paths)})")
            gc.collect()

    print("\nCalculating metrics...")
    m_hard = calculate_metrics_incremental(cm_hard_list)
    ece_score = ece_hard.compute()
    save_report(m_hard, ece_score, time_metrics)

    print("\nGenerating visualization...")
    if samples:
        fig, axes = plt.subplots(len(samples), 3, figsize=(15, 5*len(samples)))
        if len(samples) == 1: axes = [axes]
        for i, s in enumerate(samples):
            axes[i,0].imshow(s['img']); axes[i,0].set_title("Original"); axes[i,0].axis('off')
            axes[i,1].imshow(decode_mask_to_colors(s['true'])); axes[i,1].set_title("Ground Truth"); axes[i,1].axis('off')
            axes[i,2].imshow(decode_mask_to_colors(s['hard'])); axes[i,2].set_title("Hard Overlay"); axes[i,2].axis('off')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "ensemble_visualization.png", dpi=100)
        plt.close()

    print("\nDone")

if __name__ == '__main__':
    main()
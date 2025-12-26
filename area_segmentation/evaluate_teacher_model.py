"""
Teacher Model Evaluation Script
Evaluates the teacher model (PyTorch) on the final test dataset with comprehensive metrics.
"""

import os
import time
import warnings
from pathlib import Path

import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, jaccard_score
)
from tqdm import tqdm

warnings.filterwarnings('ignore')


MODEL_PATH = ""
CLEAN_PATH = ""
MASK_PATH = ""
OUTPUT_DIR = ""

# Model parameters
IMG_HEIGHT, IMG_WIDTH = 512, 512
NUM_CLASSES = 3
BATCH_SIZE = 4

# Class definitions
CLASS_NAMES = ['Background', 'Cancer', 'Other Tissue']

CLASS_MAPPING = {
    (0, 0, 0): 0,          # Black - Background
    (245, 66, 66): 1,      # Red - Cancer
    (255, 0, 0): 1,        # Red - Cancer variant
    (66, 135, 245): 2,     # Blue - Other tissue types
    (0, 110, 255): 2,
}
TEACHER_CLASS_REMAP = {
    0: 0, 1: 2, 2: 1, 3: 2, 4: 2, 5: 0
}

INVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}


class ECETracker:
    """Calculates Expected Calibration Error incrementally."""
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_counts = np.zeros(n_bins)
        self.bin_correct = np.zeros(n_bins)
        self.bin_conf_sum = np.zeros(n_bins)

    def update(self, y_true, y_probs):
        """
        y_true: (H, W) or flattened - Integer class indices
        y_probs: (H, W, Classes) or flattened - Softmax probabilities
        """
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

def benchmark_inference(model, device, input_shape=(3, 512, 512), n_warmup=10, n_runs=50):
    """Benchmarks inference speed."""
    print("\nBenchmarking Inference Time...")
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_ms = (total_time / n_runs) * 1000
    fps = 1.0 / (total_time / n_runs)
    
    print(f"   Avg Inference Time: {avg_ms:.2f} ms/image")
    print(f"   FPS:                {fps:.2f}")
    return avg_ms, fps

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
    print(f"Found {len(image_files)} valid image-mask pairs")
    return image_files, mask_files

def convert_mask_to_classes(mask_image):
    mask_classes = np.zeros((mask_image.shape[0], mask_image.shape[1]), dtype=np.uint8)
    for color, class_index in CLASS_MAPPING.items():
        match = np.all(mask_image == color, axis=-1)
        mask_classes[match] = class_index
    return mask_classes

def decode_mask_to_colors(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_index, color in INVERSE_CLASS_MAPPING.items():
        color_mask[mask == class_index] = color
    return color_mask

def load_mask(mask_path):
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    mask = convert_mask_to_classes(mask)
    return mask

def pad_to_divisible_by_32(array):
    h, w = array.shape[:2]
    pad_h = 32 - (h % 32) if h % 32 != 0 else 0
    pad_w = 32 - (w % 32) if w % 32 != 0 else 0
    if array.ndim == 3: value = [0, 0, 0] 
    else: value = 0 
    padded_array = cv2.copyMakeBorder(array, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=value)
    return padded_array

def get_transforms():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255, p=1.0),
        ToTensorV2()
    ])

def load_model(model_path, device):
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

def predict_image_with_probs(model, device, image_path):
    """
    Runs prediction and returns both the 3-class mask and 3-class probabilities.
    Includes logic to aggregate the 6 output classes into 3.
    """
    image = cv2.imread(image_path)
    if image is None: raise FileNotFoundError(f"Could not load image at {image_path}")
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_resized = cv2.resize(image_rgb, (IMG_WIDTH, IMG_HEIGHT))
    image_padded = pad_to_divisible_by_32(image_resized)
    transforms = get_transforms()
    augmented = transforms(image=image_padded)
    input_tensor = augmented['image'].unsqueeze(0).to(device, dtype=torch.float32)
    
    with torch.no_grad(): 
        output_logits = model(input_tensor) 
        probs_6class = torch.softmax(output_logits, dim=1) 
    
    probs_3class = torch.zeros((1, 3, probs_6class.shape[2], probs_6class.shape[3]), device=device)
    
    probs_3class[:, 0] = probs_6class[:, 0] + probs_6class[:, 5]
    probs_3class[:, 1] = probs_6class[:, 2]
    probs_3class[:, 2] = probs_6class[:, 1] + probs_6class[:, 3] + probs_6class[:, 4]
    
    pred_mask_3class = torch.argmax(probs_3class, dim=1)
    
    probs_np = probs_3class.squeeze().permute(1, 2, 0).cpu().numpy() 
    mask_np = pred_mask_3class.squeeze().cpu().numpy()
    
    probs_np = probs_np[:IMG_HEIGHT, :IMG_WIDTH, :]
    mask_np = mask_np[:IMG_HEIGHT, :IMG_WIDTH]
    
    return mask_np, probs_np, image_resized

def calculate_metrics(y_true_flat, y_pred_flat, ece_score, time_metrics):
    """Calculate all evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true_flat, y_pred_flat),
        'precision_weighted': precision_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0),
        'iou_weighted': jaccard_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0),
        'precision_per_class': precision_score(y_true_flat, y_pred_flat, average=None, zero_division=0),
        'recall_per_class': recall_score(y_true_flat, y_pred_flat, average=None, zero_division=0),
        'f1_per_class': f1_score(y_true_flat, y_pred_flat, average=None, zero_division=0),
        'iou_per_class': jaccard_score(y_true_flat, y_pred_flat, average=None, zero_division=0),
        'ECE': ece_score,
        'inference_time_ms': time_metrics[0],
        'fps': time_metrics[1]
    }
    return metrics

def run_full_evaluation(model, device, image_paths, mask_paths):
    print("\nStarting Evaluation Loop...")
    
    all_true_masks = []
    all_pred_masks = []
    all_images = []
    ece_tracker = ECETracker()
    
    for i in tqdm(range(len(image_paths)), desc="Evaluating"):
        try:
            pred_mask, pred_probs, img = predict_image_with_probs(model, device, image_paths[i])
            true_mask = load_mask(mask_paths[i])
            
            all_images.append(img)
            all_true_masks.append(true_mask)
            all_pred_masks.append(pred_mask)
            
            ece_tracker.update(true_mask, pred_probs)
            
        except Exception as e:
            print(f"Error: {e}")
            
    y_true_flat = np.concatenate([m.flatten() for m in all_true_masks])
    y_pred_flat = np.concatenate([m.flatten() for m in all_pred_masks])
    
    return y_true_flat, y_pred_flat, ece_tracker.compute(), all_images, all_true_masks, all_pred_masks


def print_final_summary(metrics, num_images):
    print("\n" + "="*80)
    print(" "*20 + "TEACHER MODEL EVALUATION SUMMARY")
    print("="*80)
    print(f"  Total Images:   {num_images}")
    print("\nOVERALL PERFORMANCE:")
    print(f"  Pixel Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Mean IoU:          {metrics['iou_weighted']:.4f}")
    print(f"  Mean F1-Score:     {metrics['f1_weighted']:.4f}")
    print(f"  ECE (Calibration): {metrics['ECE']:.4f}")
    print(f"  Inference Time:    {metrics['inference_time_ms']:.2f} ms/image")
    print(f"  Throughput:        {metrics['fps']:.2f} FPS")
    
    print("\nPER-CLASS PERFORMANCE:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"\n  {class_name}:")
        print(f"    Precision:  {metrics['precision_per_class'][i]:.4f}")
        print(f"    Recall:     {metrics['recall_per_class'][i]:.4f}")
        print(f"    F1-Score:   {metrics['f1_per_class'][i]:.4f}")
        print(f"    IoU:        {metrics['iou_per_class'][i]:.4f}")
    print("="*80)

def save_text_report(metrics, output_dir):
    path = os.path.join(output_dir, "evaluation_report.txt")
    with open(path, "w") as f:
        f.write("TEACHER MODEL REPORT\n")
        f.write(f"ECE: {metrics['ECE']:.4f}\n")
        f.write(f"Time: {metrics['inference_time_ms']:.2f} ms\n")
        f.write(f"Mean IoU: {metrics['iou_weighted']:.4f}\n")
    print(f" Report saved to {path}")


def main():
    print("="*80)
    print("TEACHER MODEL EVALUATION (ECE & TIME)")
    print("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    image_paths, mask_paths = get_data_paths(CLEAN_PATH, MASK_PATH)
    if not image_paths: return
    
    model = load_model(MODEL_PATH, device)
    
    time_metrics = benchmark_inference(model, device)
    
    y_true, y_pred, ece_score, imgs, t_masks, p_masks = run_full_evaluation(
        model, device, image_paths, mask_paths
    )
    
    metrics = calculate_metrics(y_true, y_pred, ece_score, time_metrics)
    
    print_final_summary(metrics, len(imgs))
    save_text_report(metrics, OUTPUT_DIR)
    

if __name__ == "__main__":
    main()
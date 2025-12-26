"""
Multi-seed Tissue Segmentation Model Evaluation
Calculates ECE and inference time for all 3 seeds (ensemble models).
Optimized for GPU execution in containerized environments.
"""

import os
import json
import warnings
import random
from pathlib import Path
import time
import numpy as np

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        print(f"  GPU devices: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU detected - running on CPU")

CLEAN_PATH = ""
MASK_PATH = ""

ENSEMBLE_CONFIGS = [
    {
        'seed': 1,
        'generalist': '',
        'specialist': ''
    },
    {
        'seed': 2,
        'generalist': '',
        'specialist': ''
    },
    {
        'seed': 3,
        'generalist': '',
        'specialist': ''
    }
]
BACKBONE = 'seresnet50'

OUTPUT_DIR = Path('')

# Model parameters
IMG_HEIGHT, IMG_WIDTH = 512, 512
NUM_CLASSES = 3
BATCH_SIZE = 4
MAX_IMAGES = None

# Class definitions
CLASS_NAMES = ['Background', 'Cancer', 'Other Tissue']
CLASS_MAPPING = {
    (0, 0, 0): 0,          # Black - Background
    (245, 66, 66): 1,      # Red - Cancer
    (66, 135, 245): 2,     # Blue - Other tissue types
}
INVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nTensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
if len(tf.config.list_physical_devices('GPU')) > 0:
    print(f"GPU Device Names: {[d.name for d in tf.config.list_physical_devices('GPU')]}")
    print(f"CUDA Available: {tf.test.is_built_with_cuda()}")

def convert_mask_to_classes(mask_image):
    """Converts a color-coded mask image to a 2D array of class indices."""
    mask_classes = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    for color, class_index in CLASS_MAPPING.items():
        match = np.all(mask_image == color, axis=-1)
        mask_classes[match] = class_index
    return mask_classes

def load_image_and_mask(image_path, mask_path):
    """Load and preprocess a single image-mask pair."""
    img = img_to_array(load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))) / 255.0
    mask = img_to_array(load_img(mask_path, target_size=(IMG_HEIGHT, IMG_WIDTH)))
    mask = convert_mask_to_classes(mask)
    return img, mask

def get_data_paths(clean_dir, mask_dir, max_images=None):
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
    
    total_found = len(image_files)
    print(f"Found {total_found} valid image-mask pairs")
    
    if max_images is not None and total_found > max_images:
        print(f"Sampling {max_images} random images...")
        indices = random.sample(range(total_found), max_images)
        image_files = [image_files[i] for i in indices]
        mask_files = [mask_files[i] for i in indices]
    
    return image_files, mask_files


def ensemble_hard_overlay(probs_generalist, probs_specialist):
    """
    Hard Overlay Strategy:
    Base is Generalist. Where Specialist predicts Cancer (Class 1), overwrite.
    Returns final mask and probability map.
    """
    mask_gen = np.argmax(probs_generalist, axis=-1)
    mask_spec = np.argmax(probs_specialist, axis=-1)
    
    final_mask = mask_gen.copy()
    
    cancer_indices = (mask_spec == 1)
    final_mask[cancer_indices] = 1
    
    final_probs = probs_generalist.copy()
    final_probs[cancer_indices] = probs_specialist[cancer_indices]
    
    return final_mask, final_probs

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

def benchmark_inference_speed_ensemble(model_gen, model_spec, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_warmup=10, num_runs=50, batch_size=1):
    """Benchmarks ensemble inference speed (generalist + specialist + overlay logic)."""
    print(f"\nBenchmarking ensemble inference speed (Batch Size: {batch_size})...")
    
    dummy_input = np.random.rand(batch_size, *input_shape).astype(np.float32)
    
    print(" Warming up...")
    for _ in range(num_warmup):
        _ = model_gen.predict(dummy_input, verbose=0)
        _ = model_spec.predict(dummy_input, verbose=0)
        
    print(f" Running {num_runs} iterations...")
    start_time = time.time()
    for _ in range(num_runs):
        pa = model_gen.predict(dummy_input, verbose=0)
        pb = model_spec.predict(dummy_input, verbose=0)
        for i in range(batch_size):
            _, _ = ensemble_hard_overlay(pa[i], pb[i])
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

def build_model():
    """Build model architecture."""
    return sm.Unet(
        backbone_name=BACKBONE,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        classes=NUM_CLASSES,
        activation='softmax',
        encoder_weights=None
    )

def load_ensemble_models(config):
    """Load both generalist and specialist models for ensemble."""
    print(f"\n--- Loading Ensemble for Seed {config['seed']} ---")
    
    print(f"Loading Generalist: {config['generalist']}")
    model_gen = build_model()
    try:
        model_gen.load_weights(config['generalist'])
        print("Generalist loaded successfully")
    except Exception as e:
        print(f"Error loading generalist: {e}")
        try:
            model_gen.load_weights(config['generalist'], skip_mismatch=True)
            print("Generalist loaded with skip_mismatch")
        except Exception as e2:
            print(f"ERROR: {e2}")
            return None, None
    
    print(f"Loading Specialist: {config['specialist']}")
    model_spec = build_model()
    try:
        model_spec.load_weights(config['specialist'])
        print("Specialist loaded successfully")
    except Exception as e:
        print(f"Error loading specialist: {e}")
        try:
            model_spec.load_weights(config['specialist'], skip_mismatch=True)
            print("Specialist loaded with skip_mismatch")
        except Exception as e2:
            print(f"ERROR: {e2}")
            return None, None
    
    return model_gen, model_spec

def calculate_ece_for_ensemble(model_gen, model_spec, image_paths, mask_paths):
    """Calculate ECE for ensemble model."""
    print("\nCollecting ensemble predictions for ECE calculation...")
    
    y_true_flat = []
    y_pred_flat = []
    y_conf_flat = []
    
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(image_paths))
        
        batch_imgs = []
        batch_masks = []
        
        for j in range(i, batch_end):
            img, mask = load_image_and_mask(image_paths[j], mask_paths[j])
            batch_imgs.append(img)
            batch_masks.append(mask)
        
        batch_imgs = np.array(batch_imgs)
        preds_gen = model_gen.predict(batch_imgs, verbose=0)
        preds_spec = model_spec.predict(batch_imgs, verbose=0)
        
        for true_mask, pred_gen, pred_spec in zip(batch_masks, preds_gen, preds_spec):
            ensemble_mask, ensemble_probs = ensemble_hard_overlay(pred_gen, pred_spec)
            
            ensemble_confidences = np.max(ensemble_probs, axis=-1)
            
            y_true_flat.append(true_mask.flatten())
            y_pred_flat.append(ensemble_mask.flatten())
            y_conf_flat.append(ensemble_confidences.flatten())
        
        processed = min((i // BATCH_SIZE + 1) * BATCH_SIZE, len(image_paths))
        if processed % 50 == 0 or processed == len(image_paths):
            print(f"Processed {processed}/{len(image_paths)} images...")
    
    y_true_flat = np.concatenate(y_true_flat)
    y_pred_flat = np.concatenate(y_pred_flat)
    y_conf_flat = np.concatenate(y_conf_flat)
    
    print(f"\nCollected {len(y_true_flat):,} pixels for ECE calculation")
    
    return y_true_flat, y_pred_flat, y_conf_flat

def evaluate_ensemble_seed(config, image_paths, mask_paths):
    """Evaluate a single ensemble seed (generalist + specialist)."""
    print("\n" + "="*80)
    print(f"EVALUATING ENSEMBLE SEED {config['seed']}")
    print("="*80)
    
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print(f"Running on GPU: {tf.config.list_physical_devices('GPU')[0].name}")
    else:
        print("Running on CPU")
    
    model_gen, model_spec = load_ensemble_models(config)
    if model_gen is None or model_spec is None:
        print("Failed to load ensemble models!")
        return None
    
    time_metrics = benchmark_inference_speed_ensemble(model_gen, model_spec, batch_size=1)
    
    y_true, y_pred, y_conf = calculate_ece_for_ensemble(model_gen, model_spec, image_paths, mask_paths)
    
    if len(y_true) > 0:
        ece_value = compute_ece(y_true, y_pred, y_conf, n_bins=10)
    else:
        ece_value = 0.0
        print("Warning: No data collected for ECE!")
    
    print("\n" + "-"*50)
    print("RESULTS")
    print("-"*50)
    print(f"Seed: {config['seed']}")
    print(f"Generalist: {config['generalist']}")
    print(f"Specialist: {config['specialist']}")
    print(f"Images Evaluated: {len(image_paths)}")
    print(f"Total Pixels: {len(y_true):,}")
    print(f"Ensemble Inference Time: {time_metrics['ms_per_image']:.2f} ms/image")
    print(f"Ensemble Throughput:     {time_metrics['fps']:.2f} FPS")
    print(f"ECE: {ece_value:.4f}")
    print("-"*50)
    
    del model_gen, model_spec
    tf.keras.backend.clear_session()
    
    return {
        'seed': config['seed'],
        'generalist_path': str(config['generalist']),
        'specialist_path': str(config['specialist']),
        'num_images': len(image_paths),
        'num_pixels': int(len(y_true)),
        'inference_ms_per_image': float(time_metrics['ms_per_image']),
        'inference_fps': float(time_metrics['fps']),
        'ece': float(ece_value)
    }

def main():
    print("="*80)
    print("MULTI-SEED ENSEMBLE MODEL EVALUATION (Generalist + Specialist)")
    print("="*80)
    print(f"Evaluating {len(ENSEMBLE_CONFIGS)} ensemble seeds")
    print(f"Output directory: {OUTPUT_DIR}")
    
    print("\nLoading data paths...")
    image_paths, mask_paths = get_data_paths(CLEAN_PATH, MASK_PATH, max_images=MAX_IMAGES)
    
    if not image_paths:
        print("ERROR: No images found!")
        return
    
    all_results = []
    for i, config in enumerate(ENSEMBLE_CONFIGS, 1):
        print(f"\n{'='*80}")
        print(f"ENSEMBLE SEED {i}/{len(ENSEMBLE_CONFIGS)}")
        print(f"{'='*80}")
        
        result = evaluate_ensemble_seed(config, image_paths, mask_paths)
        if result:
            all_results.append(result)
    
    if len(all_results) > 0:
        ece_values = [r['ece'] for r in all_results]
        inference_times = [r['inference_ms_per_image'] for r in all_results]
        fps_values = [r['inference_fps'] for r in all_results]
        
        summary = {
            'num_seeds_evaluated': len(all_results),
            'num_images_per_seed': all_results[0]['num_images'],
            'num_pixels_per_seed': all_results[0]['num_pixels'],
            'ece': {
                'mean': float(np.mean(ece_values)),
                'std': float(np.std(ece_values)),
                'min': float(np.min(ece_values)),
                'max': float(np.max(ece_values)),
                'all_values': ece_values
            },
            'inference_time_ms': {
                'mean': float(np.mean(inference_times)),
                'std': float(np.std(inference_times)),
                'min': float(np.min(inference_times)),
                'max': float(np.max(inference_times)),
                'all_values': inference_times
            },
            'fps': {
                'mean': float(np.mean(fps_values)),
                'std': float(np.std(fps_values)),
                'min': float(np.min(fps_values)),
                'max': float(np.max(fps_values)),
                'all_values': fps_values
            },
            'individual_results': all_results
        }
        
        print("\n" + "="*80)
        print("SUMMARY ACROSS ALL SEEDS")
        print("="*80)
        print(f"Seeds evaluated: {len(all_results)}")
        print(f"Images per seed: {all_results[0]['num_images']}")
        print(f"Pixels per seed: {all_results[0]['num_pixels']:,}")
        print(f"\nECE: {summary['ece']['mean']:.4f} ± {summary['ece']['std']:.4f}")
        print(f"  Range: [{summary['ece']['min']:.4f}, {summary['ece']['max']:.4f}]")
        print(f"\nInference Time: {summary['inference_time_ms']['mean']:.2f} ± {summary['inference_time_ms']['std']:.2f} ms/image")
        print(f"  Range: [{summary['inference_time_ms']['min']:.2f}, {summary['inference_time_ms']['max']:.2f}] ms")
        print(f"\nThroughput: {summary['fps']['mean']:.2f} ± {summary['fps']['std']:.2f} FPS")
        print(f"  Range: [{summary['fps']['min']:.2f}, {summary['fps']['max']:.2f}] FPS")
        print("="*80)
        
        output_file = OUTPUT_DIR / 'ensemble_all_seeds_ece_inference_metrics.json'
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\nResults saved to: {output_file}")
        print(f"ECE calibration plots saved to: {OUTPUT_DIR}")
    else:
        print("\nNo ensemble models successfully evaluated")

if __name__ == "__main__":
    main()

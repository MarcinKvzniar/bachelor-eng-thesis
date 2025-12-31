from pathlib import Path
import os
import warnings
import random
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
warnings.filterwarnings('ignore', category=UserWarning, message='.*Protobuf gencode version.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*A new version of Albumentations is available.*')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, UpSampling2D,
    concatenate, Add, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.efficientnet import EfficientNetB4

gpus = tf.config.experimental.list_physical_devices('GPU')

USE_GPU = False

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: Enabled memory growth for {len(gpus)} GPU(s)")       
        USE_GPU = True
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
        print("Falling back to CPU")
else:
    print("No GPU detected - using CPU for training")
    
    try:
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(4)
        print("CPU resources configured for training")
    except:
        print("Using default CPU configuration")

# --- Constants ---
if USE_GPU:
    SILVER_CLEAN_PATH = ""
    SILVER_MASK_PATH = ""
    
    GOLD_CLEAN_PATH = ""
    GOLD_MASK_PATH = ""
    
    MODEL_DIR = ''
    MODEL_PATH = ''
    VISUALIZATION_DIR = ''
else:
    SILVER_CLEAN_PATH = ""
    SILVER_MASK_PATH = ""
    
    GOLD_CLEAN_PATH = ""
    GOLD_MASK_PATH = ""
    
    MODEL_DIR = ''
    MODEL_PATH = ''
    VISUALIZATION_DIR = ''

IMG_HEIGHT, IMG_WIDTH = 512, 512
NUM_CLASSES = 4
BATCH_SIZE = 40 if USE_GPU else 1  
EPOCHS = 120 if USE_GPU else 2
CLASS_NAMES = ["Background", "Negative", "Positive", "Boundaries"]
CLASS_MAPPING = {
    (255, 255, 255): 0,  # White - Background
    (112, 112, 225): 1,  # Blue - Negative
    (250, 62, 62): 2,    # Red - Positive
    (0, 0, 0): 3,        # Black - Boundary
}
# Class weights to address class imbalance (Background, Negative, Positive, Boundaries)
CLASS_WEIGHTS = tf.constant([1.0, 8.1, 6.1, 4.5], dtype=tf.float32)

# Hybrid Training Oversampling Factor for GOLD dataset
GOLD_OVERSAMPLING_FACTOR = 10

# --- CASES ---
TRAIN_CASES = [
    "1", "3", "4", "5", "6", "7", "9", "10", "11", "12", "13", 
    "19095743-3c9c-4d2f-bb95-a09ccdbba298", "217361d9-3380-4172-be35-b590b5c5871f",
    "2f37bc1b-7fe8-4f80-a7b1-ee2d8668e3f2", "49209911-b4da-49b0-95f7-e87d6584c021", "5bc3e12f-adcf-448e-b5a6-deefcf265013",
    "963a1b43-d11d-47cf-b354-4ce44ffc1b4a", "adad0d5f-5554-4edb-8887-c5fff93bce9e", "aef6ca29-a218-4459-9d82-5d9bec074928",
    "ce1b9041-284c-410a-8690-96c99512435d", "e5e70662-62f6-48e6-a37b-7a8907410243"
]

VAL_CASES = ["2", "8", "b3cdae4a-ed6a-40e0-ae4c-2ccce04db861", "5f1931cd-ba90-43cc-9035-17990871ab81"]

Path(VISUALIZATION_DIR).mkdir(parents=True, exist_ok=True)

# --- Helper Functions (Mask Conversion, Augmentation) ---
def convert_mask_to_classes(mask_image):
    """Converts a color-coded mask image to a 2D array of class indices.
    Each pixel's RGB value is mapped to a class index based on exact color matches.
    """
    mask_classes = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    for color, class_index in CLASS_MAPPING.items():
        match = np.all(mask_image == color, axis=-1)
        mask_classes[match] = class_index
    return mask_classes

def augment_images(image, mask):
    """Applies a series of augmentations to the image and mask using Albumentations.
    Returns image in [0, 255] range (uint8 or float32) and mask.
    """
    augmenter = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=25, p=0.7),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    ])
    augmented = augmenter(image=image.astype(np.uint8), mask=mask)
    return augmented['image'].astype(np.float32), augmented['mask']

def get_data_from_specific_cases(clean_root, mask_root, case_list):
    """
    Searches for image-mask pairs only in the specified cases.
    """
    clean_root = Path(clean_root)
    mask_root = Path(mask_root)
    valid_pairs = []

    for case_id in case_list:
        case_dir = clean_root / case_id
        
        if case_dir.exists() and case_dir.is_dir():
            for img_file in sorted(case_dir.glob("*.png")):
                mask_name_with_suffix = img_file.stem + "_mask.png"
                mask_file = mask_root / case_id / mask_name_with_suffix
                
                if mask_file.exists():
                    valid_pairs.append((str(img_file), str(mask_file)))
    
    return valid_pairs

def prepare_hybrid_dataset():
    """
    Prepares the hybrid dataset by combining Silver and Gold datasets with manual case splits.
    """
    print("\n--- Preparing Hybrid Dataset (Manual Split) ---")
    
    silver_train = get_data_from_specific_cases(SILVER_CLEAN_PATH, SILVER_MASK_PATH, TRAIN_CASES)
    silver_val = get_data_from_specific_cases(SILVER_CLEAN_PATH, SILVER_MASK_PATH, VAL_CASES)
    
    gold_train = get_data_from_specific_cases(GOLD_CLEAN_PATH, GOLD_MASK_PATH, TRAIN_CASES)
    gold_val = get_data_from_specific_cases(GOLD_CLEAN_PATH, GOLD_MASK_PATH, VAL_CASES)
    
    print(f"Silver: {len(silver_train)} Train, {len(silver_val)} Val")
    print(f"Gold:   {len(gold_train)} Train, {len(gold_val)} Val")
    
    if not silver_train and not gold_train:
        raise ValueError("CRITICAL: No training data found! Check paths and case lists.")

    # 3. OVERSAMPLING FOR GOLD (Train Only)
    # Repeat the gold_train file list X times
    if len(gold_train) > 0:
        gold_train_oversampled = gold_train * GOLD_OVERSAMPLING_FACTOR
        print(f"Applied Oversampling x{GOLD_OVERSAMPLING_FACTOR} to Gold Train.")
        print(f"Effective Gold Train Size: {len(gold_train_oversampled)}")
    else:
        gold_train_oversampled = []
        print("Warning: No Gold training data found. Oversampling skipped.")

    # 4. Merging
    # Train: Silver + Oversampled Gold
    final_train_pairs = silver_train + gold_train_oversampled
    
    # SHUFFLING - crucial for training to avoid Gold being in one block
    random.seed(42)
    random.shuffle(final_train_pairs)
    
    # Val: Silver + Original Gold (No oversampling!)
    final_val_pairs = silver_val + gold_val
    random.shuffle(final_val_pairs) 
    
    print(f"FINAL TRAIN SIZE: {len(final_train_pairs)} (Mixed)")
    print(f"FINAL VAL SIZE:   {len(final_val_pairs)} (Unique)")
    
    if len(final_train_pairs) > 0:
        train_imgs, train_masks = zip(*final_train_pairs)
    else:
        train_imgs, train_masks = [], []
        
    if len(final_val_pairs) > 0:
        val_imgs, val_masks = zip(*final_val_pairs)
    else:
        val_imgs, val_masks = [], []
    
    return (list(train_imgs), list(train_masks)), (list(val_imgs), list(val_masks))

# --- Data Generator ---
class DataGenerator(tf.keras.utils.Sequence):
    """Generates batches of images and masks for training/validation."""

    def __init__(self, image_paths, mask_paths, batch_size, augment=False, shuffle=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size

        batch_indices = self.indices[start_idx:end_idx]
        batch_img_paths = [self.image_paths[i] for i in batch_indices]
        batch_mask_paths = [self.mask_paths[i] for i in batch_indices]

        batch_images, batch_masks = [], []

        for img_path, mask_path in zip(batch_img_paths, batch_mask_paths):
            try:
                image = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                image = img_to_array(image)

                mask_image = load_img(mask_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="rgb")
                mask_image = img_to_array(mask_image).astype(np.uint8)

                mask = convert_mask_to_classes(mask_image)

                if self.augment:
                    image, mask = augment_images(image, mask)
                
                # EfficientNet preprocessing (ImageNet normalization)
                image = image / 255.0 
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = (image - mean) / std

                batch_images.append(image)
                batch_masks.append(mask)

            except Exception as e:
                print(f"Error loading {img_path} or {mask_path}: {e}")
                continue

        masks_one_hot = to_categorical(np.array(batch_masks, dtype=np.uint8), num_classes=NUM_CLASSES)
        return np.array(batch_images, dtype=np.float32), masks_one_hot

    def on_epoch_end(self):
        """Reshuffle data after each epoch if shuffle is enabled."""
        if self.shuffle:
            np.random.shuffle(self.indices)

# --- Attention U-Net Architecture Components ---
def attention_gate(gating_signal, skip_connection, inter_channels):
    """Attention gate mechanism.

    Args:
        gating_signal: The upsampled decoder feature map (gating signal)
        skip_connection: The encoder feature map (to be attended)
        inter_channels: Number of intermediate channels for attention computation

    Returns:
        Attended feature map

    """
    # Transform gating signal
    theta_g = Conv2D(inter_channels, kernel_size=1, strides=1, padding="same")(gating_signal)
    theta_g = BatchNormalization()(theta_g)

    # Transform skip connection
    phi_x = Conv2D(inter_channels, kernel_size=1, strides=1, padding="same")(skip_connection)
    phi_x = BatchNormalization()(phi_x)

    # Add and apply activation
    add_xg = Add()([theta_g, phi_x])
    act_xg = Activation("relu")(add_xg)

    # Generate attention coefficients
    psi = Conv2D(1, kernel_size=1, strides=1, padding="same")(act_xg)
    psi = BatchNormalization()(psi)
    psi = Activation("sigmoid")(psi)

    # Apply attention coefficients to skip connection
    return Multiply()([skip_connection, psi])


# --- U-Net Building Blocks ---
def conv_block(x, filters, kernel_size=3):
    """Convolutional block: Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU."""
    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    return Activation("relu")(x)


def decoder_block(x, skip, filters, use_attention=True, dropout_rate=0.1):
    """Decoder block with optional attention gate: UpSample -> Attention -> Concatenate -> conv_block."""
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters, kernel_size=2, padding="same")(x)

    if use_attention:
        skip = attention_gate(gating_signal=x, skip_connection=skip, inter_channels=filters // 2)

    x = concatenate([x, skip], axis=-1)

    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)

    return conv_block(x, filters)


# --- Model Definition ---
def build_attention_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES, backbone="efficientnetb4"):
    """Builds a U-Net model with pretrained encoder and attention gates in the decoder.

    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        backbone: Name of the pretrained backbone ('efficientnetb4', 'efficientnetb3', etc.)

    Returns:
        Keras Model with attention U-Net architecture using pretrained encoder

    """
    keras.backend.set_image_data_format("channels_last")

    print("Creating EfficientNetB4 model with 3-channel input")
    input_layer = Input(shape=(512, 512, 3), name="input_layer")

    base_model = EfficientNetB4(include_top=False, weights=None, input_tensor=input_layer)

    weights_path = tf.keras.utils.get_file(
        "efficientnetb4_notop.h5",
        "https://storage.googleapis.com/keras-applications/efficientnetb4_notop.h5",
        cache_subdir="models",
    )

    print(f"Loading ImageNet weights from {weights_path}")
    base_model.load_weights(weights_path, skip_mismatch=True, by_name=True)
    print(f"Base model input shape: {base_model.input_shape}")

    skip_names = [
        "block2a_expand_activation",  # 128x128 (skip1)
        "block3a_expand_activation",  # 64x64 (skip2)
        "block4a_expand_activation",  # 32x32 (skip3)
        "block6a_expand_activation",  # 16x16 (skip4)
    ]

    skip_connections = [base_model.get_layer(name).output for name in skip_names]
    encoder_model = Model(inputs=base_model.input, outputs=skip_connections + [base_model.output])


    inputs = Input(shape=input_shape, name="input_layer")

    # Get all outputs (skip connections + bottleneck) by passing through encoder
    all_outputs = encoder_model(inputs, training=False)
    skip_connections = all_outputs[:-1]
    bottleneck = all_outputs[-1]

    # Reverse skip connections for decoder (highest to lowest resolution)
    skip4, skip3, skip2, skip1 = skip_connections[3], skip_connections[2], skip_connections[1], skip_connections[0]

    # Decoder (Expanding Path) with Attention Gates
    dec4 = decoder_block(bottleneck, skip4, 512, use_attention=True)  # 16 -> 32
    dec3 = decoder_block(dec4, skip3, 256, use_attention=True)  # 32 -> 64
    dec2 = decoder_block(dec3, skip2, 128, use_attention=True)  # 64 -> 128
    dec1 = decoder_block(dec2, skip1, 64, use_attention=True)  # 128 -> 256

    # Final upsampling to original resolution (256 -> 512)
    final_up = UpSampling2D(size=(2, 2))(dec1)
    final_conv = Conv2D(64, kernel_size=3, padding="same")(final_up)
    final_conv = BatchNormalization()(final_conv)
    final_conv = Activation("relu")(final_conv)

    # Output layer
    outputs = Conv2D(num_classes, kernel_size=1, padding="same", activation="softmax")(final_conv)

    return Model(inputs=inputs, outputs=outputs, name=f"attention_unet_{backbone}")

def decode_mask_to_colors(mask):
    """Helper function to convert class indices back to RGB colors."""
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    inverse_class_mapping = {v: k for k, v in CLASS_MAPPING.items()}
    for class_index, color in inverse_class_mapping.items():
        color_mask[mask == class_index] = color
    return color_mask

class EarlyStoppingWithCounter(EarlyStopping):
    """Custom EarlyStopping callback that displays epochs remaining until early stopping."""
    
    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        
        # Call parent's on_epoch_end to handle the actual early stopping logic
        super().on_epoch_end(epoch, logs)
        
        # Display counter
        if self.wait > 0:
            epochs_without_improvement = self.wait
            epochs_remaining = self.patience - self.wait
            print(f"Early Stopping: {epochs_without_improvement} epoch(s) without improvement. "
                  f"{epochs_remaining} epoch(s) remaining before stopping.")
        else:
            print(f"Early Stopping: Metric improved! Resetting counter. Patience: {self.patience} epochs.")


class VisualizePredictionsCallback(Callback):
    """Keras Callback to visualize predictions on validation data at the end of each epoch."""

    def __init__(self, validation_generator, output_dir, num_samples=3):
        super().__init__()
        self.validation_generator = validation_generator
        self.output_dir = output_dir
        self.num_samples = min(num_samples, 1) if not USE_GPU else num_samples
        self.active = len(validation_generator) > 0

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        if not self.active:
            print("VisualizePredictionsCallback disabled (empty validation generator)")
        elif not USE_GPU:
            print("VisualizePredictionsCallback set to minimal mode for CPU-only")

    def on_epoch_end(self, epoch, logs=None):
        if not self.active or self.num_samples == 0:
            return
            
        try:
            images, true_masks_one_hot = self.validation_generator[0]
            actual_samples = min(len(images), self.num_samples)
            
            if actual_samples == 0:
                return
                
            images = images[:actual_samples]
            true_masks_one_hot = true_masks_one_hot[:actual_samples]

            true_masks_indices = np.argmax(true_masks_one_hot, axis=-1)

            predictions = self.model.predict(images, verbose=0)
            predicted_masks_indices = np.argmax(predictions, axis=-1)

            fig = plt.figure(figsize=(15, actual_samples * 5))
            plt.suptitle(f'Predictions after Epoch {epoch + 1}', fontsize=16)
            
            for i in range(actual_samples):
                plt.subplot(actual_samples, 3, i * 3 + 1)
                # Denormalize ImageNet normalization for display
                img_display = images[i].copy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_display = img_display * std + mean
                img_display = np.clip(img_display, 0, 1)
                plt.imshow(img_display)
                plt.title("Original Image")
                plt.axis('off')

                plt.subplot(actual_samples, 3, i * 3 + 2)
                plt.imshow(decode_mask_to_colors(true_masks_indices[i]))
                plt.title("Ground Truth")
                plt.axis('off')

                plt.subplot(actual_samples, 3, i * 3 + 3)
                plt.imshow(decode_mask_to_colors(predicted_masks_indices[i]))
                plt.title("Prediction")
                plt.axis('off')

            plt.tight_layout(rect=(0, 0.03, 1, 0.95))

            file_path = Path(self.output_dir) / f'predictions_epoch_{epoch + 1:03d}.png'
            plt.savefig(file_path)
            plt.close(fig)
            
            del images, true_masks_one_hot, predictions, predicted_masks_indices, true_masks_indices
            K.clear_session()
            
            print(f"\nSaved prediction visualization to {file_path}")
        except Exception as e:
            print(f"Error in visualization callback: {e}")

# --- Loss and Metrics ---
def iou_score(y_true, y_pred, smooth=1e-5):
    """Calculates IoU (Jaccard) Score, ignoring background class."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Ignore background class (index 0)
    y_true_fg = y_true[..., 1:]
    y_pred_fg = y_pred[..., 1:]
    
    intersection = K.sum(y_true_fg * y_pred_fg, axis=(0, 1, 2))
    union = K.sum(y_true_fg, axis=(0, 1, 2)) + K.sum(y_pred_fg, axis=(0, 1, 2)) - intersection
    
    iou = K.mean((intersection + smooth) / (union + smooth))
    return iou

def f1_score(y_true, y_pred, smooth=1e-5):
    """Calculates F1-Score (Dice), ignoring background class."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Ignore background class (index 0)
    y_true_fg = y_true[..., 1:]
    y_pred_fg = y_pred[..., 1:]
    
    intersection = K.sum(y_true_fg * y_pred_fg, axis=(0, 1, 2))
    union = K.sum(y_true_fg, axis=(0, 1, 2)) + K.sum(y_pred_fg, axis=(0, 1, 2))
    
    dice = K.mean((2. * intersection + smooth) / (union + smooth))
    return dice

def weighted_dice_loss(y_true, y_pred, smooth=1e-5):
    """
    Weighted Dice Loss function.
    Removed normalization by sum of weights to preserve scale.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    loss = 0.0
    for i in range(NUM_CLASSES):
        y_true_i = y_true[..., i]
        y_pred_i = y_pred[..., i]
        
        intersection = K.sum(y_true_i * y_pred_i)
        union = K.sum(y_true_i) + K.sum(y_pred_i)
        
        dice_i = (2. * intersection + smooth) / (union + smooth)
        loss += CLASS_WEIGHTS[i] * (1.0 - dice_i)
        
    return loss

def weighted_focal_loss(y_true, y_pred, gamma=2.0):
    """
    Focal loss with class weights to address class imbalance.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    weights = tf.expand_dims(tf.expand_dims(CLASS_WEIGHTS, 0), 0)

    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    cross_entropy = -y_true * K.log(y_pred)
    
    loss = weights * K.pow(1.0 - y_pred, gamma) * cross_entropy
    
    return K.mean(K.sum(loss, axis=-1))

def combined_loss(y_true, y_pred):
    """
    Combines Weighted Dice Loss and Weighted Focal Loss.
    """
    loss = weighted_dice_loss(y_true, y_pred) + weighted_focal_loss(y_true, y_pred)
    return loss

metrics = [  
    f1_score,          
    iou_score,        
]

# --- Visualization Functions ---
def plot_history(history):
    """
    Plots training and validation loss and metrics over epochs.
    Saves the plots to files in the VISUALIZATION_DIR.
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(history.history['f1_score'], label='Training F1-Score')
        axes[0, 1].plot(history.history['val_f1_score'], label='Validation F1-Score')
        axes[0, 1].set_title('F1-Score (Dice)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(history.history['iou_score'], label='Training IoU')
        axes[1, 0].plot(history.history['val_iou_score'], label='Validation IoU')
        axes[1, 0].set_title('IoU Score (Jaccard)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], label='Learning Rate', color='orange')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            axes[1, 1].axis('off')

        plt.tight_layout()
        save_path = Path(VISUALIZATION_DIR) / "training_history.png"
        Path(VISUALIZATION_DIR).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved training history plot to {save_path}")

    except KeyError as e:
        print(f"Warning: Could not plot history. Missing key: {e}")
        print("Available keys:", list(history.history.keys()))
    except Exception as e:
        print(f"Error plotting history: {e}")

def visualize_predictions(model, generator, num_samples=5):
    """Visualizes Original | Ground Truth | Predicted Mask.
    Saves the visualization to a file in the VISUALIZATION_DIR.
    """
    if len(generator) == 0:
        print("Warning: Generator is empty. Cannot visualize predictions.")
        return
        
    images, true_masks = generator[0]
    actual_samples = min(len(images), num_samples)
    
    if actual_samples == 0:
        print("Warning: No images in batch. Cannot visualize predictions.")
        return
        
    images = images[:actual_samples]
    true_masks = true_masks[:actual_samples]

    print(f"Predicting on {actual_samples} validation images...")
    predictions = model.predict(images)
    predicted_masks = np.argmax(predictions, axis=-1)
    true_masks_indices = np.argmax(true_masks, axis=-1)

    fig = plt.figure(figsize=(15, actual_samples * 5))
    for i in range(actual_samples):
        plt.subplot(actual_samples, 3, i * 3 + 1)
        # Denormalize ImageNet normalization for display
        img_display = images[i].copy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = img_display * std + mean
        img_display = np.clip(img_display, 0, 1)
        plt.imshow(img_display)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(actual_samples, 3, i * 3 + 2)
        plt.imshow(decode_mask_to_colors(true_masks_indices[i]))
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(actual_samples, 3, i * 3 + 3)
        plt.imshow(decode_mask_to_colors(predicted_masks[i]))
        plt.title("Predicted Mask")
        plt.axis('off')

    plt.tight_layout()
    
    save_path = Path(VISUALIZATION_DIR) / "val_predictions.png"
    Path(VISUALIZATION_DIR).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved validation predictions to {save_path}")

def plot_confusion_matrix(model, generator):
    """Calculates and plots the normalized pixel-wise confusion matrix.
    Saves the matrix to a file in the VISUALIZATION_DIR.
    """
    if len(generator) == 0:
        print("Warning: Generator is empty. Cannot plot confusion matrix.")
        return
        
    print("Calculating confusion matrix...")
    y_true, y_pred = [], []
    
    try:
        for i in range(len(generator)):
            images, masks = generator[i]
            predictions = model.predict(images, verbose=0)
            
            y_true.extend(np.argmax(masks, axis=-1).flatten())
            y_pred.extend(np.argmax(predictions, axis=-1).flatten())

        cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
               title='Normalized Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                       ha="center", va="center",
                       color="white" if cm_normalized[i, j] > 0.5 else "black")

        fig.tight_layout()
        save_path = Path(VISUALIZATION_DIR) / "confusion_matrix.png"
        Path(VISUALIZATION_DIR).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved confusion matrix to {save_path}")
    except Exception as e:
        print(f"Error calculating confusion matrix: {e}")

# --- Main Training Logic ---
def main():
    """Main function to execute the training, evaluation, and visualization pipeline."""

    SEED = 42 
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    
    print("\n===== STARTING HYBRID TRAINING (Silver + Gold x40) [MANUAL SPLIT] =====")
    
    print("\n===== SYSTEM CONFIGURATION =====")
    print(f"Using Reproducible Seed: {SEED}")
    print(f"Using {'GPU' if USE_GPU else 'CPU'} for training")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Target epochs: {EPOCHS}")
    print("===============================\n")
    
    (train_imgs, train_msks), (val_imgs, val_msks) = prepare_hybrid_dataset()
    total_samples = len(train_imgs) + len(val_imgs)

    print(f"Total samples: {total_samples}")
    print(f"Distribution: {len(train_imgs)/total_samples:.1%} train, {len(val_imgs)/total_samples:.1%} val")

    if not USE_GPU and total_samples > 1000:
        max_train = min(100, len(train_imgs))
        max_val = min(20, len(val_imgs))

        print(f"\nCPU-only mode with large dataset detected")
        print(f"Using subset of data to avoid memory issues:")     
        print(f" - Training: {max_train}/{len(train_imgs)} images")
        print(f" - Validation: {max_val}/{len(val_imgs)} images")
        
        train_imgs, train_msks = train_imgs[:max_train], train_msks[:max_train]
        val_imgs, val_msks = val_imgs[:max_val], val_msks[:max_val]
    
    train_generator = DataGenerator(train_imgs, train_msks, BATCH_SIZE, augment=True, shuffle=True)
    val_generator = DataGenerator(val_imgs, val_msks, BATCH_SIZE, augment=False, shuffle=False)

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    
    model = build_attention_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    lr = 3e-4 if USE_GPU else 1e-6  
    weight_decay = 1e-5 if USE_GPU else 1e-7

    model.compile(
        optimizer=AdamW(learning_rate=lr, weight_decay=weight_decay, clipnorm=1.0),
        loss=combined_loss, 
        metrics=metrics
    )


    callbacks = [
        ModelCheckpoint(
            filepath=MODEL_PATH,
            save_best_only=True,
            monitor='val_f1_score',
            mode='max',
            verbose=1,
            save_weights_only=True
        ),
        ReduceLROnPlateau(monitor='val_f1_score', mode='max', factor=0.5, patience=7, min_lr=1e-7, verbose=1),
        EarlyStoppingWithCounter(monitor='val_f1_score', mode='max', patience=25, restore_best_weights=True, verbose=1),
        VisualizePredictionsCallback(val_generator, VISUALIZATION_DIR)
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print("\n--- Training Complete ---")

    print("Loading best model for evaluation...")
    model.load_weights(MODEL_PATH)

    print("\n--- Evaluation on Validation Set ---")
    val_results = model.evaluate(val_generator)
    print(f"Validation Loss: {val_results[0]:.4f}")
    for name, value in zip(model.metrics_names[1:], val_results[1:]):
        print(f"Validation {name}: {value:.4f}")

    print("\n--- Generating Visualizations ---")
    try:
        plot_history(history)
    except Exception as e:
        print(f"Error plotting history: {e}")
        
    print("\nVisualizing Predictions on Validation Data...")
    try:
        visualize_predictions(model, val_generator, num_samples=5)
    except Exception as e:
        print(f"Error visualizing predictions: {e}")
        
    print("\nGenerating Confusion Matrix for Validation Set...")
    try:
        plot_confusion_matrix(model, val_generator)
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")


if __name__ == '__main__':
    main()

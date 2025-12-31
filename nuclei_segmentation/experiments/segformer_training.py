import os
import warnings
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Check for GPU availability
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

if USE_GPU:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("No GPU detected - using CPU for training")

# --- Paths Configuration ---
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

# --- Training Configuration ---
IMG_HEIGHT, IMG_WIDTH = 512, 512
NUM_CLASSES = 4
BATCH_SIZE = 16 if USE_GPU else 1
EPOCHS = 120 if USE_GPU else 2
LEARNING_RATE = 3e-4 if USE_GPU else 1e-6
WEIGHT_DECAY = 1e-4 if USE_GPU else 1e-6

CLASS_NAMES = ["Background", "Negative", "Positive", "Boundaries"]
CLASS_MAPPING = {
    (255, 255, 255): 0,  # White - Background
    (112, 112, 225): 1,  # Blue - Negative
    (250, 62, 62): 2,    # Red - Positive
    (0, 0, 0): 3,        # Black - Boundary
}

CLASS_WEIGHTS = torch.tensor([1.0, 8.1, 6.1, 4.5], dtype=torch.float32).to(DEVICE)

GOLD_OVERSAMPLING_FACTOR = 6

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
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# --- Helper Functions ---
def convert_mask_to_classes(mask_image):
    """Converts a color-coded mask image to a 2D array of class indices."""
    mask_array = np.array(mask_image)
    mask_classes = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)
    
    for color, class_index in CLASS_MAPPING.items():
        match = np.all(mask_array == color, axis=-1)
        mask_classes[match] = class_index
    
    return mask_classes

def get_data_from_specific_cases(clean_root, mask_root, case_list):
    """Searches for image-mask pairs only in the specified cases."""
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
    """Prepares the hybrid dataset by combining Silver and Gold datasets with manual case splits."""
    print("\n--- Preparing Hybrid Dataset (Manual Split) ---")
    
    silver_train = get_data_from_specific_cases(SILVER_CLEAN_PATH, SILVER_MASK_PATH, TRAIN_CASES)
    silver_val = get_data_from_specific_cases(SILVER_CLEAN_PATH, SILVER_MASK_PATH, VAL_CASES)
    
    gold_train = get_data_from_specific_cases(GOLD_CLEAN_PATH, GOLD_MASK_PATH, TRAIN_CASES)
    gold_val = get_data_from_specific_cases(GOLD_CLEAN_PATH, GOLD_MASK_PATH, VAL_CASES)
    
    print(f"Silver: {len(silver_train)} Train, {len(silver_val)} Val")
    print(f"Gold:   {len(gold_train)} Train, {len(gold_val)} Val")
    
    if not silver_train and not gold_train:
        raise ValueError("CRITICAL: No training data found! Check paths and case lists.")

    if len(gold_train) > 0:
        gold_train_oversampled = gold_train * GOLD_OVERSAMPLING_FACTOR
        print(f"Applied Oversampling x{GOLD_OVERSAMPLING_FACTOR} to Gold Train.")
        print(f"Effective Gold Train Size: {len(gold_train_oversampled)}")
    else:
        gold_train_oversampled = []
        print("Warning: No Gold training data found. Oversampling skipped.")

    final_train_pairs = silver_train + gold_train_oversampled
    random.seed(42)
    random.shuffle(final_train_pairs)
    
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

# --- Dataset Class ---
class NucleiDataset(Dataset):
    """PyTorch Dataset for nuclei segmentation."""
    
    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=25, p=0.7),
                A.ElasticTransform(alpha=1, sigma=50, p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('RGB')
        
        image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        mask = mask.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST)
        
        image = np.array(image)
        mask = convert_mask_to_classes(mask)
        
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
        
        return image, mask

# --- Model Definition ---
def build_segformer(num_classes=NUM_CLASSES):
    """Builds a SegFormer model for semantic segmentation."""
    model_name = "nvidia/mit-b3" if USE_GPU else "nvidia/mit-b0"
    
    print(f"Building SegFormer model: {model_name}")
    
    config = SegformerConfig.from_pretrained(model_name)
    config.num_labels = num_classes
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    )
    
    return model

# --- Loss Functions ---
class CombinedLoss(nn.Module):
    """Combines Focal Loss and Dice Loss with class weights."""
    
    def __init__(self, class_weights, gamma=2.0, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.alpha = alpha
    
    def focal_loss(self, inputs, targets):
        """Focal Loss with class weights."""
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
    
    def dice_loss(self, inputs, targets):
        """Dice Loss with class weights."""
        smooth = 1e-5
        inputs = torch.softmax(inputs, dim=1)
        
        targets_one_hot = nn.functional.one_hot(targets, num_classes=NUM_CLASSES)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        loss = 0.0
        for i in range(NUM_CLASSES):
            inputs_i = inputs[:, i, :, :]
            targets_i = targets_one_hot[:, i, :, :]
            
            intersection = (inputs_i * targets_i).sum()
            union = inputs_i.sum() + targets_i.sum()
            
            dice = (2. * intersection + smooth) / (union + smooth)
            loss += self.class_weights[i] * (1.0 - dice)
        
        return loss / NUM_CLASSES
    
    def forward(self, inputs, targets):
        return self.alpha * self.focal_loss(inputs, targets) + (1 - self.alpha) * self.dice_loss(inputs, targets)

# --- Metrics ---
def calculate_iou(pred, target, num_classes=NUM_CLASSES):
    """Calculate IoU score, ignoring background class."""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Ignore background (class 0)
    for cls in range(1, num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    
    return np.nanmean(ious)

def calculate_f1(pred, target, num_classes=NUM_CLASSES):
    """Calculate F1 score (Dice), ignoring background class."""
    f1_scores = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Ignore background (class 0)
    for cls in range(1, num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        
        pred_sum = pred_inds.sum().float()
        target_sum = target_inds.sum().float()
        
        if pred_sum + target_sum == 0:
            f1_scores.append(float('nan'))
        else:
            f1 = (2. * intersection / (pred_sum + target_sum)).item()
            f1_scores.append(f1)
    
    return np.nanmean(f1_scores)

# --- Training Functions ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(pixel_values=images)
        logits = outputs.logits
        
        logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        
        loss = criterion(logits, masks)
        
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            iou = calculate_iou(preds, masks)
            f1 = calculate_f1(preds, masks)
        
        running_loss += loss.item()
        running_iou += iou
        running_f1 += f1
        
        pbar.set_postfix({'loss': loss.item(), 'iou': iou, 'f1': f1})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    epoch_f1 = running_f1 / len(dataloader)
    
    return epoch_loss, epoch_iou, epoch_f1

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(pixel_values=images)
            logits = outputs.logits
            
            logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            loss = criterion(logits, masks)
            
            preds = torch.argmax(logits, dim=1)
            iou = calculate_iou(preds, masks)
            f1 = calculate_f1(preds, masks)
            
            running_loss += loss.item()
            running_iou += iou
            running_f1 += f1
            
            pbar.set_postfix({'loss': loss.item(), 'iou': iou, 'f1': f1})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    epoch_f1 = running_f1 / len(dataloader)
    
    return epoch_loss, epoch_iou, epoch_f1

# --- Visualization Functions ---
def decode_mask_to_colors(mask):
    """Helper function to convert class indices back to RGB colors."""
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    inverse_class_mapping = {v: k for k, v in CLASS_MAPPING.items()}
    
    for class_index, color in inverse_class_mapping.items():
        color_mask[mask == class_index] = color
    
    return color_mask

def visualize_predictions(model, dataloader, device, save_path, num_samples=5):
    """Visualize predictions on validation data."""
    model.eval()
    
    images_list = []
    masks_list = []
    preds_list = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(pixel_values=images)
            logits = outputs.logits
            logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            preds = torch.argmax(logits, dim=1)
            
            # Denormalize images for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
            images_denorm = images * std + mean
            
            images_list.append(images_denorm.cpu())
            masks_list.append(masks.cpu())
            preds_list.append(preds.cpu())
            
            if len(images_list) * images.size(0) >= num_samples:
                break
    
    images_all = torch.cat(images_list, dim=0)[:num_samples]
    masks_all = torch.cat(masks_list, dim=0)[:num_samples]
    preds_all = torch.cat(preds_list, dim=0)[:num_samples]
    
    fig = plt.figure(figsize=(15, num_samples * 5))
    for i in range(num_samples):
        # Original image
        plt.subplot(num_samples, 3, i * 3 + 1)
        img = images_all[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis('off')
        
        # Ground truth
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(decode_mask_to_colors(masks_all[i].numpy()))
        plt.title("Ground Truth")
        plt.axis('off')
        
        # Prediction
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(decode_mask_to_colors(preds_all[i].numpy()))
        plt.title("Prediction")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved predictions to {save_path}")

def plot_training_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Training Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # F1 Score
    axes[0, 1].plot(history['train_f1'], label='Training F1-Score')
    axes[0, 1].plot(history['val_f1'], label='Validation F1-Score')
    axes[0, 1].set_title('F1-Score (Dice)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # IoU Score
    axes[1, 0].plot(history['train_iou'], label='Training IoU')
    axes[1, 0].plot(history['val_iou'], label='Validation IoU')
    axes[1, 0].set_title('IoU Score (Jaccard)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], label='Learning Rate', color='orange')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved training history to {save_path}")

def plot_confusion_matrix(model, dataloader, device, save_path):
    """Calculate and plot confusion matrix."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Computing confusion matrix'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(pixel_values=images)
            logits = outputs.logits
            logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
    
    cm = confusion_matrix(all_targets, all_preds, labels=range(NUM_CLASSES))
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
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved confusion matrix to {save_path}")

# --- Main Training Loop ---
def main():
    """Main training function."""
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if USE_GPU:
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print("\n===== SYSTEM CONFIGURATION =====")
    print(f"Using Reproducible Seed: {SEED}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Target epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("===============================\n")
    
    # Prepare dataset
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
    
    # Create datasets and dataloaders
    train_dataset = NucleiDataset(train_imgs, train_msks, augment=True)
    val_dataset = NucleiDataset(val_imgs, val_msks, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4 if USE_GPU else 0,
        pin_memory=USE_GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4 if USE_GPU else 0,
        pin_memory=USE_GPU
    )
    
    # Build model
    model = build_segformer(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    
    # Define loss and optimizer
    criterion = CombinedLoss(CLASS_WEIGHTS, gamma=2.0, alpha=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_f1': [], 'val_f1': [],
        'lr': []
    }
    
    # Early stopping
    best_val_f1 = 0.0
    patience = 25
    patience_counter = 0
    
    # Training loop
    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        # Train
        train_loss, train_iou, train_f1 = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_iou, val_f1 = validate_epoch(model, val_loader, criterion, DEVICE)
        
        # Update learning rate
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, F1: {val_f1:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_iou': val_iou,
                'val_loss': val_loss,
            }, MODEL_PATH)
            print(f" Saved best model with Val F1: {val_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Early Stopping: {patience_counter} epoch(s) without improvement. "
                  f"{patience - patience_counter} epoch(s) remaining before stopping.")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
        
        # Visualize predictions every 5 epochs
        if (epoch + 1) % 5 == 0:
            vis_path = Path(VISUALIZATION_DIR) / f'predictions_epoch_{epoch + 1:03d}.png'
            visualize_predictions(model, val_loader, DEVICE, vis_path, num_samples=3)
    
    print("\n--- Training Complete ---")
    
    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nBest Model Performance:")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val IoU: {checkpoint['val_iou']:.4f}")
    print(f"  Val F1: {checkpoint['val_f1']:.4f}")
    
    # Final evaluation
    print("\n--- Final Evaluation ---")
    val_loss, val_iou, val_f1 = validate_epoch(model, val_loader, criterion, DEVICE)
    print(f"Final Validation - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, F1: {val_f1:.4f}")
    
    # Generate visualizations
    print("\n--- Generating Visualizations ---")
    plot_training_history(history, Path(VISUALIZATION_DIR) / 'training_history.png')
    visualize_predictions(model, val_loader, DEVICE, Path(VISUALIZATION_DIR) / 'val_predictions.png', num_samples=5)
    plot_confusion_matrix(model, val_loader, DEVICE, Path(VISUALIZATION_DIR) / 'confusion_matrix.png')
    
    print("\n Training pipeline complete!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Visualizations saved to: {VISUALIZATION_DIR}")

if __name__ == '__main__':
    main()

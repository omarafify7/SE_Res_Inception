"""
SE-Res-Inception Training Script
================================

Robust training pipeline for the SE-Res-Inception architecture on CIFAR-100.

Features:
- Automatic CIFAR-100 download
- Mixed Precision Training (AMP) for RTX 5070 Ti tensor core utilization
- Optimized data loading with pin_memory and multiple workers
- Cosine annealing learning rate schedule
- Best model checkpointing based on validation accuracy
- tqdm progress bars with live loss/accuracy metrics
- **Mixup/CutMix data augmentation** with soft-target cross-entropy loss

Hardware Target: NVIDIA RTX 5070 Ti (Blackwell, SM_120)
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from model import SEResInception


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Training configuration - all hyperparameters in one place"""
    
    # Dataset
    DATA_DIR = "./data"
    NUM_CLASSES = 100
    
    # Training
    EPOCHS = 100
    BATCH_SIZE = 128  # RTX 5070 Ti has plenty of VRAM
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-4  # Increased from 1e-4 to combat overfitting
    DROPOUT = 0.4
    
    # Mixup/CutMix Configuration
    MIXUP_ALPHA = 1.0       # Beta distribution alpha for Mixup (1.0 is uniform)
    CUTMIX_ALPHA = 1.0      # Beta distribution alpha for CutMix
    MIXUP_CUTMIX_PROB = 0.5 # Probability of applying Mixup or CutMix (50%)
    CUTMIX_PROB = 0.5       # When augmenting, probability of CutMix vs Mixup
    
    # Data Loading (optimized for RTX 5070 Ti)
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Checkpointing
    CHECKPOINT_DIR = "./checkpoints"
    BEST_MODEL_PATH = "./checkpoints/best_model.pth"
    METRICS_PATH = "./checkpoints/metrics.npz"  # Numpy arrays for plotting
    LATEST_CHECKPOINT_PATH = "./checkpoints/latest_checkpoint.pth"  # For resume
    
    # Mixed Precision
    USE_AMP = True  # Enable for tensor core utilization


# ============================================================================
# DATA LOADING
# ============================================================================
def get_dataloaders(config: Config) -> tuple:
    """
    Create train and validation dataloaders for CIFAR-100.
    
    Data augmentation:
    - Train: RandomCrop with padding, RandomHorizontalFlip, Normalize
    - Val: Just Normalize
    
    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    
    # CIFAR-100 normalization statistics
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Download and load CIFAR-100
    print("Loading CIFAR-100 dataset...")
    train_dataset = torchvision.datasets.CIFAR100(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_dataset = torchvision.datasets.CIFAR100(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=val_transform
    )
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples:   {len(val_dataset):,}")
    
    return train_loader, val_loader, train_dataset.classes


# ============================================================================
# MIXUP / CUTMIX AUGMENTATION
# ============================================================================
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> tuple:
    """
    Apply Mixup augmentation to a batch.
    
    Mixup creates virtual training examples by linearly interpolating
    between pairs of images and their labels.
    
    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
    
    Args:
        x: Input images [B, C, H, W]
        y: Target labels [B]
        alpha: Beta distribution parameter (1.0 = uniform distribution)
    
    Returns:
        tuple: (mixed_x, y_a, y_b, lam)
            - mixed_x: Linearly interpolated images
            - y_a: Original labels
            - y_b: Shuffled labels
            - lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def rand_bbox(size: tuple, lam: float) -> tuple:
    """
    Generate random bounding box coordinates for CutMix.
    
    The box size is determined by the mixing ratio lam,
    targeting sqrt(1-lam) of the image area for the cut region.
    
    Args:
        size: Image tensor size [B, C, H, W]
        lam: Mixing coefficient (determines box area)
    
    Returns:
        tuple: (bbx1, bby1, bbx2, bby2) box coordinates
    """
    W = size[2]
    H = size[3]
    
    # Box size proportional to sqrt(1-lam)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Random center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Clip to image bounds
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> tuple:
    """
    Apply CutMix augmentation to a batch.
    
    CutMix cuts a rectangular region from one image and pastes it onto
    another, mixing labels proportional to the area of the patch.
    
    Reference: Yun et al., "CutMix: Regularization Strategy to Train 
               Strong Classifiers with Localizable Features", ICCV 2019
    
    Args:
        x: Input images [B, C, H, W]
        y: Target labels [B]
        alpha: Beta distribution parameter
    
    Returns:
        tuple: (mixed_x, y_a, y_b, lam)
            - mixed_x: CutMix-augmented images
            - y_a: Original labels
            - y_b: Shuffled labels  
            - lam: Adjusted mixing coefficient based on actual cut area
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    y_a, y_b = y, y[index]
    
    # Generate random bounding box
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    
    # Apply CutMix - replace rectangular region
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda based on actual area ratio (not the sampled lambda)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    
    return mixed_x, y_a, y_b, lam


# ============================================================================
# SOFT-TARGET CROSS ENTROPY LOSS
# ============================================================================
class SoftTargetCrossEntropy(nn.Module):
    """
    Cross-entropy loss for mixed (soft) targets from Mixup/CutMix.
    
    Standard CrossEntropyLoss expects hard labels (class indices), but
    Mixup/CutMix produce soft targets (linear combinations of two classes).
    
    This loss computes: L = -lam * log(p[y_a]) - (1-lam) * log(p[y_b])
    
    Includes label smoothing support for additional regularization.
    """
    
    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing
    
    def forward(
        self, 
        outputs: torch.Tensor, 
        targets_a: torch.Tensor, 
        targets_b: torch.Tensor, 
        lam: float
    ) -> torch.Tensor:
        """
        Compute soft-target cross entropy loss.
        
        Args:
            outputs: Model predictions [B, num_classes]
            targets_a: First target labels [B]
            targets_b: Second target labels [B]
            lam: Mixing coefficient
        
        Returns:
            Scalar loss tensor
        """
        # Apply log_softmax for numerical stability
        log_probs = F.log_softmax(outputs, dim=1)
        
        # Create one-hot encoded targets with label smoothing
        num_classes = outputs.size(1)
        
        # Smooth targets: (1-smoothing)*one_hot + smoothing/num_classes
        smooth_a = self._smooth_one_hot(targets_a, num_classes)
        smooth_b = self._smooth_one_hot(targets_b, num_classes)
        
        # Mix the softened targets
        soft_targets = lam * smooth_a + (1 - lam) * smooth_b
        
        # Compute cross-entropy: -sum(p * log(q))
        loss = torch.sum(-soft_targets * log_probs, dim=1)
        
        return loss.mean()
    
    def _smooth_one_hot(
        self, 
        targets: torch.Tensor, 
        num_classes: int
    ) -> torch.Tensor:
        """
        Create label-smoothed one-hot vectors.
        
        Args:
            targets: Class indices [B]
            num_classes: Total number of classes
        
        Returns:
            Smoothed one-hot vectors [B, num_classes]
        """
        off_value = self.label_smoothing / num_classes
        on_value = 1.0 - self.label_smoothing + off_value
        
        one_hot = torch.full(
            (targets.size(0), num_classes),
            off_value,
            device=targets.device,
            dtype=torch.float32
        )
        one_hot.scatter_(1, targets.unsqueeze(1), on_value)
        
        return one_hot


# ============================================================================
# TRAINING UTILITIES
# ============================================================================
class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate top-1 accuracy"""
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        return correct / target.size(0) * 100


def calculate_mixed_accuracy(
    output: torch.Tensor, 
    targets_a: torch.Tensor, 
    targets_b: torch.Tensor, 
    lam: float
) -> float:
    """
    Calculate accuracy for mixed samples.
    
    For Mixup/CutMix, we count a prediction as partially correct
    based on whether it matches either of the mixed classes,
    weighted by the mixing coefficient.
    
    Args:
        output: Model predictions [B, num_classes]
        targets_a: First target labels [B]
        targets_b: Second target labels [B]
        lam: Mixing coefficient
    
    Returns:
        Weighted accuracy percentage
    """
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct_a = pred.eq(targets_a).float()
        correct_b = pred.eq(targets_b).float()
        accuracy = lam * correct_a + (1 - lam) * correct_b
        return accuracy.mean().item() * 100


class MetricsTracker:
    """
    Tracks and persists training metrics as numpy arrays for plotting.
    
    Saves:
    - train_loss: Training loss per epoch
    - train_acc: Training accuracy per epoch
    - val_loss: Validation loss per epoch
    - val_acc: Validation accuracy per epoch
    - epochs: Epoch numbers (for x-axis plotting)
    
    Supports checkpoint resume by loading existing metrics from disk.
    """
    
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.epochs = []
        
        # Try to load existing metrics if resuming
        self.load()
    
    def load(self) -> bool:
        """Load metrics from disk if they exist. Returns True if loaded."""
        if os.path.exists(self.save_path):
            try:
                data = np.load(self.save_path)
                self.train_loss = data['train_loss'].tolist()
                self.train_acc = data['train_acc'].tolist()
                self.val_loss = data['val_loss'].tolist()
                self.val_acc = data['val_acc'].tolist()
                self.epochs = data['epochs'].tolist()
                print(f"Loaded metrics from {self.save_path} ({len(self.epochs)} epochs)")
                return True
            except Exception as e:
                print(f"Warning: Could not load metrics: {e}")
                return False
        return False
    
    def update(self, epoch: int, train_loss: float, train_acc: float, 
               val_loss: float, val_acc: float):
        """Add metrics for a completed epoch."""
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.save()
    
    def save(self):
        """Save metrics to disk as numpy arrays."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        np.savez(
            self.save_path,
            train_loss=np.array(self.train_loss),
            train_acc=np.array(self.train_acc),
            val_loss=np.array(self.val_loss),
            val_acc=np.array(self.val_acc),
            epochs=np.array(self.epochs)
        )
    
    def get_last_epoch(self) -> int:
        """Return the last completed epoch, or 0 if none."""
        return self.epochs[-1] if self.epochs else 0
    
    def get_arrays(self) -> dict:
        """Return all metrics as numpy arrays for plotting."""
        return {
            'epochs': np.array(self.epochs),
            'train_loss': np.array(self.train_loss),
            'train_acc': np.array(self.train_acc),
            'val_loss': np.array(self.val_loss),
            'val_acc': np.array(self.val_acc)
        }


# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion_standard: nn.Module,
    criterion_mixup: SoftTargetCrossEntropy,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: Config
) -> tuple:
    """
    Train for one epoch with Mixed Precision (AMP) and Mixup/CutMix.
    
    Augmentation Strategy:
    - 50% of batches: Apply Mixup or CutMix (50/50 split between them)
    - 50% of batches: Standard training with basic augmentations
    
    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.train()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d} [Train]", leave=False)
    
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Probability switch: Apply Mixup/CutMix or standard augmentation
        use_mixup_cutmix = random.random() < config.MIXUP_CUTMIX_PROB
        
        if use_mixup_cutmix:
            # Choose between Mixup and CutMix
            if random.random() < config.CUTMIX_PROB:
                # Apply CutMix
                mixed_images, targets_a, targets_b, lam = cutmix_data(
                    images, targets, config.CUTMIX_ALPHA
                )
            else:
                # Apply Mixup
                mixed_images, targets_a, targets_b, lam = mixup_data(
                    images, targets, config.MIXUP_ALPHA
                )
            
            # Mixed Precision forward pass with mixed images
            with autocast(device_type='cuda', enabled=config.USE_AMP):
                outputs = model(mixed_images)
                loss = criterion_mixup(outputs, targets_a, targets_b, lam)
            
            # Calculate mixed accuracy
            acc = calculate_mixed_accuracy(outputs, targets_a, targets_b, lam)
        else:
            # Standard training (no Mixup/CutMix)
            with autocast(device_type='cuda', enabled=config.USE_AMP):
                outputs = model(images)
                loss = criterion_standard(outputs, targets)
            
            # Standard accuracy
            acc = calculate_accuracy(outputs, targets)
        
        # Mixed Precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.2f}%'
        })
    
    return loss_meter.avg, acc_meter.avg


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Config
) -> tuple:
    """
    Validate the model (no Mixup/CutMix during validation).
    
    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.eval()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    pbar = tqdm(val_loader, desc="          [Val  ]", leave=False)
    
    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Mixed Precision inference
            with autocast(device_type='cuda', enabled=config.USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            # Metrics
            acc = calculate_accuracy(outputs, targets)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))
            
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.2f}%'
            })
    
    return loss_meter.avg, acc_meter.avg


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    best_acc: float,
    path: str,
    is_best: bool = True
):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(checkpoint, path)
    
    # Also save as latest checkpoint for resume (overwrites each epoch)
    latest_path = os.path.join(os.path.dirname(path), 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device
) -> tuple:
    """
    Load checkpoint and restore training state.
    
    Returns:
        tuple: (start_epoch, best_acc)
    """
    if not os.path.exists(path):
        return 0, 0.0
    
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    
    print(f"Resumed from epoch {start_epoch} (best acc: {best_acc:.2f}%)")
    return start_epoch, best_acc


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def main():
    """Main training entry point"""
    
    config = Config()
    
    # Print training configuration
    print("=" * 60)
    print("SE-Res-Inception Training")
    print("=" * 60)
    
    # Device setup
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, training on CPU (will be slow!)")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    
    print(f"Mixed Precision (AMP): {'Enabled' if config.USE_AMP else 'Disabled'}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Weight Decay: {config.WEIGHT_DECAY}")
    print(f"Mixup/CutMix Probability: {config.MIXUP_CUTMIX_PROB:.0%}")
    print(f"Epochs: {config.EPOCHS}")
    print("=" * 60)
    
    # Data
    train_loader, val_loader, class_names = get_dataloaders(config)
    
    # Model
    model = SEResInception(num_classes=config.NUM_CLASSES, dropout=config.DROPOUT)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")
    print("=" * 60)
    
    # Loss Functions
    # Standard cross-entropy for non-augmented batches and validation
    criterion_standard = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Soft-target cross-entropy for Mixup/CutMix batches
    criterion_mixup = SoftTargetCrossEntropy(label_smoothing=0.1)
    
    # Optimizer with increased weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY  # Now 5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    # Mixed Precision Scaler
    scaler = GradScaler(enabled=config.USE_AMP)
    
    # Initialize metrics tracker (will load existing metrics if resuming)
    metrics = MetricsTracker(config.METRICS_PATH)
    
    # Try to resume from latest checkpoint
    start_epoch, best_acc = load_checkpoint(
        config.LATEST_CHECKPOINT_PATH,
        model, optimizer, scheduler, scaler, device
    )
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(start_epoch + 1, config.EPOCHS + 1):
        epoch_start = time.time()
        
        # Train with Mixup/CutMix
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion_standard, criterion_mixup,
            optimizer, scaler, device, epoch, config
        )
        
        # Validate (no Mixup/CutMix)
        val_loss, val_acc = validate(model, val_loader, criterion_standard, device, config)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Epoch time
        epoch_time = time.time() - epoch_start
        
        # Save metrics (persisted as numpy arrays for plotting)
        metrics.update(epoch, train_loss, train_acc, val_loss, val_acc)
        
        # Calculate generalization gap
        gen_gap = train_acc - val_acc
        
        # Print epoch summary
        print(
            f"Epoch {epoch:3d}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Gap: {gen_gap:.1f}% | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_acc,
                config.BEST_MODEL_PATH, is_best=True
            )
            print(f"  -> New best model saved! (Val Acc: {best_acc:.2f}%)")
        else:
            # Save latest checkpoint for resume (even if not best)
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_acc,
                config.LATEST_CHECKPOINT_PATH, is_best=False
            )
    
    # Training complete
    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Total Training Time: {total_time/60:.1f} minutes")
    print(f"Model saved to: {config.BEST_MODEL_PATH}")
    print(f"Metrics saved to: {config.METRICS_PATH}")
    print("=" * 60)
    
    # Print metrics file usage hint
    print("\nTo plot training curves:")
    print("  import numpy as np")
    print("  import matplotlib.pyplot as plt")
    print(f"  data = np.load('{config.METRICS_PATH}')")
    print("  plt.plot(data['epochs'], data['train_loss'], label='Train Loss')")
    print("  plt.plot(data['epochs'], data['val_loss'], label='Val Loss')")
    print("  plt.legend(); plt.show()")


if __name__ == "__main__":
    main()

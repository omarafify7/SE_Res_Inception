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

Hardware Target: NVIDIA RTX 5070 Ti (Blackwell, SM_120)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
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
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.4
    
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
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: Config
) -> tuple:
    """
    Train for one epoch with Mixed Precision (AMP).
    
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
        
        # Mixed Precision forward pass
        with autocast(device_type='cuda', enabled=config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        # Mixed Precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        acc = calculate_accuracy(outputs, targets)
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
    Validate the model.
    
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
    
    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
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
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, config
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, config)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Epoch time
        epoch_time = time.time() - epoch_start
        
        # Save metrics (persisted as numpy arrays for plotting)
        metrics.update(epoch, train_loss, train_acc, val_loss, val_acc)
        
        # Print epoch summary
        print(
            f"Epoch {epoch:3d}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
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

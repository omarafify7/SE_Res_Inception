"""
Baseline Model Training Script
===============================

Trains ResNet-18 or GoogLeNet on CIFAR-100 using the EXACT same training
pipeline as the SE-Res-Inception model (train.py) for fair comparison.

Usage:
    python -m baselines.train_baseline --model resnet18
    python -m baselines.train_baseline --model googlenet
    python -m baselines.train_baseline --model resnet18 --dry-run
"""

import os
import sys
import time
import random
import argparse
import torch  # ty: ignore[unresolved-import]
import torch.nn as nn  # ty: ignore[unresolved-import]
import torch.optim as optim  # ty: ignore[unresolved-import]
from torch.utils.data import DataLoader  # ty: ignore[unresolved-import]
from torch.amp import autocast, GradScaler  # ty: ignore[unresolved-import]
import torchvision  # ty: ignore[unresolved-import]
import torchvision.transforms as transforms  # ty: ignore[unresolved-import]
import torchvision.models as models  # ty: ignore[unresolved-import]
from tqdm import tqdm  # ty: ignore[unresolved-import]

# Add project root to path so we can import from train.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import (  # ty: ignore[unresolved-import]
    SoftTargetCrossEntropy,
    AverageMeter,
    MetricsTracker,
    mixup_data,
    cutmix_data,
    calculate_accuracy,
    calculate_mixed_accuracy,
)


# ============================================================================
# CONFIGURATION (mirrors train.py Config for CIFAR-100)
# ============================================================================
class BaselineConfig:
    """Training configuration -- mirrors train.py Config for CIFAR-100."""

    # Dataset
    MEAN = (0.5071, 0.4867, 0.4408)
    STD = (0.2675, 0.2565, 0.2761)
    IMG_SIZE = 32
    NUM_CLASSES = 100
    DATA_DIR = "./data"

    # Hyperparameters (identical to train.py)
    BATCH_SIZE = 80
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-4
    EPOCHS = 100

    # Augmentation (identical to train.py)
    MIXUP_CUTMIX_PROB = 0.5
    CUTMIX_PROB = 0.5
    MIXUP_ALPHA = 1.0
    CUTMIX_ALPHA = 1.0

    # Hardware
    NUM_WORKERS = 4
    PIN_MEMORY = True
    USE_AMP = True


# ============================================================================
# MODEL FACTORY
# ============================================================================
def build_model(model_name: str, num_classes: int = 100):
    """
    Build a baseline model with the final FC layer adjusted for CIFAR-100.

    Args:
        model_name: One of 'resnet18' or 'googlenet'.
        num_classes: Number of output classes.

    Returns:
        nn.Module ready for training.
    """
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, num_classes)
        return model

    if model_name == "googlenet":
        model = models.googlenet(weights=None, aux_logits=True)
        model.fc = nn.Linear(1024, num_classes)
        # Also fix the auxiliary classifiers so they output num_classes
        if model.aux1 is not None:
            model.aux1.fc2 = nn.Linear(
                model.aux1.fc2.in_features, num_classes
            )
        if model.aux2 is not None:
            model.aux2.fc2 = nn.Linear(
                model.aux2.fc2.in_features, num_classes
            )
        return model

    raise ValueError(f"Unknown model: {model_name}. Choose 'resnet18' or 'googlenet'.")


# ============================================================================
# DATA LOADING (mirrors train.py get_dataloaders for CIFAR-100)
# ============================================================================
def get_dataloaders(config: BaselineConfig):
    """Prepare CIFAR-100 DataLoaders with the same augmentation as train.py."""
    print("Preparing CIFAR-100 DataLoaders...")

    transform_train = transforms.Compose([
        transforms.RandomCrop(config.IMG_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])

    train_set = torchvision.datasets.CIFAR100(
        root=config.DATA_DIR, train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR100(
        root=config.DATA_DIR, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
    )

    print(f"Train samples: {len(train_set):,}")
    print(f"Val samples:   {len(test_set):,}")

    return train_loader, test_loader


# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_one_epoch(
    model,
    train_loader,
    criterion_standard,
    criterion_mixup,
    optimizer,
    scaler,
    device,
    epoch: int,
    config: BaselineConfig,
    model_name: str,
) -> tuple:
    """
    Train for one epoch with AMP and Mixup/CutMix.
    Mirrors train.py train_one_epoch exactly.

    For GoogLeNet, auxiliary logits are handled by adding aux losses.
    """
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    is_googlenet = model_name == "googlenet"

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d} [Train]", leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        use_mixup_cutmix = random.random() < config.MIXUP_CUTMIX_PROB

        if use_mixup_cutmix:
            if random.random() < config.CUTMIX_PROB:
                mixed_images, targets_a, targets_b, lam = cutmix_data(
                    images, targets, config.CUTMIX_ALPHA
                )
            else:
                mixed_images, targets_a, targets_b, lam = mixup_data(
                    images, targets, config.MIXUP_ALPHA
                )

            with autocast(device_type="cuda", enabled=config.USE_AMP):
                output = model(mixed_images)
                # GoogLeNet returns GoogLeNetOutputs with aux logits during training
                if is_googlenet and hasattr(output, "logits"):
                    main_out = output.logits
                    loss = criterion_mixup(main_out, targets_a, targets_b, lam)
                    if output.aux_logits1 is not None:
                        loss += 0.3 * criterion_mixup(
                            output.aux_logits1, targets_a, targets_b, lam
                        )
                    if output.aux_logits2 is not None:
                        loss += 0.3 * criterion_mixup(
                            output.aux_logits2, targets_a, targets_b, lam
                        )
                else:
                    main_out = output
                    loss = criterion_mixup(main_out, targets_a, targets_b, lam)

            acc = calculate_mixed_accuracy(main_out, targets_a, targets_b, lam)
        else:
            with autocast(device_type="cuda", enabled=config.USE_AMP):
                output = model(images)
                if is_googlenet and hasattr(output, "logits"):
                    main_out = output.logits
                    loss = criterion_standard(main_out, targets)
                    if output.aux_logits1 is not None:
                        loss += 0.3 * criterion_standard(output.aux_logits1, targets)
                    if output.aux_logits2 is not None:
                        loss += 0.3 * criterion_standard(output.aux_logits2, targets)
                else:
                    main_out = output
                    loss = criterion_standard(main_out, targets)

            acc = calculate_accuracy(main_out, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "acc": f"{acc_meter.avg:.2f}%",
        })

    return loss_meter.avg, acc_meter.avg


def validate(
    model,
    val_loader,
    criterion,
    device,
    config: BaselineConfig,
    model_name: str,
) -> tuple:
    """Validate the model (no Mixup/CutMix). Mirrors train.py validate."""
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    is_googlenet = model_name == "googlenet"

    pbar = tqdm(val_loader, desc="          [Val  ]", leave=False)

    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=config.USE_AMP):
                output = model(images)
                # During eval, GoogLeNet should return a plain tensor,
                # but handle the named-tuple case defensively.
                if is_googlenet and hasattr(output, "logits"):
                    output = output.logits
                loss = criterion(output, targets)

            acc = calculate_accuracy(output, targets)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))

            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "acc": f"{acc_meter.avg:.2f}%",
            })

    return loss_meter.avg, acc_meter.avg


# ============================================================================
# CHECKPOINT HELPERS
# ============================================================================
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_acc, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_acc": best_acc,
    }
    torch.save(checkpoint, path)

    latest_path = os.path.join(
        os.path.dirname(path),
        os.path.basename(path).replace("_best.pth", "_latest.pth"),
    )
    torch.save(checkpoint, latest_path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """Load checkpoint and restore training state."""
    if not os.path.exists(path):
        return 0, 0.0

    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint["epoch"]
    best_acc = checkpoint["best_acc"]

    print(f"Resumed from epoch {start_epoch} (best acc: {best_acc:.2f}%)")
    return start_epoch, best_acc


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train baseline models on CIFAR-100")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet18", "googlenet"],
        help="Baseline model to train",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Override learning rate"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run 1 epoch for quick testing"
    )
    args = parser.parse_args()

    config = BaselineConfig()

    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.dry_run:
        config.EPOCHS = 1
        config.BATCH_SIZE = 32
        print("DRY RUN MODE ENABLED: 1 Epoch, Batch Size 32")

    model_name = args.model

    # Paths
    checkpoint_dir = os.path.join("checkpoints", "baselines")
    best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
    latest_path = os.path.join(checkpoint_dir, f"{model_name}_latest.pth")
    metrics_path = os.path.join(checkpoint_dir, f"{model_name}_metrics.npz")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Banner
    print("=" * 60)
    print(f"Baseline Training: {model_name.upper()}")
    print("=" * 60)

    # Device
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, training on CPU (will be slow!)")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"Device: {torch.cuda.get_device_name(0)}")

    print(f"Mixed Precision (AMP): {'Enabled' if config.USE_AMP else 'Disabled'}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Weight Decay: {config.WEIGHT_DECAY}")
    print(f"Mixup/CutMix Probability: {config.MIXUP_CUTMIX_PROB:.0%}")
    print(f"Epochs: {config.EPOCHS}")
    print("=" * 60)

    # Data
    train_loader, val_loader = get_dataloaders(config)

    # Model
    model = build_model(model_name, num_classes=config.NUM_CLASSES)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}")
    print(f"Parameters: {total_params:,}")
    print("=" * 60)

    # Loss
    criterion_standard = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_mixup = SoftTargetCrossEntropy(label_smoothing=0.1)

    # Optimizer (identical to train.py)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # AMP scaler
    scaler = GradScaler(enabled=config.USE_AMP)

    # Metrics tracker
    metrics = MetricsTracker(metrics_path)

    # Resume
    start_epoch, best_acc = load_checkpoint(
        latest_path, model, optimizer, scheduler, scaler, device
    )

    # Training loop
    start_time = time.time()

    for epoch in range(start_epoch + 1, config.EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion_standard, criterion_mixup,
            optimizer, scaler, device, epoch, config, model_name,
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion_standard, device, config, model_name,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start

        metrics.update(epoch, train_loss, train_acc, val_loss, val_acc)

        gen_gap = train_acc - val_acc

        print(
            f"Epoch {epoch:3d}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Gap: {gen_gap:.1f}% | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_acc, best_path
            )
            print(f"  -> New best model saved! (Val Acc: {best_acc:.2f}%)")
        else:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_acc, latest_path
            )

    total_time = time.time() - start_time
    print("=" * 60)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Total Training Time: {total_time / 60:.1f} minutes")
    print(f"Model saved to: {best_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Confusion Matrix Visualization
==============================

WHAT IT SHOWS:
--------------
A 200x200 heatmap where each cell (i, j) shows how often class i was predicted as class j.
- Diagonal = correct predictions (should be bright)
- Off-diagonal = misclassifications (errors)

WHAT TO EXPECT:
---------------
1. Strong diagonal line = model learned most classes well
2. Bright clusters off-diagonal = systematic confusion between similar classes
   (e.g., different dog breeds, similar furniture types)
3. Dark rows = classes with low recall (model rarely predicts them correctly)
4. Dark columns = classes with low precision (model over-predicts them)

HOW TO INTERPRET:
-----------------
- Look for "blocks" of confusion → semantically similar classes
- Find the darkest diagonal cells → your hardest classes
- Find bright off-diagonal cells → classes that look alike to the model

Usage: python confusion_matrix.py
Output: ./outputs/confusion_matrix.png
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import SEResInception


def get_class_names(data_dir: str = "./data") -> list:
    """Load Tiny ImageNet class names (wnids)."""
    train_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'train')
    dataset = torchvision.datasets.ImageFolder(root=train_dir)
    return dataset.classes


def load_model(checkpoint_path: str, num_classes: int = 200, device: torch.device = None):
    """Load trained model from checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SEResInception(num_classes=num_classes, dropout=0.0)  # No dropout for inference
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Best accuracy: {checkpoint.get('best_acc', 'N/A'):.2f}%")
    
    return model


def get_validation_loader(data_dir: str = "./data", batch_size: int = 64) -> DataLoader:
    """Create validation data loader."""
    val_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'val')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ])
    
    val_set = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return val_loader


def compute_confusion_matrix(model, dataloader, num_classes: int, device: torch.device):
    """Compute confusion matrix from model predictions."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Computing predictions"):
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            targets = targets.numpy()
            
            for true, pred in zip(targets, predictions):
                confusion[true, pred] += 1
    
    return confusion


def plot_confusion_matrix(confusion: np.ndarray, class_names: list, output_path: str):
    """Plot and save confusion matrix."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Normalize by row (true class) for better visualization
    confusion_normalized = confusion.astype(float) / (confusion.sum(axis=1, keepdims=True) + 1e-8)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Full confusion matrix (zoomed out)
    ax1 = axes[0]
    im1 = ax1.imshow(confusion_normalized, cmap='Blues', aspect='auto')
    ax1.set_title('Confusion Matrix (200 classes)\nBrighter = More Predictions', fontsize=14)
    ax1.set_xlabel('Predicted Class')
    ax1.set_ylabel('True Class')
    plt.colorbar(im1, ax=ax1, label='Proportion')
    
    # Find top confused pairs for annotation
    np.fill_diagonal(confusion_normalized, 0)  # Ignore diagonal
    flat_idx = np.argsort(confusion_normalized.ravel())[-20:]  # Top 20 confusions
    top_confusions = []
    for idx in flat_idx:
        i, j = np.unravel_index(idx, confusion_normalized.shape)
        if confusion_normalized[i, j] > 0.05:  # Only significant confusions
            top_confusions.append((i, j, confusion_normalized[i, j]))
    
    # Summary statistics
    ax2 = axes[1]
    ax2.axis('off')
    
    # Compute per-class accuracy
    per_class_acc = np.diag(confusion) / (confusion.sum(axis=1) + 1e-8)
    overall_acc = np.diag(confusion).sum() / confusion.sum()
    
    summary_text = f"""
CONFUSION MATRIX SUMMARY
========================

Overall Accuracy: {overall_acc*100:.2f}%

Per-Class Accuracy Statistics:
  Mean:   {per_class_acc.mean()*100:.2f}%
  Std:    {per_class_acc.std()*100:.2f}%
  Min:    {per_class_acc.min()*100:.2f}%
  Max:    {per_class_acc.max()*100:.2f}%

Top 10 Most Confused Pairs:
(True Class → Predicted Class: Rate)
"""
    for i, j, rate in sorted(top_confusions, key=lambda x: -x[2])[:10]:
        summary_text += f"  {class_names[i][:12]:12} → {class_names[j][:12]:12}: {rate*100:.1f}%\n"
    
    ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrix to {output_path}")
    
    # Also save raw data
    np.save(output_path.replace('.png', '_raw.npy'), confusion)
    print(f"Saved raw matrix to {output_path.replace('.png', '_raw.npy')}")


def main():
    # Configuration
    checkpoint_path = "./checkpoints/tiny_imagenet/best_model.pth"
    output_path = "./outputs/confusion_matrix.png"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and data
    model = load_model(checkpoint_path, device=device)
    dataloader = get_validation_loader()
    class_names = get_class_names()
    
    # Compute and plot
    confusion = compute_confusion_matrix(model, dataloader, num_classes=200, device=device)
    plot_confusion_matrix(confusion, class_names, output_path)
    
    print("\n✅ Confusion matrix complete!")
    print("Check the output image for patterns of confusion between similar classes.")


if __name__ == "__main__":
    main()

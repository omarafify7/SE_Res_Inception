"""
Per-Class Accuracy Analysis
===========================

WHAT IT SHOWS:
--------------
A detailed breakdown of model performance for each of the 200 classes.
Shows which classes the model excels at and which it struggles with.

WHAT TO EXPECT:
---------------
1. High variance in per-class accuracy (some classes 90%+, others <40%)
2. "Easy" classes: objects with distinct shapes/colors (e.g., goldfish, pizza)
3. "Hard" classes: fine-grained distinctions (e.g., different dog breeds)
4. Low-sample classes may have unstable accuracy

HOW TO INTERPRET:
-----------------
- Bottom 20 classes → candidates for data augmentation or architecture improvements
- Top 20 classes → model has learned strong features for these
- Classes with 0% accuracy → severe confusion with another class
- Pattern in hard classes → may reveal systematic weakness (e.g., all textures, all animals)

Usage: python per_class_accuracy.py
Output: ./outputs/per_class_accuracy.png
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import SEResInception


def get_class_names(data_dir: str = "./data") -> list:
    """Load Tiny ImageNet class names."""
    train_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'train')
    dataset = torchvision.datasets.ImageFolder(root=train_dir)
    return dataset.classes


def load_wnid_to_words(data_dir: str = "./data") -> dict:
    """Load mapping from wnid to human-readable class names."""
    words_file = os.path.join(data_dir, 'tiny-imagenet-200', 'words.txt')
    wnid_to_words = {}
    
    if os.path.exists(words_file):
        with open(words_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    wnid_to_words[parts[0]] = parts[1].split(',')[0]  # Take first word
    
    return wnid_to_words


def load_model(checkpoint_path: str, num_classes: int = 200, device: torch.device = None):
    """Load trained model from checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SEResInception(num_classes=num_classes, dropout=0.0)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
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


def compute_per_class_accuracy(model, dataloader, num_classes: int, device: torch.device):
    """Compute per-class accuracy."""
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            targets = targets.numpy()
            
            for true, pred in zip(targets, predictions):
                total[true] += 1
                if true == pred:
                    correct[true] += 1
    
    accuracy = correct / (total + 1e-8)
    return accuracy, correct, total


def plot_per_class_accuracy(accuracy: np.ndarray, class_names: list, 
                            wnid_to_words: dict, output_path: str):
    """Create comprehensive per-class accuracy visualization."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracy)
    sorted_acc = accuracy[sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]
    
    # Create human-readable names
    sorted_readable = [wnid_to_words.get(n, n)[:20] for n in sorted_names]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 14))
    
    # 1. Histogram of accuracies
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist(accuracy * 100, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(accuracy.mean() * 100, color='red', linestyle='--', 
                label=f'Mean: {accuracy.mean()*100:.1f}%')
    ax1.axvline(np.median(accuracy) * 100, color='green', linestyle='--',
                label=f'Median: {np.median(accuracy)*100:.1f}%')
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_ylabel('Number of Classes')
    ax1.set_title('Distribution of Per-Class Accuracy')
    ax1.legend()
    
    # 2. Bottom 20 classes (hardest)
    ax2 = fig.add_subplot(2, 2, 2)
    bottom_20_idx = range(20)
    bars = ax2.barh(range(20), sorted_acc[:20] * 100, color='coral')
    ax2.set_yticks(range(20))
    ax2.set_yticklabels([sorted_readable[i] for i in bottom_20_idx], fontsize=9)
    ax2.set_xlabel('Accuracy (%)')
    ax2.set_title('Bottom 20 Classes (Hardest)')
    ax2.set_xlim(0, 100)
    
    # Add accuracy values on bars
    for i, (bar, acc) in enumerate(zip(bars, sorted_acc[:20])):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{acc*100:.1f}%', va='center', fontsize=8)
    
    # 3. Top 20 classes (easiest)
    ax3 = fig.add_subplot(2, 2, 3)
    top_20_idx = range(180, 200)
    bars = ax3.barh(range(20), sorted_acc[180:200] * 100, color='mediumseagreen')
    ax3.set_yticks(range(20))
    ax3.set_yticklabels([sorted_readable[i] for i in top_20_idx], fontsize=9)
    ax3.set_xlabel('Accuracy (%)')
    ax3.set_title('Top 20 Classes (Easiest)')
    ax3.set_xlim(0, 100)
    
    for i, (bar, acc) in enumerate(zip(bars, sorted_acc[180:200])):
        ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{acc*100:.1f}%', va='center', fontsize=8)
    
    # 4. Summary statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    summary_text = f"""
PER-CLASS ACCURACY SUMMARY
==========================

Statistics:
  Mean Accuracy:    {accuracy.mean()*100:.2f}%
  Median Accuracy:  {np.median(accuracy)*100:.2f}%
  Std Dev:          {accuracy.std()*100:.2f}%
  Min:              {accuracy.min()*100:.2f}%
  Max:              {accuracy.max()*100:.2f}%

Distribution:
  Classes > 80%:    {(accuracy > 0.8).sum()} ({(accuracy > 0.8).sum()/2:.1f}%)
  Classes 60-80%:   {((accuracy >= 0.6) & (accuracy <= 0.8)).sum()}
  Classes 40-60%:   {((accuracy >= 0.4) & (accuracy < 0.6)).sum()}
  Classes < 40%:    {(accuracy < 0.4).sum()} ({(accuracy < 0.4).sum()/2:.1f}%)

INTERPRETATION:
--------------
• High variance = model specializes in some classes
• Low-accuracy classes may have:
  - Similar visual appearance to other classes
  - Less distinctive features
  - More diverse within-class variation

• Consider targeted augmentation for hard classes
• Classes with 0% may need architecture changes
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved per-class accuracy to {output_path}")
    
    # Save detailed CSV
    csv_path = output_path.replace('.png', '.csv')
    with open(csv_path, 'w') as f:
        f.write("wnid,readable_name,accuracy,rank\n")
        for rank, idx in enumerate(sorted_indices):
            name = class_names[idx]
            readable = wnid_to_words.get(name, name)
            f.write(f"{name},{readable},{accuracy[idx]*100:.2f},{rank+1}\n")
    print(f"Saved detailed CSV to {csv_path}")


def main():
    checkpoint_path = "./checkpoints/tiny_imagenet/best_model.pth"
    output_path = "./outputs/per_class_accuracy.png"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_model(checkpoint_path, device=device)
    dataloader = get_validation_loader()
    class_names = get_class_names()
    wnid_to_words = load_wnid_to_words()
    
    accuracy, correct, total = compute_per_class_accuracy(
        model, dataloader, num_classes=200, device=device
    )
    
    plot_per_class_accuracy(accuracy, class_names, wnid_to_words, output_path)
    
    print("\n✅ Per-class accuracy analysis complete!")


if __name__ == "__main__":
    main()

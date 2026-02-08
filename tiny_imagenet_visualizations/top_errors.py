"""
Top Errors Gallery
==================

WHAT IT SHOWS:
--------------
A gallery of the model's most confident wrong predictions.
These are images where the model was VERY sure (high confidence) but still wrong.

WHAT TO EXPECT:
---------------
1. Images that genuinely look ambiguous to humans
2. Mislabeled images in the dataset
3. Images with multiple objects (e.g., cat in front of a car → model says car)
4. Feature mimicry (e.g., a spotted dog predicted as leopard)

HOW TO INTERPRET:
-----------------
- High-confidence errors → model has learned spurious correlations
- Low-confidence errors → model is appropriately uncertain
- Repeated class confusion → architectural limitation for those features
- Dataset issues → if you see clear mislabels, dataset quality is a factor

This visualization is valuable for:
1. Finding dataset bugs (mislabeled images)
2. Understanding model failure modes
3. Identifying classes that need more augmentation

Usage: python top_errors.py
Output: ./outputs/top_errors_gallery.png
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
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import SEResInception


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model."""
    model = SEResInception(num_classes=200, dropout=0.0)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def get_class_names_and_words(data_dir: str = "./data"):
    """Load class names and human-readable words."""
    train_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'train')
    dataset = torchvision.datasets.ImageFolder(root=train_dir)
    class_names = dataset.classes
    
    # Load human-readable names
    words_file = os.path.join(data_dir, 'tiny-imagenet-200', 'words.txt')
    wnid_to_words = {}
    if os.path.exists(words_file):
        with open(words_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    wnid_to_words[parts[0]] = parts[1].split(',')[0]
    
    return class_names, wnid_to_words


def find_top_errors(model, data_dir: str, device: torch.device, top_k: int = 32):
    """Find the most confident wrong predictions."""
    val_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'val')
    
    # Preprocessing transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ])
    
    # Raw transform for visualization
    raw_transform = transforms.Compose([transforms.ToTensor()])
    
    val_set = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)
    raw_set = torchvision.datasets.ImageFolder(root=val_dir, transform=raw_transform)
    
    errors = []  # (confidence, idx, true_label, pred_label, probs)
    
    print("Finding confident errors...")
    with torch.no_grad():
        for idx in tqdm(range(len(val_set))):
            img_tensor, true_label = val_set[idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)[0]
            pred_label = probs.argmax().item()
            confidence = probs[pred_label].item()
            
            if pred_label != true_label:
                # Store top-5 predictions for analysis
                top5_probs, top5_idx = probs.topk(5)
                errors.append({
                    'confidence': confidence,
                    'idx': idx,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'true_prob': probs[true_label].item(),
                    'top5_probs': top5_probs.cpu().numpy(),
                    'top5_idx': top5_idx.cpu().numpy()
                })
    
    # Sort by confidence (highest first)
    errors.sort(key=lambda x: -x['confidence'])
    
    return errors[:top_k], val_set, raw_set


def create_error_gallery(errors: list, val_set, raw_set, class_names: list,
                         wnid_to_words: dict, output_path: str):
    """Create a gallery of top errors with detailed information."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    n_errors = len(errors)
    n_cols = 4
    n_rows = (n_errors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
    
    for i, error in enumerate(errors):
        ax = axes[i]
        
        # Get raw image
        raw_img, _ = raw_set[error['idx']]
        img_np = raw_img.permute(1, 2, 0).numpy()
        
        ax.imshow(img_np)
        ax.axis('off')
        
        # Get readable names
        true_name = wnid_to_words.get(class_names[error['true_label']], 
                                       class_names[error['true_label']])[:15]
        pred_name = wnid_to_words.get(class_names[error['pred_label']], 
                                       class_names[error['pred_label']])[:15]
        
        # Create title with prediction info
        title = (f"TRUE: {true_name}\n"
                f"PRED: {pred_name} ({error['confidence']*100:.1f}%)\n"
                f"True rank: {np.where(error['top5_idx'] == error['true_label'])[0]}")
        
        # Check if true label is in top-5
        if error['true_label'] in error['top5_idx']:
            true_rank = np.where(error['top5_idx'] == error['true_label'])[0][0] + 1
            title = (f"TRUE: {true_name}\n"
                    f"PRED: {pred_name} ({error['confidence']*100:.1f}%)\n"
                    f"True was #{true_rank} ({error['true_prob']*100:.1f}%)")
        else:
            title = (f"TRUE: {true_name}\n"
                    f"PRED: {pred_name} ({error['confidence']*100:.1f}%)\n"
                    f"True not in top-5 ({error['true_prob']*100:.1f}%)")
        
        ax.set_title(title, fontsize=9, color='red')
    
    # Hide unused subplots
    for i in range(len(errors), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Top {len(errors)} Most Confident Wrong Predictions\n"
                 "(Model was VERY sure but still wrong)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved error gallery to {output_path}")
    
    # Also create a text report
    report_path = output_path.replace('.png', '_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("TOP ERRORS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Analyze common failure patterns
        pred_counts = {}
        true_counts = {}
        confusion_pairs = {}
        
        for error in errors:
            true_name = wnid_to_words.get(class_names[error['true_label']], 
                                          class_names[error['true_label']])
            pred_name = wnid_to_words.get(class_names[error['pred_label']], 
                                          class_names[error['pred_label']])
            
            pred_counts[pred_name] = pred_counts.get(pred_name, 0) + 1
            true_counts[true_name] = true_counts.get(true_name, 0) + 1
            
            pair = f"{true_name} -> {pred_name}"
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        f.write("MOST FREQUENTLY PREDICTED (incorrectly):\n")
        for name, count in sorted(pred_counts.items(), key=lambda x: -x[1])[:10]:
            f.write(f"  {name}: {count} times\n")
        
        f.write("\nMOST FREQUENTLY WRONG ABOUT:\n")
        for name, count in sorted(true_counts.items(), key=lambda x: -x[1])[:10]:
            f.write(f"  {name}: {count} times\n")
        
        f.write("\nMOST COMMON CONFUSION PAIRS:\n")
        for pair, count in sorted(confusion_pairs.items(), key=lambda x: -x[1])[:10]:
            f.write(f"  {pair}: {count} times\n")
    
    print(f"Saved error report to {report_path}")


def main():
    checkpoint_path = "./checkpoints/tiny_imagenet/best_model.pth"
    output_path = "./outputs/top_errors_gallery.png"
    data_dir = "./data"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_model(checkpoint_path, device)
    class_names, wnid_to_words = get_class_names_and_words(data_dir)
    
    errors, val_set, raw_set = find_top_errors(model, data_dir, device, top_k=32)
    
    print(f"\nFound {len(errors)} confident errors")
    print(f"Highest confidence error: {errors[0]['confidence']*100:.1f}%")
    
    create_error_gallery(errors, val_set, raw_set, class_names, wnid_to_words, output_path)
    
    print("\n✅ Top errors analysis complete!")
    print("Look for patterns in the errors - do certain classes get confused?")


if __name__ == "__main__":
    main()

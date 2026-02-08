"""
t-SNE/UMAP Embedding Visualization
==================================

WHAT IT SHOWS:
--------------
Visualizes the high-dimensional feature space (before the final classification layer)
reduced to 2D. Each point is an image, colored by its true class.

WHAT TO EXPECT:
---------------
1. Clusters = images the model sees as similar
2. Tight clusters = strong feature learning for that class
3. Overlapping clusters = classes the model confuses
4. Outliers = unusual/hard examples

HOW TO INTERPRET:
-----------------
- Well-separated clusters → model has learned discriminative features
- Mixed clusters → model struggles to distinguish those classes
- Long "chains" → possible data quality issues or transformation artifacts
- If semantic groups cluster together (animals, furniture) → hierarchical learning

t-SNE vs UMAP:
- t-SNE: Better at preserving local structure (clusters)
- UMAP: Better at preserving global structure (relationships between clusters)

Usage: python tsne_embeddings.py
Output: ./outputs/embedding_tsne.png and ./outputs/embedding_umap.png
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# Try to import UMAP (optional)
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not installed. Install with: pip install umap-learn")

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


def get_validation_loader(data_dir: str = "./data", batch_size: int = 64) -> DataLoader:
    """Create validation data loader."""
    val_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'val')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ])
    
    val_set = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return val_loader, val_set.classes


class FeatureExtractor:
    """Hook-based feature extractor for the penultimate layer."""
    
    def __init__(self, model):
        self.model = model
        self.features = None
        
        # Hook the global average pooling output
        # In SEResInception, features are after GAP but before FC
        self._register_hook()
    
    def _register_hook(self):
        def hook(module, input, output):
            self.features = output.detach()
        
        # The fc layer's input is what we want
        self.model.fc.register_forward_hook(lambda m, i, o: None)  # Dummy
        
        # Actually, let's hook into the forward and get features
        self.original_forward = self.model.forward
        
        def hooked_forward(x):
            # Run through the model but capture pre-FC features
            # Stage 1
            x = self.model.stem(x)
            x = self.model.inception1(x)
            x = self.model.inception2(x)
            x = self.model.pool1(x)
            # Stage 2
            x = self.model.inception3(x)
            x = self.model.inception4(x)
            # Stage 3
            x = self.model.inception5(x)
            x = self.model.inception6(x)
            x = self.model.inception7(x)
            x = self.model.pool2(x)
            # Stage 4
            x = self.model.inception8(x)
            x = self.model.inception9(x)
            # Classifier head
            x = self.model.global_pool(x)
            x = x.view(x.size(0), -1)
            self.features = x.detach()
            x = self.model.dropout(x)
            x = self.model.fc(x)
            return x
        
        self.model.forward = hooked_forward


def extract_features(model, dataloader, device) -> tuple:
    """Extract features for all validation images."""
    feature_extractor = FeatureExtractor(model)
    
    all_features = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            
            all_features.append(feature_extractor.features.cpu().numpy())
            all_labels.append(labels.numpy())
            all_predictions.append(predictions.cpu().numpy())
    
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)
    
    return features, labels, predictions


def plot_embedding(embedding: np.ndarray, labels: np.ndarray, predictions: np.ndarray,
                   class_names: list, output_path: str, method: str = "t-SNE"):
    """Create embedding visualization."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create figure with two views
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # Left: Color by true class (sample of classes for visibility)
    ax1 = axes[0]
    
    # Select top 20 most common classes for clearer visualization
    unique_labels, counts = np.unique(labels, return_counts=True)
    top_20_classes = unique_labels[np.argsort(-counts)[:20]]
    
    mask = np.isin(labels, top_20_classes)
    
    scatter = ax1.scatter(embedding[mask, 0], embedding[mask, 1], 
                         c=labels[mask], cmap='tab20', alpha=0.6, s=10)
    ax1.set_title(f'{method} Embedding - Colored by True Class (Top 20)', fontsize=12)
    ax1.set_xlabel(f'{method} Dimension 1')
    ax1.set_ylabel(f'{method} Dimension 2')
    
    # Add legend for a few classes
    handles = []
    for i, cls in enumerate(top_20_classes[:10]):
        color = plt.cm.tab20(i / 20)
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=color, markersize=8,
                                   label=class_names[cls][:15]))
    ax1.legend(handles=handles, loc='upper right', fontsize=8)
    
    # Right: Color by correct/incorrect
    ax2 = axes[1]
    correct = labels == predictions
    
    ax2.scatter(embedding[~correct, 0], embedding[~correct, 1],
               c='red', alpha=0.4, s=10, label=f'Wrong ({(~correct).sum()})')
    ax2.scatter(embedding[correct, 0], embedding[correct, 1],
               c='green', alpha=0.4, s=10, label=f'Correct ({correct.sum()})')
    
    ax2.set_title(f'{method} Embedding - Correct vs Incorrect Predictions', fontsize=12)
    ax2.set_xlabel(f'{method} Dimension 1')
    ax2.set_ylabel(f'{method} Dimension 2')
    ax2.legend(loc='upper right')
    
    # Add summary
    accuracy = correct.mean() * 100
    fig.suptitle(f'{method} Visualization of Model Features\n'
                 f'Overall Accuracy: {accuracy:.2f}% | Points: {len(labels):,}',
                 fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {method} embedding to {output_path}")


def main():
    checkpoint_path = "./checkpoints/tiny_imagenet/best_model.pth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_model(checkpoint_path, device)
    dataloader, class_names = get_validation_loader()
    
    print("Extracting features...")
    features, labels, predictions = extract_features(model, dataloader, device)
    print(f"Feature shape: {features.shape}")
    
    # t-SNE
    print("\nComputing t-SNE (this may take a few minutes)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embedding_tsne = tsne.fit_transform(features)
    plot_embedding(embedding_tsne, labels, predictions, class_names,
                  "./outputs/embedding_tsne.png", "t-SNE")
    
    # UMAP (if available)
    if HAS_UMAP:
        print("\nComputing UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding_umap = reducer.fit_transform(features)
        plot_embedding(embedding_umap, labels, predictions, class_names,
                      "./outputs/embedding_umap.png", "UMAP")
    
    # Save features for later analysis
    np.savez("./outputs/features.npz", 
             features=features, labels=labels, predictions=predictions)
    print("\nSaved features to ./outputs/features.npz")
    
    print("\n✅ Embedding visualization complete!")
    print("Look for tight clusters (good) and overlapping regions (confusion).")


if __name__ == "__main__":
    main()

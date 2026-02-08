"""
Grad-CAM Visualization
======================

WHAT IT SHOWS:
--------------
Gradient-weighted Class Activation Mapping (Grad-CAM) highlights which regions
of an image the model focuses on when making predictions.

A heatmap is overlaid on the original image:
- RED/YELLOW = high importance (model "looks" here)
- BLUE/PURPLE = low importance (ignored regions)

WHAT TO EXPECT:
---------------
1. Correct predictions: Heatmap should highlight the actual object
2. Wrong predictions: Heatmap may focus on background or wrong features
3. Good model: Attention aligns with human intuition about what's important
4. Overfitting signs: Focusing on textures/backgrounds instead of objects

HOW TO INTERPRET:
-----------------
- If model focuses on object → good feature learning
- If model focuses on background → possible dataset bias (e.g., all cats on couches)
- If attention is diffuse → model may be uncertain
- For wrong predictions, Grad-CAM reveals WHY the model was confused

Usage: python grad_cam.py
Output: ./outputs/grad_cam_gallery.png
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
from PIL import Image
import cv2

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import SEResInception


class GradCAM:
    """
    Grad-CAM implementation for SE-Res-Inception.
    Computes gradient-weighted activation maps from the last convolutional layer.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for the input image.
        
        Args:
            input_tensor: Preprocessed image tensor [1, C, H, W]
            target_class: Class to explain (None = use predicted class)
        
        Returns:
            cam: Heatmap normalized to [0, 1] with shape [H, W]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # Global average pooling
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive contributions
        
        # Upsample to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class, output.softmax(dim=1)[0, target_class].item()


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model."""
    model = SEResInception(num_classes=200, dropout=0.0)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def get_sample_images(data_dir: str = "./data", num_correct: int = 8, num_wrong: int = 8):
    """Get sample images for visualization (mix of correct and incorrect predictions)."""
    val_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'val')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ])
    
    val_set = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)
    
    # Also need raw images for visualization
    raw_transform = transforms.Compose([transforms.ToTensor()])
    raw_set = torchvision.datasets.ImageFolder(root=val_dir, transform=raw_transform)
    
    return val_set, raw_set


def overlay_cam_on_image(image: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image."""
    # Convert CAM to colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay
    overlaid = alpha * heatmap + (1 - alpha) * image
    overlaid = np.clip(overlaid, 0, 1)
    
    return overlaid


def create_grad_cam_gallery(model, val_set, raw_set, device, output_path: str,
                            num_samples: int = 16):
    """Create a gallery of Grad-CAM visualizations."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get the last convolutional layer (before global average pooling)
    # In SEResInception, this is inception9 (the final inception block)
    target_layer = model.inception9
    grad_cam = GradCAM(model, target_layer)
    
    # Get class names
    class_names = val_set.classes
    
    # Collect samples
    correct_samples = []
    wrong_samples = []
    
    indices = np.random.permutation(len(val_set))
    
    for idx in indices:
        if len(correct_samples) >= num_samples // 2 and len(wrong_samples) >= num_samples // 2:
            break
        
        img_tensor, true_label = val_set[idx]
        raw_img, _ = raw_set[idx]
        
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            pred_label = output.argmax(dim=1).item()
        
        sample = (idx, img_tensor, raw_img, true_label, pred_label)
        
        if pred_label == true_label and len(correct_samples) < num_samples // 2:
            correct_samples.append(sample)
        elif pred_label != true_label and len(wrong_samples) < num_samples // 2:
            wrong_samples.append(sample)
    
    # Create visualization
    all_samples = correct_samples + wrong_samples
    n_cols = 4
    n_rows = len(all_samples) // n_cols
    
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(16, n_rows * 4))
    
    for i, (idx, img_tensor, raw_img, true_label, pred_label) in enumerate(all_samples):
        row = (i // n_cols) * 2
        col = i % n_cols
        
        # Generate Grad-CAM
        cam, _, confidence = grad_cam.generate(img_tensor, target_class=pred_label)
        
        # Convert raw image for display
        raw_np = raw_img.permute(1, 2, 0).numpy()
        
        # Original image
        ax_orig = axes[row, col]
        ax_orig.imshow(raw_np)
        ax_orig.axis('off')
        
        is_correct = true_label == pred_label
        color = 'green' if is_correct else 'red'
        status = '✓' if is_correct else '✗'
        ax_orig.set_title(f"{status} True: {class_names[true_label][:10]}\n"
                         f"Pred: {class_names[pred_label][:10]} ({confidence*100:.1f}%)",
                         fontsize=9, color=color)
        
        # Grad-CAM overlay
        ax_cam = axes[row + 1, col]
        overlay = overlay_cam_on_image(raw_np, cam)
        ax_cam.imshow(overlay)
        ax_cam.axis('off')
        ax_cam.set_title("Grad-CAM", fontsize=9)
    
    plt.suptitle("Grad-CAM Visualization\nTop rows: Correct predictions | Bottom rows: Wrong predictions",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved Grad-CAM gallery to {output_path}")


def main():
    checkpoint_path = "./checkpoints/tiny_imagenet/best_model.pth"
    output_path = "./outputs/grad_cam_gallery.png"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_model(checkpoint_path, device)
    val_set, raw_set = get_sample_images()
    
    create_grad_cam_gallery(model, val_set, raw_set, device, output_path)
    
    print("\n✅ Grad-CAM visualization complete!")
    print("Compare where the model 'looks' for correct vs incorrect predictions.")


if __name__ == "__main__":
    main()

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# --- IMPORT YOUR MODEL ---
from model import SEResInception

# --- CONFIGURATION ---
CHECKPOINT_PATH = "./checkpoints/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_CLASS_NAME = "otter"  # The class we want to test
TARGET_LAYER_NAME = "layer4" # Usually the last block of the network (adjust based on your model print)

# CIFAR-100 Classes (Fine)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hooks to capture gradients and activations
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        # 1. Forward Pass
        output = self.model(x)
        self.model.zero_grad()
        
        # 2. Backward Pass for the specific class
        score = output[0, class_idx]
        score.backward()

        # 3. Generate CAM
        gradients = self.gradients[0].cpu().data.numpy() # [C, H, W]
        activations = self.activations[0].cpu().data.numpy() # [C, H, W]

        # Global Average Pooling of gradients (weights)
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU (we only care about positive influence)
        cam = np.maximum(cam, 0)
        
        # Resize to input image size (32x32)
        cam = cv2.resize(cam, (32, 32))
        
        # Normalize to 0-1
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)
        return cam, output

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def denormalize(tensor):
    # Reverse CIFAR normalization for display
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)

def main():
    # 1. Load Model
    model = SEResInception()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    # --- AUTO-DETECT LAST LAYER ---
    # We try to grab the last ResInception block. 
    # Adjust this logic if your model structure is different.
    # Assuming your model has a list/sequential of blocks named 'layers' or similar.
    target_layer = None
    
    # Try finding the last Sequential block
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Sequential):
            # Grab the last block in the last sequence
            target_layer = module[-1] 
    
    # Fallback: if you know the name, e.g., model.blocks[8]
    if target_layer is None:
        print("Could not auto-detect layer. Defaulting to last known module.")
        target_layer = list(model.modules())[-2] # Second to last (usually before FC)

    print(f"Hooking into layer: {target_layer}")
    grad_cam = GradCAM(model, target_layer)

    # 2. Get Data (Find an Otter)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    # Find the first image of the target class
    target_idx = CIFAR100_CLASSES.index(TARGET_CLASS_NAME)
    img_tensor, label = next((img, lbl) for img, lbl in testset if lbl == target_idx)
    
    input_tensor = img_tensor.unsqueeze(0).to(DEVICE) # Add batch dim

    # 3. Generate Heatmap
    mask, output = grad_cam(input_tensor, target_idx)
    
    # Get prediction
    pred_idx = output.argmax(dim=1).item()
    pred_name = CIFAR100_CLASSES[pred_idx]
    
    print(f"True Class: {TARGET_CLASS_NAME}")
    print(f"Predicted: {pred_name}")

    # 4. Visualize
    original_img = denormalize(img_tensor)
    visualization = show_cam_on_image(original_img, mask)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original_img)
    axes[0].set_title(f"Original ({TARGET_CLASS_NAME})")
    axes[0].axis('off')

    axes[1].imshow(visualization)
    axes[1].set_title(f"GradCAM (Pred: {pred_name})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_result.png')
    plt.show()

if __name__ == "__main__":
    main()
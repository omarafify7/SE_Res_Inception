import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# --- IMPORT YOUR MODEL ---
# Ensure model.py is in the same directory
from model import SEResInception 

# --- CONFIGURATION ---
CHECKPOINT_PATH = "./checkpoints/best_model.pth"
BATCH_SIZE = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CIFAR-100 SUPERCLASS MAPPING ---
# CIFAR-100 has 100 "fine" classes and 20 "coarse" superclasses.
# This list maps index i (fine) to index j (coarse).
# E.g. apple (0) -> fruit_and_vegetables (4)
FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,  # 0-9
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11, # 10-19
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, # 20-29
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5, # 30-39
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, # 40-49
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17, # 50-59
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, # 60-69
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13, # 70-79
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, # 80-89
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13  # 90-99
]

SUPERCLASS_NAMES = [
    "Aquatic mammals", "Fish", "Flowers", "Food containers", 
    "Fruit/Veg", "Electronics", "Furniture", "Insects", 
    "Large carnivores", "Large man-made", "Large natural", 
    "Large omnivores", "Medium mammals", "Non-insect invert", 
    "People", "Reptiles", "Small mammals", "Trees", 
    "Vehicles 1", "Vehicles 2"
]

def load_data():
    # Standard normalization for validation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return testloader

def evaluate():
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    
    # Initialize Model
    model = SEResInception() # Adjust args if your init takes parameters
    
    # Load Weights safely
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    # Handle if checkpoint saves 'model_state_dict' or just the model
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_targets = []

    print("Running Inference...")
    with torch.no_grad():
        for inputs, targets in load_data():
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # --- CONVERT TO SUPERCLASSES ---
    print("Mapping to Superclasses...")
    super_preds = [FINE_TO_COARSE[p] for p in all_preds]
    super_targets = [FINE_TO_COARSE[t] for t in all_targets]

    # --- CALCULATE SUPERCLASS ACCURACY ---
    correct = sum(p == t for p, t in zip(super_preds, super_targets))
    total = len(super_targets)
    superclass_accuracy = 100.0 * correct / total
    print(f"\n{'='*50}")
    print(f"SUPERCLASS ACCURACY: {correct}/{total} = {superclass_accuracy:.2f}%")
    print(f"{'='*50}\n")

    # --- PLOT CONFUSION MATRIX ---
    cm = confusion_matrix(super_targets, super_preds)
    # Normalize by row (true class)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=False, fmt=".2f", cmap="Blues",
                xticklabels=SUPERCLASS_NAMES, yticklabels=SUPERCLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('CIFAR-100 Superclass Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    print("Saving plot to 'superclass_confusion.png'...")
    plt.savefig('superclass_confusion.png')
    plt.show()

if __name__ == "__main__":
    evaluate()
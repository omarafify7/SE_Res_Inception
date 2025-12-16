"""
SE-Res-Inception FastAPI Inference Server
==========================================

Production-ready API for serving CIFAR-100 predictions using the SE-Res-Inception model.
Optimized for NVIDIA RTX 5070 Ti with CUDA acceleration.

Endpoints:
    POST /predict - Accept image upload and return top-5 predictions with confidence scores

Author: MLOps Engineer
Hardware: NVIDIA RTX 5070 Ti (CUDA 12.x)
"""

import io
import time
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms

# Import the model architecture
from model import SEResInception


# =============================================================================
# CIFAR-100 FINE-GRAINED LABELS (100 classes)
# =============================================================================
CIFAR100_LABELS: List[str] = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
    'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly',
    'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
    'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
    'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew',
    'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe',
    'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# =============================================================================
# GLOBAL MODEL STATE
# =============================================================================
model: SEResInception = None
device: torch.device = None
preprocess: transforms.Compose = None


def load_model() -> None:
    """
    Load the SE-Res-Inception model and weights onto the GPU.
    Called once during application startup.
    """
    global model, device, preprocess
    
    # Set device to CUDA (RTX 5070 Ti)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    if not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU. Performance will be degraded.")
    else:
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA Version: {torch.version.cuda}")
    
    # Initialize model architecture
    model = SEResInception(num_classes=100)
    
    # Load pretrained weights
    checkpoint_path = 'checkpoints/best_model.pth'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"[INFO] Loaded model state from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("[INFO] Loaded model state from checkpoint")
            else:
                # Assume the dict is the state_dict itself
                model.load_state_dict(checkpoint)
                print("[INFO] Loaded model state dict directly")
        else:
            # If checkpoint is not a dict, assume it's the state dict
            model.load_state_dict(checkpoint)
            print("[INFO] Loaded model state dict")
            
    except FileNotFoundError:
        raise RuntimeError(f"Checkpoint not found at: {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
    
    # Move model to GPU and set to evaluation mode
    model = model.to(device)
    model.eval()
    print(f"[INFO] Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Define preprocessing pipeline
    # Using ImageNet normalization as specified, resize to 224x224
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    print("[INFO] Preprocessing pipeline initialized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup: Load model
    print("=" * 60)
    print("SE-Res-Inception Inference Server Starting...")
    print("=" * 60)
    load_model()
    print("=" * 60)
    print("Server ready to accept requests!")
    print("=" * 60)
    
    yield
    
    # Shutdown: Cleanup
    print("[INFO] Shutting down server...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[INFO] Server shutdown complete")


# Initialize FastAPI application with lifespan
app = FastAPI(
    title="SE-Res-Inception CIFAR-100 API",
    description="GPU-accelerated image classification API using SE-Res-Inception architecture",
    version="1.0.0",
    lifespan=lifespan
)


def preprocess_image(image_data: bytes) -> torch.Tensor:
    """
    Preprocess uploaded image for model inference.
    
    Args:
        image_data: Raw bytes of the uploaded image
        
    Returns:
        Preprocessed tensor ready for model inference (1, 3, 224, 224)
        
    Raises:
        HTTPException: If image cannot be processed
    """
    try:
        # Open image and convert to RGB
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing transformations
        tensor = preprocess(image)
        
        # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
        tensor = tensor.unsqueeze(0)
        
        return tensor
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}. Please upload a valid image file (JPEG, PNG, etc.)"
        )


@app.get("/")
async def root() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "SE-Res-Inception V2",
        "dataset": "CIFAR-100",
        "device": str(device)
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Detailed health check with GPU info."""
    info = {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }
    
    if torch.cuda.is_available():
        info["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024, 2),
            "memory_cached_mb": round(torch.cuda.memory_reserved(0) / 1024 / 1024, 2)
        }
    
    return info


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Perform image classification using the SE-Res-Inception model.
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
        
    Returns:
        JSON response containing:
        - top_5_predictions: List of {class_name, confidence_percent}
        - inference_time_ms: Time taken for the forward pass
        
    Raises:
        HTTPException: If file is not a valid image
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image file."
        )
    
    # Read file contents
    try:
        image_data = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {str(e)}"
        )
    
    if len(image_data) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )
    
    # Preprocess image
    input_tensor = preprocess_image(image_data)
    
    # Move input to GPU
    input_tensor = input_tensor.to(device)
    
    # Perform inference with no gradient computation
    with torch.no_grad():
        # Synchronize GPU before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        # Forward pass
        logits = model(input_tensor)
        
        # Synchronize GPU after forward pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Calculate inference time in milliseconds
        inference_time_ms = (end_time - start_time) * 1000
        
        # Convert logits to probabilities
        probabilities = F.softmax(logits, dim=1)
        
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probabilities, k=5, dim=1)
        
        # Convert to Python lists
        top5_probs = top5_probs.squeeze(0).cpu().tolist()
        top5_indices = top5_indices.squeeze(0).cpu().tolist()
    
    # Build response with class names and confidence percentages
    top_5_predictions = [
        {
            "class_name": CIFAR100_LABELS[idx],
            "confidence_percent": round(prob * 100, 2)
        }
        for idx, prob in zip(top5_indices, top5_probs)
    ]
    
    return JSONResponse(content={
        "top_5_predictions": top_5_predictions,
        "inference_time_ms": round(inference_time_ms, 2)
    })


@app.get("/labels")
async def get_labels() -> Dict[str, Any]:
    """Return the list of all CIFAR-100 class labels."""
    return {
        "num_classes": len(CIFAR100_LABELS),
        "labels": CIFAR100_LABELS
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

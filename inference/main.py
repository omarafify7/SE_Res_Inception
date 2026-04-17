"""
SE-Res-Inception Internal Inference Microservice
=================================================

Internal API for serving image classification predictions using the
SE-Res-Inception model. This service is not exposed publicly; it sits
behind an API gateway.

Endpoints:
    GET  /health           - Health check (Docker healthcheck)
    POST /internal/predict - Accept image upload, return top-5 predictions

Configuration (environment variables):
    CHECKPOINT_PATH  - Path to model weights (default: checkpoints/best_model.pth)
    NUM_CLASSES      - Number of output classes (default: 100)
    DATASET          - Dataset profile for normalization (default: cifar100)
    DEV_MODE         - Set to "1" to enable /docs and /redoc
"""

import io
import os
import time
from contextlib import asynccontextmanager

import torch  # ty: ignore[unresolved-import]
import torch.nn.functional as F  # ty: ignore[unresolved-import]
from PIL import Image  # ty: ignore[unresolved-import]
from fastapi import FastAPI, UploadFile, File, HTTPException  # ty: ignore[unresolved-import]
from pydantic import BaseModel  # ty: ignore[unresolved-import]
from torchvision import transforms  # ty: ignore[unresolved-import]

# Import the model architecture (model.py lives at the project root;
# in Docker the file is copied alongside this module)
from model import SEResInception


# =============================================================================
# Configuration from environment
# =============================================================================
CHECKPOINT_PATH: str = os.environ.get("CHECKPOINT_PATH", "checkpoints/best_model.pth")
NUM_CLASSES: int = int(os.environ.get("NUM_CLASSES", "100"))
DATASET: str = os.environ.get("DATASET", "cifar100").lower()
DEV_MODE: bool = os.environ.get("DEV_MODE", "0") == "1"

# =============================================================================
# Dataset-specific normalization and image size
# =============================================================================
DATASET_CONFIGS: dict = {
    "cifar100": {
        "image_size": 32,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "tiny_imagenet": {
        "image_size": 64,
        "mean": (0.4802, 0.4481, 0.3975),
        "std": (0.2302, 0.2265, 0.2262),
    },
    "coco_crops": {
        "image_size": 128,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}

# =============================================================================
# CIFAR-100 fine-grained labels (100 classes)
# =============================================================================
CIFAR100_LABELS: list[str] = [
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm",
]

# =============================================================================
# Pydantic response models
# =============================================================================

class Prediction(BaseModel):
    class_name: str
    confidence_percent: float


class PredictionResponse(BaseModel):
    top_5_predictions: list[Prediction]
    inference_time_ms: float


# =============================================================================
# Global model state
# =============================================================================
model: SEResInception | None = None
device: torch.device | None = None
preprocess: transforms.Compose | None = None
labels: list[str] = []


def _get_labels(dataset: str, num_classes: int) -> list[str]:
    """Return class labels for the configured dataset."""
    if dataset == "cifar100":
        return CIFAR100_LABELS
    # For other datasets, use generic class names
    return [f"class_{i}" for i in range(num_classes)]


def load_model() -> None:
    """Load the SE-Res-Inception model and weights onto the best available device."""
    global model, device, preprocess, labels

    # Resolve device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA Version: {torch.version.cuda}")
    else:
        print("[WARNING] CUDA not available, falling back to CPU.")

    # Resolve dataset config
    if DATASET not in DATASET_CONFIGS:
        raise RuntimeError(
            f"Unknown DATASET '{DATASET}'. "
            f"Supported: {', '.join(DATASET_CONFIGS.keys())}"
        )

    ds_cfg = DATASET_CONFIGS[DATASET]

    # Initialize model
    model = SEResInception(num_classes=NUM_CLASSES)

    # Load checkpoint
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"[INFO] Loaded model state from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
                print("[INFO] Loaded model state from checkpoint")
            else:
                model.load_state_dict(checkpoint)
                print("[INFO] Loaded model state dict directly")
        else:
            model.load_state_dict(checkpoint)
            print("[INFO] Loaded model state dict")

    except FileNotFoundError:
        raise RuntimeError(f"Checkpoint not found at: {CHECKPOINT_PATH}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    model = model.to(device)
    model.eval()
    print(f"[INFO] Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Build preprocessing pipeline from dataset config
    preprocess = transforms.Compose([
        transforms.Resize((ds_cfg["image_size"], ds_cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(ds_cfg["mean"]), std=list(ds_cfg["std"])),
    ])
    print(f"[INFO] Preprocessing pipeline initialized (dataset={DATASET}, size={ds_cfg['image_size']}x{ds_cfg['image_size']})")

    # Resolve labels
    labels = _get_labels(DATASET, NUM_CLASSES)


# =============================================================================
# Application lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the inference service."""
    print("=" * 60)
    print("SE-Res-Inception Internal Inference Service Starting...")
    print("=" * 60)
    load_model()
    print("=" * 60)
    print("Service ready to accept requests!")
    print("=" * 60)

    yield

    print("[INFO] Shutting down service...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[INFO] Service shutdown complete")


# =============================================================================
# FastAPI application
# =============================================================================

app = FastAPI(
    title="SE-Res-Inception Inference Service",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if DEV_MODE else None,
    redoc_url="/redoc" if DEV_MODE else None,
)


# =============================================================================
# Helper
# =============================================================================

def preprocess_image(image_data: bytes) -> torch.Tensor:
    """Open raw image bytes, apply transforms, return a batched tensor."""
    if preprocess is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        image = Image.open(io.BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = preprocess(image)
        return tensor.unsqueeze(0)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {e}. Please upload a valid image file (JPEG, PNG, etc.)",
        )


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health_check() -> dict:
    """Minimal health check for Docker / orchestrator probes."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.post("/internal/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Run inference on an uploaded image and return top-5 predictions.

    This endpoint is internal-only; the API gateway proxies public
    traffic here.
    """
    # Validate content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image file.",
        )

    # Read file
    try:
        image_data = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    if len(image_data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Preprocess
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    input_tensor = preprocess_image(image_data).to(device)

    # Inference with accurate GPU timing
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        logits = model(input_tensor)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        probabilities = F.softmax(logits, dim=1)
        top5_probs, top5_indices = torch.topk(probabilities, k=5, dim=1)

        top5_probs = top5_probs.squeeze(0).cpu().tolist()
        top5_indices = top5_indices.squeeze(0).cpu().tolist()

    # Build typed response
    top_5 = [
        Prediction(
            class_name=labels[idx] if idx < len(labels) else f"class_{idx}",
            confidence_percent=round(prob * 100, 2),
        )
        for idx, prob in zip(top5_indices, top5_probs)
    ]

    return PredictionResponse(
        top_5_predictions=top_5,
        inference_time_ms=round(inference_time_ms, 2),
    )


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    import uvicorn  # ty: ignore[unresolved-import]

    uvicorn.run(app, host="0.0.0.0", port=8000)

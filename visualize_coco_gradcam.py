"""
COCO-Crops Grad-CAM Visualization
==================================

Generates two visualization artifacts from the trained SE-Res-Inception
COCO checkpoint (checkpoints/coco/best_model.pth):

  1. outputs/coco_gradcam_gallery.png   -- full narrative gallery (8 classes,
                                           original + heatmap per example)
  2. outputs/coco_gradcam_thumbnail.png -- portfolio-ready wide hero image
                                           (6 classes, side-by-side pairs)

Heatmaps come from Grad-CAM hooked into the final ResInceptionBlock
(model.inception9), which captures the post-SE-attention feature maps.

Usage:
    python visualize_coco_gradcam.py
"""

from __future__ import annotations

import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.coco_crops import CocoCropsDataset
from model import SEResInception


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "./checkpoints/coco/best_model.pth"
COCO_ROOT = r"C:\Users\Omar\Documents\datasets\coco2014"
COCO_VAL_IMG_DIR = os.path.join(COCO_ROOT, "images", "val2014")
COCO_VAL_ANN = os.path.join(COCO_ROOT, "annotations", "instances_val2014.json")

OUTPUT_DIR = "./outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 128
NUM_CLASSES = 80
SEED = 42

# ImageNet normalization (used during COCO training)
COCO_MEAN = (0.485, 0.456, 0.406)
COCO_STD = (0.229, 0.224, 0.225)

# Curated categories for the gallery (visually distinctive, variety of object types)
GALLERY_CATEGORIES = ["person", "dog", "cat", "bird", "car", "pizza", "elephant", "umbrella"]

# Tighter selection for the portfolio thumbnail (most visually striking)
THUMBNAIL_CATEGORIES = ["elephant", "dog", "pizza", "person", "cat", "car"]

# How many candidate crops to evaluate per category before picking the best
CANDIDATES_PER_CLASS = 40


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
class GradCAM:
    """Grad-CAM on a given target layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _m, _i, output):
        self.activations = output

    def _save_gradient(self, _m, _gi, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        """Return heatmap (H, W) normalized to [0, 1] and the logits."""
        output = self.model(x)
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        assert self.gradients is not None and self.activations is not None
        grads = self.gradients[0].detach().cpu().numpy()      # (C, H, W)
        acts = self.activations[0].detach().cpu().numpy()     # (C, H, W)

        weights = np.mean(grads, axis=(1, 2))                 # (C,)
        cam = np.tensordot(weights, acts, axes=1)             # (H, W)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        cam = (cam - cam.min()) / (cam.max() + 1e-7)
        return cam, output


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization and convert (C, H, W) tensor -> (H, W, C) image in [0, 1]."""
    mean = np.array(COCO_MEAN)
    std = np.array(COCO_STD)
    img = tensor.detach().permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    return np.clip(img, 0.0, 1.0)


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Overlay a JET heatmap on an RGB image (both float [0, 1])."""
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = np.float32(heatmap_color[:, :, ::-1]) / 255.0  # BGR -> RGB
    blended = heatmap_color * alpha + img * (1 - alpha)
    blended = np.clip(blended, 0.0, 1.0)
    return blended


def select_best_examples(model, dataset, grad_cam, target_categories):
    """
    For each target category, scan CANDIDATES_PER_CLASS random crops with that
    label and pick the one the model classifies most confidently and correctly.

    Returns dict: {category_name: {orig, overlay, pred_class, conf, true_class}}
    """
    # Label index per category name
    name_to_idx = {name: idx for idx, name in enumerate(dataset.categories)}
    results = {}

    # Build per-class index list from the dataset (fast, no disk IO)
    print("Indexing dataset by class...")
    class_to_samples: dict[int, list[int]] = {name_to_idx[n]: [] for n in target_categories}
    for i, (_, _, category_id) in enumerate(dataset.samples):
        label = dataset.label_map[category_id]
        if label in class_to_samples:
            class_to_samples[label].append(i)

    for category in target_categories:
        label = name_to_idx[category]
        idxs = class_to_samples[label]
        if not idxs:
            print(f"  [skip] No crops for '{category}'")
            continue

        random.seed(SEED + label)
        candidates = random.sample(idxs, k=min(CANDIDATES_PER_CLASS, len(idxs)))
        print(f"  Scanning {len(candidates)} candidates for '{category}' (label {label})...")

        best_idx, best_conf = None, -1.0
        for idx in candidates:
            img_tensor, lbl = dataset[idx]
            inp = img_tensor.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(inp)
                probs = F.softmax(logits, dim=1)
                conf, pred = probs.max(dim=1)
            if pred.item() == label and conf.item() > best_conf:
                best_conf = conf.item()
                best_idx = idx

        if best_idx is None:
            print(f"  [skip] No confident correct prediction for '{category}'")
            continue

        # Recompute gradcam for the winner (needs grads, so no no_grad)
        img_tensor, lbl = dataset[best_idx]
        inp = img_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)
        heatmap, logits = grad_cam(inp, label)

        probs = F.softmax(logits, dim=1)
        conf = probs[0, label].item()
        pred_idx = int(logits.argmax(dim=1).item())
        pred_name = dataset.categories[pred_idx]

        orig = denormalize(img_tensor)
        overlay = overlay_heatmap(orig, heatmap, alpha=0.55)

        results[category] = {
            "orig": orig,
            "overlay": overlay,
            "heatmap": heatmap,
            "true_class": category,
            "pred_class": pred_name,
            "conf": conf * 100.0,
        }
        print(f"  -> Picked crop idx {best_idx} with {conf*100:.1f}% confidence")

    return results


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
def render_full_gallery(results: dict, output_path: str):
    """4x2 grid (4 categories x 2 columns original|heatmap per row)."""
    categories = list(results.keys())
    n = len(categories)
    cols = 2  # (original, heatmap) pairs
    rows_per_col = (n + 1) // 2

    fig = plt.figure(figsize=(12, 3.2 * rows_per_col), facecolor="white")
    gs = gridspec.GridSpec(
        rows_per_col, 4,
        width_ratios=[1, 1, 1, 1],
        wspace=0.06, hspace=0.28,
    )

    fig.suptitle(
        "SE-Res-Inception V2 — Grad-CAM on COCO-Crops (128×128)",
        fontsize=15, fontweight="bold", y=0.995,
    )

    for i, category in enumerate(categories):
        r = i // 2
        c = (i % 2) * 2
        data = results[category]

        ax_orig = fig.add_subplot(gs[r, c])
        ax_orig.imshow(data["orig"])
        ax_orig.axis("off")
        ax_orig.set_title(
            f"{category}\n(conf {data['conf']:.1f}%)",
            fontsize=10, color="#2c3e50", fontweight="600",
        )

        ax_cam = fig.add_subplot(gs[r, c + 1])
        ax_cam.imshow(data["overlay"])
        ax_cam.axis("off")
        ax_cam.set_title("Grad-CAM", fontsize=10, color="#c0392b", fontweight="600")

    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


def render_thumbnail(results: dict, output_path: str):
    """
    Wide portfolio hero: 2 rows x 3 columns. Each cell shows original +
    overlay side-by-side, dark aesthetic.
    """
    categories = [c for c in THUMBNAIL_CATEGORIES if c in results][:6]
    if len(categories) < 6:
        # pad with whatever we have
        extras = [c for c in results if c not in categories]
        categories.extend(extras[: 6 - len(categories)])
    categories = categories[:6]

    # Figure: wide aspect ratio (3:1 effective per cell) for portfolio card
    fig = plt.figure(figsize=(18, 8), facecolor="#0f1419")
    gs = gridspec.GridSpec(2, 6, wspace=0.05, hspace=0.18)

    fig.suptitle(
        "SE-Res-Inception V2  ·  channel attention on COCO-Crops",
        fontsize=18, fontweight="bold", color="#eaeaea", y=0.97,
    )

    for i, category in enumerate(categories):
        data = results[category]
        row = i // 3
        col_base = (i % 3) * 2

        ax_orig = fig.add_subplot(gs[row, col_base])
        ax_orig.imshow(data["orig"])
        ax_orig.axis("off")
        ax_orig.set_title(
            f"{category}  ({data['conf']:.0f}%)",
            fontsize=12, color="#eaeaea", fontweight="600", pad=6,
        )

        ax_cam = fig.add_subplot(gs[row, col_base + 1])
        ax_cam.imshow(data["overlay"])
        ax_cam.axis("off")

    # footer caption
    fig.text(
        0.5, 0.03,
        "Grad-CAM heatmaps from the final SE-attention Inception block  ·  8.86 M params  ·  84.23% top-1 on COCO-Crops",
        ha="center", color="#9aa5b1", fontsize=11, style="italic",
    )

    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#0f1419")
    plt.close(fig)
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    model = SEResInception(num_classes=NUM_CLASSES, input_size=IMAGE_SIZE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).eval()
    print(f"Loaded (epoch {ckpt.get('epoch', '?')}, best_acc {ckpt.get('best_acc', 0):.2f}%)")

    # Target layer: final ResInceptionBlock (post-SE-attention features)
    target_layer = model.inception9
    grad_cam = GradCAM(model, target_layer)

    print(f"Loading COCO val dataset: {COCO_VAL_IMG_DIR}")
    dataset = CocoCropsDataset(
        root=COCO_VAL_IMG_DIR,
        ann_file=COCO_VAL_ANN,
        split="val",
        image_size=IMAGE_SIZE,
    )
    print(f"Val crops: {len(dataset):,}")
    print(f"Classes: {dataset.num_classes}")

    # Select best examples for each gallery category (union of gallery + thumbnail)
    target_categories = list(dict.fromkeys(GALLERY_CATEGORIES + THUMBNAIL_CATEGORIES))
    results = select_best_examples(model, dataset, grad_cam, target_categories)

    # Render outputs
    gallery_path = os.path.join(OUTPUT_DIR, "coco_gradcam_gallery.png")
    render_full_gallery(
        {k: v for k, v in results.items() if k in GALLERY_CATEGORIES},
        gallery_path,
    )

    thumbnail_path = os.path.join(OUTPUT_DIR, "coco_gradcam_thumbnail.png")
    render_thumbnail(results, thumbnail_path)

    print("\nDone.")
    print(f"  Gallery:   {gallery_path}")
    print(f"  Thumbnail: {thumbnail_path}")


if __name__ == "__main__":
    main()

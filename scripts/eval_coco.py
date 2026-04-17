"""Evaluate the trained COCO-Crops SE-Res-Inception checkpoint.

Measures top-1, top-5, CPU+GPU inference latency on the COCO 2014 val
bbox-crop set. Emits outputs/coco_results.csv.
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.coco_crops import get_coco_dataloaders
from model import SEResInception


CHECKPOINT = "checkpoints/coco/best_model.pth"
COCO_ROOT = r"C:\Users\Omar\Documents\datasets\coco2014"
OUTPUT_DIR = "outputs"
IMG_SIZE = 128
BATCH_SIZE = 64

LATENCY_TOTAL = 100
LATENCY_WARMUP = 10
LATENCY_COOLDOWN = 10


def evaluate(model, loader, device):
    correct1 = correct5 = total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            _, pred1 = logits.max(dim=1)
            _, pred5 = logits.topk(5, dim=1)
            correct1 += pred1.eq(targets).sum().item()
            correct5 += pred5.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
            total += targets.size(0)
    return correct1 / total * 100.0, correct5 / total * 100.0


def measure_latency(model, device, device_type):
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    times = []
    with torch.no_grad():
        for _ in range(LATENCY_TOTAL):
            if device_type == "gpu":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device_type == "gpu":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    mid = times[LATENCY_WARMUP : LATENCY_TOTAL - LATENCY_COOLDOWN]
    return float(np.mean(mid))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SEResInception(num_classes=80, input_size=IMG_SIZE)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded {CHECKPOINT} (epoch {ckpt.get('epoch', '?')}, params {params_m:.2f}M)")

    _, val_loader, _, _ = get_coco_dataloaders(
        root=COCO_ROOT, image_size=IMG_SIZE, batch_size=BATCH_SIZE, num_workers=12
    )

    model = model.to(device).eval()
    print("Computing top-1 / top-5 accuracy on COCO val bbox-crops...")
    top1, top5 = evaluate(model, val_loader, device)
    print(f"  Top-1: {top1:.2f}%  |  Top-5: {top5:.2f}%")

    print("Measuring GPU inference latency...")
    gpu_ms = measure_latency(model.to(device), device, "gpu")

    print("Measuring CPU inference latency...")
    cpu_model = model.to("cpu").eval()
    cpu_ms = measure_latency(cpu_model, torch.device("cpu"), "cpu")

    print(f"  CPU: {cpu_ms:.2f} ms   |   GPU: {gpu_ms:.2f} ms")

    csv_path = os.path.join(OUTPUT_DIR, "coco_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "input_size", "num_classes", "params_M",
                    "top1_acc", "top5_acc", "inference_ms_cpu", "inference_ms_gpu", "epochs"])
        w.writerow(["SE-Res-Inception", "coco_crops", IMG_SIZE, 80,
                    round(params_m, 2), round(top1, 2), round(top5, 2),
                    round(cpu_ms, 2), round(gpu_ms, 2), ckpt.get("epoch", 50)])
    print(f"Saved {csv_path}")

    print("\n" + "=" * 72)
    print(f"{'Model':<22}{'Dataset':<14}{'Top-1':>8}{'Top-5':>8}"
          f"{'Params':>10}{'CPU ms':>10}{'GPU ms':>10}")
    print("-" * 72)
    print(f"{'SE-Res-Inception':<22}{'coco_crops':<14}{top1:>7.2f}%{top5:>7.2f}%"
          f"{params_m:>9.2f}M{cpu_ms:>10.2f}{gpu_ms:>10.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()

"""
Baseline Comparison Script
===========================

Loads trained model checkpoints (SE-Res-Inception + baselines), evaluates
each on the CIFAR-100 test set, measures inference latency, and produces
a comparison CSV and grouped bar chart.

Usage:
    python -m baselines.plot_comparison
"""

import os
import sys
import csv
import time

import numpy as np  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
import torch.nn as nn  # ty: ignore[unresolved-import]
import torchvision  # ty: ignore[unresolved-import]
import torchvision.transforms as transforms  # ty: ignore[unresolved-import]
import torchvision.models as models  # ty: ignore[unresolved-import]
import matplotlib  # ty: ignore[unresolved-import]
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SEResInception  # ty: ignore[unresolved-import]


# ============================================================================
# CONSTANTS
# ============================================================================
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
NUM_CLASSES = 100
IMG_SIZE = 32
BATCH_SIZE = 100

# Checkpoint paths
OUR_MODEL_PATH = os.path.join("checkpoints", "best_model.pth")
BASELINE_DIR = os.path.join("checkpoints", "baselines")
OUTPUT_DIR = "outputs"

# Latency measurement config
LATENCY_TOTAL_PASSES = 100
LATENCY_WARMUP = 10
LATENCY_COOLDOWN = 10


# ============================================================================
# MODEL BUILDERS
# ============================================================================
def load_our_model(device):
    """Load the SE-Res-Inception model from checkpoint."""
    model = SEResInception(num_classes=NUM_CLASSES)
    checkpoint_path = OUR_MODEL_PATH

    if not os.path.exists(checkpoint_path):
        # Try cifar100 subdirectory
        checkpoint_path = os.path.join("checkpoints", "cifar100", "best_model.pth")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"Loaded SE-Res-Inception from {checkpoint_path}")
    else:
        print(f"WARNING: No checkpoint found at {OUR_MODEL_PATH}, using random weights")

    model = model.to(device)
    model.eval()
    return model


def load_baseline_resnet18(device):
    """Load ResNet-18 baseline from checkpoint."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, NUM_CLASSES)

    checkpoint_path = os.path.join(BASELINE_DIR, "resnet18_best.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"Loaded ResNet-18 from {checkpoint_path}")
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}, using random weights")

    model = model.to(device)
    model.eval()
    return model


def load_baseline_googlenet(device):
    """Load GoogLeNet baseline from checkpoint."""
    model = models.googlenet(weights=None, aux_logits=False)
    model.fc = nn.Linear(1024, NUM_CLASSES)

    checkpoint_path = os.path.join(BASELINE_DIR, "googlenet_best.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # Filter out aux classifier keys if present (trained with aux_logits=True)
        filtered_state = {
            k: v for k, v in state_dict.items()
            if not k.startswith("aux1.") and not k.startswith("aux2.")
        }
        model.load_state_dict(filtered_state, strict=False)
        print(f"Loaded GoogLeNet from {checkpoint_path}")
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}, using random weights")

    model = model.to(device)
    model.eval()
    return model


# ============================================================================
# EVALUATION
# ============================================================================
def get_test_loader():
    """Get CIFAR-100 test set DataLoader."""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    test_set = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return test_loader


def evaluate_accuracy(model, test_loader, device):
    """Compute top-1 and top-5 accuracy on the test set."""
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            # Handle GoogLeNet named-tuple output defensively
            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            # Top-1
            _, pred_top1 = outputs.max(dim=1)
            correct_top1 += pred_top1.eq(targets).sum().item()

            # Top-5
            _, pred_top5 = outputs.topk(5, dim=1, largest=True, sorted=True)
            correct_top5 += pred_top5.eq(targets.unsqueeze(1)).any(dim=1).sum().item()

            total += targets.size(0)

    top1 = correct_top1 / total * 100.0
    top5 = correct_top5 / total * 100.0
    return top1, top5


def count_parameters(model):
    """Count total trainable parameters in millions."""
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


def measure_latency(model, device, device_type: str):
    """
    Measure inference latency (ms) with a single 32x32 input.

    Runs LATENCY_TOTAL_PASSES forward passes and averages the middle
    portion (dropping first LATENCY_WARMUP and last LATENCY_COOLDOWN).
    """
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    timings = []

    with torch.no_grad():
        for _ in range(LATENCY_TOTAL_PASSES):
            if device_type == "gpu":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(dummy)

            if device_type == "gpu":
                torch.cuda.synchronize()

            elapsed = (time.perf_counter() - start) * 1000.0  # ms
            timings.append(elapsed)

    # Drop warmup and cooldown
    measured = timings[LATENCY_WARMUP: LATENCY_TOTAL_PASSES - LATENCY_COOLDOWN]
    return float(np.mean(measured))


# ============================================================================
# PLOTTING
# ============================================================================
def plot_comparison(results: list[dict], output_path: str):
    """
    Create a grouped bar chart with 3 subplots:
    (1) Top-1 Accuracy, (2) Parameters (M), (3) Inference Latency (ms).

    Our model is highlighted with a distinct color.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    model_names = [r["model"] for r in results]
    top1_accs = [r["top1_acc"] for r in results]
    params = [r["params_M"] for r in results]

    # Use GPU latency if available, else CPU
    has_gpu = any(r["inference_ms_gpu"] > 0 for r in results)
    if has_gpu:
        latencies = [r["inference_ms_gpu"] for r in results]
        latency_label = "Inference Latency (ms, GPU)"
    else:
        latencies = [r["inference_ms_cpu"] for r in results]
        latency_label = "Inference Latency (ms, CPU)"

    # Colors: our model gets a distinct color
    colors = []
    for name in model_names:
        if "SE-Res-Inception" in name:
            colors.append("#2196F3")  # Blue for our model
        else:
            colors.append("#90A4AE")  # Gray for baselines

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Subplot 1: Top-1 Accuracy
    bars1 = axes[0].bar(model_names, top1_accs, color=colors, edgecolor="white", width=0.6)
    axes[0].set_title("Top-1 Accuracy (%)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    for bar, val in zip(bars1, top1_accs):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    # Subplot 2: Parameters (M)
    bars2 = axes[1].bar(model_names, params, color=colors, edgecolor="white", width=0.6)
    axes[1].set_title("Parameters (M)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Millions")
    for bar, val in zip(bars2, params):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{val:.1f}M", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    # Subplot 3: Inference Latency
    bars3 = axes[2].bar(model_names, latencies, color=colors, edgecolor="white", width=0.6)
    axes[2].set_title(latency_label, fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Milliseconds")
    for bar, val in zip(bars3, latencies):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    # Rotate x labels for readability
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "SE-Res-Inception vs. Baselines on CIFAR-100",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison chart to {output_path}")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Device
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"Device: {device}")

    # Test data
    test_loader = get_test_loader()

    # Define models to evaluate
    model_configs = [
        ("SE-Res-Inception", load_our_model),
        ("ResNet-18", load_baseline_resnet18),
        ("GoogLeNet", load_baseline_googlenet),
    ]

    results = []

    for name, loader_fn in model_configs:
        print(f"\nEvaluating {name}...")
        model = loader_fn(device)

        # Parameter count
        params_m = count_parameters(model)
        print(f"  Parameters: {params_m:.2f}M")

        # Accuracy
        top1, top5 = evaluate_accuracy(model, test_loader, device)
        print(f"  Top-1 Accuracy: {top1:.2f}%")
        print(f"  Top-5 Accuracy: {top5:.2f}%")

        # Latency -- CPU
        model_cpu = model.to("cpu")
        model_cpu.eval()
        latency_cpu = measure_latency(model_cpu, torch.device("cpu"), "cpu")
        print(f"  Inference Latency (CPU): {latency_cpu:.2f} ms")

        # Latency -- GPU (if available)
        latency_gpu = 0.0
        if use_gpu:
            model_gpu = model.to(device)
            model_gpu.eval()
            latency_gpu = measure_latency(model_gpu, device, "gpu")
            print(f"  Inference Latency (GPU): {latency_gpu:.2f} ms")

        results.append({
            "model": name,
            "params_M": round(params_m, 2),
            "top1_acc": round(top1, 2),
            "top5_acc": round(top5, 2),
            "inference_ms_cpu": round(latency_cpu, 2),
            "inference_ms_gpu": round(latency_gpu, 2),
        })

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "baseline_comparison.csv")
    fieldnames = ["model", "params_M", "top1_acc", "top5_acc", "inference_ms_cpu", "inference_ms_gpu"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved comparison CSV to {csv_path}")

    # Plot
    chart_path = os.path.join(OUTPUT_DIR, "baseline_comparison.png")
    plot_comparison(results, chart_path)

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Model':<22} {'Params(M)':>10} {'Top-1':>8} {'Top-5':>8} {'CPU(ms)':>10} {'GPU(ms)':>10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['model']:<22} {r['params_M']:>10.2f} {r['top1_acc']:>7.2f}% "
            f"{r['top5_acc']:>7.2f}% {r['inference_ms_cpu']:>10.2f} {r['inference_ms_gpu']:>10.2f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()

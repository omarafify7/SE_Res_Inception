"""
SE-Res-Inception Training Curves Plotter
=========================================

Loads metrics from checkpoints and plots training/validation curves.
Run this after training (or during training to see progress).

Usage:
    python plot_metrics.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(metrics_path: str = "./checkpoints/metrics.npz"):
    """
    Load and plot training curves from saved metrics.
    
    Args:
        metrics_path: Path to the metrics.npz file
    """
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file not found at {metrics_path}")
        print("Run train.py first to generate metrics.")
        return
    
    # Load metrics
    data = np.load(metrics_path)
    epochs = data['epochs']
    train_loss = data['train_loss']
    train_acc = data['train_acc']
    val_loss = data['val_loss']
    val_acc = data['val_acc']
    
    print(f"Loaded metrics for {len(epochs)} epochs")
    print(f"Best Val Accuracy: {val_acc.max():.2f}% (Epoch {epochs[val_acc.argmax()]})")
    print(f"Final Val Accuracy: {val_acc[-1]:.2f}%")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('SE-Res-Inception Training Curves', fontsize=14, fontweight='bold')
    
    # Plot Loss
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=1)
    
    # Plot Accuracy
    ax2.plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_acc, 'r-', label='Val Accuracy', linewidth=2)
    ax2.axhline(y=val_acc.max(), color='g', linestyle='--', alpha=0.5, 
                label=f'Best Val: {val_acc.max():.2f}%')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=1)
    
    plt.tight_layout()
    
    # Save figure
    output_path = "./checkpoints/training_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")
    
    # Show plot
    plt.show()


if __name__ == "__main__":
    plot_training_curves()

# Tiny ImageNet Visualization Suite

This folder contains scripts to analyze your trained SE-Res-Inception model on Tiny ImageNet.

## Quick Start

Run all visualizations:

```bash
conda activate se_res_inception
python tiny_imagenet_visualizations/run_all.py
```

Or run individual scripts from the project root:

```bash
python tiny_imagenet_visualizations/confusion_matrix.py
python tiny_imagenet_visualizations/per_class_accuracy.py
python tiny_imagenet_visualizations/top_errors.py
python tiny_imagenet_visualizations/grad_cam.py
python tiny_imagenet_visualizations/tsne_embeddings.py
```

## Outputs

All outputs are saved to `./outputs/`:

| File                     | Description                         |
| ------------------------ | ----------------------------------- |
| `confusion_matrix.png`   | 200x200 heatmap of class confusions |
| `per_class_accuracy.png` | Best/worst classes breakdown        |
| `top_errors_gallery.png` | Most confident wrong predictions    |
| `grad_cam_gallery.png`   | Where the model "looks" on images   |
| `embedding_tsne.png`     | 2D feature space visualization      |

## Interpretation Guide

### Confusion Matrix

- **Bright diagonal** = good per-class accuracy
- **Off-diagonal clusters** = systematic confusion (similar classes)
- **Dark rows** = low recall (model misses that class)

### Per-Class Accuracy

- **Bottom 20** = candidates for targeted improvement
- **Top 20** = strong feature learning
- **High variance** = model specializes in certain features

### Top Errors

- Look for patterns: are errors random or systematic?
- High-confidence errors suggest learned spurious correlations
- May reveal dataset labeling issues

### Grad-CAM

- **Object-focused attention** = good feature learning
- **Background-focused** = possible dataset bias
- Compare correct vs incorrect predictions

### t-SNE/UMAP

- **Tight clusters** = discriminative features
- **Overlapping clusters** = confused classes
- **Semantic groupings** = hierarchical learning

## Dependencies

Core (should already be installed):

- torch, torchvision, numpy, matplotlib, tqdm

Optional:

```bash
pip install umap-learn opencv-python seaborn scikit-learn
```

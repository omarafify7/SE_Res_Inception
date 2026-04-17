"""Tests for CocoCropsDataset (COCO 2014 bounding-box crops)."""

import os
import random

import pytest  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]

COCO_ROOT = r"C:\Users\Omar\Documents\datasets\coco2014"
COCO_TRAIN_ANN = os.path.join(COCO_ROOT, "annotations", "instances_train2014.json")
COCO_TRAIN_IMGS = os.path.join(COCO_ROOT, "images", "train2014")

coco_available = os.path.isdir(COCO_ROOT) and os.path.isfile(COCO_TRAIN_ANN)


@pytest.mark.skipif(not coco_available, reason="COCO 2014 data not found")
def test_coco_crops_basic():
    """Test CocoCropsDataset on a small subset."""
    from datasets.coco_crops import CocoCropsDataset

    dataset = CocoCropsDataset(
        root=COCO_TRAIN_IMGS,
        ann_file=COCO_TRAIN_ANN,
        split="train",
        image_size=128,
    )

    # dataset should contain annotations
    assert len(dataset) > 0, "Dataset is empty"

    # 80 COCO categories
    assert dataset.num_classes == 80

    # Sample 100 random items spread across the dataset (COCO annotations
    # are grouped by image, so sequential sampling hits few categories).
    random.seed(42)
    n = min(100, len(dataset))
    indices = random.sample(range(len(dataset)), n)

    labels_seen = set()
    for i in indices:
        img, label = dataset[i]

        # correct tensor shape: (3, 128, 128)
        assert img.shape == (3, 128, 128), f"Wrong shape at index {i}: {img.shape}"
        assert isinstance(img, torch.Tensor)

        # label in valid range
        assert 0 <= label <= 79, f"Label out of range at index {i}: {label}"

        labels_seen.add(label)

    # at least 5 distinct classes in 100 randomly sampled crops
    assert len(labels_seen) >= 5, (
        f"Expected >= 5 unique labels in {n} random samples, got {len(labels_seen)}"
    )

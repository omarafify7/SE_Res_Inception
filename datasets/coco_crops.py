"""COCO 2014 bounding-box crops dataset for 80-class image classification."""

import os

from torch.utils.data import Dataset, DataLoader  # ty: ignore[unresolved-import]
from torchvision import transforms  # ty: ignore[unresolved-import]
from PIL import Image  # ty: ignore[unresolved-import]
from pycocotools.coco import COCO  # ty: ignore[unresolved-import]


class CocoCropsDataset(Dataset):
    """Dataset that extracts bounding-box crops from COCO 2014 images,
    reformulating object detection as an 80-class classification task.

    Each sample is a single object crop resized to ``image_size x image_size``.
    """

    def __init__(
        self,
        root,
        ann_file,
        split="train",
        min_area=1024,
        pad=0.1,
        image_size=128,
        transform=None,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.min_area = min_area
        self.pad = pad
        self.image_size = image_size
        self.transform = transform

        # ---- load COCO annotations ----
        self.coco = COCO(ann_file)

        # ---- build contiguous label map (category_id -> 0..79) ----
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats = sorted(cats, key=lambda c: c["id"])  # deterministic order
        self.label_map = {cat["id"]: idx for idx, cat in enumerate(cats)}
        self.categories = [cat["name"] for cat in cats]
        self.num_classes = len(self.categories)  # 80

        # ---- collect valid annotations ----
        self.samples = []  # list of (image_id, bbox, category_id)
        all_ann_ids = self.coco.getAnnIds()
        all_anns = self.coco.loadAnns(all_ann_ids)

        for ann in all_anns:
            # skip crowd annotations
            if ann.get("iscrowd", 0) == 1:
                continue
            # skip tiny boxes
            if ann["area"] < self.min_area:
                continue
            x, y, w, h = ann["bbox"]
            if w < 16 or h < 16:
                continue
            self.samples.append((ann["image_id"], ann["bbox"], ann["category_id"]))

        # ---- default transform ----
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    # -----------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # -----------------------------------------------------------------
    def __getitem__(self, idx):
        image_id, bbox, category_id = self.samples[idx]

        # load image
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        # unpack bbox and apply padding
        x, y, w, h = bbox
        pad_x = self.pad * w
        pad_y = self.pad * h

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)

        crop = img.crop((x1, y1, x2, y2))

        # apply transform (transform pipeline already includes Resize)
        tensor = self.transform(crop)

        label = self.label_map[category_id]
        return tensor, label


# =====================================================================
# Helper: ready-made DataLoaders
# =====================================================================

def get_coco_dataloaders(root, image_size=128, batch_size=64, num_workers=4):
    """Create train and val DataLoaders for COCO crops.

    Parameters
    ----------
    root : str
        Path to the COCO 2014 root directory (contains ``annotations/``
        and ``images/``).
    image_size : int
        Target spatial size for each crop.
    batch_size : int
        Mini-batch size.
    num_workers : int
        DataLoader worker processes.

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    num_classes : int  (80)
    class_names : list[str]
    """

    train_ann = os.path.join(root, "annotations", "instances_train2014.json")
    val_ann = os.path.join(root, "annotations", "instances_val2014.json")
    train_img_dir = os.path.join(root, "images", "train2014")
    val_img_dir = os.path.join(root, "images", "val2014")

    # ---- training augmentation ----
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=image_size // 8),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # ---- validation: plain resize + normalize ----
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = CocoCropsDataset(
        root=train_img_dir,
        ann_file=train_ann,
        split="train",
        image_size=image_size,
        transform=train_transform,
    )

    val_dataset = CocoCropsDataset(
        root=val_img_dir,
        ann_file=val_ann,
        split="val",
        image_size=image_size,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loader, val_loader, train_dataset.num_classes, train_dataset.categories

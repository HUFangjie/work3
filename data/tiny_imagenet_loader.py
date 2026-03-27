# data/tiny_imagenet_loader.py
"""
Tiny-ImageNet-200 dataset loader.

Expected directory layout (recommended):
    <data_root>/tiny-imagenet-200/
        train/<wnid>/images/*.JPEG
        val/images/*.JPEG
        val/val_annotations.txt

Notes:
- Tiny-ImageNet does not provide labels for a separate test set. In practice,
  the official validation split is commonly used as the "test" set.
- Some community copies reorganize val into class subfolders. We support both:
    (A) val/<wnid>/images/*.JPEG (ImageFolder)
    (B) val/images + val_annotations.txt (custom dataset)
"""

from __future__ import annotations

import os
from typing import Dict, Tuple, List, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def get_tiny_imagenet_transforms(image_size: int = 64) -> Tuple[transforms.Compose, transforms.Compose]:
    """Standard Tiny-ImageNet transforms (ImageNet normalization)."""
    train_tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )
    test_tfm = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )
    return train_tfm, test_tfm


class TinyImageNetVal(Dataset):
    """Validation split loader using val_annotations.txt."""

    def __init__(
        self,
        images_dir: str,
        annotations_path: str,
        class_to_idx: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.transform = transform
        self.class_to_idx = dict(class_to_idx)

        # Parse annotations: <img>	<wnid>	...
        samples: List[Tuple[str, int]] = []
        with open(annotations_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img_name, wnid = parts[0], parts[1]
                if wnid not in self.class_to_idx:
                    # Skip unknown classes (should not happen with official data)
                    continue
                label = int(self.class_to_idx[wnid])
                img_path = os.path.join(images_dir, img_name)
                if os.path.isfile(img_path):
                    samples.append((img_path, label))

        if len(samples) == 0:
            raise RuntimeError(
                f"TinyImageNetVal found 0 samples. Check: {images_dir} and {annotations_path}"
            )

        self.samples = samples
        # for DataManager label extraction
        self.targets = [y for _, y in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _resolve_root(data_root: str) -> str:
    candidates = [
        os.path.join(data_root, "tiny-imagenet-200"),
        os.path.join(data_root, "tiny_imagenet_200"),
        data_root,
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "train")) and os.path.isdir(os.path.join(c, "val")):
            return c
    # fallback to first candidate
    return candidates[0]


def get_tiny_imagenet_datasets(
    data_root: str,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
) -> Tuple[Dataset, Dataset]:
    """
    Returns:
      train_dataset: ImageFolder over train/<wnid>/images
      test_dataset:  validation split (either ImageFolder over val/<wnid>/images, or custom val loader)

    """
    root = _resolve_root(data_root)
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Tiny-ImageNet train dir not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Tiny-ImageNet val dir not found: {val_dir}")

    # Train split: ImageFolder works (classes are wnids)
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)

    # Validation split: try ImageFolder if val is organized into class subfolders
    try_val_as_imagefolder = any(
        os.path.isdir(os.path.join(val_dir, d)) for d in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, d)) and d not in {"images"}
    )
    if try_val_as_imagefolder:
        val_dataset = datasets.ImageFolder(root=val_dir, transform=test_transform)
        return train_dataset, val_dataset

    # Otherwise, use val/images + val_annotations.txt
    images_dir = os.path.join(val_dir, "images")
    ann_path = os.path.join(val_dir, "val_annotations.txt")
    if os.path.isdir(images_dir) and os.path.isfile(ann_path):
        val_dataset = TinyImageNetVal(
            images_dir=images_dir,
            annotations_path=ann_path,
            class_to_idx=train_dataset.class_to_idx,
            transform=test_transform,
        )
        return train_dataset, val_dataset

    raise FileNotFoundError(
        "Tiny-ImageNet validation structure not recognized. Expected either: "
        "(A) val/<wnid>/images/*.JPEG or (B) val/images + val/val_annotations.txt. "
        f"Got val_dir={val_dir}"
    )

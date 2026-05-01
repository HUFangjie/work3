from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def _assert_exists(path: Path, msg: str) -> None:
    if not path.exists():
        raise FileNotFoundError(msg)


def get_dataset(dataset: str, data_root: str) -> Tuple[Dataset, int, int, int]:
    root = Path(data_root)
    name = dataset.lower()

    if name == "cifar10":
        _assert_exists(root / "cifar-10-batches-py", "Dataset not found. Please place CIFAR-10 under analysis/data/cifar-10-batches-py/.")
        tfm = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        ds = datasets.CIFAR10(root=str(root), train=True, download=False, transform=tfm)
        return ds, 10, 3, 32

    if name == "mnist":
        _assert_exists(root / "MNIST", "Dataset not found. Please place MNIST under analysis/data/MNIST/.")
        tfm = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        ds = datasets.MNIST(root=str(root), train=True, download=False, transform=tfm)
        return ds, 10, 1, 32

    if name == "tinyimagenet":
        tiny_root = root / "tiny-imagenet-200" / "train"
        _assert_exists(tiny_root, "Dataset not found. Please place Tiny-ImageNet under analysis/data/tiny-imagenet-200/.")
        tfm = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        ds = datasets.ImageFolder(root=str(tiny_root), transform=tfm)
        return ds, 200, 3, 64

    raise ValueError(f"Unsupported dataset: {dataset}")

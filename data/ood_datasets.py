# data/ood_datasets.py
"""
OOD dataset support for FD robustness evaluation.

Typical choices:
  - CIFAR-10 in-distribution -> SVHN as OOD
  - MNIST/Fashion-MNIST in-distribution -> the other as OOD
"""

from __future__ import annotations

from typing import Optional

import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_ood_loader(
    in_distribution_dataset: str,
    root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = False,
) -> DataLoader:
    """
    Construct an OOD DataLoader matched to the in-distribution dataset.

    Args:
        in_distribution_dataset: "cifar10", "fmnist", "mnist", etc.
        root: root directory for torchvision datasets.
        batch_size: batch size.
        num_workers: dataloader workers.
        shuffle: whether to shuffle.

    Returns:
        DataLoader for an OOD dataset.
    """
    name = in_distribution_dataset.lower()

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    if name == "cifar10":
        # Use SVHN (test split) as OOD
        ds = datasets.SVHN(
            root=os.path.join(root, "svhn"),
            split="test",
            download=True,
            transform=tfm,
        )
    elif name in ["fmnist", "fashion_mnist"]:
        # Use MNIST as OOD (if you view Fashion-MNIST as ID)
        ds = datasets.MNIST(
            root=os.path.join(root, "mnist"),
            train=False,
            download=True,
            transform=tfm,
        )
    elif name == "mnist":
        # Use Fashion-MNIST as OOD
        ds = datasets.FashionMNIST(
            root=os.path.join(root, "fmnist"),
            train=False,
            download=True,
            transform=tfm,
        )
    else:
        raise ValueError(f"Unsupported in-distribution dataset for OOD loader: {name}")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader

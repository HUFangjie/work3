# data/mnist_loader.py
"""
MNIST dataset loader.

Interface:
    get_mnist_datasets(data_root, train_transform, test_transform)
      -> (train_dataset, test_dataset)
"""

from typing import Tuple

from torchvision import datasets
from torchvision.transforms import Compose


def get_mnist_datasets(
    data_root: str,
    train_transform: Compose,
    test_transform: Compose,
) -> Tuple[datasets.MNIST, datasets.MNIST]:
    """Load MNIST train/test datasets."""
    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    return train_dataset, test_dataset

# data/cifar10_loader.py
"""
CIFAR-10 dataset loader.

Interface:
    get_cifar10_datasets(data_root, train_transform, test_transform)
      -> (train_dataset, test_dataset)
"""

from typing import Tuple

from torchvision import datasets
from torchvision.transforms import Compose


def get_cifar10_datasets(
    data_root: str,
    train_transform: Compose,
    test_transform: Compose,
) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """
    Load CIFAR-10 train and test datasets.

    Args:
        data_root: Path to store/download data.
        train_transform: Transform applied to training images.
        test_transform: Transform applied to test images.

    Returns:
        (train_dataset, test_dataset)
    """
    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    return train_dataset, test_dataset

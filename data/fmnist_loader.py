# data/fmnist_loader.py
"""
Fashion-MNIST dataset loader.

Interface:
    get_fmnist_datasets(data_root, train_transform, test_transform)
      -> (train_dataset, test_dataset)
"""

from typing import Tuple

from torchvision import datasets
from torchvision.transforms import Compose


def get_fmnist_datasets(
    data_root: str,
    train_transform: Compose,
    test_transform: Compose,
) -> Tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    """
    Load Fashion-MNIST train and test datasets.

    Args:
        data_root: Path to store/download data.
        train_transform: Transform applied to training images.
        test_transform: Transform applied to test images.

    Returns:
        (train_dataset, test_dataset)
    """
    train_dataset = datasets.FashionMNIST(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    return train_dataset, test_dataset

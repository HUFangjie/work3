# data/pathmnist_loader.py

from __future__ import annotations

from typing import Any, Tuple


def get_pathmnist_datasets(
    data_root: str,
    train_transform: Any,
    test_transform: Any,
) -> Tuple[object, object]:
    """Return (train_dataset, test_dataset) for PathMNIST (MedMNIST).

    Automatic download is handled by MedMNIST (download=True).
    """
    try:
        from medmnist import PathMNIST
    except Exception as e:
        raise ImportError(
            "PathMNIST requires the `medmnist` package. Install via: pip install medmnist"
        ) from e

    train_dataset = PathMNIST(
        split="train",
        root=data_root,
        transform=train_transform,
        download=True,
    )
    test_dataset = PathMNIST(
        split="test",
        root=data_root,
        transform=test_transform,
        download=True,
    )
    return train_dataset, test_dataset

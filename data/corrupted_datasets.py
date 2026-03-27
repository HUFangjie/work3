# data/corrupted_datasets.py
"""
Corrupted datasets support for robustness evaluation.

This module provides two modes of corrupted data:

1. Offline corruption using CIFAR-10-C / MNIST-C style `.npy` files
   (if they are available on disk).

2. On-the-fly corruption that applies simple perturbations to the
   original test set (Fashion-MNIST, CIFAR-10, MNIST), so that we
   can evaluate corruption robustness even when the C-variants are
   not downloaded.

We intentionally only support **three** corruption types for the
on-the-fly mode to keep things simple and controlled:

    - "gaussian_noise"
    - "brightness"
    - "blur"

These are enough to stress-test robustness without relying on the
full CIFAR10-C / MNIST-C suite.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. CIFAR-10-C style dataset (if files are present)
# ---------------------------------------------------------------------------


class CIFAR10C(Dataset):
    """
    CIFAR-10-C style corruption dataset.

    Expected directory layout:

        root/
          cifar10_c/
            gaussian_noise.npy
            brightness.npy
            blur.npy
            ...
            labels.npy

    Each corruption .npy file is of shape [5*N, 32, 32, 3],
    where severity ∈ {1,...,5} selects a contiguous chunk
    of length N. labels.npy is of shape [N] or [5*N].
    """

    def __init__(
        self,
        root: str,
        corruption_type: str,
        severity: int,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        if severity < 1 or severity > 5:
            raise ValueError(f"severity must be in [1,5], got {severity}")

        data_dir = os.path.join(root, "cifar10_c")
        data_path = os.path.join(data_dir, f"{corruption_type}.npy")
        labels_path = os.path.join(data_dir, "labels.npy")

        if not os.path.isfile(data_path):
            raise FileNotFoundError(
                f"CIFAR10-C corruption file not found: {data_path}"
            )
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(
                f"CIFAR10-C labels file not found: {labels_path}"
            )

        data = np.load(data_path)  # [5*N, H, W, C]
        labels = np.load(labels_path)

        # Handle labels shape: [N] or [5*N]
        if labels.shape[0] * 5 == data.shape[0]:
            # labels are for severity-aggregated data; tile them
            labels = np.tile(labels, 5)
        elif labels.shape[0] != data.shape[0]:
            raise ValueError(
                f"Unexpected labels shape in CIFAR10-C: "
                f"data={data.shape}, labels={labels.shape}"
            )

        # Slice the block corresponding to this severity
        num_total = data.shape[0]
        if num_total % 5 != 0:
            raise ValueError(
                f"CIFAR10-C data first dimension should be 5*N, got {num_total}"
            )
        n = num_total // 5
        start = (severity - 1) * n
        end = severity * n

        self.data = data[start:end]
        self.targets = labels[start:end]
        self.transform = transform

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        img = self.data[idx]  # [H, W, C], uint8
        target = int(self.targets[idx])

        # HWC -> CHW, float in [0,1]
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float() / 255.0

        if self.transform is not None:
            img = self.transform(img)

        return img, target


# ---------------------------------------------------------------------------
# 2. MNIST-C style dataset (if files are present)
# ---------------------------------------------------------------------------


class MNISTC(Dataset):
    """
    MNIST-C style corruption dataset.

    Expected directory layout:

        root/
          mnist_c/
            gaussian_noise.npy
            brightness.npy
            blur.npy
            ...
            labels.npy

    Each corruption .npy file is of shape [5*N, H, W] or [5*N, H, W, 1].
    """

    def __init__(
        self,
        root: str,
        corruption_type: str,
        severity: int,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        if severity < 1 or severity > 5:
            raise ValueError(f"severity must be in [1,5], got {severity}")

        data_dir = os.path.join(root, "mnist_c")
        data_path = os.path.join(data_dir, f"{corruption_type}.npy")
        labels_path = os.path.join(data_dir, "labels.npy")

        if not os.path.isfile(data_path):
            raise FileNotFoundError(
                f"MNIST-C corruption file not found: {data_path}"
            )
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(
                f"MNIST-C labels file not found: {labels_path}"
            )

        data = np.load(data_path)  # [5*N, H, W] or [5*N, H, W, 1]
        labels = np.load(labels_path)

        # Handle labels shape: [N] or [5*N]
        if labels.shape[0] * 5 == data.shape[0]:
            labels = np.tile(labels, 5)
        elif labels.shape[0] != data.shape[0]:
            raise ValueError(
                f"Unexpected labels shape in MNIST-C: "
                f"data={data.shape}, labels={labels.shape}"
            )

        num_total = data.shape[0]
        if num_total % 5 != 0:
            raise ValueError(
                f"MNIST-C data first dimension should be 5*N, got {num_total}"
            )
        n = num_total // 5
        start = (severity - 1) * n
        end = severity * n

        self.data = data[start:end]
        self.targets = labels[start:end]
        self.transform = transform

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        img = self.data[idx]  # [H, W] or [H, W, 1], uint8
        target = int(self.targets[idx])

        if img.ndim == 2:
            # [H, W] -> [1, H, W]
            img = img[None, ...]
        elif img.ndim == 3:
            # [H, W, 1] -> [1, H, W]
            img = np.transpose(img, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected MNIST-C image shape: {img.shape}")

        img = torch.from_numpy(img).float() / 255.0

        if self.transform is not None:
            img = self.transform(img)

        return img, target


# ---------------------------------------------------------------------------
# 3. On-the-fly corruption wrapper
# ---------------------------------------------------------------------------


_SUPPORTED_ONLINE_CORRUPTIONS = {"gaussian_noise", "brightness", "blur"}


class OnTheFlyCorruptedDataset(Dataset):
    """
    Wrap a base dataset and apply simple corruptions on the fly.

    This is used in two cases:
      - For Fashion-MNIST, where no official FMNIST-C exists, so
        we corrupt the Fashion-MNIST test set directly.
      - As a fallback for CIFAR-10 or MNIST when the C-variant
        .npy files are not present.

    We only support a small set of corruption_type values:
        {"gaussian_noise", "brightness", "blur"}.
    """

    def __init__(self, base_dataset: Dataset, corruption_type: str, severity: int):
        super().__init__()
        if severity < 1 or severity > 5:
            raise ValueError(f"severity must be in [1,5], got {severity}")
        if corruption_type not in _SUPPORTED_ONLINE_CORRUPTIONS:
            raise ValueError(
                f"Unsupported corruption_type for on-the-fly corruption: "
                f"{corruption_type}. Supported: {_SUPPORTED_ONLINE_CORRUPTIONS}"
            )

        self.base = base_dataset
        self.corruption_type = corruption_type
        self.severity = severity

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        # We expect base_dataset to return (tensor [C,H,W] in [0,1], label).
        x, y = self.base[idx]
        if not torch.is_tensor(x):
            raise TypeError(
                "OnTheFlyCorruptedDataset expects base_dataset to return "
                "torch.Tensor images; got type {}".format(type(x))
            )
        x = self._apply_corruption(x)
        return x, y

    # -------------------- corruption implementations -------------------- #

    def _apply_corruption(self, x: torch.Tensor) -> torch.Tensor:
        if self.corruption_type == "gaussian_noise":
            return self._apply_gaussian_noise(x)
        elif self.corruption_type == "brightness":
            return self._apply_brightness(x)
        elif self.corruption_type == "blur":
            return self._apply_blur(x)
        else:
            # Should not reach here because __init__ guards the type.
            return x

    def _apply_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Additive Gaussian noise with severity-dependent std.
        """
        std_map = {1: 0.1, 2: 0.15, 3: 0.2, 4: 0.3, 5: 0.4}
        std = std_map.get(self.severity, 0.2)
        noise = torch.randn_like(x) * std
        return (x + noise).clamp(0.0, 1.0)

    def _apply_brightness(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple brightness shift: x -> x + delta, then clamp to [0,1].
        Positive delta makes image brighter, negative makes darker.
        """
        # Map severity to brightness shift in [0, 0.5]
        delta_map = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}
        delta = delta_map.get(self.severity, 0.3)
        return (x + delta).clamp(0.0, 1.0)

    def _apply_blur(self, x: torch.Tensor) -> torch.Tensor:
        """
        Approximate blur via average pooling with a small kernel.
        """
        # Use larger kernel for higher severity.
        kernel_map = {1: 3, 2: 3, 3: 5, 4: 5, 5: 7}
        k = kernel_map.get(self.severity, 3)

        # avg_pool2d expects [N,C,H,W]
        x4 = x.unsqueeze(0)
        # padding='same' equivalent: pad=(k//2, k//2)
        x_blur = F.avg_pool2d(x4, kernel_size=k, stride=1, padding=k // 2)
        return x_blur.squeeze(0)


# ---------------------------------------------------------------------------
# 4. Public loader helper
# ---------------------------------------------------------------------------


def get_corrupted_loader(
    dataset_name: str,
    root: str,
    corruption_type: str,
    severity: int,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = False,
) -> DataLoader:
    """
    Construct a DataLoader for corrupted data.

    - For CIFAR-10:
        * We first try CIFAR10-C .npy files under `root/cifar10_c/`.
        * If they are missing, we fall back to on-the-fly corruption
          over the CIFAR-10 test set.

    - For Fashion-MNIST (fmnist / fashion_mnist):
        * There is no official FMNIST-C, so we *always* use
          on-the-fly corruption over the Fashion-MNIST test set.

    - For MNIST:
        * We first try MNIST-C .npy files under `root/mnist_c/`.
        * If they are missing, we fall back to on-the-fly corruption
          over the MNIST test set.
    """
    dataset_name = dataset_name.lower()

    # ---------------- CIFAR-10 ---------------- #
    if dataset_name == "cifar10":
        try:
            ds: Dataset = CIFAR10C(
                root=root,
                corruption_type=corruption_type,
                severity=severity,
                transform=None,
            )
        except FileNotFoundError:
            # Fallback: online corruption on CIFAR-10 test set
            base_test = datasets.CIFAR10(
                root=root,
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )
            ds = OnTheFlyCorruptedDataset(
                base_dataset=base_test,
                corruption_type=corruption_type,
                severity=severity,
            )

    # --------------- Fashion-MNIST --------------- #
    elif dataset_name in {"fmnist", "fashion_mnist"}:
        # Always use Fashion-MNIST test set with on-the-fly corruption.
        base_test = datasets.FashionMNIST(
            root=root,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        ds = OnTheFlyCorruptedDataset(
            base_dataset=base_test,
            corruption_type=corruption_type,
            severity=severity,
        )

    # ----------------- MNIST ----------------- #
    elif dataset_name == "mnist":
        try:
            ds = MNISTC(
                root=root,
                corruption_type=corruption_type,
                severity=severity,
                transform=None,
            )
        except FileNotFoundError:
            # Fallback: online corruption on MNIST test set
            base_test = datasets.MNIST(
                root=root,
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )
            ds = OnTheFlyCorruptedDataset(
                base_dataset=base_test,
                corruption_type=corruption_type,
                severity=severity,
            )

    else:
        raise ValueError(f"Unsupported dataset for corrupted loader: {dataset_name}")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader

# core/data_manager.py
"""
DataManager: central place to load datasets and create
  - public dataset/dataloader
  - per-client private datasets/dataloaders
  - (later) corrupted and OOD datasets

For Step 1, we only handle clean train/test data and Non-IID partitioning.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from data.transforms import get_fmnist_transforms, get_cifar10_transforms, get_pathmnist_transforms
from data.fmnist_loader import get_fmnist_datasets
from data.cifar10_loader import get_cifar10_datasets
from data.pathmnist_loader import get_pathmnist_datasets
from data.tiny_imagenet_loader import get_tiny_imagenet_datasets, get_tiny_imagenet_transforms
from core.partition import create_partitioner


class DataManager:
    """
    DataManager handles:
      - Loading raw train/test datasets for FEMNIST/CIFAR10
      - Splitting train into: public set + private pool
      - Partitioning private pool Non-IID across clients
      - Creating DataLoaders for public set and each client
    """

    def __init__(self, config: Dict):
        self.config = config
        data_cfg = config["data_config"]

        self.dataset_name: str = data_cfg["dataset"].lower()
        self.data_root: str = data_cfg["data_root"]
        self.num_clients: int = data_cfg["num_clients"]
        self.public_ratio: float = data_cfg["public_ratio"]
        self.batch_size_private: int = data_cfg["batch_size_private"]
        self.batch_size_public: int = data_cfg["batch_size_public"]
        self.num_workers: int = data_cfg.get("num_workers", 0)
        self.val_ratio: float = data_cfg.get("val_ratio", 0.1)

        seed: int = config.get("seed", 42)
        self.rng = np.random.RandomState(seed)

        # Placeholders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.public_dataset = None

        self.client_private_indices: Dict[int, List[int]] = {}
        self.client_private_loaders: Dict[int, DataLoader] = {}
        self.public_loader: Optional[DataLoader] = None

        # --- Main steps ---
        self._load_raw_datasets()
        self._split_train_val()
        self._create_public_and_private()
        self._create_client_loaders()
        self._create_public_loader()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_raw_datasets(self) -> None:
        """
        Load raw train/test datasets with appropriate transforms.
        """
        if self.dataset_name in ["fmnist", "fashion_mnist"]:
            train_transform, test_transform = get_fmnist_transforms()
            train_dataset, test_dataset = get_fmnist_datasets(
                data_root=self.data_root,
                train_transform=train_transform,
                test_transform=test_transform,
            )
        elif self.dataset_name == "cifar10":
            train_transform, test_transform = get_cifar10_transforms()
            train_dataset, test_dataset = get_cifar10_datasets(
                data_root=self.data_root,
                train_transform=train_transform,
                test_transform=test_transform,
            )
        elif self.dataset_name in ["pathmnist", "path-mnist"]:
            train_transform, test_transform = get_pathmnist_transforms(image_size=28)
            train_dataset, test_dataset = get_pathmnist_datasets(
                data_root=self.data_root,
                train_transform=train_transform,
                test_transform=test_transform,
            )
        elif self.dataset_name in ["tiny_imagenet", "tiny-imagenet", "tinyimagenet"]:
            # Tiny-ImageNet-200 (64x64). We treat its official validation set as "test".
            train_transform, test_transform = get_tiny_imagenet_transforms(image_size=64)
            train_dataset, test_dataset = get_tiny_imagenet_datasets(
                data_root=self.data_root,
                train_transform=train_transform,
                test_transform=test_transform,
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


    def _split_train_val(self) -> None:
        """
        Split the original training dataset into train (for federation)
        and validation. For simplicity we use a random split with
        fraction `val_ratio` for validation.

        Note: public data is later taken from the train portion,
        not from validation.
        """
        num_samples = len(self.train_dataset)
        indices = np.arange(num_samples)
        self.rng.shuffle(indices)

        val_size = int(num_samples * self.val_ratio)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        # Subset for federation + validation
        self.val_dataset = Subset(self.train_dataset, val_indices)
        # reuse train_dataset for indices train_indices via Subset when needed
        self._federated_train_indices = train_indices

    def _create_public_and_private(self) -> None:
        """
        From the federated train indices, create:
          - public_dataset: shared by all clients
          - private_pool_indices: to be partitioned Non-IID
        """
        federated_indices = self._federated_train_indices
        num_fed_samples = len(federated_indices)
        num_public = int(num_fed_samples * self.public_ratio)

        # Shuffle before split
        self.rng.shuffle(federated_indices)

        public_indices = federated_indices[:num_public]
        private_pool_indices = federated_indices[num_public:]

        # Public dataset: simple subset of original train_dataset
        self.public_dataset = Subset(self.train_dataset, public_indices)

        # Private pool: we will partition these indices across clients
        # We need labels for these indices
        labels = self._get_labels_for_indices(private_pool_indices)

        # Create partitioner
        data_cfg = self.config["data_config"]
        partitioner = create_partitioner(
            partition_type=data_cfg["partition_type"],
            dirichlet_alpha=data_cfg.get("dirichlet_alpha", 0.5),
            num_shards=data_cfg.get("num_shards", 40),
            label_separation_classes_per_client=data_cfg.get(
                "label_separation_classes_per_client", 2
            ),
        )

        client_to_local_idx = partitioner.partition(
            labels=labels,
            num_clients=self.num_clients,
            rng=self.rng,
        )

        # Map local indices (0..len(private_pool_indices)-1) back to global
        for cid, local_list in client_to_local_idx.items():
            global_indices = [private_pool_indices[i] for i in local_list]
            self.client_private_indices[cid] = global_indices

    def _create_client_loaders(self) -> None:
        """
        Create DataLoader for each client's private dataset.
        """
        for cid, indices in self.client_private_indices.items():
            subset = Subset(self.train_dataset, indices)
            loader = DataLoader(
                subset,
                batch_size=self.batch_size_private,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            self.client_private_loaders[cid] = loader

    def _create_public_loader(self) -> None:
        """
        Create DataLoader for the public dataset, shared by all clients.
        """
        loader = DataLoader(
            self.public_dataset,
            batch_size=self.batch_size_public,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.public_loader = loader

    def _get_labels_for_indices(self, indices: np.ndarray) -> List[int]:
        """
        Extract labels for given indices from the underlying train_dataset.

        Handles both torchvision-style datasets (with .targets or .labels)
        and generic custom datasets (where __getitem__ returns (x, y)).
        """
        # torchvision datasets usually expose labels via .targets or .labels
        if hasattr(self.train_dataset, "targets"):
            all_labels = np.array(self.train_dataset.targets)
        elif hasattr(self.train_dataset, "labels"):
            all_labels = np.array(self.train_dataset.labels)
        else:
            # fallback: query __getitem__ (slower, but generic)
            labels = []
            for idx in indices:
                _, y = self.train_dataset[idx]
                labels.append(int(y))
            return labels

        sel = all_labels[indices]
        # MedMNIST-style labels can be shaped (N, 1); squeeze to 1D ints.
        try:
            if getattr(sel, "ndim", 1) > 1:
                sel = sel.squeeze(-1)
        except Exception:
            pass
        return [int(v) for v in sel.tolist()]

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------

    def get_public_loader(self) -> DataLoader:
        if self.public_loader is None:
            raise RuntimeError("Public loader has not been created.")
        return self.public_loader

    def get_client_private_loader(self, client_id: int) -> DataLoader:
        if client_id not in self.client_private_loaders:
            raise KeyError(f"Client {client_id} does not exist.")
        return self.client_private_loaders[client_id]

    def get_val_loader(self, batch_size: Optional[int] = None) -> DataLoader:
        if batch_size is None:
            batch_size = self.batch_size_private
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_test_loader(self, batch_size: Optional[int] = None) -> DataLoader:
        if batch_size is None:
            batch_size = self.batch_size_private
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_num_clients(self) -> int:
        return self.num_clients

    def get_public_dataset_size(self) -> int:
        return len(self.public_dataset)

    def get_client_dataset_size(self, client_id: int) -> int:
        return len(self.client_private_indices[client_id])

    def summary(self) -> str:
        """
        Return a short text summary of the partitioning result.
        """
        sizes = {cid: len(idx) for cid, idx in self.client_private_indices.items()}
        total_private = sum(sizes.values())
        return (
            f"DataManager summary:\n"
            f"  Dataset: {self.dataset_name}\n"
            f"  Num clients: {self.num_clients}\n"
            f"  Public samples: {len(self.public_dataset)}\n"
            f"  Private samples (total): {total_private}\n"
            f"  Private per client: {sizes}\n"
        )

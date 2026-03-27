# core/partition.py
"""
Non-IID partitioning strategies for federated learning.

We define a BasePartitioner interface and a few concrete implementations:
  - DirichletPartitioner
  - ShardPartitioner
  - LabelSeparationPartitioner

Each partitioner takes the list/array of labels for all samples in the
(private) training set, and outputs a mapping:
    client_id -> list_of_sample_indices
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

import numpy as np


class BasePartitioner(ABC):
    """
    Abstract base class for dataset partitioning.
    """

    @abstractmethod
    def partition(
        self,
        labels: Sequence[int],
        num_clients: int,
        rng: np.random.RandomState,
    ) -> Dict[int, List[int]]:
        """
        Partition dataset indices into num_clients shards.

        Args:
            labels: Sequence of integer labels for all samples.
            num_clients: Number of clients.
            rng: NumPy RandomState for reproducibility.

        Returns:
            A dict mapping client_id (0..num_clients-1) -> list of indices.
        """
        raise NotImplementedError


class DirichletPartitioner(BasePartitioner):
    """
    Dirichlet-based Non-IID partitioner.

    For each class k, its samples are distributed across clients according to a
    Dirichlet(alpha) distribution. Smaller alpha => more skew.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def partition(
        self,
        labels: Sequence[int],
        num_clients: int,
        rng: np.random.RandomState,
    ) -> Dict[int, List[int]]:
        labels = np.array(labels)
        num_samples = len(labels)
        client_indices = {cid: [] for cid in range(num_clients)}

        unique_labels = np.unique(labels)
        for c in unique_labels:
            # indices of samples with label c
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)

            # sample Dirichlet proportions for this class
            proportions = rng.dirichlet(alpha=[self.alpha] * num_clients)
            # convert proportions to counts
            counts = (proportions * len(idx_c)).astype(int)

            # due to rounding, we may have leftover
            diff = len(idx_c) - np.sum(counts)
            # distribute leftover indices one-by-one
            for i in range(diff):
                counts[i % num_clients] += 1

            start = 0
            for cid, count in enumerate(counts):
                if count > 0:
                    client_indices[cid].extend(idx_c[start:start + count].tolist())
                    start += count

        # Shuffle indices within each client
        for cid in range(num_clients):
            rng.shuffle(client_indices[cid])

        return client_indices


class ShardPartitioner(BasePartitioner):
    """
    Shard-based Non-IID partitioner.

    Typical usage: sort samples by label, split into num_shards shards,
    then assign shards to clients (each client gets num_shards // num_clients).
    """

    def __init__(self, num_shards: int):
        self.num_shards = num_shards

    def partition(
        self,
        labels: Sequence[int],
        num_clients: int,
        rng: np.random.RandomState,
    ) -> Dict[int, List[int]]:
        labels = np.array(labels)
        num_samples = len(labels)

        # sort indices by label
        indices = np.arange(num_samples)
        sorted_indices = indices[np.argsort(labels)]

        if self.num_shards > num_samples:
            raise ValueError(
                f"num_shards={self.num_shards} > num_samples={num_samples} is not allowed."
            )

        shard_size = num_samples // self.num_shards
        shards = []

        for i in range(self.num_shards):
            start = i * shard_size
            if i == self.num_shards - 1:
                # last shard takes remaining samples
                end = num_samples
            else:
                end = (i + 1) * shard_size
            shards.append(sorted_indices[start:end])

        rng.shuffle(shards)

        # assign shards to clients as evenly as possible
        client_indices = {cid: [] for cid in range(num_clients)}

        # repeat assignment if num_shards < num_clients
        for i, shard in enumerate(shards):
            cid = i % num_clients
            client_indices[cid].extend(shard.tolist())

        # Shuffle within each client
        for cid in range(num_clients):
            rng.shuffle(client_indices[cid])

        return client_indices


class LabelSeparationPartitioner(BasePartitioner):
    """
    Extreme label separation partitioner.

    The idea is to simulate clients that only have data from a few classes.
    Implementation strategy (simple):

      1. Compute unique labels and shuffle them.
      2. Assign classes to clients in a round-robin fashion in groups of
         `classes_per_client`.
      3. For each class, all its samples go ONLY to the assigned client.

    Note: This means each class is **exclusive** to one client. In general,
    this implies num_clients * classes_per_client >= num_classes; if not,
    some clients may receive fewer classes, but the mapping remains valid.
    """

    def __init__(self, classes_per_client: int = 2):
        self.classes_per_client = classes_per_client

    def partition(
        self,
        labels: Sequence[int],
        num_clients: int,
        rng: np.random.RandomState,
    ) -> Dict[int, List[int]]:
        labels = np.array(labels)
        unique_labels = list(np.unique(labels))
        rng.shuffle(unique_labels)
        num_classes = len(unique_labels)

        # Map from class -> client
        class_to_client = {}
        client_indices = {cid: [] for cid in range(num_clients)}

        # Assign classes in blocks of size classes_per_client
        # E.g., clients: 0,1,2; classes_per_client=2
        #   client 0 -> classes[0:2]
        #   client 1 -> classes[2:4], ...
        # remaining classes -> wrap around
        idx = 0
        for c in unique_labels:
            cid = (idx // self.classes_per_client) % num_clients
            class_to_client[c] = cid
            idx += 1

        # Assign each sample to corresponding client's list
        for i, y in enumerate(labels):
            cid = class_to_client[y]
            client_indices[cid].append(i)

        # Shuffle within each client
        for cid in range(num_clients):
            rng.shuffle(client_indices[cid])

        return client_indices


def create_partitioner(
    partition_type: str,
    dirichlet_alpha: float = 0.5,
    num_shards: int = 40,
    label_separation_classes_per_client: int = 2,
) -> BasePartitioner:
    """
    Factory function to create a partitioner instance based on type.

    Args:
        partition_type: "dirichlet", "shard", or "label_separation"
        dirichlet_alpha: alpha for DirichletPartitioner
        num_shards: number of shards for ShardPartitioner
        label_separation_classes_per_client: #classes per client for label separation

    Returns:
        An instance of BasePartitioner subclass.
    """
    partition_type = partition_type.lower()
    if partition_type == "dirichlet":
        return DirichletPartitioner(alpha=dirichlet_alpha)
    elif partition_type == "shard":
        return ShardPartitioner(num_shards=num_shards)
    elif partition_type == "label_separation":
        return LabelSeparationPartitioner(
            classes_per_client=label_separation_classes_per_client
        )
    else:
        raise ValueError(f"Unknown partition_type: {partition_type}")

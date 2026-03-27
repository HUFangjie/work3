# models/fmnist_cnn.py
"""
Lightweight CNN for Fashion-MNIST-like datasets (28x28 grayscale).

This architecture is intentionally simple but strong enough as a baseline
for federated experiments. It supports width scaling and dropout.

We compute the flatten_dim dynamically using a dummy forward pass, to avoid
shape mismatch bugs when the conv/pool configuration changes.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FMNISTCNN(nn.Module):
    """
    A small CNN for 28x28 grayscale images.

    Args:
        input_channels: Number of input channels (default 1 for grayscale).
        num_classes: Number of output classes (10 for FMNIST, 62 for FEMNIST).
        width_mult: Width multiplier for channels (e.g., 0.5, 1.0, 2.0).
        dropout: Dropout probability for fully-connected layers.
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        width_mult: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.dropout_p = dropout

        def c(ch: int) -> int:
            # helper to scale channels by width_mult
            return int(ch * width_mult)

        # Conv block 1: 1x28x28 -> 32x28x28 -> 32x14x14
        self.conv1 = nn.Conv2d(input_channels, c(32), kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c(32))

        # Conv block 2: 32x14x14 -> 64x14x14 -> 64x7x7
        self.conv2 = nn.Conv2d(c(32), c(64), kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c(64))

        # Conv block 3: 64x7x7 -> 128x7x7 -> 128x3x3 (or 4x4 depending on pooling)
        self.conv3 = nn.Conv2d(c(64), c(128), kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c(128))

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # --------- 动态计算 flatten_dim ---------
        flatten_dim = self._infer_flatten_dim()
        # ----------------------------------------

        self.fc1 = nn.Linear(flatten_dim, c(256))
        self.fc2 = nn.Linear(c(256), num_classes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolutional feature extractor only,
        without the final fully-connected layers.
        """
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2)  # 28 -> 14

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2)  # 14 -> 7

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        # 这里使用 kernel_size=2, padding=0，得到 7 -> 3
        x = F.max_pool2d(x, kernel_size=2)  # 7 -> 3

        return x

    def _infer_flatten_dim(self) -> int:
        """
        Run a dummy tensor through the conv blocks to infer flatten_dim.
        Assumes input images are 28x28 (FMNIST/FEMNIST style).
        """
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, 28, 28)
            feat = self._forward_features(dummy)
            flatten_dim = feat.view(1, -1).size(1)
        return flatten_dim

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc2(x)
        return logits

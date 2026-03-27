# models/cifar10_cnn.py
"""
Residual-VGG style CNN for CIFAR-10 (32x32 RGB).

- 通道规模与原始版本保持一致：64 -> 128 -> 256（可用 width_mult 缩放）
- 每个 block 内加入轻量残差 BasicBlock，提升梯度稳定性和收敛速度
- 结构仍然是 3 个 block + 3 次 2x2 池化 -> 4x4，再接全连接 512 -> num_classes
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    2 层 3x3 Conv + BN 的 BasicBlock，带可选 dropout 和 1x1 shortcut。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 第一层卷积的 stride（这里我们都用 stride=1）
        dropout: Dropout prob（针对中间特征做 Dropout2d）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        # 如果通道数变化，或者 stride != 1，则用 1x1 conv 做 shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class CIFAR10CNN(nn.Module):
    """
    Residual-VGG style CNN for CIFAR-10.

    Args:
        input_channels: 输入通道数（CIFAR-10 为 3）
        num_classes: 类别数（CIFAR-10 为 10）
        width_mult: 通道宽度缩放系数（例如 0.75 / 1.0 / 1.25）
        dropout: block 内部特征的 Dropout prob（典型 0.0–0.3）
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        width_mult: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        def c(ch: int) -> int:
            # 通道缩放工具
            return max(1, int(ch * width_mult))

        # --- Block 1: 3x32x32 -> 64x32x32 -> 64x16x16 ---
        self.conv1_in = nn.Conv2d(
            input_channels,
            c(64),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1_in = nn.BatchNorm2d(c(64))
        # 残差部分保持 64 通道
        self.block1 = ResidualBlock(
            in_channels=c(64),
            out_channels=c(64),
            stride=1,
            dropout=dropout,
        )

        # --- Block 2: 64x16x16 -> 128x16x16 -> 128x8x8 ---
        self.block2 = ResidualBlock(
            in_channels=c(64),
            out_channels=c(128),
            stride=1,
            dropout=dropout,
        )

        # --- Block 3: 128x8x8 -> 256x8x8 -> 256x4x4 ---
        self.block3 = ResidualBlock(
            in_channels=c(128),
            out_channels=c(256),
            stride=1,
            dropout=dropout,
        )

        # 三次 2x2 池化：32 -> 16 -> 8 -> 4
        flatten_dim = c(256) * 4 * 4

        self.fc1 = nn.Linear(flatten_dim, c(512))
        self.fc2 = nn.Linear(c(512), num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem conv: 3 -> 64, 32x32
        x = F.relu(self.bn1_in(self.conv1_in(x)), inplace=True)

        # Block 1 + pool: 32 -> 16
        x = self.block1(x)
        x = F.max_pool2d(x, 2)  # 32 -> 16

        # Block 2 + pool: 16 -> 8
        x = self.block2(x)
        x = F.max_pool2d(x, 2)  # 16 -> 8

        # Block 3 + pool: 8 -> 4
        x = self.block3(x)
        x = F.max_pool2d(x, 2)  # 8 -> 4

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x), inplace=True))
        logits = self.fc2(x)
        return logits


if __name__ == "__main__":
    # 简单自检
    model = CIFAR10CNN()
    x = torch.randn(8, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)  # 期望: [8, 10]

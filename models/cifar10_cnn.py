# models/cifar10_cnn.py
"""
Modernized CIFAR-10 CNN (ResNet-style) for 32x32 RGB.

Design goals:
- Stronger feature extractor than plain VGG-like stack.
- Stable optimization with BN + residual connections.
- Parameter-efficient head via global average pooling (reduces overfitting).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


def _make_stage(
    in_ch: int,
    out_ch: int,
    num_blocks: int,
    first_stride: int,
    dropout: float,
) -> nn.Sequential:
    blocks = [BasicBlock(in_ch, out_ch, stride=first_stride, dropout=dropout)]
    for _ in range(1, num_blocks):
        blocks.append(BasicBlock(out_ch, out_ch, stride=1, dropout=dropout))
    return nn.Sequential(*blocks)


class CIFAR10CNN(nn.Module):
    """
    ResNet-20-like backbone tailored for CIFAR-10:
      stem(3->64) -> stage1(64, 3 blocks) ->
      stage2(128, 3 blocks, downsample) ->
      stage3(256, 3 blocks, downsample) ->
      GAP -> FC(num_classes)
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
            return max(8, int(ch * width_mult))

        c1, c2, c3 = c(64), c(128), c(256)

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.stage1 = _make_stage(c1, c1, num_blocks=3, first_stride=1, dropout=dropout)
        self.stage2 = _make_stage(c1, c2, num_blocks=3, first_stride=2, dropout=dropout)  # 32 -> 16
        self.stage3 = _make_stage(c2, c3, num_blocks=3, first_stride=2, dropout=dropout)  # 16 -> 8

        self.head_drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = torch.flatten(x, 1)
        x = self.head_drop(x)
        return self.fc(x)


if __name__ == "__main__":
    m = CIFAR10CNN()
    t = torch.randn(4, 3, 32, 32)
    y = m(t)
    print(y.shape)  # [4, 10]


# models/tiny_imagenet_wrn.py
"""WideResNet variants for Tiny-ImageNet federated distillation.

These models are intentionally smaller than ImageNet ResNet-34/50 but wider than
plain CIFAR CNNs. They work well for 64x64 Tiny-ImageNet inputs because they keep
an early high-resolution stem and use global average pooling for the 200-way head.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBasicBlock(nn.Module):
    """Pre-activation WideResNet block."""

    def __init__(self, in_planes: int, out_planes: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = float(dropout)

        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x), inplace=True)
        shortcut = self.shortcut(out if not isinstance(self.shortcut, nn.Identity) else x)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        if self.dropout > 0.0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out)
        return out + shortcut


class TinyImageNetWideResNet(nn.Module):
    """WideResNet for 64x64 Tiny-ImageNet.

    Args:
        depth: WRN depth, must satisfy ``depth = 6n + 4`` (e.g. 28, 40).
        widen_factor: width multiplier k. ``4`` is a good FD baseline;
            ``8`` is stronger but more memory intensive.
        dropout: dropout inside residual blocks.
        input_channels: number of image channels.
        num_classes: Tiny-ImageNet uses 200 classes.
    """

    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 4,
        dropout: float = 0.1,
        input_channels: int = 3,
        num_classes: int = 200,
        **_: object,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"WideResNet depth must be 6n+4, got {depth}.")
        n = (depth - 4) // 6
        widths: Sequence[int] = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(input_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = self._make_layer(widths[1], n, stride=1, dropout=dropout)  # 64x64
        self.block2 = self._make_layer(widths[2], n, stride=2, dropout=dropout)  # 32x32
        self.block3 = self._make_layer(widths[3], n, stride=2, dropout=dropout)  # 16x16
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)

        self._init_weights()

    def _make_layer(self, out_planes: int, num_blocks: int, stride: int, dropout: float) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(WideBasicBlock(self.in_planes, out_planes, stride=s, dropout=dropout))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, output_size=1)
        out = torch.flatten(out, 1)
        return self.fc(out)


def wrn28_4_tiny(**kwargs: object) -> TinyImageNetWideResNet:
    return TinyImageNetWideResNet(depth=28, widen_factor=4, dropout=float(kwargs.pop("dropout", 0.1)), **kwargs)


def wrn28_8_tiny(**kwargs: object) -> TinyImageNetWideResNet:
    return TinyImageNetWideResNet(depth=28, widen_factor=8, dropout=float(kwargs.pop("dropout", 0.1)), **kwargs)


if __name__ == "__main__":
    model = wrn28_4_tiny(num_classes=200)
    x = torch.randn(2, 3, 64, 64)
    print(model(x).shape)

from __future__ import annotations

from typing import Dict, List

import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 8 * 8, 256), nn.ReLU(), nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.net(x)


def _resnet(name: str, in_ch: int, num_classes: int):
    model = getattr(models, name)(num_classes=num_classes)
    if in_ch == 1:
        model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def _wrn_fallback(in_ch: int, num_classes: int):
    # fallback to wide_resnet50_2 as wrn28_10-like wide backbone proxy
    model = models.wide_resnet50_2(num_classes=num_classes)
    if in_ch == 1:
        model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def assign_architectures(num_clients: int, mode: str, seed: int) -> List[str]:
    if mode == "homogeneous":
        return ["resnet18"] * num_clients

    # deterministic ratios 50/30/20
    n18 = int(round(num_clients * 0.5))
    n34 = int(round(num_clients * 0.3))
    nwrn = num_clients - n18 - n34
    archs = ["resnet18"] * n18 + ["resnet34"] * n34 + ["wrn28_10"] * nwrn
    # deterministic shuffle by seed
    import random
    rng = random.Random(seed)
    rng.shuffle(archs)
    return archs


def build_model(arch: str, in_ch: int, num_classes: int):
    if arch == "resnet18":
        return _resnet("resnet18", in_ch, num_classes)
    if arch == "resnet34":
        return _resnet("resnet34", in_ch, num_classes)
    if arch == "wrn28_10":
        return _wrn_fallback(in_ch, num_classes)
    if arch == "smallcnn":
        return SmallCNN(in_ch, num_classes)
    raise ValueError(arch)

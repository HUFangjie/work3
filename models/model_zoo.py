
# models/model_zoo.py
"""
Model zoo for the Fed-T3-FD project.

Single entry point:

    get_model(name, **kwargs) -> nn.Module

Supported names:
  - "mnist_cnn"        : MNISTCNN (28x28 grayscale)
  - "fmnist_cnn"       : FMNISTCNN (28x28 grayscale)
  - "femnist_cnn"      : alias of FMNISTCNN
  - "cifar10_cnn"      : CIFAR10CNN (32x32 RGB)
  - "resnet18_tiny"    : ResNet-18 with CIFAR-style stem (3x3 conv, stride=1, no maxpool) for 64x64 inputs
  - "resnet34_tiny"    : ResNet-34 with CIFAR-style stem for 64x64 inputs
  - "resnet18_imagenet": torchvision ResNet-18 standard ImageNet stem (7x7, stride=2, maxpool)
  - "resnet34_imagenet": torchvision ResNet-34 standard ImageNet stem
  - "resnet50_tiny"    : ResNet-50 with CIFAR-style stem for 64x64 inputs
  - "resnet50_imagenet": torchvision ResNet-50 standard ImageNet stem
"""

from __future__ import annotations

from typing import Any, Dict, Callable

import torch
import torch.nn as nn

from models.fmnist_cnn import FMNISTCNN
from models.mnist_cnn import MNISTCNN
from models.cifar10_cnn import CIFAR10CNN


def _tv_resnet_builder(name: str) -> Callable[..., nn.Module]:
    """
    Return a torchvision ResNet constructor with weights disabled,
    compatible across torchvision versions.
    """
    try:
        from torchvision.models import resnet18, resnet34, resnet50
    except Exception as e:
        raise ImportError("torchvision is required for ResNet models.") from e

    name = name.lower()
    if name == "resnet18":
        return resnet18
    if name == "resnet34":
        return resnet34
    if name == "resnet50":
        return resnet50
    raise ValueError(f"Unsupported torchvision resnet: {name}. Supported: resnet18/resnet34/resnet50")


def _build_resnet(
    arch: str,
    input_channels: int,
    num_classes: int,
    tiny_stem: bool,
) -> nn.Module:
    """
    Build ResNet with either:
      - tiny_stem=True: 3x3 stride1 conv1, no maxpool (better for 32/64 inputs)
      - tiny_stem=False: standard ImageNet stem (7x7 stride2 + maxpool)
    """
    ctor = _tv_resnet_builder(arch)

    # torchvision API compatibility: weights vs pretrained
    try:
        model = ctor(weights=None, num_classes=num_classes)
    except TypeError:
        model = ctor(pretrained=False, num_classes=num_classes)

    # adjust input channels if needed
    if input_channels != 3:
        # replace conv1 while preserving other hyperparams
        old = model.conv1
        model.conv1 = nn.Conv2d(
            input_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )

    if tiny_stem:
        # CIFAR-style stem: 3x3 stride=1 padding=1, remove maxpool
        model.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()

    # classifier head already matches num_classes via ctor
    return model


_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "mnist_cnn": MNISTCNN,
    "fmnist_cnn": FMNISTCNN,
    "femnist_cnn": FMNISTCNN,
    "cifar10_cnn": CIFAR10CNN,
    # ResNet entries are handled in get_model (special kwargs)
}


def get_model(name: str, **kwargs: Any) -> nn.Module:
    """
    Instantiate a model by name.

    Common kwargs (recommended from config.model_config):
      - input_channels: int
      - num_classes: int
      - width_mult: float (CNN only)
      - dropout: float (CNN only)
    """
    name = name.lower()

    # ResNet family
    if name in ("resnet18_tiny", "resnet34_tiny", "resnet50_tiny", "resnet18_imagenet", "resnet34_imagenet", "resnet50_imagenet"):
        input_channels = int(kwargs.get("input_channels", 3))
        num_classes = int(kwargs.get("num_classes", 1000))
        if name.startswith("resnet18"):
            arch = "resnet18"
        elif name.startswith("resnet34"):
            arch = "resnet34"
        else:
            arch = "resnet50"
        tiny_stem = name.endswith("_tiny")
        return _build_resnet(arch=arch, input_channels=input_channels, num_classes=num_classes, tiny_stem=tiny_stem)

    # CNN registry
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name: {name}. Available: {list(_MODEL_REGISTRY.keys()) + ['resnet18_tiny','resnet34_tiny','resnet50_tiny','resnet18_imagenet','resnet34_imagenet','resnet50_imagenet']}"
        )
    model_cls = _MODEL_REGISTRY[name]
    return model_cls(**kwargs)


def adapt_model_config_for_dataset(
    dataset: str,
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Adjust model_config based on dataset, while remaining robust when users
    switch datasets via CLI but forget to update model_config.

    Policy:
      - If a key is missing, fill it.
      - If a key exists but looks like a legacy/default value from another dataset,
        override it to a sensible dataset default.
    """
    dataset = dataset.lower()
    cfg = dict(model_config)  # shallow copy

    def _maybe_set(key: str, value, override_if: Callable[[Any], bool]):
        if key not in cfg or override_if(cfg.get(key)):
            cfg[key] = value

    if dataset in ["mnist", "minist"]:
        _maybe_set("input_channels", 1, lambda v: v not in (1,))
        _maybe_set("num_classes", 10, lambda v: v not in (10,))
        _maybe_set("name", "mnist_cnn", lambda v: str(v).lower() != "mnist_cnn")
    elif dataset in ["fmnist", "fashion_mnist"]:
        _maybe_set("input_channels", 1, lambda v: v not in (1,))
        _maybe_set("num_classes", 10, lambda v: v not in (10,))
        _maybe_set("name", "fmnist_cnn", lambda v: str(v).lower() not in ("fmnist_cnn", "femnist_cnn"))
    elif dataset in ["femnist", "emnist"]:
        _maybe_set("input_channels", 1, lambda v: v not in (1,))
        _maybe_set("num_classes", 62, lambda v: v not in (62,))
        _maybe_set("name", "femnist_cnn", lambda v: str(v).lower() != "femnist_cnn")
    elif dataset == "cifar10":
        _maybe_set("input_channels", 3, lambda v: v not in (3,))
        _maybe_set("num_classes", 10, lambda v: v not in (10,))
        _maybe_set("name", "cifar10_cnn", lambda v: str(v).lower() not in ("cifar10_cnn", "resnet18_tiny", "resnet34_tiny", "resnet18_imagenet", "resnet34_imagenet", "resnet50_tiny", "resnet50_imagenet"))
    elif dataset in ["tiny_imagenet", "tiny-imagenet", "tinyimagenet"]:
        _maybe_set("input_channels", 3, lambda v: v not in (3,))
        _maybe_set("num_classes", 200, lambda v: v != 200)
        _maybe_set("name", "resnet34_tiny", lambda v: str(v).lower() not in ("resnet18_tiny", "resnet34_tiny", "resnet50_tiny", "resnet18_imagenet", "resnet34_imagenet", "resnet50_imagenet"))
    elif dataset in ["pathmnist", "path-mnist"]:
        _maybe_set("input_channels", 3, lambda v: v not in (3,))
        _maybe_set("num_classes", 9, lambda v: v != 9)
        _maybe_set("name", "resnet18_tiny", lambda v: str(v).lower() not in ("resnet18_tiny", "resnet34_tiny", "resnet18_imagenet", "resnet34_imagenet"))

    else:
        # Keep user-specified values
        pass

    return cfg

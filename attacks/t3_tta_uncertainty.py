# attacks/t3_tta_uncertainty.py
"""
Test-Time Augmentation (TTA) based epistemic/aleatoric uncertainty estimator.

Given a model and a batch of inputs, apply K different augmentations
and measure:
  - aleatoric entropy: average entropy over augmentations
  - variance: variance of probabilities across augmentations
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class TTAUncertaintyEstimator:
    """
    TTA-based uncertainty estimator.

    Args:
        model: client's model (eval mode, no grad inside this module).
        dataset_name: e.g., "fmnist", "cifar10" (for default augmentations).
        tta_type: "weak" or "strong".
        num_augments: number of TTA samples per input.
        custom_transform: optional externally provided transform; if not None,
                         this overrides (dataset_name, tta_type) defaults.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset_name: str = "fmnist",
        tta_type: Literal["weak", "strong"] = "weak",
        num_augments: int = 4,
        custom_transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.model = model    # 模型引用存进来
        self.model.eval()     # 模型切到 evaluation 模式
        self.dataset_name = dataset_name.lower()
        self.tta_type = tta_type   # 记录 TTA 强度类型 "weak" / "strong"
        self.num_augments = int(num_augments)  # 一个样本要做几次增强（K）

        if custom_transform is not None:
            self.transform = custom_transform
        else:
            self.transform = self._build_default_transform()

    # ------------------------------------------------------------------ #
    # Transform building
    # ------------------------------------------------------------------ #
    def _build_default_transform(self) -> transforms.Compose:
        """
        Default torchvision-style transform for TTA.

        这一版保留 Weak/Strong 两级强度，但允许通过 custom_transform 完全覆盖。
        """
        if self.dataset_name in ["fmnist", "fashion_mnist"]:
            if self.tta_type == "strong":
                # 15° + 10% 平移 + 缩放+ RandomErasing 随机擦除一块区域。
                tfm = transforms.Compose(
                    [
                        transforms.RandomAffine(
                            degrees=90,
                            translate=(0.8, 0.8),
                            scale=(0.2, 1.8),
                            shear=45
                        ),
                        transforms.RandomErasing(p=0.3),
                    ]
                )
            else:  # weak
                # 10°以内小旋转 + 5% 平移
                tfm = transforms.Compose(
                    [
                        transforms.RandomAffine(
                            degrees=10,
                            translate=(0.05, 0.05),
                        ),
                    ]
                )
        elif self.dataset_name == "cifar10":
            if self.tta_type == "strong":
                tfm = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ColorJitter(
                            brightness=0.2,
                            contrast=0.2,
                            saturation=0.2,
                            hue=0.1,
                        ),
                    ]
                )
            else:  # weak
                tfm = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(p=0.5),
                    ]
                )
        else:
            tfm = transforms.Compose([])
        return tfm

    # ------------------------------------------------------------------ #
    # Public APIs
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def estimate_aleatoric_and_variance(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate aleatoric entropy and variance per sample.

        Args:
            x: [B, C, H, W] on device.

        Returns:
            H_aleatoric: [B]  近似输入的 aleatoric entropy
            var_per_sample: [B]  TTA 概率的方差（proxy for epistemic）
        """
        B = x.size(0)
        device = x.device

        if self.num_augments <= 1:
            # 没有 TTA 时，退化为单次前向：aleatoric = H(p), variance = 0
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            H = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)  # [B]
            return H, torch.zeros(B, device=device)

        probs_list = []
        ent_list = []

        for _ in range(self.num_augments):
            # 逐样本做数据增强（transform 大多不支持 4D batch）
            x_cpu = x.detach().cpu()
            aug_imgs = []
            for i in range(B):
                img = self.transform(x_cpu[i])
                aug_imgs.append(img)
            x_aug_cpu = torch.stack(aug_imgs, dim=0)
            x_aug = x_aug_cpu.to(device)

            logits = self.model(x_aug)          # [B,C]
            probs = F.softmax(logits, dim=-1)   # [B,C]
            probs_list.append(probs)

            ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)  # [B]
            ent_list.append(ent)

        probs_stack = torch.stack(probs_list, dim=0)  # [K,B,C]
        ent_stack = torch.stack(ent_list, dim=0)      # [K,B]

        mean_probs = probs_stack.mean(dim=0)          # [B,C]
        H_aleatoric = ent_stack.mean(dim=0)           # [B]

        # Epistemic 部分用概率的方差做 proxy
        var_across_aug = (probs_stack - mean_probs.unsqueeze(0)) ** 2  # [K,B,C]
        var_per_sample = var_across_aug.mean(dim=(0, 2))               # [B]

        return H_aleatoric.to(device), var_per_sample.to(device)

    @torch.no_grad()
    def compute_variance(self, x: torch.Tensor) -> torch.Tensor:
        """
        保留原接口：只返回 variance，内部复用 estimate_aleatoric_and_variance。
        """
        _, var = self.estimate_aleatoric_and_variance(x)
        return var

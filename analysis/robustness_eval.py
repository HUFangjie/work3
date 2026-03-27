# analysis/robustness_eval.py
"""
Evaluation utilities for ID calibration / confidence analysis.

现在的设计简化为：给定任意 DataLoader，计算以下指标：
  - accuracy
  - Avg. Conf.（所有样本的平均 max softmax prob）
  - ECE（Expected Calibration Error）
  - KS（正确 vs 错误置信度分布的 KS 统计量）
  - mean entropy
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.metrics import (
    compute_accuracy,
    compute_ece,
    compute_confidence_and_predictions,
    compute_ks_confidence_correct_vs_error,
)
from utils.tensor_utils import entropy_from_logits


@torch.no_grad()
def _eval_loader_basic(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_bins: int = 15,
) -> Dict[str, float]:
    """
    Evaluate basic metrics on a given loader:
      - accuracy
      - average confidence (Avg. Conf.)
      - ECE (num_bins)
      - KS (correct vs error confidence distributions)
      - mean entropy
    """
    model.eval()

    all_logits = []
    all_targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())

    if len(all_logits) == 0:
        return {
            "accuracy": 0.0,
            "avg_conf": 0.0,
            "ece": 0.0,
            "ks": 0.0,
            "mean_entropy": 0.0,
        }

    logits = torch.cat(all_logits, dim=0)   # [N,C]
    targets = torch.cat(all_targets, dim=0)  # [N]

    acc = compute_accuracy(logits, targets)
    ece = compute_ece(logits, targets, num_bins=num_bins)
    confidences, _ = compute_confidence_and_predictions(logits)
    avg_conf = float(confidences.mean().item())
    ks = compute_ks_confidence_correct_vs_error(logits, targets)
    ent = entropy_from_logits(logits).mean().item()

    return {
        "accuracy": float(acc),
        "avg_conf": float(avg_conf),
        "ece": float(ece),
        "ks": float(ks),
        "mean_entropy": float(ent),
    }


def evaluate_on_corruptions(
    model: nn.Module,
    device: torch.device,
    get_loader_fn,
    corruption_types: List[str],
    severities: List[int],
    num_bins: int = 15,
) -> Dict[Tuple[str, int], Dict[str, float]]:
    """
    （保留接口以兼容旧代码）在给定的 corrupted loaders 上做同样的 ID 指标评估。

    Args:
        model: PyTorch model.
        device: torch.device.
        get_loader_fn: a function (corruption_type, severity) -> DataLoader.
        corruption_types: list of corruption type names.
        severities: list of severities (1..5).
        num_bins: ECE 分桶数。

    Returns:
        results: dict keyed by (corruption_type, severity) -> metric dict
                 metric dict keys:
                   "accuracy", "avg_conf", "ece", "ks", "mean_entropy".
    """
    results: Dict[Tuple[str, int], Dict[str, float]] = {}

    for c_type in corruption_types:
        for sev in severities:
            loader = get_loader_fn(c_type, sev)
            metrics = _eval_loader_basic(model, loader, device, num_bins=num_bins)
            results[(c_type, sev)] = metrics

    return results


def evaluate_on_ood(
    model: nn.Module,
    device: torch.device,
    ood_loader: DataLoader,
    num_bins: int = 15,
) -> Dict[str, float]:
    """
    （保留接口以兼容旧代码）在 OOD loader 上计算同样的一套指标。

    Returns:
        dict with keys: "accuracy", "avg_conf", "ece", "ks", "mean_entropy".
    """
    return _eval_loader_basic(model, ood_loader, device, num_bins=num_bins)

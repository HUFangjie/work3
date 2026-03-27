# analysis/visualize_ood.py
"""
Visualization utilities for OOD and calibration analysis.

NOTE: 这些函数只是生成 matplotlib 图形，不会自动保存/显示，
由调用者决定 plt.show() 或 fig.savefig(...)。
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import torch

from core.metrics import compute_calibration_bins
from utils.tensor_utils import entropy_from_logits


def plot_entropy_histograms(
    id_logits: torch.Tensor,
    ood_logits: torch.Tensor,
    num_bins: int = 50,
    title: str = "Entropy on ID vs OOD",
) -> plt.Figure:
    """
    Plot histograms of entropy on ID vs OOD sets.

    Args:
        id_logits: [N_id, C]
        ood_logits: [N_ood, C]

    Returns:
        matplotlib Figure.
    """
    id_ent = entropy_from_logits(id_logits)
    ood_ent = entropy_from_logits(ood_logits)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        id_ent.cpu().numpy(),
        bins=num_bins,
        alpha=0.6,
        label="ID",
        density=True,
    )
    ax.hist(
        ood_ent.cpu().numpy(),
        bins=num_bins,
        alpha=0.6,
        label="OOD",
        density=True,
    )
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_reliability_diagram(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 15,
    title: str = "Reliability Diagram",
) -> plt.Figure:
    """
    Plot reliability diagram for given logits/targets.

    Args:
        logits: [N,C]
        targets: [N]

    Returns:
        matplotlib Figure.
    """
    bins = compute_calibration_bins(logits, targets, num_bins=num_bins)
    conf = bins["bin_confidence"].cpu().numpy()
    acc = bins["bin_accuracy"].cpu().numpy()

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly Calibrated")
    ax.bar(
        bins["bin_edges"][:-1].cpu().numpy(),
        acc,
        width=1.0 / num_bins,
        align="edge",
        edgecolor="black",
        alpha=0.7,
        label="Empirical Accuracy",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig

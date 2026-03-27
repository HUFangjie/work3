# attacks/utils.py
"""
Utility functions for T3 and other logit-space attacks.
"""

from __future__ import annotations

from typing import Tuple

import torch

from utils.tensor_utils import (
    entropy_from_logits,
    js_divergence,
    topk_indices,
)


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Entropy per sample from logits. Shape: [batch].
    """
    return entropy_from_logits(logits)


def compute_top1_margin(logits: torch.Tensor) -> torch.Tensor:
    """
    Top-1 margin: logit[top1] - logit[top2], per sample.

    Args:
        logits: [batch, num_classes]

    Returns:
        margins: [batch]
    """
    values, indices = topk_indices(logits, k=2, dim=-1)  # [B,2], [B,2]
    top1 = values[:, 0]
    top2 = values[:, 1]
    return top1 - top2


def topk_consistency(
    base_logits: torch.Tensor,
    adv_logits: torch.Tensor,
    k: int = 1,
) -> torch.Tensor:
    """
    Check top-k index consistency between base and adv logits.

    Args:
        base_logits: [batch, num_classes]
        adv_logits: [batch, num_classes]
        k: top-k

    Returns:
        mask: [batch] bool tensor; True iff the sets of top-k indices match.
    """
    _, base_idx = topk_indices(base_logits, k=k, dim=-1)
    _, adv_idx = topk_indices(adv_logits, k=k, dim=-1)

    # treat top-k indices as sets: sorted & compare
    base_sorted, _ = torch.sort(base_idx, dim=-1)
    adv_sorted, _ = torch.sort(adv_idx, dim=-1)
    return (base_sorted == adv_sorted).all(dim=-1)


def js_div_from_logits(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Thin wrapper for JS divergence on logits. Shape: [batch].
    """
    return js_divergence(p_logits, q_logits)


def quantile_threshold(
    values: torch.Tensor,
    q: float,
) -> float:
    """
    Compute empirical quantile threshold for 1D tensor (on CPU or device).

    Args:
        values: 1D tensor
        q: float in [0,1], e.g., 0.8 means 80% quantile.

    Returns:
        threshold (float).
    """
    v = values.detach().cpu()
    k = max(int(round(q * (v.numel() - 1))), 0)
    sorted_v, _ = torch.sort(v)
    return float(sorted_v[k].item())

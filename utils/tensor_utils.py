# utils/tensor_utils.py
"""
Common tensor utilities used across attacks, defenses, and analysis.

This module should NOT depend on any project-specific classes
(Server, Client, etc.). It only uses PyTorch tensors.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def logits_to_probs(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Convert logits to probabilities via softmax.

    Args:
        logits: Tensor of shape (..., num_classes)
        dim:    Dimension along which to apply softmax.

    Returns:
        probs: Tensor of same shape as logits, where values sum to 1 along `dim`.
    """
    return F.softmax(logits, dim=dim)


def probs_to_logits(probs: torch.Tensor, eps: float = 1e-12, dim: int = -1) -> torch.Tensor:
    """
    Convert probabilities to logits via log.

    Args:
        probs: Probability tensor (values in [0, 1]).
        eps:   Numerical stability term.
        dim:   Class dimension (unused, kept for symmetry).

    Returns:
        logits: log(probs).
    """
    clipped = probs.clamp(min=eps, max=1.0)
    return torch.log(clipped)


def entropy_from_probs(probs: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute Shannon entropy (in nats) from probabilities.

    Args:
        probs: Tensor of probabilities summing to 1 along `dim`.
        dim:   Dimension along which to compute entropy.
        eps:   Numerical stability term.

    Returns:
        ent: Tensor of entropies with one less dimension than probs along `dim`.
    """
    clipped = probs.clamp(min=eps, max=1.0)
    log_p = torch.log(clipped)
    ent = -torch.sum(clipped * log_p, dim=dim)
    return ent


def entropy_from_logits(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute entropy from logits directly via softmax.

    Args:
        logits: Tensor of logits.
        dim:    Class dimension.

    Returns:
        ent: Tensor of entropies.
    """
    probs = logits_to_probs(logits, dim=dim)
    return entropy_from_probs(probs, dim=dim)


def kl_divergence(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    dim: int = -1,
    log_base: float = 2.0,
) -> torch.Tensor:
    """
    Compute KL(P || Q) between two distributions given logits.

    Args:
        p_logits: logits for P
        q_logits: logits for Q
        dim:      class dimension
        log_base: if 2.0, returns KL in bits; if e, returns nats.

    Returns:
        kl: Tensor of KL divergences.
    """
    p_log_prob = F.log_softmax(p_logits, dim=dim)
    q_log_prob = F.log_softmax(q_logits, dim=dim)
    p_prob = p_log_prob.exp()

    kl_nats = torch.sum(p_prob * (p_log_prob - q_log_prob), dim=dim)
    if log_base is None or log_base == torch.e:
        return kl_nats
    else:
        return kl_nats / torch.log(torch.tensor(log_base, device=kl_nats.device))


def js_divergence(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    dim: int = -1,
    log_base: float = 2.0,
) -> torch.Tensor:
    """
    Compute Jensen–Shannon divergence between two distributions given logits.

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M),
    where M = 0.5 * (P + Q).

    Args:
        p_logits: logits for P
        q_logits: logits for Q
        dim:      class dimension
        log_base: base of logarithm for output units.

    Returns:
        js: Tensor of JS divergences.
    """
    p_log_prob = F.log_softmax(p_logits, dim=dim)
    q_log_prob = F.log_softmax(q_logits, dim=dim)
    p_prob = p_log_prob.exp()
    q_prob = q_log_prob.exp()

    m_prob = 0.5 * (p_prob + q_prob)
    m_log_prob = torch.log(m_prob + 1e-12)

    kl_p_m = torch.sum(p_prob * (p_log_prob - m_log_prob), dim=dim)
    kl_q_m = torch.sum(q_prob * (q_log_prob - m_log_prob), dim=dim)
    js_nats = 0.5 * (kl_p_m + kl_q_m)

    if log_base is None or log_base == torch.e:
        return js_nats
    else:
        return js_nats / torch.log(torch.tensor(log_base, device=js_nats.device))


def l2_norm(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    Compute L2 norm along a given dimension.

    Args:
        tensor: Input tensor.
        dim:    Dimension along which to compute norm.
        keepdim: Whether to keep the reduced dimension.

    Returns:
        norms: Tensor of norms.
    """
    return torch.norm(tensor, p=2, dim=dim, keepdim=keepdim)


def topk_indices(logits: torch.Tensor, k: int = 1, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get top-k values and indices from logits along a given dimension.

    Args:
        logits: Input logits.
        k:      Number of top elements.
        dim:    Dimension to take top-k over.

    Returns:
        (values, indices): Tensors of top-k values and their indices.
    """
    return torch.topk(logits, k=k, dim=dim)

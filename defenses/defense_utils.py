# defenses/defense_utils.py
"""
Utility functions for defenses, e.g., entropy computation, trimming helpers.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from utils.tensor_utils import entropy_from_logits


def compute_mean_entropy_per_client(
    client_logits: Dict[int, torch.Tensor],
) -> Dict[int, float]:
    """
    Compute mean entropy for each client's logits over the current batch.

    Returns:
        dict: client_id -> mean_entropy
    """
    mean_ent = {}
    for cid, logits in client_logits.items():
        ent = entropy_from_logits(logits)  # [batch]
        mean_ent[cid] = float(ent.mean().item())
    return mean_ent


def filter_clients_by_entropy(
    client_logits: Dict[int, torch.Tensor],
    max_entropy: float,
) -> Dict[int, torch.Tensor]:
    """
    Keep only clients whose mean entropy <= max_entropy.
    If all clients are filtered out, return original dict.
    """
    mean_ent = compute_mean_entropy_per_client(client_logits)
    kept = {
        cid: logits
        for cid, logits in client_logits.items()
        if mean_ent[cid] <= max_entropy
    }
    if len(kept) == 0:
        # fallback: keep all
        return client_logits
    return kept

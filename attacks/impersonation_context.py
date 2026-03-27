# attacks/impersonation_context.py
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

# CPU cache for current public batch (per round/batch)
_BENIGN_POOL: Optional[Dict[int, torch.Tensor]] = None
_CACHE_TARGET: Optional[torch.Tensor] = None


def clear_benign_pool() -> None:
    global _BENIGN_POOL, _CACHE_TARGET
    _BENIGN_POOL = None
    _CACHE_TARGET = None


def set_benign_pool(pool: Dict[int, torch.Tensor]) -> None:
    """
    pool: dict client_id -> logits tensor (preferably CPU) shape [B,C]
    """
    global _BENIGN_POOL, _CACHE_TARGET
    _BENIGN_POOL = pool
    _CACHE_TARGET = None


@torch.no_grad()
def get_farthest_benign_logits(device: torch.device) -> Optional[torch.Tensor]:
    """
    Implements Eq.(4) from Li et al. 2024:
      id = argmax_i sum_j ||l_i - l_j||_2

    We compute on CPU for stability and move to `device` when returning.
    Returns logits tensor [B,C] on `device`, or None if pool unavailable.
    """
    global _BENIGN_POOL, _CACHE_TARGET
    if _BENIGN_POOL is None or len(_BENIGN_POOL) == 0:
        return None

    if _CACHE_TARGET is not None:
        return _CACHE_TARGET.to(device)

    ids = list(_BENIGN_POOL.keys())
    mats = [(_BENIGN_POOL[i].detach().cpu().float()) for i in ids]  # [B,C] each
    X = torch.stack([m.reshape(-1) for m in mats], dim=0)  # [N, B*C]

    # Pairwise distances (N x N)
    D = torch.cdist(X, X, p=2)  # float32 on CPU
    scores = D.sum(dim=1)       # [N]
    farthest_idx = int(torch.argmax(scores).item())

    target = mats[farthest_idx].to(device).type_as(_BENIGN_POOL[ids[farthest_idx]])
    _CACHE_TARGET = target.detach().clone()
    return _CACHE_TARGET.to(device)

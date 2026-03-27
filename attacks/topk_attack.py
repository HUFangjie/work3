# attacks/topk_attack.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from attacks.base_attack import BaseAttack


def normalize_logits_minmax(
    logits: torch.Tensor,
    low: float = -10.0,
    high: float = 10.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Per-sample min-max normalize logits to [low, high] along class dim.
    Preserves ordering within each sample.

    logits: [B, C]
    """
    vmin = logits.min(dim=-1, keepdim=True).values
    vmax = logits.max(dim=-1, keepdim=True).values
    denom = (vmax - vmin).clamp_min(eps)
    scaled = (logits - vmin) / denom  # [0,1]
    return scaled * (high - low) + low


class TopKLogitAttack(BaseAttack):
    """
    Li et al. 2024 Top-k attack for knowledge/logits in FD:
      1) Normalize logits to [-10, 10]
      2) Add a negative perturbation delta to top-k components (per sample)
    """

    def __init__(
        self,
        is_malicious: bool,
        cfg: Optional[Dict[str, Any]] = None,
        client_id: Optional[int] = None,
        model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(is_malicious=is_malicious, cfg=cfg, client_id=client_id, model=model)

        sub = (self.cfg or {}).get("topk", {})
        self.k: int = int(sub.get("k", 3))
        self.delta: float = float(sub.get("delta", -10.0))
        self.normalize: bool = bool(sub.get("normalize", True))
        self.norm_low: float = float(sub.get("norm_low", -10.0))
        self.norm_high: float = float(sub.get("norm_high", 10.0))

    def attack_logits(
        self,
        x_public: torch.Tensor,
        logits: torch.Tensor,
        round_idx: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> torch.Tensor:
        if not self.is_malicious:
            return logits

        adv = logits.clone()

        # (optional) normalize to [-10,10] as described in the paper
        if self.normalize:
            adv = normalize_logits_minmax(adv, low=self.norm_low, high=self.norm_high)

        # add delta to top-k entries per sample
        _, idx = torch.topk(adv, k=min(self.k, adv.size(-1)), dim=-1)  # [B,k]
        src = torch.full(idx.shape, self.delta, device=adv.device, dtype=adv.dtype)
        adv.scatter_add_(dim=-1, index=idx, src=src)
        return adv

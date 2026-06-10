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
        self.rank_weighted: bool = bool(sub.get("rank_weighted", True))
        self.promote_non_top1: bool = bool(sub.get("promote_non_top1", True))
        self.promote_strength: float = float(sub.get("promote_strength", 0.8))

    def attack_logits(
        self,
        x_public: torch.Tensor,
        logits: torch.Tensor,
        y_public: Optional[torch.Tensor] = None,
        round_idx: Optional[int] = None,
        global_step: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not self.is_malicious:
            return logits

        adv = logits.clone()

        # (optional) normalize to [-10,10] as described in the paper
        if self.normalize:
            adv = normalize_logits_minmax(adv, low=self.norm_low, high=self.norm_high)

        # top-k indices
        k = min(self.k, adv.size(-1))
        topv, idx = torch.topk(adv, k=k, dim=-1)  # [B,k]

        # rank-aware penalties to break internal ordering instead of uniform subtraction
        if self.rank_weighted:
            # strongest suppression on top-1, weaker on lower ranks
            rank_w = torch.linspace(1.0, 0.35, steps=k, device=adv.device, dtype=adv.dtype).unsqueeze(0)  # [1,k]
            penalties = (self.delta * rank_w).expand_as(idx).contiguous()  # [B,k], self.delta is usually negative
        else:
            penalties = torch.full_like(topv, self.delta)
        adv.scatter_add_(dim=-1, index=idx, src=penalties)

        # additionally promote a non-top1 class so argmax is more likely to flip
        if self.promote_non_top1 and adv.size(-1) > 1:
            top1_idx = idx[:, 0]  # [B]
            # choose target as original top-(k+1) if exists, else current smallest class
            if adv.size(-1) > k:
                target_idx = torch.topk(adv, k=k + 1, dim=-1).indices[:, -1]
            else:
                target_idx = torch.argmin(adv, dim=-1)
            # avoid accidentally selecting top1
            same = target_idx.eq(top1_idx)
            if same.any():
                target_idx = torch.where(same, (target_idx + 1) % adv.size(-1), target_idx)

            gap = (adv.gather(1, top1_idx.unsqueeze(1)) - adv.gather(1, target_idx.unsqueeze(1))).clamp_min(0.0)
            boost = (gap + torch.abs(torch.as_tensor(self.delta, device=adv.device, dtype=adv.dtype))) * self.promote_strength
            adv.scatter_add_(dim=-1, index=target_idx.unsqueeze(1), src=boost)
        return adv

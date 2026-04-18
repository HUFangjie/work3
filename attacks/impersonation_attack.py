# attacks/impersonation_attack.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from attacks.base_attack import BaseAttack
from attacks.impersonation_context import get_farthest_benign_logits


class ImpersonationAttack(BaseAttack):
    """
    Li et al. 2024 Impersonation attack:
      - requires benign clients' logits
      - all attackers mimic the benign knowledge that is farthest from others
        (Eq.(4) in the paper).
    """

    # marker for FD loop to know it should stage benign logits first
    requires_benign_pool: bool = True

    def __init__(
        self,
        is_malicious: bool,
        cfg: Optional[Dict[str, Any]] = None,
        client_id: Optional[int] = None,
        model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(is_malicious=is_malicious, cfg=cfg, client_id=client_id, model=model)
        sub = (self.cfg or {}).get("impersonation", {})
        # Strength knobs (default keeps previous behavior)
        self.logit_scale: float = float(sub.get("logit_scale", 1.0))
        self.top1_boost: float = float(sub.get("top1_boost", 0.0))
        max_abs = sub.get("max_abs_logit", None)
        self.max_abs_logit: Optional[float] = float(max_abs) if max_abs is not None else None

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

        target = get_farthest_benign_logits(device=logits.device)
        if target is None:
            # fallback: if pool not available, do nothing (safe default)
            return logits
        if target.shape != logits.shape:
            # Safety fallback: align to current micro-batch shape.
            # This should rarely happen after forcing full-batch mode for impersonation.
            if target.dim() == logits.dim() and target.size(-1) == logits.size(-1):
                target = target[: logits.size(0)]
            else:
                return logits
        adv = target.to(logits.device).type_as(logits)

        # Optional strength amplification:
        # 1) globally scale logits (sharper confidence if >1)
        if self.logit_scale != 1.0:
            adv = adv * self.logit_scale
        # 2) optionally boost current top-1 class margin
        if self.top1_boost != 0.0:
            pred = adv.argmax(dim=-1)
            adv = adv.clone()
            src = torch.full_like(pred.unsqueeze(-1).float(), self.top1_boost)
            adv.scatter_add_(dim=-1, index=pred.unsqueeze(-1), src=src)
        # 3) optional clipping for numerical stability
        if self.max_abs_logit is not None and self.max_abs_logit > 0:
            adv = torch.clamp(adv, -self.max_abs_logit, self.max_abs_logit)

        return adv

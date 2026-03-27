# attacks/naive_sharpening_attack.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from attacks.base_attack import BaseAttack


class NaiveSharpeningAttack(BaseAttack):
    """Naive sharpening: scale logits to make predictions more confident.

    This preserves argmax (accuracy) while typically increasing over-confidence
    (reducing entropy) by scaling logits with a factor > 1 (equivalently temperature < 1).
    """

    def __init__(
        self,
        is_malicious: bool,
        cfg: Optional[Dict[str, Any]] = None,
        client_id: Optional[int] = None,
        model=None,
    ) -> None:
        super().__init__(is_malicious=is_malicious, cfg=cfg, client_id=client_id, model=model)
        self.scale: float = float(self.cfg.get("scale", 3.0))  # >1 sharpens
        self.clip: float = float(self.cfg.get("clip", 20.0))

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
        adv = logits * self.scale
        if self.clip > 0:
            adv = torch.clamp(adv, -self.clip, self.clip)
        return adv

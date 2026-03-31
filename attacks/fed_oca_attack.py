# attacks/fed_oca_attack.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from attacks.base_attack import BaseAttack


class FedOCAAttack(BaseAttack):
    """Federated Over-Confidence Attack (Fed-OCA).

    Port of the 'overconfidence' idea to federated distillation:
      - Increase confidence of the predicted class without changing argmax.
      - Achieved by scaling logits (positive factor) and optionally adding margin
        to the top-1 logit.

    This is label-free (does not require y_public).
    """

    def __init__(
        self,
        is_malicious: bool,
        cfg: Optional[Dict[str, Any]] = None,
        client_id: Optional[int] = None,
        model=None,
    ) -> None:
        super().__init__(is_malicious=is_malicious, cfg=cfg, client_id=client_id, model=model)
        sub = (self.cfg or {}).get("fed_oca", {})
        self.scale: float = float(sub.get("scale", self.cfg.get("scale", 2.5)))
        self.margin: float = float(sub.get("margin", self.cfg.get("margin", 0.0)))
        self.clip: float = float(sub.get("clip", self.cfg.get("clip", 20.0)))

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
        if self.margin != 0.0:
            pred = adv.argmax(dim=-1)
            adv = adv.clone()
            adv.scatter_add_(dim=-1, index=pred.unsqueeze(-1), src=torch.full_like(pred.unsqueeze(-1).float(), self.margin))
        if self.clip > 0:
            adv = torch.clamp(adv, -self.clip, self.clip)
        return adv

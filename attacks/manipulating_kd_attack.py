# attacks/manipulating_kd_attack.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from attacks.base_attack import BaseAttack


class ManipulatingKDAttack(BaseAttack):
    """Manipulate KD signal by crafting a 'sharpened then biased' teacher distribution.

    Implementation:
      1) Convert logits -> probs.
      2) Apply temperature sharpening (T<1).
      3) Optionally add a small bias towards the predicted class to further amplify confidence.
      4) Convert back to logits via log(p + eps).

    This keeps argmax unchanged (since bias is towards current argmax), but
    alters calibration/uncertainty.
    """

    def __init__(
        self,
        is_malicious: bool,
        cfg: Optional[Dict[str, Any]] = None,
        client_id: Optional[int] = None,
        model=None,
    ) -> None:
        super().__init__(is_malicious=is_malicious, cfg=cfg, client_id=client_id, model=model)
        self.temperature: float = float(self.cfg.get("temperature", 0.5))  # <1 sharper
        self.bias: float = float(self.cfg.get("bias", 0.02))  # prob mass added to argmax
        self.eps: float = float(self.cfg.get("eps", 1e-8))

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

        T = max(self.temperature, 1e-6)
        probs = F.softmax(logits / T, dim=-1)  # sharpened

        if self.bias > 0:
            pred = probs.argmax(dim=-1)
            probs = probs * (1.0 - self.bias)
            probs.scatter_add_(dim=-1, index=pred.unsqueeze(-1), src=torch.full_like(pred.unsqueeze(-1).float(), self.bias))
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        adv_logits = torch.log(probs.clamp_min(self.eps))
        return adv_logits

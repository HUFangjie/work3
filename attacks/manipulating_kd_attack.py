# attacks/manipulating_kd_attack.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from attacks.base_attack import BaseAttack


class ManipulatingKDAttack(BaseAttack):
    """Manipulate KD signal with directional wrong-class transfer + entropy control."""

    def __init__(
        self,
        is_malicious: bool,
        cfg: Optional[Dict[str, Any]] = None,
        client_id: Optional[int] = None,
        model=None,
    ) -> None:
        super().__init__(is_malicious=is_malicious, cfg=cfg, client_id=client_id, model=model)
        mk_cfg = (self.cfg or {}).get("manipulating_kd", {})
        self.temperature: float = float(mk_cfg.get("temperature", 1.2))  # >1 smoother, less obvious
        self.transfer_mass: float = float(mk_cfg.get("transfer_mass", 0.20))
        self.target_offset: int = int(mk_cfg.get("target_offset", 1))
        self.entropy_floor_ratio: float = float(mk_cfg.get("entropy_floor_ratio", 0.35))
        self.min_prob: float = float(mk_cfg.get("min_prob", 1e-4))
        self.eps: float = float(mk_cfg.get("eps", 1e-8))

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
        probs = F.softmax(logits / T, dim=-1)
        B, C = probs.shape

        pred = probs.argmax(dim=-1)  # current most likely (often correct) class
        off = self.target_offset % max(1, C)
        if off == 0:
            off = 1
        target = (pred + off) % C

        # move probability mass from predicted class -> wrong target class
        mass = probs.gather(1, pred.unsqueeze(1)) * float(max(0.0, min(1.0, self.transfer_mass)))
        adv_probs = probs.clone()
        adv_probs.scatter_add_(1, pred.unsqueeze(1), -mass)
        adv_probs.scatter_add_(1, target.unsqueeze(1), mass)

        # entropy floor: mix with tempered benign distribution to avoid zero-entropy outlier
        benign_probs = F.softmax(logits, dim=-1)
        mix = float(max(0.0, min(1.0, self.entropy_floor_ratio)))
        adv_probs = (1.0 - mix) * adv_probs + mix * benign_probs

        # avoid long-tail collapse to exact same log-probability line
        adv_probs = adv_probs.clamp_min(self.min_prob)
        adv_probs = adv_probs / adv_probs.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        adv_logits = torch.log(adv_probs.clamp_min(self.eps))
        return adv_logits

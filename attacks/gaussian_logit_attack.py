# attacks/gaussian_logit_attack.py
"""
Simple Gaussian logit attack as a baseline.

For malicious clients:
    adv_logits = logits + N(0, sigma^2)

For benign clients:
    adv_logits = logits
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from attacks.base_attack import BaseAttack


class GaussianLogitAttack(BaseAttack):
    def __init__(
        self,
        is_malicious: bool,
        cfg: Optional[Dict[str, Any]] = None,
        client_id: Optional[int] = None,
        model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(
            is_malicious=is_malicious,
            cfg=cfg,
            client_id=client_id,
            model=model,
        )
        # 从总攻配置中取出 Gaussian 子配置
        g_cfg = (cfg or {}).get("gaussian", {})
        self.sigma: float = float(g_cfg.get("sigma", 0.1))

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

        # logits 可能在 CPU 或 GPU，保持原设备
        device = logits.device
        noise = torch.randn_like(logits, device=device) * self.sigma
        adv_logits = logits + noise
        return adv_logits

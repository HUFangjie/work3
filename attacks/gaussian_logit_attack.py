# attacks/gaussian_logit_attack.py
"""Adaptive Gaussian-style logit poisoning.

This implementation avoids naive zero-mean random noise and supports:
1) max_prediction_masking: suppress current top-1 class and boost a directed target.
2) targeted_mean_shift: apply positive-mean directional shift + adaptive noise.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from attacks.base_attack import BaseAttack


class GaussianLogitAttack(BaseAttack):
    def __init__(
        self,
        is_malicious: bool,
        cfg: Optional[Dict[str, Any]] = None,
        client_id: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__(is_malicious=is_malicious, cfg=cfg, client_id=client_id, model=model)
        g_cfg = (cfg or {}).get("gaussian", {})
        self.mode: str = str(g_cfg.get("mode", "targeted_mean_shift")).lower()
        self.sigma: float = float(g_cfg.get("sigma", 0.1))
        self.scale_with_span: bool = bool(g_cfg.get("scale_with_span", True))
        self.span_scale: float = float(g_cfg.get("span_scale", 0.35))
        self.span_bias: float = float(g_cfg.get("span_bias", 0.10))
        self.min_scale: float = float(g_cfg.get("min_scale", 0.5))
        self.max_scale: Optional[float] = g_cfg.get("max_scale", None)
        self.target_offset: int = int(g_cfg.get("target_offset", 1))
        self.mask_strength: float = float(g_cfg.get("mask_strength", 1.0))
        self.mix_with_original: float = float(g_cfg.get("mix_with_original", 0.15))

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

        num_classes = int(logits.shape[-1])
        top1 = torch.argmax(logits, dim=-1)
        offset = self.target_offset % max(1, num_classes)
        if offset == 0:
            offset = 1
        target = (top1 + offset) % num_classes

        # adaptive amplitude from benign logit span
        row_max = logits.max(dim=1, keepdim=True).values
        row_min = logits.min(dim=1, keepdim=True).values
        row_span = (row_max - row_min).clamp(min=1e-6)
        if self.scale_with_span:
            amp = row_span * self.span_scale + self.span_bias
        else:
            amp = torch.full_like(row_span, float(self.sigma))
        amp = amp.clamp(min=self.min_scale)
        if self.max_scale is not None:
            amp = amp.clamp(max=float(self.max_scale))

        if self.mode == "max_prediction_masking":
            adv_logits = logits.clone()
            adv_logits.scatter_(1, top1.unsqueeze(1), logits.gather(1, top1.unsqueeze(1)) - self.mask_strength * amp)
            adv_logits.scatter_(1, target.unsqueeze(1), logits.gather(1, target.unsqueeze(1)) + self.mask_strength * amp)
            noise = torch.randn_like(logits) * (self.sigma * amp)
            adv_logits = adv_logits + noise
        else:
            # targeted_mean_shift: non-zero mean directional perturbation + adaptive random component
            dir_vec = torch.zeros_like(logits)
            dir_vec.scatter_(1, top1.unsqueeze(1), -1.0)
            dir_vec.scatter_(1, target.unsqueeze(1), +1.0)
            mean_shift = amp * dir_vec
            noise = torch.randn_like(logits) * (self.sigma * amp)
            adv_logits = logits + mean_shift + noise

        if self.mix_with_original > 0.0:
            m = max(0.0, min(1.0, self.mix_with_original))
            adv_logits = (1.0 - m) * adv_logits + m * logits
        return adv_logits

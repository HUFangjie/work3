# attacks/label_flip_attack.py
"""Label-flip attack with stronger FD poisoning.

For malicious clients:
- private training: flip labels (classic local label-poisoning).
- public uplink logits: perform directed target flipping, using `y_public` when
  available; otherwise use model top-1 pseudo labels and flip to a fixed wrong
  direction. Forged logits are scaled from the original logit amplitude.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from attacks.base_attack import BaseAttack


class LabelFlipAttack(BaseAttack):
    """Classic label-flip attack.

    Core behavior:
    - On private supervised training, malicious clients replace true labels
      with wrong labels (cyclic mapping by default): y -> (y + 1) % K.
    - Logit attack path is kept for FD uplink poisoning compatibility.
    """
    def __init__(
        self,
        is_malicious: bool,
        cfg: Optional[Dict[str, Any]] = None,
        client_id: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__(is_malicious=is_malicious, cfg=cfg, client_id=client_id, model=model)
        lf_cfg = (self.cfg or {}).get("label_flip", {})
        self.flip_probability = float(lf_cfg.get("flip_probability", 1.0))
        self.target_logit = float(lf_cfg.get("target_logit", 10.0))
        self.non_target_logit = float(lf_cfg.get("non_target_logit", -10.0))
        self.use_hard_target = bool(lf_cfg.get("use_hard_target", True))
        self.fixed_target_offset = int(lf_cfg.get("fixed_target_offset", 1))
        self.amplitude_scale = float(lf_cfg.get("amplitude_scale", 1.5))
        self.amplitude_bias = float(lf_cfg.get("amplitude_bias", 0.5))
        self.min_amplitude = float(lf_cfg.get("min_amplitude", 2.0))
        self.mix_with_original = float(lf_cfg.get("mix_with_original", 0.2))

    def attack_private_labels(
        self,
        y: torch.Tensor,
        num_classes: Optional[int] = None,
    ) -> torch.Tensor:
        if (not self.is_malicious) or self.flip_probability <= 0.0:
            return y

        y_flat = y.long().view(-1)
        if num_classes is None:
            if y_flat.numel() == 0:
                return y
            num_classes = int(torch.max(y_flat).item()) + 1

        if self.flip_probability >= 1.0:
            mask = torch.ones_like(y_flat, dtype=torch.bool)
        else:
            mask = torch.rand(y_flat.shape[0], device=y_flat.device) < self.flip_probability

        y_adv = y_flat.clone()
        y_adv[mask] = (y_adv[mask] + 1) % int(num_classes)
        return y_adv.view_as(y)

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

        if self.flip_probability <= 0.0:
            return logits

        num_classes = int(logits.shape[-1])
        adv_logits = logits.clone()

        # 每个样本按概率决定是否执行 flip
        if self.flip_probability >= 1.0:
            flip_mask = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        else:
            flip_mask = (torch.rand(logits.shape[0], device=logits.device) < self.flip_probability)

        # Target source: real public label if available; otherwise top-1 pseudo label.
        if y_public is not None:
            src = y_public.to(logits.device).long().view(-1)[: logits.shape[0]]
        else:
            src = torch.argmax(logits, dim=-1)

        offset = self.fixed_target_offset % max(1, num_classes)
        if offset == 0:
            offset = 1
        target = (src + offset) % num_classes

        # Dynamic amplitude from current logits (avoid static +10/-10).
        # Use per-sample range so forged logits stay adaptive but strong.
        row_max = logits.max(dim=1, keepdim=True).values
        row_min = logits.min(dim=1, keepdim=True).values
        row_amp = (row_max - row_min).clamp(min=1e-6)
        amp = (row_amp * self.amplitude_scale + self.amplitude_bias).clamp(min=self.min_amplitude)

        if self.use_hard_target:
            forged = torch.full_like(logits, 0.0)
            forged = forged - amp
            forged.scatter_(1, target.unsqueeze(1), amp)
            if self.mix_with_original > 0.0:
                mix = max(0.0, min(1.0, self.mix_with_original))
                forged = (1.0 - mix) * forged + mix * logits
        else:
            forged = torch.roll(logits, shifts=offset, dims=-1)

        adv_logits[flip_mask] = forged[flip_mask]
        return adv_logits

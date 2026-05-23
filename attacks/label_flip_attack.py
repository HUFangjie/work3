# attacks/label_flip_attack.py
"""Label-flip attack in logit space.

This attack supports two modes:
1) If ``y_public`` is available, perform *true* label flipping against those
   labels by constructing targeted logits for wrong classes.
2) Otherwise, fallback to a class-dimension permutation (roll).
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

        # 如果拿到了真实标签，做“标签翻转”而不是单纯滚动通道
        if y_public is not None:
            y = y_public.to(logits.device).long().view(-1)
            y = y[: logits.shape[0]]
            target = (y + 1) % num_classes
            if self.use_hard_target:
                forged = torch.full_like(logits, self.non_target_logit)
                forged.scatter_(1, target.unsqueeze(1), self.target_logit)
            else:
                forged = torch.roll(logits, shifts=1, dims=-1)
            adv_logits[flip_mask] = forged[flip_mask]
            return adv_logits

        # 没有标签时 fallback 到固定置换
        rolled = torch.roll(logits, shifts=1, dims=-1)
        adv_logits[flip_mask] = rolled[flip_mask]
        return adv_logits

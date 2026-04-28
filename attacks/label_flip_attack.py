# attacks/label_flip_attack.py
"""
Logit-space label-flip style attack.

Since we only operate on logits (no ground-truth labels here), we simulate
label flipping by applying a fixed permutation to the class dimension.

For malicious clients:
    adv_logits[:, k] = logits[:, (k - 1) mod K]  (i.e., roll along class dim)

For benign clients:
    adv_logits = logits
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from attacks.base_attack import BaseAttack


class LabelFlipAttack(BaseAttack):
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

        # 沿类别维度做循环位移，实现固定的标签置换
        adv_logits = torch.roll(logits, shifts=1, dims=-1)
        return adv_logits

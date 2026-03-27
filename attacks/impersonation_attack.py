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

    def attack_logits(
        self,
        x_public: torch.Tensor,
        logits: torch.Tensor,
        round_idx: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> torch.Tensor:
        if not self.is_malicious:
            return logits

        target = get_farthest_benign_logits(device=logits.device)
        if target is None:
            # fallback: if pool not available, do nothing (safe default)
            return logits
        return target.to(logits.device).type_as(logits)

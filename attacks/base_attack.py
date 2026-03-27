# attacks/base_attack.py
"""
Base class for logit-space attacks in federated distillation.

Design:
  - Attack is per-client and can keep internal state.
  - Attack works on (x_public, logits) and may use the client's model
    (for TTA, etc.), but SHOULD NOT modify model parameters.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseAttack:
    """
    Base attack class.

    Args:
        is_malicious: whether this client is adversarial.
        cfg: dict of attack-wide configuration (config["attack_config"]).
        client_id: integer client id (for logging / per-client state).
        model: optional reference to the client's model (for TTA-based attacks).
    """

    def __init__(
        self,
        is_malicious: bool,
        cfg: Optional[Dict[str, Any]] = None,
        client_id: Optional[int] = None,
        model: Optional[nn.Module] = None,
    ) -> None:
        self.is_malicious = bool(is_malicious)
        self.cfg = cfg or {}
        self.client_id = client_id
        self.model = model

        # 尝试从 model 推断设备
        if model is not None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

    def attack_logits(
        self,
        x_public: torch.Tensor,
        logits: torch.Tensor,
        y_public: Optional[torch.Tensor] = None,
        round_idx: Optional[int] = None,
        global_step: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Default attack: identity mapping. Child classes override this.

        Args:
            x_public: public batch (tensor on any device).
            logits: model logits on public batch.
            round_idx: current communication round (optional).
            global_step: global step counter (optional).

        Returns:
            adv_logits: adversarial logits (same shape as logits).
        """
        return logits

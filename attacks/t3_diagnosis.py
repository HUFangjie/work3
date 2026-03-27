# attacks/t3_diagnosis.py
"""
Adaptive diagnosis module for T3.

Given logits on a public batch, compute entropy per sample and
select "hard" samples via a dynamic quantile threshold.
"""

from __future__ import annotations

from typing import Dict

import torch

from attacks.utils import compute_entropy, quantile_threshold
import logging

logger = logging.getLogger()


class AdaptiveDiagnosis:
    """
    Dynamic quantile-based hard sample selector.

    Args:
        rho: attack budget (fraction of samples to select), e.g., 0.2
    """

    def __init__(self, rho: float = 0.2) -> None:
        self.rho = float(rho)

    def select_hard_samples(
        self,
        logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Select hard samples based on entropy.

        Args:
            logits: [batch, num_classes]

        Returns:
            dict with keys:
              - "mask": bool tensor [batch], True for selected samples
              - "threshold": scalar tensor (entropy threshold tau_t)
              - "entropies": [batch] tensor of entropies
        """
        # 1. compute entropy per sample
        ent = compute_entropy(logits)  # [B]

        # 2. compute dynamic threshold: top rho in entropy
        #    i.e., select samples with entropy >= tau_t
        if ent.numel() == 0:
            mask = torch.zeros_like(ent, dtype=torch.bool)
            tau_t = torch.tensor(0.0, device=logits.device)
            return {"mask": mask, "threshold": tau_t, "entropies": ent}

        # we want upper (1 - rho) quantile if rho is attack budget on "hard" tail
        # e.g., rho=0.2 -> select top 20% highest entropies
        q = 1.0 - self.rho
        q = min(max(q, 0.0), 1.0)
        tau_val = quantile_threshold(ent, q)
        tau_t = torch.tensor(tau_val, device=logits.device, dtype=logits.dtype)

        mask = ent >= tau_t

        # 兜底：如果没有样本被选中，则至少选出 entropy 最大的一个
        if mask.sum() == 0:
            max_idx = torch.argmax(ent)
            mask[max_idx] = True

        # if self.rho > 0:
        #     logger.info(
        #         f"[T3-debug][diag_mask] B={ent.numel()}, hard={int(mask.sum().item())}"
        #     )

        return {
            "mask": mask,
            "threshold": tau_t,
            "entropies": ent,
        }

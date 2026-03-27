# defenses/defense_trimean.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from defenses.base_defense import BaseDefense


class TriMeanDefense(BaseDefense):
    """Coordinate-wise tri-mean aggregation.

    tri-mean = (Q1 + 2*Median + Q3) / 4
    computed per coordinate over clients.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__(device=device)

    def aggregate(
        self,
        client_logits: Dict[int, torch.Tensor],
        y_public: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not client_logits:
            raise ValueError("TriMeanDefense received empty client_logits.")

        mats = [t.detach().float() for t in client_logits.values()]  # each [B, C]
        X = torch.stack(mats, dim=0)  # [n, B, C]
        q1 = torch.quantile(X, 0.25, dim=0)
        q2 = torch.quantile(X, 0.50, dim=0)
        q3 = torch.quantile(X, 0.75, dim=0)
        return (q1 + 2.0 * q2 + q3) / 4.0

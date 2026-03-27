# defenses/defense_mkrum.py
from __future__ import annotations

from typing import Any, Dict, Optional

import math
import torch

from defenses.base_defense import BaseDefense


class MKrumDefense(BaseDefense):
    """Multi-Krum aggregation for logit tensors.

    Parameters (from defense_config["mkrum"]):
      - byz_frac: float, expected upper bound fraction of Byzantine clients (default 0.2)
      - f: int, explicit Byzantine upper bound (overrides byz_frac if provided)
      - m: int, number of selected clients to average (default n - f)

    Notes:
      - Distances are computed on flattened logits.
      - Output is the mean of selected m clients (Multi-Krum).
    """

    def __init__(
        self,
        device: torch.device,
        byz_frac: float = 0.2,
        f: Optional[int] = None,
        m: Optional[int] = None,
    ) -> None:
        super().__init__(device=device)
        self.byz_frac = float(byz_frac)
        self.f = f
        self.m = m

    @staticmethod
    def _flatten_logits(client_logits: Dict[int, torch.Tensor]) -> tuple[list[int], torch.Tensor]:
        cids = list(client_logits.keys())
        mats = [client_logits[cid].detach().float().view(-1) for cid in cids]
        X = torch.stack(mats, dim=0)  # [n, D]
        return cids, X

    def aggregate(
        self,
        client_logits: Dict[int, torch.Tensor],
        y_public: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not client_logits:
            raise ValueError("MKrumDefense received empty client_logits.")

        cids, X = self._flatten_logits(client_logits)
        n = X.size(0)

        if n == 1:
            return client_logits[cids[0]].detach()

        # Infer f and m.
        f = int(self.f) if self.f is not None else int(max(0, math.floor(self.byz_frac * n)))
        # Krum requires n >= 2f + 3 to be meaningful.
        f = min(f, max(0, (n - 3) // 2))

        m = int(self.m) if self.m is not None else max(1, n - f)
        m = max(1, min(m, n))

        # Pairwise squared distances [n, n]
        d = torch.cdist(X, X, p=2) ** 2

        # For each i, score = sum of distances to (n - f - 2) closest others.
        nb_count = max(1, n - f - 2)
        scores = []
        for i in range(n):
            di = d[i]
            vals, _ = torch.sort(di)
            score = vals[1 : 1 + nb_count].sum()  # exclude self
            scores.append(score)
        scores_t = torch.stack(scores)

        # Select m clients with smallest scores.
        _, sel_idx = torch.topk(scores_t, k=m, largest=False)
        sel_idx = sel_idx.tolist()

        sel_logits = [client_logits[cids[i]].detach().float() for i in sel_idx]
        out = torch.stack(sel_logits, dim=0).mean(dim=0)
        return out

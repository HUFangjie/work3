# defenses/defense_fedgraphguard.py
from __future__ import annotations

from typing import Any, Dict, Optional

import math
import torch

from defenses.base_defense import BaseDefense


class FedGraphGuardDefense(BaseDefense):
    """Graph-similarity defense for aggregating distillation logits.

    This implementation deliberately avoids sklearn's spectral embedding path.
    Tiny-ImageNet logits can produce sparse / disconnected k-NN graphs, which
    triggers repeated warnings in sklearn ("Graph is not fully connected...").
    Instead, we build a dense, positive-affinity graph with a small floor and
    use graph-degree trust scores to select and weight consistent clients.
    """

    def __init__(
        self,
        device: torch.device,
        keep_ratio: float = 0.7,
        min_clients_kept: int = 2,
        similarity_temperature: float = 0.5,
        affinity_floor: float = 1e-3,
        weight_temperature: float = 0.5,
        normalize_logits: bool = True,
    ) -> None:
        super().__init__(device=device)
        self.keep_ratio = float(keep_ratio)
        self.min_clients_kept = int(min_clients_kept)
        self.similarity_temperature = float(similarity_temperature)
        self.affinity_floor = float(affinity_floor)
        self.weight_temperature = float(weight_temperature)
        self.normalize_logits = bool(normalize_logits)

    def _client_features(self, client_logits: Dict[int, torch.Tensor]) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        client_ids = list(client_logits.keys())
        mats = [client_logits[cid].detach().float().to(self.device) for cid in client_ids]
        X = torch.stack([m.reshape(-1) for m in mats], dim=0)

        if self.normalize_logits:
            X = X - X.mean(dim=1, keepdim=True)
            X = X / X.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)

        return client_ids, X, torch.stack(mats, dim=0)

    def _dense_affinity(self, X: torch.Tensor) -> torch.Tensor:
        if self.normalize_logits:
            sim = (X @ X.t()).clamp(min=-1.0, max=1.0)
        else:
            Xn = X - X.mean(dim=1, keepdim=True)
            Xn = Xn / Xn.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            sim = (Xn @ Xn.t()).clamp(min=-1.0, max=1.0)

        temp = max(self.similarity_temperature, 1e-12)
        affinity = torch.exp((sim - 1.0) / temp)
        floor = max(self.affinity_floor, 0.0)
        if floor > 0.0:
            affinity = affinity.clamp_min(floor)
        affinity.fill_diagonal_(0.0)
        return affinity

    def aggregate(
        self,
        client_logits: Dict[int, torch.Tensor],
        y_public: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not client_logits:
            raise ValueError("FedGraphGuardDefense received empty client_logits.")

        _, X, stacked = self._client_features(client_logits)
        num_clients = int(stacked.shape[0])
        if num_clients == 1:
            return stacked[0]

        affinity = self._dense_affinity(X)
        trust = affinity.sum(dim=1) / max(num_clients - 1, 1)

        keep_n = int(math.ceil(num_clients * min(max(self.keep_ratio, 0.0), 1.0)))
        keep_n = max(min(keep_n, num_clients), min(self.min_clients_kept, num_clients))
        keep_idx = torch.topk(trust, k=keep_n, largest=True).indices

        temp = max(self.weight_temperature, 1e-12)
        kept_trust = trust[keep_idx]
        weights = torch.softmax((kept_trust - kept_trust.max()) / temp, dim=0)
        return (weights.view(-1, 1, 1) * stacked[keep_idx]).sum(dim=0)

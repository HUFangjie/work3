# defenses/defense_fedtgd.py
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.cluster import DBSCAN

from defenses.base_defense import BaseDefense


class FedTGDDefense(BaseDefense):
    """FedTGD robust aggregation for federated distillation logits.

    Steps:
      1) Normalize logits to [-10, 10] (optional).
      2) Top-k truncation per class-column across public samples.
      3) DBSCAN clustering on truncated logits vectors.
      4) Cosine similarity screening of cluster representatives, then average kept clusters.
    """

    def __init__(
        self,
        device: torch.device,
        k: int = 3,
        eps: float = 5.0,
        min_samples: int = 1,
        normalize_logits: bool = True,
    ) -> None:
        super().__init__(device=device)
        self.k = int(k)
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.normalize_logits = bool(normalize_logits)

    @staticmethod
    def _normalize_to_range(x: torch.Tensor, lo: float = -10.0, hi: float = 10.0) -> torch.Tensor:
        xmin = x.min()
        xmax = x.max()
        if float((xmax - xmin).abs().item()) < 1e-12:
            return torch.clamp(x, lo, hi)
        x01 = (x - xmin) / (xmax - xmin)
        return x01 * (hi - lo) + lo

    @staticmethod
    def _topk_truncate_per_column(L: torch.Tensor, k: int) -> torch.Tensor:
        H, D = L.shape
        k = max(1, min(k, H))
        V = torch.zeros_like(L)
        idx = torch.topk(L, k=k, dim=0, largest=True, sorted=False).indices  # [k, D]
        for j in range(D):
            V[idx[:, j], j] = L[idx[:, j], j]
        return V

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
        na = torch.norm(a, p=2).clamp_min(eps)
        nb = torch.norm(b, p=2).clamp_min(eps)
        return float(torch.dot(a, b).item() / (na * nb).item())

    @staticmethod
    def _threshold_t(scores: np.ndarray) -> float:
        # T = Q3 - 1.5*(Q3-Q1) (paper)
        q1 = float(np.quantile(scores, 0.25))
        q3 = float(np.quantile(scores, 0.75))
        return q3 - 1.5 * (q3 - q1)

    def aggregate(
        self,
        client_logits: Dict[int, torch.Tensor],
        y_public: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not client_logits:
            raise ValueError("FedTGDDefense received empty client_logits.")

        cids = list(client_logits.keys())
        Ls = [client_logits[cid].detach().float() for cid in cids]  # [H, D]
        if self.normalize_logits:
            Ls = [self._normalize_to_range(L) for L in Ls]

        Vs = [self._topk_truncate_per_column(L, self.k) for L in Ls]

        V_flat = np.stack([V.view(-1).cpu().numpy() for V in Vs], axis=0)
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="euclidean").fit_predict(V_flat)
        uniq = sorted(set(int(x) for x in labels.tolist()))
        clusters: Dict[int, list[int]] = {lab: [] for lab in uniq}
        for i, lab in enumerate(labels.tolist()):
            clusters[int(lab)].append(i)

        L_star, V_star = [], []
        for lab in uniq:
            idxs = clusters[lab]
            L_star.append(torch.stack([Ls[i] for i in idxs], dim=0).mean(dim=0))
            V_star.append(torch.stack([Vs[i] for i in idxs], dim=0).mean(dim=0))

        p = len(L_star)
        if p == 1:
            return L_star[0]

        V_vec = [v.view(-1) for v in V_star]
        scores = []
        for j in range(p):
            s = 0.0
            for k in range(p):
                if k == j:
                    continue
                s += self._cosine(V_vec[k], V_vec[j])
            scores.append(s / max(1, p - 1))

        scores_np = np.array(scores, dtype=np.float64)
        T = self._threshold_t(scores_np) if p >= 4 else float(np.min(scores_np))

        kept = [L_star[j] for j in range(p) if scores[j] > T]
        if len(kept) == 0:
            kept = L_star
        return torch.stack(kept, dim=0).mean(dim=0)

# defenses/defense_fedmdr.py
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from defenses.base_defense import BaseDefense


class FedMDRDefense(BaseDefense):
    """FedMDR-style robust aggregation for federated distillation logits.

    Core steps (paper-aligned):
      1) Compute each client's accuracy on labeled public data (y_public required).
      2) Dynamic trimming via boxplot lower fence on accuracies.
      3) Softmax mapping (temperature rho) -> weights.
      4) Weighted geometric median (Weiszfeld) over flattened logits.
    """

    def __init__(
        self,
        device: torch.device,
        rho: float = 10.0,
        max_iter: int = 50,
        eps: float = 1e-6,
        trim_on_weights: bool = True,
    ) -> None:
        super().__init__(device=device)
        self.rho = float(rho)
        self.max_iter = int(max_iter)
        self.eps = float(eps)
        self.trim_on_weights = bool(trim_on_weights)

    @staticmethod
    def _boxplot_lower_fence(arr: np.ndarray) -> float:
        q1 = float(np.quantile(arr, 0.25))
        q3 = float(np.quantile(arr, 0.75))
        return q1 - 1.5 * (q3 - q1)

    def _trimmed_softmax_weights(self, acc: np.ndarray) -> np.ndarray:
        thr_acc = self._boxplot_lower_fence(acc)
        keep = acc > thr_acc
        if keep.sum() == 0:
            keep[:] = True

        acc_kept = acc.copy()
        acc_kept[~keep] = -1e9

        mx = float(np.max(acc_kept[keep]))
        exps = np.zeros_like(acc_kept, dtype=np.float64)
        exps[keep] = np.exp(self.rho * (acc_kept[keep] - mx))
        denom = float(exps.sum()) if float(exps.sum()) > 0 else 1.0
        w = exps / denom

        if self.trim_on_weights:
            w_pos = w[w > 0]
            if w_pos.size >= 4:
                thr_w = self._boxplot_lower_fence(w_pos)
                w[w < max(0.0, thr_w)] = 0.0
                s = float(w.sum())
                if s > 0:
                    w = w / s
        return w

    def _weighted_geometric_median(self, X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Start from weighted mean
        y = (w[:, None] * X).sum(dim=0)

        for _ in range(self.max_iter):
            diff = X - y[None, :]
            dist = torch.norm(diff, p=2, dim=1).clamp_min(self.eps)
            num = ((w / dist)[:, None] * X).sum(dim=0)
            den = (w / dist).sum().clamp_min(self.eps)
            y_next = num / den
            if torch.norm(y_next - y, p=2).item() < 1e-5:
                y = y_next
                break
            y = y_next
        return y

    def aggregate(
        self,
        client_logits: Dict[int, torch.Tensor],
        y_public: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not client_logits:
            raise ValueError("FedMDRDefense received empty client_logits.")
        if y_public is None:
            raise ValueError("FedMDRDefense requires y_public (labels for the public batch).")

        y = y_public.detach().view(-1).cpu()
        mats = []
        accs = []

        for _, lg in client_logits.items():
            lg_f = lg.detach().float()
            mats.append(lg_f)
            pred = lg_f.argmax(dim=-1).view(-1).cpu()
            accs.append(float((pred == y).float().mean().item()))

        acc_np = np.array(accs, dtype=np.float64)
        w_np = self._trimmed_softmax_weights(acc_np)
        w = torch.tensor(w_np, dtype=torch.float32, device=mats[0].device)

        if float(w.sum().item()) <= 0:
            return torch.stack(mats, dim=0).mean(dim=0)

        X = torch.stack([m.view(-1) for m in mats], dim=0)  # [n, D]
        y_med = self._weighted_geometric_median(X, w)
        return y_med.view_as(mats[0])

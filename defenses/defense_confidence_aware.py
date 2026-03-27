from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

import torch

from defenses.base_defense import BaseDefense


class ConfidenceAwareDefense(BaseDefense):
    """Confidence-aware client downweighting for FD logit aggregation."""

    def __init__(
        self,
        device: torch.device,
        tau_conf: float = 0.9,
        hist_window: int = 5,
        beta: float = 2.0,
        eps: float = 1e-12,
        lambdas: Optional[List[float]] = None,
    ) -> None:
        super().__init__(device=device)
        self.tau_conf = float(tau_conf)
        self.hist_window = int(hist_window)
        self.beta = float(beta)
        self.eps = float(eps)
        self.lambdas = torch.tensor(lambdas or [1.0] * 9, dtype=torch.float32)
        if self.lambdas.numel() != 9:
            raise ValueError("ConfidenceAwareDefense expects exactly 9 lambda weights.")

        self.history: Dict[int, Dict[str, Deque]] = defaultdict(
            lambda: {
                "mean_conf": deque(maxlen=self.hist_window),
                "mean_entropy": deque(maxlen=self.hist_window),
                "high_conf_ratio": deque(maxlen=self.hist_window),
                "entropy_hist": deque(maxlen=self.hist_window),
            }
        )

        self.last_suspicion_scores: Dict[int, float] = {}
        self.last_weights: Dict[int, float] = {}

    def _wasserstein_1d(self, a: torch.Tensor, b: torch.Tensor, quantiles: int = 256) -> float:
        a = a.detach().float().view(-1)
        b = b.detach().float().view(-1)
        if a.numel() == 0 or b.numel() == 0:
            return 0.0

        q = torch.linspace(0.0, 1.0, steps=max(8, int(quantiles)), device=a.device)
        qa = torch.quantile(a, q)
        qb = torch.quantile(b.to(a.device), q)
        return float(torch.mean(torch.abs(qa - qb)).item())

    @staticmethod
    def _safe_mean(values: Deque) -> float:
        if len(values) == 0:
            return 0.0
        return float(sum(float(v) for v in values) / len(values))

    def aggregate(
        self,
        client_logits: Dict[int, torch.Tensor],
        y_public: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not client_logits:
            raise ValueError("ConfidenceAwareDefense received empty client_logits.")

        cids = list(client_logits.keys())
        logits_list = [client_logits[cid].detach().float() for cid in cids]

        mean_conf: Dict[int, float] = {}
        high_conf_ratio: Dict[int, float] = {}
        mean_entropy: Dict[int, float] = {}
        entropy_samples: Dict[int, torch.Tensor] = {}

        for cid, logits in zip(cids, logits_list):
            p = torch.softmax(logits, dim=-1)
            conf = torch.max(p, dim=-1).values
            ent = -(p * torch.log(p + self.eps)).sum(dim=-1)

            mean_conf[cid] = float(conf.mean().item())
            high_conf_ratio[cid] = float((conf > self.tau_conf).float().mean().item())
            mean_entropy[cid] = float(ent.mean().item())
            entropy_samples[cid] = ent.detach()

        entropy_pool = torch.cat([entropy_samples[cid] for cid in cids], dim=0)
        tau_ent = float(torch.quantile(entropy_pool, 0.2).item())

        features = []
        for cid in cids:
            ent_k = entropy_samples[cid]
            low_entropy_ratio = float((ent_k < tau_ent).float().mean().item())
            peer_entropy_dist = self._wasserstein_1d(ent_k, entropy_pool)

            h = self.history[cid]
            if len(h["mean_conf"]) > 0:
                delta_conf = abs(mean_conf[cid] - self._safe_mean(h["mean_conf"]))
                delta_entropy = abs(mean_entropy[cid] - self._safe_mean(h["mean_entropy"]))
                delta_high_ratio = abs(high_conf_ratio[cid] - self._safe_mean(h["high_conf_ratio"]))

                hist_entropy_ref = torch.cat(list(h["entropy_hist"]), dim=0)
                hist_entropy_dist = self._wasserstein_1d(ent_k, hist_entropy_ref)
            else:
                delta_conf = 0.0
                delta_entropy = 0.0
                delta_high_ratio = 0.0
                hist_entropy_dist = 0.0

            features.append(
                [
                    mean_conf[cid],
                    high_conf_ratio[cid],
                    mean_entropy[cid],
                    low_entropy_ratio,
                    peer_entropy_dist,
                    delta_conf,
                    delta_entropy,
                    delta_high_ratio,
                    hist_entropy_dist,
                ]
            )

        feat = torch.tensor(features, dtype=torch.float32)
        g = feat.clone()
        g[:, 2] = -g[:, 2]

        med = torch.median(g, dim=0).values
        mad = torch.median(torch.abs(g - med.unsqueeze(0)), dim=0).values + self.eps
        z = (g - med.unsqueeze(0)) / mad.unsqueeze(0)

        lambdas = self.lambdas.to(z.device)
        suspicion = (torch.relu(z) * lambdas.unsqueeze(0)).sum(dim=1)

        raw_w = torch.exp(-self.beta * suspicion)
        w = raw_w / raw_w.sum().clamp_min(self.eps)

        stacked = torch.stack(logits_list, dim=0)
        agg = (w.view(-1, 1, 1) * stacked).sum(dim=0)

        self.last_suspicion_scores = {cid: float(s.item()) for cid, s in zip(cids, suspicion)}
        self.last_weights = {cid: float(ww.item()) for cid, ww in zip(cids, w)}

        for cid in cids:
            h = self.history[cid]
            h["mean_conf"].append(mean_conf[cid])
            h["mean_entropy"].append(mean_entropy[cid])
            h["high_conf_ratio"].append(high_conf_ratio[cid])
            h["entropy_hist"].append(entropy_samples[cid].detach().cpu())

        return agg

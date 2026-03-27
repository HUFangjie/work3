# defenses/cronus_defense.py
"""
Cronus-style robust logit aggregation.

High-level idea:
  1) For each public batch, stack per-client logits: [N_clients, B, C].
  2) Transform logits -> probabilities with temperature scaling.
  3) Compute a robust "center" distribution per sample (here: coordinate-wise median).
  4) For each client, compute distance to center (L2 in probability space).
  5) Convert distances into trust weights, optionally trim the farthest clients.
  6) Aggregate ORIGINAL logits with these weights.

This approximates the behavior of Cronus-style defenses:
  - Clients whose predictions are inconsistent with the consensus are down-weighted.
  - Robust to a minority of strongly poisoned logits.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from defenses.base_defense import BaseDefense


class CronusDefense(BaseDefense):
    """
    Cronus-style robust logit aggregator.

    Args:
        device: torch.device.
        temperature: temperature for softmax when computing probabilities.
        gamma: sharpness of trust weights; larger -> more aggressively down-weight far clients.
        trimming_fraction: fraction of clients to drop per sample (e.g., 0.1 -> drop top 10% farthest).
        min_clients_kept: minimum number of clients that must remain per sample after trimming.
    """

    def __init__(
        self,
        device: torch.device,
        temperature: float = 1.0,
        gamma: float = 2.0,
        trimming_fraction: float = 0.0,
        min_clients_kept: int = 2,
    ) -> None:
        super().__init__(device=device)
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.trimming_fraction = float(trimming_fraction)
        self.min_clients_kept = int(min_clients_kept)

    def aggregate(
        self,
        client_logits: Dict[int, torch.Tensor],
        y_public: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Robust aggregation of logits for a single public batch.

        Args:
            client_logits: dict {client_id -> [B, C] logits}.

        Returns:
            teacher_logits: [B, C] aggregated logits.
        """
        if len(client_logits) == 0:
            raise ValueError("CronusDefense.aggregate() received empty client_logits.")

        # 1) stack logits: [N, B, C]
        #    按 client_id 排序只是为了结果稳定可复现
        client_ids = sorted(client_logits.keys())
        logits_list = [client_logits[cid].to(self.device) for cid in client_ids]
        stacked = torch.stack(logits_list, dim=0)  # [N, B, C]

        N, B, C = stacked.shape

        # 2) 转成概率空间（更适合做几何/距离）
        probs = F.softmax(stacked / self.temperature, dim=-1)  # [N, B, C]

        # 3) robust center：按客户端维度取概率的 coordinate-wise median
        center = probs.median(dim=0).values  # [B, C]

        # 4) 每个客户端到 center 的 L2 距离：dist[i,b] = ||p_i(b,.) - center(b,.)||_2
        dists = torch.norm(probs - center.unsqueeze(0), dim=-1)  # [N, B]

        # 5) 基于距离计算信任权重：w = exp(-gamma * dist)
        weights = torch.exp(-self.gamma * dists)  # [N, B]

        # 6) 可选 trimming：每个样本丢掉最远的 eta*N 个客户端
        if self.trimming_fraction > 0.0 and N > self.min_clients_kept:
            k_trim = int(self.trimming_fraction * N)
            k_trim = min(max(k_trim, 0), N - self.min_clients_kept)
            if k_trim > 0:
                # 对每个样本 b，找到距离最大的 k_trim 个客户端，把权重置零
                for b in range(B):
                    # dists[:, b]: [N]
                    _, idx_sorted = torch.sort(dists[:, b], descending=True)
                    to_drop = idx_sorted[:k_trim]
                    weights[to_drop, b] = 0.0

        # 7) 对每个样本 b 做权重归一化
        weight_sum = weights.sum(dim=0, keepdim=True).clamp_min(1e-12)  # [1, B]
        norm_w = weights / weight_sum  # [N, B]

        # 8) 加权聚合“原始 logits”而不是 probabilities
        #    teacher_logits[b, c] = sum_i norm_w[i,b] * stacked[i,b,c]
        norm_w_expanded = norm_w.unsqueeze(-1)  # [N, B, 1]
        teacher_logits = (norm_w_expanded * stacked).sum(dim=0)  # [B, C]

        return teacher_logits

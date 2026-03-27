# defenses/defense_entropy_clip.py
"""
EntropyClipDefense: simple example defense that drops clients whose
mean entropy is too high, then averages remaining logits.

This is不是严谨的论文级防御，只是一个示例占位，后续可以替换为更复杂逻辑。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from defenses.base_defense import BaseDefense
from defenses.defense_utils import filter_clients_by_entropy


class EntropyClipDefense(BaseDefense):
    def __init__(self, device: torch.device, max_entropy: float = 2.5) -> None:
        super().__init__(device)
        self.max_entropy = float(max_entropy)

    def aggregate(
        self,
        client_logits: Dict[int, torch.Tensor],
        y_public: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # 1. 按平均 entropy 过滤客户端
        filtered = filter_clients_by_entropy(client_logits, max_entropy=self.max_entropy)
        # 2. 对剩余客户端 logits 做简单平均
        logits_list = [logits.to(self.device) for logits in filtered.values()]
        stacked = torch.stack(logits_list, dim=0)
        return stacked.mean(dim=0)

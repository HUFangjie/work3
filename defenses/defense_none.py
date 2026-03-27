# defenses/defense_none.py
"""
NoDefense: simple average aggregation of logits.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from defenses.base_defense import BaseDefense


class NoDefense(BaseDefense):
    """
    Simple average over client logits.
    """

    def aggregate(
        self,
        client_logits: Dict[int, torch.Tensor],
        y_public: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        logits_list = [logits.to(self.device) for logits in client_logits.values()]
        stacked = torch.stack(logits_list, dim=0)  # [num_clients, B, C]
        return stacked.mean(dim=0)

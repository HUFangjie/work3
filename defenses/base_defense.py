# defenses/base_defense.py
"""
Base class for server-side defenses during logit aggregation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseDefense(ABC):
    """
    Base defense class.

    A defense takes per-client logits on a public batch and returns
    aggregated teacher logits.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @abstractmethod
    def aggregate(
        self,
        client_logits: Dict[int, torch.Tensor],
        y_public: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            client_logits: dict mapping client_id -> logits [batch, num_classes]

        Returns:
            teacher_logits: [batch, num_classes]
        """
        raise NotImplementedError

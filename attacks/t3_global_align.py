# attacks/t3_global_align.py
"""
Global distribution alignment module for T3.

Instead of matching only the mean entropy, we keep a history of
entropy samples and approximate the 1D Wasserstein-1 distance between
the current entropy distribution and the reference history.
"""

from __future__ import annotations

from typing import Optional

import torch


class GlobalAligner:
    """
    Track a history of reference entropies.

    Args:
        max_history: maximum number of entropy samples to keep.
    """

    def __init__(
        self,
        max_history: int = 1024,
    ) -> None:
        self.max_history = int(max_history)
        self._history: Optional[torch.Tensor] = None  # stored on CPU

    def update(self, entropies: torch.Tensor) -> None:
        """
        Update reference distribution using a batch of entropies [B].
        We only store them as CPU tensor in a FIFO buffer.
        """
        ent = entropies.detach().view(-1).cpu()
        if self._history is None:
            self._history = ent
        else:
            self._history = torch.cat([self._history, ent], dim=0)

        if self._history.numel() > self.max_history:
            self._history = self._history[-self.max_history :]

    def get_ref_distribution(self) -> Optional[torch.Tensor]:
        """
        Return a 1D tensor of historical entropies (on CPU), or None.
        """
        return self._history

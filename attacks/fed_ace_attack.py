# attacks/fed_ace_attack.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from attacks.base_attack import BaseAttack


class FedACEAttack(BaseAttack):
    """Federated ACE-style calibration sabotage (Fed-ACE).

    Goal (logit-space analogue):
      - Keep predicted labels unchanged (so accuracy largely preserved).
      - Increase miscalibration by:
          * making incorrect predictions more confident (push top-1 logit up)
          * making correct predictions less confident while keeping argmax (pull top-1 down a bit)

    This requires y_public to determine correctness w.r.t ground truth.
    If y_public is not provided, we fallback to Fed-OCA behavior (overconfidence).
    """

    def __init__(
        self,
        is_malicious: bool,
        cfg: Optional[Dict[str, Any]] = None,
        client_id: Optional[int] = None,
        model=None,
    ) -> None:
        super().__init__(is_malicious=is_malicious, cfg=cfg, client_id=client_id, model=model)
        sub = (self.cfg or {}).get("fed_ace", {})
        self.delta_wrong: float = float(sub.get("delta_wrong", self.cfg.get("delta_wrong", 6.0)))
        self.delta_correct: float = float(sub.get("delta_correct", self.cfg.get("delta_correct", 1.5)))
        self.clip: float = float(sub.get("clip", self.cfg.get("clip", 20.0)))

    @staticmethod
    def _safe_pull_top1(logits_row: torch.Tensor, top1: int, delta: float) -> torch.Tensor:
        """Reduce top1 logit by at most the margin to keep argmax unchanged."""
        # Find second best logit
        vals, idx = torch.topk(logits_row, k=2, dim=-1)
        top1_val = vals[0]
        top2_val = vals[1]
        # We can reduce top1 down to just above top2
        max_pull = (top1_val - top2_val - 1e-6).clamp_min(0.0)
        pull = torch.tensor(delta, device=logits_row.device).clamp_max(max_pull)
        out = logits_row.clone()
        out[top1] = out[top1] - pull
        return out

    def attack_logits(
        self,
        x_public: torch.Tensor,
        logits: torch.Tensor,
        y_public: Optional[torch.Tensor] = None,
        round_idx: Optional[int] = None,
        global_step: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not self.is_malicious:
            return logits

        adv = logits.detach().clone()

        pred = adv.argmax(dim=-1)
        if y_public is None:
            # Fallback: overconfidence on predicted class
            adv.scatter_add_(dim=-1, index=pred.unsqueeze(-1), src=torch.full_like(pred.unsqueeze(-1).float(), self.delta_wrong))
        else:
            y = y_public.view(-1).to(pred.device)
            correct = pred.eq(y)

            # Increase confidence for wrong predictions
            if (~correct).any():
                adv.scatter_add_(
                    dim=-1,
                    index=pred[~correct].unsqueeze(-1),
                    src=torch.full_like(pred[~correct].unsqueeze(-1).float(), self.delta_wrong),
                )

            # Decrease confidence for correct predictions (without changing argmax)
            if correct.any():
                for i in torch.where(correct)[0].tolist():
                    p = int(pred[i].item())
                    adv[i] = self._safe_pull_top1(adv[i], p, self.delta_correct)

        if self.clip > 0:
            adv = torch.clamp(adv, -self.clip, self.clip)
        return adv

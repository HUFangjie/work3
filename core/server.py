# core/server.py
from __future__ import annotations

from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from defenses.base_defense import BaseDefense


class Server:
    """
    Federated distillation server with a REAL global student model.

    Responsibilities:
      1) Aggregate client logits on public data (possibly after defenses).
      2) Train a server-side global student model to fit the aggregated logits.
      3) Provide the student for evaluation.
    """

    def __init__(
        self,
        device: torch.device,
        defense: BaseDefense,
        student_model: nn.Module,
        fd_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.device = device
        self.defense = defense

        self.student_model = student_model.to(device)

        fd_config = fd_config or {}
        self.temperature: float = float(fd_config.get("kd_temperature", 1.0))
        self.kd_alpha: float = float(fd_config.get("kd_alpha", 1.0))

        opt_name = str(fd_config.get("server_optimizer", fd_config.get("optimizer", "sgd"))).lower()
        lr = float(fd_config.get("server_lr", fd_config.get("lr", 0.01)))
        momentum = float(fd_config.get("momentum", 0.9))
        weight_decay = float(fd_config.get("weight_decay", 5e-4))

        if opt_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.student_model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.student_model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported server optimizer: {opt_name}")

    # -------------------------
    # FD core: aggregation
    # -------------------------
    def aggregate_logits(
        self,
        client_logits: Dict[int, torch.Tensor],
        y_public: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate client logits on the public batch via the defense object.
        Returns a tensor (usually on CPU depending on defense implementation).
        """
        return self.defense.aggregate(client_logits, y_public=y_public)

    # -------------------------
    # FD core: student distill
    # -------------------------
    def distill_student_on_public(
        self,
        x_public: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> float:
        """
        One KD step on server student:
          minimize KL( softmax(teacher/T) || softmax(student/T) ) * T^2
        """
        self.student_model.train()
        self.optimizer.zero_grad()

        x_public = x_public.to(self.device)

        # teacher logits might be on CPU; move to device
        teacher_logits = teacher_logits.to(self.device).detach()

        T = self.temperature
        student_logits = self.student_model(x_public)

        log_p_s = F.log_softmax(student_logits / T, dim=-1)
        p_t = F.softmax(teacher_logits / T, dim=-1)

        kd_loss = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
        loss = self.kd_alpha * kd_loss

        loss.backward()
        self.optimizer.step()

        return float(loss.detach().cpu().item())

    def get_student(self) -> nn.Module:
        return self.student_model

# core/client.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from attacks.base_attack import BaseAttack


class Client:
    """
    Federated distillation client.

    Args:
        client_id: integer ID of the client.
        model: local model (nn.Module).
        private_loader: DataLoader for the client's private data.
        device: torch.device on which the model runs.
        fd_config: configuration dict (config["fd_config"]).
        attack: optional BaseAttack object; if None, use identity attack.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        private_loader: DataLoader,
        device: torch.device,
        fd_config: Dict[str, Any],
        attack: Optional[BaseAttack] = None,
    ) -> None:
        self.client_id = client_id
        # ------------------------------------------------------------------
        # Memory note (important for many-client FD):
        #   Keeping *all* client models resident on GPU will quickly exhaust
        #   memory, especially for large backbones (e.g., ResNet-50 on
        #   Tiny-ImageNet). We therefore support optional CPU offloading.
        #
        # Default behavior:
        #   - If running on CUDA, we *prefer* CPU residency and only move the
        #     model to GPU for the specific computation (public logits / local
        #     training) and then move it back.
        #   - This is controlled by fd_config["offload_clients_to_cpu"].
        #
        # This change is intentionally minimal and does not alter FD logic.
        # ------------------------------------------------------------------
        self.device = device
        self.fd_config = fd_config

        self.offload_to_cpu: bool = bool(fd_config.get("offload_clients_to_cpu", device.type == "cuda"))

        # Keep model on CPU by default when offloading is enabled.
        self.model = model.to("cpu") if self.offload_to_cpu else model.to(device)
        self.private_loader = private_loader

        self.temperature: float = fd_config.get("kd_temperature", 1.0)
        self.kd_alpha: float = fd_config.get("kd_alpha", 1.0)

        opt_name = fd_config.get("optimizer", "sgd").lower()
        lr = fd_config.get("lr", 0.01)
        momentum = fd_config.get("momentum", 0.9)
        weight_decay = fd_config.get("weight_decay", 5e-4)

        if opt_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        self.ce_loss = nn.CrossEntropyLoss()

        # Track whether optimizer state has been moved to CUDA (for offload efficiency)
        self._opt_state_on_cuda: bool = False

        # 攻击模块（可以是恒等映射）
        if attack is None:
            self.attack = BaseAttack(is_malicious=False, cfg=None, client_id=client_id)
        else:
            self.attack = attack

    # ------------------------------------------------------------------
    # Device / memory helpers
    # ------------------------------------------------------------------
    def _move_optimizer_state(self, device: torch.device) -> None:
        """Move optimizer state tensors to the given device.

        This is necessary when offloading models between CPU/GPU; otherwise
        momentum/Adam buffers can remain on GPU and leak memory.
        """
        try:
            for st in self.optimizer.state.values():
                if isinstance(st, dict):
                    for k, v in list(st.items()):
                        if torch.is_tensor(v):
                            st[k] = v.to(device)
        except Exception:
            # Best-effort; if something unexpected happens, do not crash.
            return

    def _ensure_model_on_infer(self, device: torch.device) -> None:
        """Ensure *model* lives on `device` (inference path).

        For public-logits inference we do NOT need optimizer state on GPU.
        Moving optimizer buffers each micro-batch is a major slowdown.
        """
        try:
            cur = next(self.model.parameters()).device
        except StopIteration:
            cur = device
        if cur != device:
            self.model.to(device)

    def _ensure_model_on_train(self, device: torch.device) -> None:
        """Ensure model + optimizer state live on `device` (training path)."""
        try:
            cur = next(self.model.parameters()).device
        except StopIteration:
            cur = device
        if cur != device:
            self.model.to(device)
        # optimizer state follows training device to keep momentum/Adam buffers consistent
        if device.type == "cuda":
            self._move_optimizer_state(device)
            self._opt_state_on_cuda = True
        else:
            self._move_optimizer_state(device)
            self._opt_state_on_cuda = False

    def _maybe_offload(self) -> None:
        """Offload model back to CPU when enabled.

        We only move optimizer state back if it has been moved to CUDA before.
        """
        if self.offload_to_cpu and self.device.type == "cuda":
            self.model.to("cpu")
            if self._opt_state_on_cuda:
                self._move_optimizer_state(torch.device("cpu"))
                self._opt_state_on_cuda = False
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])

    # ------------------------------------------------------------------
    # Federated distillation related methods
    # ------------------------------------------------------------------
    def compute_public_logits(
        self,
        x_public: torch.Tensor,
        y_public: Optional[torch.Tensor] = None,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute (possibly attacked) logits on a batch of public data.

        Returns CPU tensor for easier "communication".
        """
        self.model.eval()

        # Micro-batch inference to cap activation peak.
        micro_bs = int(self.fd_config.get("public_logits_micro_bs", 32))
        micro_bs = max(1, micro_bs)
        use_amp = bool(self.fd_config.get("public_logits_amp", True))

        # Ensure model is on GPU for the actual forward if needed (model only).
        self._ensure_model_on_infer(self.device)

        uplink_dtype = str(self.fd_config.get("uplink_logits_dtype", "float32")).lower()

        # IMPORTANT: do NOT wrap the whole block in inference_mode.
        # T3-style attacks run gradient-based optimization in logit-space.
        # If we compute adversarial logits under inference_mode/no_grad,
        # loss.backward() will fail with "does not require grad".
        need_attack_grad = bool(getattr(self.attack, "is_malicious", False))

        adv_chunks: list[torch.Tensor] = []
        for xb in x_public.split(micro_bs, dim=0):
            xb = xb.to(self.device, non_blocking=True)

            # Forward pass does not need gradients.
            with torch.no_grad():
                if use_amp and (self.device.type == "cuda"):
                    with torch.cuda.amp.autocast():
                        logits = self.model(xb)
                else:
                    logits = self.model(xb)

            # Apply attack in logit space.
            # For malicious clients, explicitly enable grad so PGD can backprop
            # through the perturbation variable (not through the model).
            if need_attack_grad:
                with torch.enable_grad():
                    adv_logits = self.attack.attack_logits(
                        x_public=xb,
                        logits=logits,
                        y_public=y_public,
                        round_idx=round_idx,
                    )
            else:
                # benign / non-gradient attacks
                adv_logits = self.attack.attack_logits(
                    x_public=xb,
                    logits=logits,
                    y_public=y_public,
                    round_idx=round_idx,
                )

            # Immediately move to CPU to avoid GPU accumulation.
            out_cpu = adv_logits.detach().float().cpu()
            if uplink_dtype in ("float16", "fp16", "half"):
                out_cpu = out_cpu.half()
            adv_chunks.append(out_cpu)
            del xb, logits, adv_logits

        self._maybe_offload()
        return torch.cat(adv_chunks, dim=0) if len(adv_chunks) > 1 else adv_chunks[0]

    def distill_on_public(
        self,
        x_public: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> float:
        # Distillation is training => keep on target device during this call.
        self._ensure_model_on_train(self.device)
        self.model.train()
        self.optimizer.zero_grad()

        x_public = x_public.to(self.device)
        teacher_logits = teacher_logits.to(self.device)

        T = self.temperature
        student_logits = self.model(x_public)
        log_p_s = F.log_softmax(student_logits / T, dim=-1)
        p_t = F.softmax(teacher_logits / T, dim=-1)

        kd_loss = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

        loss = self.kd_alpha * kd_loss
        loss.backward()
        self.optimizer.step()

        self._maybe_offload()

        return float(loss.detach().cpu().item())

    def train_on_private(self, local_epochs: int = 1) -> float:
        # Private training is training => keep on target device during this call.
        self._ensure_model_on_train(self.device)
        self.model.train()
        total_loss = 0.0
        total_steps = 0

        for _ in range(local_epochs):
            for x, y in self.private_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                if y.ndim > 1:

                    y = y.view(-1)

                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.ce_loss(logits, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_steps += 1

        if total_steps == 0:
            return 0.0
        avg = total_loss / float(total_steps)
        self._maybe_offload()
        return avg

    # ------------------------------------------------------------------
    def eval_mode(self) -> None:
        self.model.eval()

    def train_mode(self) -> None:
        self.model.train()

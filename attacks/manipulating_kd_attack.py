# attacks/manipulating_kd_attack.py
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from attacks.base_attack import BaseAttack


class ManipulatingKDAttack(BaseAttack):
    """Full-local ManipulatingKD attack with shared malicious logits.

    This attack is intentionally computed from the pre-attack logits of all
    selected clients.  The optimized tensor is then uploaded identically by
    every malicious client.
    """

    def __init__(
        self,
        is_malicious: bool,
        cfg: Optional[Dict[str, Any]] = None,
        client_id: Optional[int] = None,
        model=None,
    ) -> None:
        super().__init__(is_malicious=is_malicious, cfg=cfg, client_id=client_id, model=model)
        mk_cfg = (self.cfg or {}).get("manipulating_kd", {})
        self.tau: float = float(mk_cfg.get("tau", 5.0))
        self.num_ascent_steps: int = int(mk_cfg.get("num_ascent_steps", 80))
        self.attack_lr: float = float(mk_cfg.get("attack_lr", 0.35))
        self.dual_lr: float = float(mk_cfg.get("dual_lr", 0.1))
        self.init_ratio: float = float(mk_cfg.get("init_ratio", 0.10))
        self.boundary_ratio: float = float(mk_cfg.get("boundary_ratio", 1.0))
        self.num_restarts: int = int(mk_cfg.get("num_restarts", 4))
        self.grad_eps: float = float(mk_cfg.get("grad_eps", 1e-8))
        self.eps: float = float(mk_cfg.get("eps", 1e-12))
        self.last_overhead: Dict[str, Any] = {}

    def attack_logits(
        self,
        x_public: torch.Tensor,
        logits: torch.Tensor,
        y_public: Optional[torch.Tensor] = None,
        round_idx: Optional[int] = None,
        global_step: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """ManipulatingKD needs all clients' raw logits; per-client calls are identity."""
        return logits

    def _project_to_mse_ball(
        self,
        logits: torch.Tensor,
        center: torch.Tensor,
        radius: torch.Tensor,
    ) -> torch.Tensor:
        """Project each sample's logits into its per-sample MSE stealth ball."""
        diff = logits - center
        mse = (diff ** 2).mean(dim=-1, keepdim=True)
        scale = torch.sqrt(radius.unsqueeze(-1).clamp_min(0.0) / mse.clamp_min(self.eps))
        scale = torch.minimum(scale, torch.ones_like(scale))
        return center + diff * scale

    def _normalize_per_sample(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize each sample vector to unit RMS to avoid tiny KL gradients."""
        rms = torch.sqrt((tensor ** 2).mean(dim=-1, keepdim=True)).clamp_min(self.grad_eps)
        return tensor / rms

    def _kl_reference_to_poisoned(
        self,
        reference_logits: torch.Tensor,
        poisoned_aggregate: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        ref_prob = torch.softmax(reference_logits / tau, dim=-1)
        poisoned_log_prob = torch.log_softmax(poisoned_aggregate / tau, dim=-1)
        return (
            ref_prob
            * (
                torch.log(ref_prob.clamp_min(self.eps))
                - poisoned_log_prob
            )
        ).sum(dim=-1)

    def _select_best_candidate(
        self,
        candidates: torch.Tensor,
        reference_logits: torch.Tensor,
        benign_sum: torch.Tensor,
        num_malicious: int,
        num_clients: int,
        tau: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pick the highest-KL candidate independently for each public sample."""
        poisoned_aggregate = (benign_sum.unsqueeze(0) + num_malicious * candidates) / num_clients
        kl = self._kl_reference_to_poisoned(
            reference_logits=reference_logits.unsqueeze(0),
            poisoned_aggregate=poisoned_aggregate,
            tau=tau,
        )
        best_idx = kl.argmax(dim=0)
        gather_idx = best_idx.view(1, -1, 1).expand(1, candidates.shape[1], candidates.shape[2])
        best_logits = candidates.gather(dim=0, index=gather_idx).squeeze(0)
        best_kl = kl.gather(dim=0, index=best_idx.view(1, -1)).squeeze(0)
        return best_logits, best_kl

    def build_shared_malicious_logits(
        self,
        all_client_logits: torch.Tensor,
        malicious_ids: Sequence[int],
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Optimize one shared malicious upload from all clients' raw logits.

        Args:
            all_client_logits: Tensor[M, N, L] containing pre-attack raw logits.
            malicious_ids: Malicious client indices within the first dimension of
                all_client_logits.
            round_idx: Current communication round for overhead logging.

        Returns:
            Tensor[N, L] to be uploaded by every malicious client.
        """
        t0 = time.perf_counter()

        if all_client_logits.dim() != 3:
            raise ValueError(
                "ManipulatingKD expects all_client_logits with shape [M, N, L], "
                f"got {tuple(all_client_logits.shape)}"
            )

        M = int(all_client_logits.shape[0])
        N = int(all_client_logits.shape[1])
        malicious_ids = [int(i) for i in malicious_ids]
        C = len(malicious_ids)
        if C == 0:
            return all_client_logits.mean(dim=0).detach()
        if min(malicious_ids) < 0 or max(malicious_ids) >= M:
            raise ValueError(f"malicious_ids must index the M={M} clients in all_client_logits")

        tau = max(float(self.tau), 1e-12)
        all_client_logits = all_client_logits.detach().float()

        reference_logits = all_client_logits.mean(dim=0)
        center_logits = all_client_logits.mean(dim=0)

        distance = ((all_client_logits - center_logits.unsqueeze(0)) ** 2).mean(dim=-1)
        max_ref_distance = distance.max(dim=0).values
        radius = C * max_ref_distance
        radius_sqrt = torch.sqrt(radius.clamp_min(self.eps)).unsqueeze(-1)

        malicious_set = set(malicious_ids)
        benign_ids = [i for i in range(M) if i not in malicious_set]
        if len(benign_ids) > 0:
            benign_sum = all_client_logits[benign_ids].sum(dim=0)
        else:
            benign_sum = torch.zeros_like(reference_logits)

        # Keep the paper's malicious-mean initialization, but do not rely on
        # local KL gradients from that point.  We first build several feasible
        # boundary candidates and choose the strongest one per sample; this uses
        # the full stealth budget immediately and avoids weak attacks caused by
        # the near-zero KL gradient around the reference aggregate.
        base_logits = (
            all_client_logits[malicious_ids]
            .mean(dim=0)
            .detach()
            .clone()
        )
        base_logits = self._project_to_mse_ball(
            logits=base_logits,
            center=center_logits,
            radius=radius,
        )

        candidates = [base_logits]

        direction = self._normalize_per_sample(torch.randn_like(base_logits))
        jitter = self.init_ratio * radius_sqrt * direction
        candidates.append(
            self._project_to_mse_ball(
                logits=base_logits + jitter,
                center=center_logits,
                radius=radius,
            )
        )

        ref_prob = torch.softmax(reference_logits / tau, dim=-1)

        top_class = ref_prob.argmax(dim=-1, keepdim=True)
        bottom_class = ref_prob.argmin(dim=-1, keepdim=True)
        top_bottom_direction = torch.zeros_like(base_logits)
        top_bottom_direction.scatter_add_(1, top_class, -torch.ones_like(top_class, dtype=base_logits.dtype))
        top_bottom_direction.scatter_add_(1, bottom_class, torch.ones_like(bottom_class, dtype=base_logits.dtype))
        candidates.append(
            center_logits
            + self.boundary_ratio * radius_sqrt * self._normalize_per_sample(top_bottom_direction)
        )

        anti_prob_direction = ref_prob.mean(dim=-1, keepdim=True) - ref_prob
        candidates.append(
            center_logits
            + self.boundary_ratio * radius_sqrt * self._normalize_per_sample(anti_prob_direction)
        )

        anti_logit_direction = reference_logits.mean(dim=-1, keepdim=True) - reference_logits
        candidates.append(
            center_logits
            + self.boundary_ratio * radius_sqrt * self._normalize_per_sample(anti_logit_direction)
        )

        for _ in range(max(0, self.num_restarts)):
            candidates.append(
                center_logits
                + self.boundary_ratio
                * radius_sqrt
                * self._normalize_per_sample(torch.randn_like(base_logits))
            )

        candidate_logits = torch.stack(
            [
                self._project_to_mse_ball(logits=cand, center=center_logits, radius=radius)
                for cand in candidates
            ],
            dim=0,
        )

        with torch.no_grad():
            shared_malicious_logits, best_kl = self._select_best_candidate(
                candidates=candidate_logits,
                reference_logits=reference_logits,
                benign_sum=benign_sum,
                num_malicious=C,
                num_clients=M,
                tau=tau,
            )
            best_logits = shared_malicious_logits.detach().clone()

        shared_malicious_logits = shared_malicious_logits.detach().clone().requires_grad_(True)

        dual = torch.zeros(N, device=all_client_logits.device)
        step_scale = max(float(self.attack_lr), 0.0) * radius_sqrt

        for _ in range(max(0, self.num_ascent_steps)):
            poisoned_aggregate = (benign_sum + C * shared_malicious_logits) / M
            kl = self._kl_reference_to_poisoned(
                reference_logits=reference_logits,
                poisoned_aggregate=poisoned_aggregate,
                tau=tau,
            )

            attack_distance = ((shared_malicious_logits - center_logits) ** 2).mean(dim=-1)
            violation = attack_distance - radius

            lagrangian = (kl - dual.detach() * violation).mean()
            grad = torch.autograd.grad(lagrangian, shared_malicious_logits)[0]
            grad = self._normalize_per_sample(grad)

            with torch.no_grad():
                improved = kl.detach() > best_kl
                best_kl = torch.where(improved, kl.detach(), best_kl)
                best_logits = torch.where(improved.unsqueeze(-1), shared_malicious_logits.detach(), best_logits)

                # Projected normalized ascent is deliberately used in addition
                # to the dual update: the KL gradient near the reference is very
                # small, so raw gradient ascent often never reaches the useful
                # stealth boundary within a public batch.
                shared_malicious_logits = shared_malicious_logits + step_scale * grad
                shared_malicious_logits = self._project_to_mse_ball(
                    logits=shared_malicious_logits,
                    center=center_logits,
                    radius=radius,
                )

                post_distance = ((shared_malicious_logits - center_logits) ** 2).mean(dim=-1)
                post_violation = post_distance - radius
                dual += self.dual_lr * post_violation.detach()
                dual.clamp_(min=0.0)

            shared_malicious_logits = shared_malicious_logits.detach().clone().requires_grad_(True)

        with torch.no_grad():
            poisoned_aggregate = (benign_sum + C * shared_malicious_logits) / M
            final_kl = self._kl_reference_to_poisoned(
                reference_logits=reference_logits,
                poisoned_aggregate=poisoned_aggregate,
                tau=tau,
            ).detach()
            improved = final_kl > best_kl
            shared_malicious_logits = torch.where(
                improved.unsqueeze(-1),
                shared_malicious_logits.detach(),
                best_logits,
            )
            final_kl = torch.where(improved, final_kl, best_kl)

            final_attack_distance = ((shared_malicious_logits - center_logits) ** 2).mean(dim=-1)
            final_violation = final_attack_distance - radius
            self.last_overhead = {
                "round": -1 if round_idx is None else int(round_idx),
                "t_total_s": float(time.perf_counter() - t0),
                "t_diag_s": 0.0,
                "t_tta_s": 0.0,
                "t_pgd_s": float(time.perf_counter() - t0),
                "hard_cnt": int((final_violation > 0).sum().item()),
                "mean_kl": float(final_kl.mean().item()) if N > 0 else 0.0,
                "max_violation": float(final_violation.max().item()) if N > 0 else 0.0,
            }
            return shared_malicious_logits.detach().clone()

# attacks/t3_variance_opt.py
"""
Variance-aware adversarial optimization in logit space for T3.

We treat logits as attack variables and apply PGD in L_infinity norm
to maximize "sharpness" (overconfidence) while:
  - keeping aleatoric entropy close to a target (stealth),
  - focusing more on high-epistemic examples (variance-aware),
  - aligning the entropy distribution with a global reference (W1).
"""

from __future__ import annotations

from typing import Optional

import torch

from attacks.utils import compute_entropy
import logging

logger = logging.getLogger()


# 用于近似计算两个一维分布（这里是熵分布）之间的 Wasserstein-1 距离
def wasserstein_1d(
    current: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    Approximate 1D Wasserstein-1 distance between two entropy distributions.

    Args:
        current: [B] current entropies (on device).
        reference: [N] reference entropies (any device).

    Returns:
        scalar tensor, differentiable w.r.t current.
    """
    c = current.view(-1)  # 展平成一维向量
    r = reference.view(-1).to(current.device)

    if c.numel() == 0 or r.numel() == 0:
        return torch.tensor(0.0, device=current.device, dtype=current.dtype)

    c_sorted, _ = torch.sort(c)     # 排序
    r_sorted, _ = torch.sort(r)

    n = min(c_sorted.numel(), r_sorted.numel())     # 对齐长度
    c_sorted = c_sorted[:n]
    r_sorted = r_sorted[:n]

    return torch.mean(torch.abs(c_sorted - r_sorted))    # 距离计算


class VarianceAwareLogitOptimizer:
    """
    PGD-style optimizer in logit space.

    Objective (conceptual) per sample i:
        L_total = (ent_i)                    # utility: push toward overconfidence
                  + lambda_stealth * (ent_i - H_ale_i)^2
                  + lambda_align * W1(ent, history)   (global level)
    and we use variance_weights to modulate how strongly we attack high-epistemic
    examples (variance-aware), controlled by lambda_epistemic.
    """

    def __init__(
        self,
        epsilon: float = 0.5,
        step_size: float = 0.1,
        num_steps: int = 100,         # 扰动更新次数
        lambda_stealth: float = 1.0,
        lambda_epistemic: float = 1.0,
        lambda_align: float = 0.0,
        debug: bool = False,
    ) -> None:
        self.epsilon = float(epsilon)
        self.step_size = float(step_size)
        self.num_steps = int(num_steps)
        self.lambda_stealth = float(lambda_stealth)
        self.lambda_epistemic = float(lambda_epistemic)
        self.lambda_align = float(lambda_align)
        self.debug = bool(debug)

    def optimize(
        self,
        base_logits: torch.Tensor,
        variance_weights: Optional[torch.Tensor] = None,
        target_aleatoric_entropy: Optional[torch.Tensor] = None,
        ref_entropies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run PGD in logit space.

        Args:
            base_logits: [B,C] logits from client's model (detached).
            variance_weights: [B] non-negative weights; if None, use uniform.
            target_aleatoric_entropy: [B] or scalar, aleatoric entropy target per sample.
            ref_entropies: [N] historical entropy samples for W1 alignment.

        Returns:
            adv_logits: [B,C] adversarial logits.
        """
        # print("lambda_stealth", self.lambda_stealth)
        # print("lambda_align", self.lambda_align)
        # print("lambda_align", self.lambda_epistemic)
        device = base_logits.device
        z0 = base_logits.detach()    # z0 常量PGD 的“起点”

        # PGD perturbation variable  初始扰动 0
        delta = torch.zeros_like(z0, device=device, requires_grad=True)

        if self.debug:
            with torch.no_grad():
                ent_init = compute_entropy(z0)
                ent_init_mean = float(ent_init.mean().item())
        else:
            ent_init_mean = 0.0

        # variance-based weights，用于“攻击力度调度”
        if variance_weights is not None:
            w = variance_weights.detach().to(device)
            w = w + 1e-6
            w = w / w.mean()
        else:
            w = torch.ones(z0.size(0), device=device)

        # 处理 target_aleatoric_entropy
        if target_aleatoric_entropy is not None:
            t_ale = target_aleatoric_entropy.detach().to(device)
            if t_ale.dim() == 0:
                t_ale = t_ale.expand(z0.size(0))
            else:
                t_ale = t_ale.view(-1)
                if t_ale.numel() != z0.size(0):
                    # broadcast as scalar
                    t_ale = t_ale.mean().expand(z0.size(0))
        else:
            t_ale = None

        for _ in range(self.num_steps):
            adv_logits = z0 + delta     #计算当前的恶意 logits
            ent = compute_entropy(adv_logits)  # [B] 对当前恶意 logits 计算每个样本的熵

            # 1) Utility: 让总熵下降（过度自信）
            l_utility = ent  # [B]

            # 2) Stealth: 对齐 aleatoric 熵目标 (Eq.13)
            if t_ale is not None:
                l_stealth = (ent - t_ale) ** 2  # [B] 算当前熵和目标 aleatoric 熵之间的平方差
            else:
                l_stealth = torch.zeros_like(ent)

            # 3) Variance-aware weighting (Epistemic)
            #    高 variance 样本（epistemic 大）→ 权重更高
            #    lambda_epistemic 控制这部分的强度
            w_eff = 1.0 + self.lambda_epistemic * (w - 1.0)
            weighted_utility = l_utility * w_eff

            loss = weighted_utility.mean() + self.lambda_stealth * l_stealth.mean()

            # 4) Global distribution alignment via 1D W1
            if ref_entropies is not None and self.lambda_align > 0.0:
                w1 = wasserstein_1d(ent, ref_entropies.to(device))
                loss = loss + self.lambda_align * w1

            # PGD step (gradient descent on loss)
            loss.backward()

            with torch.no_grad():
                grad_sign = delta.grad.sign()
                delta.data = delta.data - self.step_size * grad_sign
                delta.data.clamp_(-self.epsilon, self.epsilon)

            delta.grad.zero_()


        if self.debug:
            with torch.no_grad():
                adv_tmp = z0 + delta
                ent_final = compute_entropy(adv_tmp)
                ent_final_mean = float(ent_final.mean().item())

                if variance_weights is not None:
                    w_mean = float(w.mean().item())
                    w_min = float(w.min().item())
                    w_max = float(w.max().item())
                else:
                    w_mean = w_min = w_max = 1.0

                if t_ale is not None:
                    l_stealth_final = (ent_final - t_ale) ** 2
                    stealth_mean = float(l_stealth_final.mean().item())
                else:
                    stealth_mean = 0.0

                # logger.info(
                #     f"[T3-debug][pgd] B={z0.size(0)}, "
                #     f"ent_init={ent_init_mean:.4f}, "
                #     f"ent_final={ent_final_mean:.4f}, "
                #     f"stealth_mean={stealth_mean:.4f}, "
                #     f"w_mean={w_mean:.4f}, w_min={w_min:.4f}, w_max={w_max:.4f}, "
                #     f"lambda_stealth={self.lambda_stealth}, "
                #     f"lambda_epistemic={self.lambda_epistemic}, "
                #     f"lambda_align={self.lambda_align}"
                # )

        adv_logits = (z0 + delta).detach()
        return adv_logits

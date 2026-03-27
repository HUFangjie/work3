# attacks/t3_attack.py
"""
T3 attack: temperature-guided logit poisoning with
  - adaptive diagnosis (entropy-based hard sample selection),
  - TTA-based aleatoric/epistemic estimation,
  - variance-aware logit PGD optimization,
  - global distribution alignment regularization.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import time

import torch
import torch.nn as nn

from attacks.base_attack import BaseAttack
from attacks.t3_diagnosis import AdaptiveDiagnosis
from attacks.t3_tta_uncertainty import TTAUncertaintyEstimator
from attacks.t3_variance_opt import VarianceAwareLogitOptimizer
from attacks.t3_global_align import GlobalAligner
from attacks.utils import compute_entropy
import logging

logger = logging.getLogger()



class T3Attack(BaseAttack):
    def __init__(
        self,
        is_malicious: bool,
        cfg: Optional[Dict[str, Any]] = None,
        client_id: Optional[int] = None,
        model: Optional[nn.Module] = None,
        dataset_name: str = "fmnist",
    ) -> None:
        super().__init__(
            is_malicious=is_malicious,
            cfg=cfg,
            client_id=client_id,
            model=model,
        )

        self.dataset_name = dataset_name.lower()
        t3_cfg = (cfg or {}).get("t3", {})

        self.rho = float(t3_cfg.get("rho", 0.2))             # hard sample selection
        self.lambda_stealth = float(t3_cfg.get("lambda_stealth", 1.0))
        self.lambda_epistemic = float(t3_cfg.get("lambda_epistemic", 1.0))
        self.lambda_align = float(t3_cfg.get("lambda_align", 0.0))
        self.epsilon = float(t3_cfg.get("epsilon", 0.5))
        self.num_steps = int(t3_cfg.get("pgd_steps", 10))
        self.step_size = float(t3_cfg.get("pgd_step_size", 0.1))
        self.tta_type = t3_cfg.get("tta_type", "weak")
        self.num_tta = int(t3_cfg.get("tta_num_augments", 4))
        self.history_window = int(t3_cfg.get("history_window", 1024))

        self.debug: bool = bool(t3_cfg.get("debug", False))

        # 子模块：诊断
        self.diagnosis = AdaptiveDiagnosis(rho=self.rho)

        # 子模块：TTA 估计（如果 model 不存在，则退化为 None）
        if self.model is not None:
            self.tta_estimator = TTAUncertaintyEstimator(
                model=self.model,
                dataset_name=self.dataset_name,
                tta_type=self.tta_type,
                num_augments=self.num_tta,
                custom_transform=None,  # 可通过外部注入自定义 transform
            )
        else:
            self.tta_estimator = None

        # 子模块：方差感知优化
        self.optimizer = VarianceAwareLogitOptimizer(
            epsilon=self.epsilon,
            step_size=self.step_size,
            num_steps=self.num_steps,
            lambda_stealth=self.lambda_stealth,
            lambda_epistemic=self.lambda_epistemic,
            lambda_align=self.lambda_align,
            debug=self.debug,
        )

        # 子模块：全局分布对齐（维护参考熵分布）
        self.aligner = GlobalAligner(max_history=self.history_window)

    # ------------------------------------------------------------------ #
    # 主攻击逻辑
    # ------------------------------------------------------------------ #
    def attack_logits(
        self,
        x_public: torch.Tensor,
        logits: torch.Tensor,
        round_idx: Optional[int] = None,
        global_step: Optional[int] = None,
        y_public: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply T3 attack to the logits (for malicious clients).

        Benign clients: identity mapping.
        """
        if (not self.is_malicious) or (self.model is None):
            return logits

        device = logits.device
        x = x_public.to(device)
        base_logits = logits.to(device)

        # ------------------------------
        # Overhead accounting (per call)
        # ------------------------------
        t_call0 = time.perf_counter()
        t_diag_s = 0.0
        t_tta_s = 0.0
        t_pgd_s = 0.0
        hard_cnt = 0

        # 0) 更新全局参考分布：使用 base logits 的熵作为“准-benign”参考
        base_ent = compute_entropy(base_logits)  # [B]
        self.aligner.update(base_ent.detach())
        ref_dist = self.aligner.get_ref_distribution()  # 1D or None

        # 1) 自适应诊断：选 hard samples（高熵）
        t0 = time.perf_counter()
        diag_info = self.diagnosis.select_hard_samples(base_logits)
        t_diag_s += time.perf_counter() - t0
        mask = diag_info["mask"]
        tau_t = diag_info["threshold"]
        ent_all = diag_info["entropies"]
        hard_cnt = int(mask.sum().item())

        if self.debug:
            num_hard = int(mask.sum().item())
            mean_ent_all = float(ent_all.mean().item())
            mean_ent_hard = float(ent_all[mask].mean().item()) if num_hard > 0 else 0.0
            # logger.info(
            #     f"[T3-debug][diag] client={self.client_id}, round={round_idx}, "
            #     f"B={ent_all.numel()}, hard={num_hard}, "
            #     f"tau={float(tau_t.item()):.4f}, "
            #     f"ent_mean={mean_ent_all:.4f}, ent_hard_mean={mean_ent_hard:.4f}"
            # )

        if mask.sum() == 0:
            # 没有 hard samples，直接返回原始 logits
            self.last_overhead = {
                "round": int(round_idx) if round_idx is not None else -1,
                "client_id": int(self.client_id) if self.client_id is not None else -1,
                "t_total_s": float(time.perf_counter() - t_call0),
                "t_diag_s": float(t_diag_s),
                "t_tta_s": float(t_tta_s),
                "t_pgd_s": float(t_pgd_s),
                "hard_cnt": int(hard_cnt),
            }
            return logits

        hard_logits = base_logits[mask]  # [Bh,C]
        hard_x = x[mask]                 # [Bh,...]
        # if self.debug:
        #     logger.info(
        #         f"[T3-debug][hard_x] client={self.client_id}, round={round_idx}, "
        #         f"hard_shape={tuple(hard_x.shape)}"
        #     )

        # 2) TTA 不确定性估计：得到 H_aleatoric 和 variance（只在 hard subset 上）
        if self.tta_estimator is not None:
            t0 = time.perf_counter()
            H_ale_hard, var_hard = self.tta_estimator.estimate_aleatoric_and_variance(
                hard_x
            )  # [Bh], [Bh]
            t_tta_s += time.perf_counter() - t0
            if self.debug:
                H_mean = float(H_ale_hard.mean().item())
                H_std = float(H_ale_hard.std().item())
                v_mean = float(var_hard.mean().item())
                v_min = float(var_hard.min().item())
                v_max = float(var_hard.max().item())
                # logger.info(
                #     f"[T3-debug][tta] client={self.client_id}, round={round_idx}, "
                #     f"H_ale_mean={H_mean:.4f}, H_ale_std={H_std:.4f}, "
                #     f"var_mean={v_mean:.6f}, var_min={v_min:.6f}, var_max={v_max:.6f}"
                # )
        else:
            H_ale_hard = None
            var_hard = None

        # 3) variance-aware PGD：在 hard logits 上做对抗优化
        t0 = time.perf_counter()
        adv_hard_logits = self.optimizer.optimize(
            base_logits=hard_logits,
            variance_weights=var_hard,
            target_aleatoric_entropy=H_ale_hard,
            ref_entropies=ref_dist,
        )
        t_pgd_s += time.perf_counter() - t0
        # logger.info(f"adv_hard_logits:{adv_hard_logits}")

        # 4) 把 hard 样本的 adv logits 写回到整 batch 中
        adv_logits_all = base_logits.clone()
        adv_logits_all[mask] = adv_hard_logits

        # --- 记录一次 PGD 前后熵的变化（仅恶意客户端会走到这里） ---
        try:
            with torch.no_grad():
                ent_before = compute_entropy(base_logits)  # [B]
                ent_after = compute_entropy(adv_logits_all)  # [B]
                mean_before = float(ent_before.mean().item())
                mean_after = float(ent_after.mean().item())
                delta = mean_after - mean_before

                # hard 子集上的熵变化
                hard_before = ent_before[mask]
                hard_after = ent_after[mask]
                mean_before_h = float(hard_before.mean().item())
                mean_after_h = float(hard_after.mean().item())
                delta_h = mean_after_h - mean_before_h

            cid = getattr(self, "client_id", None)
            # logger.info(
            #     f"[T3] client={cid}, round={round_idx} | "
            #     f"entropy_before={mean_before:.4f}, "
            #     f"entropy_after={mean_after:.4f}, "
            #     f"delta={delta:.4f}"
            # )

            # if self.debug:
            #     logger.info(
            #         f"[T3-debug][entropy_hard] client={cid}, round={round_idx} | "
            #         f"hard_before={mean_before_h:.4f}, "
            #         f"hard_after={mean_after_h:.4f}, "
            #         f"hard_delta={delta_h:.4f}"
            #     )
        except Exception as e:
            print(f"[T3] entropy logging failed: {e}")



        self.last_overhead = {
            "round": int(round_idx) if round_idx is not None else -1,
            "client_id": int(self.client_id) if self.client_id is not None else -1,
            "t_total_s": float(time.perf_counter() - t_call0),
            "t_diag_s": float(t_diag_s),
            "t_tta_s": float(t_tta_s),
            "t_pgd_s": float(t_pgd_s),
            "hard_cnt": int(hard_cnt),
        }
        return adv_logits_all

# attacks/__init__.py
"""
Attack factory: create Attack objects for each client based on config.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from attacks.base_attack import BaseAttack
from attacks.gaussian_logit_attack import GaussianLogitAttack
from attacks.label_flip_attack import LabelFlipAttack
from attacks.t3_attack import T3Attack
from attacks.topk_attack import TopKLogitAttack
from attacks.impersonation_attack import ImpersonationAttack
from attacks.naive_sharpening_attack import NaiveSharpeningAttack
from attacks.manipulating_kd_attack import ManipulatingKDAttack
from attacks.fed_ace_attack import FedACEAttack
from attacks.fed_oca_attack import FedOCAAttack


def create_attack(
    attack_config: Dict[str, Any],
    client_id: int,
    is_malicious: bool,
    model: Optional[nn.Module] = None,
    dataset_name: str = "fmnist",
) -> BaseAttack:
    """
    Create an attack object for a given client.

    Args:
        attack_config: config["attack_config"] dict.
        client_id: integer id of the client.
        is_malicious: whether this client is adversarial.
        model: client's model (for TTA-based attacks like T3).
        dataset_name: name of dataset (for choosing TTA transforms, etc.).

    Returns:
        A BaseAttack subclass instance.
    """
    enabled = attack_config.get("enabled", False)
    name = attack_config.get("name", "none").lower()

    if (not enabled) or name == "none":
        return BaseAttack(
            is_malicious=False,
            cfg=attack_config,
            client_id=client_id,
            model=model,
        )

    if name == "gaussian":
        return GaussianLogitAttack(
            is_malicious=is_malicious,
            cfg=attack_config,
            client_id=client_id,
            model=model,
        )
    elif name == "label_flip":
        return LabelFlipAttack(
            is_malicious=is_malicious,
            cfg=attack_config,
            client_id=client_id,
            model=model,
        )
    elif name == "t3":
        return T3Attack(
            is_malicious=is_malicious,
            cfg=attack_config,
            client_id=client_id,
            model=model,
            dataset_name=dataset_name,
        )

    elif name == "topk":
        return TopKLogitAttack(
        is_malicious=is_malicious,
        cfg=attack_config,
        client_id=client_id,
        model=model,
    )

    elif name in ("impersonation", "impersonate"):
        return ImpersonationAttack(
            is_malicious=is_malicious,
            cfg=attack_config,
            client_id=client_id,
            model=model,
        )

    else:
        # 未知攻击类型，fallback 到最安全的恒等映射
        return BaseAttack(
            is_malicious=is_malicious,
            cfg=attack_config,
            client_id=client_id,
            model=model,
        )

    if name == "naive_sharpening":
        return NaiveSharpeningAttack(is_malicious=is_malicious, cfg=cfg, client_id=client_id, model=model)

    if name == "manipulating_kd":
        return ManipulatingKDAttack(is_malicious=is_malicious, cfg=cfg, client_id=client_id, model=model)

    if name == "fed_ace":
        return FedACEAttack(is_malicious=is_malicious, cfg=cfg, client_id=client_id, model=model)

    if name == "fed_oca":
        return FedOCAAttack(is_malicious=is_malicious, cfg=cfg, client_id=client_id, model=model)


